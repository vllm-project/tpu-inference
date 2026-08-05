# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from tpu_inference.core.sched.compact_mamba_scheduler import (
    _SelectedMambaPositionSchedulerMixin, _split_at_next_mamba_cached_position)


class TestSplitAtNextMambaCachedPosition:

    @pytest.fixture
    def req(self):
        return SimpleNamespace(num_computed_tokens=0)

    def test_unset_positions_preserve_native_chunk(self, req):
        assert _split_at_next_mamba_cached_position(req, 8192) == 8192

    def test_prefill_stops_at_first_crossed_position(self, req):
        assert _split_at_next_mamba_cached_position(
            req,
            8192,
            cached_positions=frozenset({4096, 6144}),
        ) == 4096

    def test_prefill_ending_at_position_is_unchanged(self, req):
        assert _split_at_next_mamba_cached_position(
            req,
            4096,
            cached_positions=frozenset({4096}),
        ) == 4096

    def test_prefill_before_next_position_is_unchanged(self, req):
        assert _split_at_next_mamba_cached_position(
            req,
            2048,
            cached_positions=frozenset({4096}),
        ) == 2048

    def test_prefill_uses_next_position_after_current_state(self, req):
        req.num_computed_tokens = 4096

        assert _split_at_next_mamba_cached_position(
            req,
            4096,
            cached_positions=frozenset({4096, 6144}),
        ) == 2048

    def test_new_prefix_hits_are_included_in_current_position(self, req):
        assert _split_at_next_mamba_cached_position(
            req,
            8192,
            num_new_local_computed_tokens=4096,
            cached_positions=frozenset({4096, 8192}),
        ) == 4096

    def test_nonpositive_native_chunk_is_unchanged(self, req):
        assert _split_at_next_mamba_cached_position(
            req,
            0,
            cached_positions=frozenset({4096}),
        ) == 0

    def test_position_inside_mm_placeholder_is_skipped(self, req):
        req.mm_features = [
            SimpleNamespace(
                mm_position=SimpleNamespace(offset=2048, length=4096))
        ]

        assert _split_at_next_mamba_cached_position(
            req,
            12288,
            cached_positions=frozenset({4096, 8192}),
        ) == 8192

    @pytest.mark.parametrize("position", [4096, 8192])
    def test_position_at_mm_placeholder_edge_is_retained(self, req, position):
        req.mm_features = [
            SimpleNamespace(
                mm_position=SimpleNamespace(offset=4096, length=4096))
        ]

        assert _split_at_next_mamba_cached_position(
            req,
            12288,
            cached_positions=frozenset({position}),
        ) == position


class TestSelectedMambaPositionSchedulerMixin:

    @pytest.mark.parametrize(
        ("num_computed_tokens", "cached_positions", "encoder_offset"),
        [
            pytest.param(0, frozenset({4096}), 4096, id="fresh-request"),
            pytest.param(
                4096,
                frozenset({4096, 8192}),
                8192,
                id="local-prefix-hit",
            ),
        ],
    )
    def test_encoder_scheduling_is_capped_before_future_input(
        self,
        num_computed_tokens,
        cached_positions,
        encoder_offset,
    ):

        class NativeScheduler:

            def __init__(self):
                self.native_calls = []

            def _try_schedule_encoder_inputs(
                self,
                request,
                num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens=0,
            ):
                self.native_calls.append((
                    request,
                    num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens,
                ))
                scheduled = [
                    i for i, feature in enumerate(request.mm_features)
                    if feature.mm_position.offset < num_computed_tokens +
                    num_new_tokens
                ]
                return scheduled, num_new_tokens, encoder_compute_budget, []

        class Scheduler(_SelectedMambaPositionSchedulerMixin, NativeScheduler):
            pass

        scheduler = Scheduler()
        scheduler._tpu_mamba_cached_positions = cached_positions
        request = SimpleNamespace(
            num_computed_tokens=0,
            mm_features=[
                SimpleNamespace(mm_position=SimpleNamespace(
                    offset=encoder_offset, length=1024))
            ],
        )

        result = scheduler._try_schedule_encoder_inputs(
            request,
            num_computed_tokens=num_computed_tokens,
            num_new_tokens=8192,
            encoder_compute_budget=2048,
        )

        assert result == ([], 4096, 2048, [])
        assert scheduler.native_calls == [(request, num_computed_tokens, 4096,
                                           2048, 0)]

    def test_native_alignment_runs_before_selected_position_split(self):

        class NativeScheduler:

            def __init__(self):
                self.native_calls = []

            def _mamba_block_aligned_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens=0,
                num_external_computed_tokens=0,
            ):
                self.native_calls.append((
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    num_external_computed_tokens,
                ))
                return 8192

        class Scheduler(_SelectedMambaPositionSchedulerMixin, NativeScheduler):
            pass

        scheduler = Scheduler()
        scheduler._tpu_mamba_cached_positions = frozenset({4096})
        request = SimpleNamespace(num_computed_tokens=0)

        assert scheduler._mamba_block_aligned_split(request, 16384) == 4096
        assert scheduler.native_calls == [(request, 16384, 0, 0)]
