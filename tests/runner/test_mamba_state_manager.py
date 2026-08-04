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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vllm.v1.kv_cache_interface import MambaSpec

import tpu_inference.runner.mamba_state_manager as mamba_state_manager_module
from tpu_inference.runner.mamba_state_manager import MambaStateManager
from tpu_inference.runner.utils import ForbidCompile


class TestMambaStateManager:

    @pytest.fixture
    def manager(self):
        runner = MagicMock()
        runner.cache_config.enable_prefix_caching = True
        runner.cache_config.mamba_cache_mode = "align"
        runner.kv_cache_manager.uses_compact_mamba_state = False
        runner.speculative_config = None
        runner.max_num_reqs = 4
        runner.dp_size = 1
        runner.mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]),
                                        ("data", ))

        mamba_spec = MagicMock(spec=MambaSpec)
        mamba_spec.block_size = 4
        mamba_spec.num_speculative_blocks = 0
        attention_group = MagicMock()
        attention_group.kv_cache_spec = MagicMock()
        attention_group.layer_names = ["attention.0"]
        mamba_group = MagicMock()
        mamba_group.kv_cache_spec = mamba_spec
        mamba_group.layer_names = ["gdn.0", "gdn.1"]
        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_groups = [attention_group, mamba_group]

        request = MagicMock()
        request.num_computed_tokens = 8
        request.block_ids = ([0, 1, 2], [2, 3, 4])
        runner.requests = {"request-0": request}
        runner.input_batch.req_ids = ["request-0"]
        runner.input_batch.num_reqs = 1
        runner.layer_name_to_kvcache_index = {"gdn.0": 0, "gdn.1": 1}

        conv_state = jnp.arange(16, dtype=jnp.float32).astype(
            jnp.bfloat16).reshape(8, 2)
        recurrent_state = jnp.arange(32, dtype=jnp.float32).reshape(8, 2, 2)
        second_conv_state = (jnp.arange(24, dtype=jnp.float32) + 100).astype(
            jnp.bfloat16).reshape(8, 3)
        second_recurrent_state = (jnp.arange(48, dtype=jnp.float32) +
                                  200).reshape(8, 2, 3)

        def commit_state(state):
            sharding = jax.sharding.NamedSharding(
                runner.mesh,
                jax.sharding.PartitionSpec("data",
                                           *([None] * (state.ndim - 1))),
            )
            return jax.device_put(state, sharding)

        runner.kv_caches = [
            (commit_state(conv_state), commit_state(recurrent_state)),
            (commit_state(second_conv_state),
             commit_state(second_recurrent_state)),
        ]

        manager = MambaStateManager(runner)
        manager.initialize(kv_cache_config)
        return manager

    @staticmethod
    def _add_mamba_group(manager, layer_name, cache_idx):
        mamba_spec = MagicMock(spec=MambaSpec)
        mamba_spec.block_size = 4
        mamba_spec.num_speculative_blocks = 0
        mamba_group = MagicMock()
        mamba_group.kv_cache_spec = mamba_spec
        mamba_group.layer_names = [layer_name]
        gid = len(manager.kv_cache_config.kv_cache_groups)
        manager.kv_cache_config.kv_cache_groups.append(mamba_group)
        manager.mamba_groups[gid] = mamba_spec
        manager.current_state_block_ids[gid] = {}
        manager.runner.layer_name_to_kvcache_index[layer_name] = cache_idx
        return gid

    def test_preprocess_restores_all_states_in_one_jit(self, manager,
                                                       monkeypatch):
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = set()
        scheduler_output.num_scheduled_tokens = {"request-0": 1}
        scheduler_output.assigned_dp_rank = {"request-0": 0}

        old_caches = tuple(
            tuple(np.asarray(state).copy() for state in states)
            for states in manager.runner.kv_caches)
        old_formats = tuple(
            tuple(state.format for state in states)
            for states in manager.runner.kv_caches)
        copy_calls = 0
        copy_state_blocks = mamba_state_manager_module._copy_state_blocks

        def record_copy(*args, **kwargs):
            nonlocal copy_calls
            copy_calls += 1
            return copy_state_blocks(*args, **kwargs)

        monkeypatch.setattr(mamba_state_manager_module, "_copy_state_blocks",
                            record_copy)
        manager.preprocess(scheduler_output)

        # Eight cached tokens end at logical block 1 (physical block 3).
        # Scheduling token nine moves the running state to logical block 2
        # (physical block 4), so every GDN state must be restored there by one
        # compiled call. Padded 0->0 copies leave the null block unchanged.
        assert copy_calls == 1
        for new_states, old_states, expected_formats in zip(
                manager.runner.kv_caches, old_caches, old_formats,
                strict=True):
            for new_state, old_state, expected_format in zip(new_states,
                                                             old_states,
                                                             expected_formats,
                                                             strict=True):
                np.testing.assert_array_equal(new_state[4], old_state[3])
                np.testing.assert_array_equal(new_state[0], old_state[0])
                np.testing.assert_array_equal(new_state[5], old_state[5])
                assert new_state.format == expected_format
        assert manager.get_current_state_block_id(1, "request-0") == 4
        assert manager.mamba_state_idx["request-0"] == 2

    @pytest.mark.parametrize(
        ("max_num_reqs", "expected"),
        [
            (1, (1, )),
            (6, (1, 2, 4, 6)),
            (8, (1, 2, 4, 8)),
        ],
    )
    def test_copy_bucket_sizes_include_exact_upper_bound(
            self, max_num_reqs, expected):
        assert (mamba_state_manager_module._get_copy_bucket_sizes(max_num_reqs)
                == expected)

    @pytest.mark.parametrize(
        ("num_copies", "max_num_reqs", "expected"),
        [
            (1, 64, 1),
            (2, 64, 2),
            (3, 64, 4),
            (33, 64, 64),
            (5, 6, 6),
        ],
    )
    def test_copy_bucket_selection_rounds_up(self, num_copies, max_num_reqs,
                                             expected):
        assert (mamba_state_manager_module._get_copy_bucket_size(
            num_copies, max_num_reqs) == expected)

    def test_precompile_copy_state_blocks_warms_all_runtime_buckets(
            self, manager):
        old_caches = tuple(
            tuple(np.asarray(state).copy() for state in states)
            for states in manager.runner.kv_caches)

        compiled_states = manager.precompile_copy_state_blocks()

        assert compiled_states is not None
        for new_states, old_states in zip(manager.runner.kv_caches,
                                          old_caches,
                                          strict=True):
            for new_state, old_state in zip(new_states,
                                            old_states,
                                            strict=True):
                np.testing.assert_array_equal(new_state, old_state)

        with ForbidCompile():
            for num_copies in range(1, manager.runner.max_num_reqs + 1):
                copies = [(1, dst, 0) for dst in range(1, num_copies + 1)]
                manager._apply_copies({1: copies})

        for new_states, old_states in zip(manager.runner.kv_caches,
                                          old_caches,
                                          strict=True):
            for new_state, old_state in zip(new_states,
                                            old_states,
                                            strict=True):
                np.testing.assert_array_equal(new_state[4], old_state[1])

    def test_copy_uses_one_bucket_across_all_groups(self, manager):
        second_gid = self._add_mamba_group(manager, "gdn.2", 2)
        manager.runner.kv_caches.append(
            tuple(state + 1000 for state in manager.runner.kv_caches[0]))
        index_shapes = []

        def record_copy_shapes(states_by_group, src_ids_by_group,
                               dst_ids_by_group):
            index_shapes.append((
                tuple(ids.shape[0] for ids in src_ids_by_group),
                tuple(ids.shape[0] for ids in dst_ids_by_group),
            ))
            return states_by_group

        manager._copy_state_blocks_fn = record_copy_shapes
        manager._apply_copies({
            1: [(1, 2, 0), (1, 3, 0), (1, 4, 0)],
            second_gid: [(1, 2, 0)],
        })

        assert index_shapes == [((4, 4), (4, 4))]

    def test_preprocess_keeps_current_block_without_copy(
            self, manager, monkeypatch):
        manager.runner.requests["request-0"].num_computed_tokens = 9
        manager.mamba_state_idx["request-0"] = 2
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = set()
        scheduler_output.num_scheduled_tokens = {"request-0": 1}
        scheduler_output.assigned_dp_rank = {"request-0": 0}
        old_states = manager.runner.kv_caches[0]
        copy_state_blocks = MagicMock()
        monkeypatch.setattr(mamba_state_manager_module, "_copy_state_blocks",
                            copy_state_blocks)

        manager.preprocess(scheduler_output)

        copy_state_blocks.assert_not_called()
        for actual, expected in zip(manager.runner.kv_caches[0], old_states):
            np.testing.assert_array_equal(actual, expected)

    def test_preprocess_tracks_group_local_current_blocks(self, manager):
        self._add_mamba_group(manager, "gdn.2", 2)
        third_states = tuple(state + 1000
                             for state in manager.runner.kv_caches[0])
        expected_third_states = tuple(
            np.asarray(state).copy() for state in third_states)
        manager.runner.kv_caches.append(third_states)
        manager.runner.requests["request-0"].block_ids = (
            [0, 1, 2],
            [2, 3, 4],
            [5, 6, 7],
        )
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = set()
        scheduler_output.num_scheduled_tokens = {"request-0": 1}
        scheduler_output.assigned_dp_rank = {"request-0": 0}

        manager.preprocess(scheduler_output)

        assert manager.get_current_state_block_id(1, "request-0") == 4
        assert manager.get_current_state_block_id(2, "request-0") == 7
        for actual, expected in zip(manager.runner.kv_caches[2],
                                    expected_third_states,
                                    strict=True):
            np.testing.assert_array_equal(actual[7], expected[6])

    def test_preprocess_rejects_null_current_block(self, manager):
        manager.runner.requests["request-0"].block_ids = ([0, 1, 2], [2, 3, 0])
        manager.runner.requests["request-0"].num_computed_tokens = 9
        manager.mamba_state_idx["request-0"] = 2
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = set()
        scheduler_output.num_scheduled_tokens = {"request-0": 1}
        scheduler_output.assigned_dp_rank = {"request-0": 0}

        with pytest.raises(RuntimeError, match="reserved null block"):
            manager.preprocess(scheduler_output)

    def test_preprocess_rejects_missing_layer_mapping(self, manager):
        manager.runner.layer_name_to_kvcache_index.clear()
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = set()
        scheduler_output.num_scheduled_tokens = {"request-0": 1}
        scheduler_output.assigned_dp_rank = {"request-0": 0}

        with pytest.raises(RuntimeError, match="missing KV-cache mappings"):
            manager.preprocess(scheduler_output)

    def test_preprocess_rejects_shared_source_and_running_block(self, manager):
        manager.runner.requests["request-0"].block_ids = ([0, 1, 2], [2, 3, 3])
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = set()
        scheduler_output.num_scheduled_tokens = {"request-0": 1}
        scheduler_output.assigned_dp_rank = {"request-0": 0}

        with pytest.raises(RuntimeError, match="same physical block 3"):
            manager.preprocess(scheduler_output)

    def test_postprocess_rejects_shared_running_and_snapshot_block(
            self, manager):
        manager.runner.requests["request-0"].block_ids = ([0, 1, 2], [2, 3, 3])
        manager.mamba_state_idx["request-0"] = 1
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {"request-0": 4}
        scheduler_output.assigned_dp_rank = {"request-0": 0}

        with pytest.raises(RuntimeError, match="same physical block 3"):
            manager.postprocess(scheduler_output)

    def test_copy_rejects_duplicate_destination_blocks(self, manager):
        with pytest.raises(RuntimeError, match="duplicate destination blocks"):
            manager._apply_copies({1: [(3, 4, 0), (3, 4, 0)]})

    def test_copy_keeps_inactive_group_in_fixed_jit_tree(
            self, manager, monkeypatch):
        inactive_gid = self._add_mamba_group(manager, "gdn.2", 2)
        inactive_states = (
            jnp.arange(32,
                       dtype=jnp.float32).astype(jnp.bfloat16).reshape(8, 4),
            jnp.arange(24, dtype=jnp.float32).reshape(8, 3),
        )
        manager.runner.kv_caches.append(inactive_states)
        expected_inactive_states = tuple(
            np.asarray(state).copy() for state in inactive_states)
        group_sizes = []
        copy_state_blocks = mamba_state_manager_module._copy_state_blocks

        def record_copy(states_by_group, *args, **kwargs):
            group_sizes.append(
                tuple(len(group_states) for group_states in states_by_group))
            return copy_state_blocks(states_by_group, *args, **kwargs)

        monkeypatch.setattr(mamba_state_manager_module, "_copy_state_blocks",
                            record_copy)
        manager._apply_copies({1: [(3, 4, 0)], inactive_gid: []})

        assert group_sizes == [(2, 1)]
        for actual, expected in zip(manager.runner.kv_caches[2],
                                    expected_inactive_states,
                                    strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_copy_rejects_cache_shared_across_groups(self, manager):
        duplicate_gid = self._add_mamba_group(manager, "gdn.2", 0)

        with pytest.raises(RuntimeError, match="duplicate cache indices: 0"):
            manager._apply_copies({1: [(3, 4, 0)], duplicate_gid: []})

    def test_finished_only_cycle_clears_request_bookkeeping(self, manager):
        manager.mamba_state_idx["request-0"] = 2
        manager.current_state_block_ids[1]["request-0"] = 4
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = set()
        scheduler_output.preempted_req_ids = set()
        scheduler_output.finished_req_ids = {"request-0"}

        manager.update_request_lifecycle(scheduler_output)

        assert "request-0" not in manager.mamba_state_idx
        assert "request-0" not in manager.current_state_block_ids[1]

    def test_lifecycle_accepts_list_resumed_ids_from_dp_scheduler(
            self, manager):
        manager.mamba_state_idx["request-0"] = 2
        manager.current_state_block_ids[1]["request-0"] = 4
        scheduler_output = MagicMock()
        scheduler_output.scheduled_cached_reqs.resumed_req_ids = ["request-0"]
        scheduler_output.preempted_req_ids = []
        scheduler_output.finished_req_ids = set()

        manager.update_request_lifecycle(scheduler_output)

        assert "request-0" not in manager.mamba_state_idx
        assert "request-0" not in manager.current_state_block_ids[1]

    def test_speculative_decoding_is_rejected(self, manager):
        manager.runner.speculative_config = MagicMock()

        with pytest.raises(NotImplementedError,
                           match="does not yet support speculative"):
            manager.initialize(manager.kv_cache_config)
