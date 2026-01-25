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

from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from tpu_inference import envs
from tpu_inference.runner.profiler import (
    BatchCompositionStats, InferencePhase, PhasedBasedProfiler,
    determine_phase_from_batch_composition_stats, get_batch_composition_stats)


class MockInputBatch:

    def __init__(self, req_ids, num_computed_tokens_cpu):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = np.array(num_computed_tokens_cpu)


class MockSchedulerOutput:

    def __init__(self, num_scheduled_tokens):
        self.num_scheduled_tokens = num_scheduled_tokens


@pytest.mark.parametrize(
    "scenario, num_reqs, req_ids, computed, scheduled, expected_prefill, expected_decode",
    [
        ("prefill_only", 2, [101, 102], [0, 0], {
            101: 50,
            102: 100
        }, 150, 0),
        ("decode_only", 3, [201, 202, 203], [10, 20, 5], {
            201: 1,
            202: 1,
            203: 1
        }, 0, 3),
        ("mixed_batch", 4, [301, 302, 303, 304], [0, 10, 0, 20], {
            301: 100,
            302: 1,
            303: 50,
            304: 1
        }, 150, 2),
        ("chunked_prefill", 2, [401, 402], [50, 10], {
            401: 50,
            402: 1
        }, 50, 1),
    ])
def test_get_batch_composition_stats(scenario, num_reqs, req_ids, computed,
                                     scheduled, expected_prefill,
                                     expected_decode):
    """Tests get_batch_composition_stats for various scenarios."""
    input_batch = MockInputBatch(req_ids, computed)
    scheduler_output = MockSchedulerOutput(scheduled)
    total_tokens = sum(scheduled.values())

    stats = get_batch_composition_stats(
        input_batch=input_batch,
        total_num_scheduled_tokens=total_tokens,
        num_reqs=num_reqs,
        padded_total_num_scheduled_tokens=total_tokens + 8,
        scheduler_output=scheduler_output)

    assert stats.num_prefill_tokens == expected_prefill
    assert stats.num_decode_tokens == expected_decode
    assert stats.num_reqs == num_reqs
    assert stats.total_num_scheduled_tokens == total_tokens


@pytest.mark.parametrize("prefill_tokens, total_tokens, expected_phase", [
    (90, 100, InferencePhase.PREFILL_HEAVY),
    (89, 100, InferencePhase.NOT_DETERMINED),
    (15, 100, InferencePhase.DECODE_HEAVY),
    (50, 100, InferencePhase.BALANCED),
    (70, 100, InferencePhase.NOT_DETERMINED),
    (30, 100, InferencePhase.NOT_DETERMINED),
    (40, 100, InferencePhase.BALANCED),
    (50, 100, InferencePhase.BALANCED),
    (60, 100, InferencePhase.BALANCED),
    (100, 100, InferencePhase.PREFILL_HEAVY),
    (20, 100, InferencePhase.DECODE_HEAVY),
    (21, 100, InferencePhase.NOT_DETERMINED),
    (0, 100, InferencePhase.DECODE_HEAVY),
])
def test_determine_phase_from_batch_composition_stats(prefill_tokens,
                                                      total_tokens,
                                                      expected_phase):
    """Tests the phase determination logic based on prefill ratios."""
    stats = BatchCompositionStats(
        total_num_scheduled_tokens=total_tokens,
        num_prefill_tokens=prefill_tokens,
        num_decode_tokens=0,
        padded_total_num_scheduled_tokens=0,
        num_reqs=0,
    )
    phase = determine_phase_from_batch_composition_stats(stats)
    assert phase == expected_phase


@pytest.fixture
def profiler_fixture(tmp_path):
    """Fixture to set up a PhasedBasedProfiler with mocked dependencies."""
    target_module = "tpu_inference.runner.profiler"
    with patch(f"{target_module}.jax.profiler.start_trace") as mock_start, \
         patch(f"{target_module}.jax.profiler.stop_trace") as mock_stop, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch(f"{target_module}.datetime") as mock_datetime, \
         patch(f"{target_module}.InferencePhase", InferencePhase), \
         patch(f"{target_module}.determine_phase_from_batch_composition_stats") as mock_determine_phase:

        mock_now = MagicMock()
        mock_now.strftime.return_value = "2024_01_01_12_00_00"
        mock_datetime.datetime.now.return_value = mock_now

        profiler = PhasedBasedProfiler(profile_dir=str(tmp_path))
        profiler.num_steps_to_profile_for = envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

        yield {
            "profiler": profiler,
            "mock_start": mock_start,
            "mock_stop": mock_stop,
            "mock_file": mock_file,
            "mock_determine_phase": mock_determine_phase,
        }


def test_phased_profiler_full_cycle(profiler_fixture):
    """Tests a full start-step-stop profiling cycle for one phase."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_stop = profiler_fixture["mock_stop"]
    mock_file = profiler_fixture["mock_file"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    stats = BatchCompositionStats(
        total_num_scheduled_tokens=100,
        num_prefill_tokens=0,
        num_decode_tokens=0,
        padded_total_num_scheduled_tokens=0,
        num_reqs=2,
    )

    # 1. Start profiling on PREFILL_HEAVY phase
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    mock_start.assert_called_once()
    assert profiler.profiling_n_steps_left == envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR
    assert profiler.current_phase == "prefill_heavy"
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]
    assert mock_file().write.call_count == 1  # Wrote stats on start

    # 2. Step profiling (N-1 steps)
    for i in range(envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
        assert profiler.profiling_n_steps_left == envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1 - i
        mock_start.assert_called_once()  # Not called again
        mock_stop.assert_not_called()

    # 3. Final step stops profiling
    profiler.step(stats)
    mock_stop.assert_called_once()
    assert profiler.profiling_n_steps_left == 0
    assert profiler.current_phase == InferencePhase.NOT_DETERMINED
    assert mock_file(
    ).write.call_count == envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR + 1


def test_phased_profiler_ignores_initial_request(profiler_fixture):
    """Tests that profiling is not triggered for initial small requests."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY

    profiler.step(
        BatchCompositionStats(
            total_num_scheduled_tokens=1,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            padded_total_num_scheduled_tokens=0,
            num_reqs=1,
        ))
    mock_start.assert_not_called()

    profiler.step(
        BatchCompositionStats(
            total_num_scheduled_tokens=100,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            padded_total_num_scheduled_tokens=0,
            num_reqs=1,
        ))
    mock_start.assert_not_called()

    profiler.step(
        BatchCompositionStats(
            total_num_scheduled_tokens=1,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            padded_total_num_scheduled_tokens=0,
            num_reqs=2,
        ))
    mock_start.assert_not_called()

    profiler.step(
        BatchCompositionStats(
            total_num_scheduled_tokens=2,
            num_prefill_tokens=0,
            num_decode_tokens=0,
            padded_total_num_scheduled_tokens=0,
            num_reqs=2,
        ))
    mock_start.assert_called_once()


def test_phased_profiler_handles_all_phases(profiler_fixture):
    """Tests that the profiler can profile all defined phases sequentially."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_stop = profiler_fixture["mock_stop"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    stats = BatchCompositionStats(
        total_num_scheduled_tokens=100,
        num_prefill_tokens=0,
        num_decode_tokens=0,
        padded_total_num_scheduled_tokens=0,
        num_reqs=2,
    )
    phases_to_profile = [
        InferencePhase.PREFILL_HEAVY, InferencePhase.DECODE_HEAVY,
        InferencePhase.BALANCED
    ]

    for i, phase in enumerate(phases_to_profile):
        # Start profiling for the new phase
        mock_determine_phase.return_value = phase
        profiler.step(stats)
        assert mock_start.call_count == i + 1
        assert profiler.current_phase == phase.name.lower()
        assert profiler.inference_phase_seen[phase]

        # Step until profiling stops for this phase
        for _ in range(envs.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
            profiler.step(stats)

        assert mock_stop.call_count == i + 1
        assert profiler.current_phase == InferencePhase.NOT_DETERMINED

    # After all phases seen, should not start again
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    assert mock_start.call_count == len(phases_to_profile)
