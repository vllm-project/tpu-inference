# SPDX-License-Identifier: Apache-2.0
import io
import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.interpreters import pxla
from jax._src.pallas.utils import next_power_of_2

from tpu_inference.runner.utils import (
    PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR, ContinuousBatchStatsLogger,
    ForbidCompile, InferencePhase, LatencyTracker, PhasedBasedProfiler,
    determine_phase_from_batch_composition_stats, get_batch_composition_stats,
    get_padded_num_reqs_with_upper_limit, get_padded_token_len,
    get_req_paddings, get_token_paddings)


def test_min_token_size_alignment():
    """
    Simulates the logic in tpu_runner.py to verify that non-power-of-two 
    Data Parallel (DP) sizes are correctly aligned.
    """
    # Scenario 1: dp_size=5, kv_packing=1
    dp_size = 5
    kv_packing = 1
    min_token_size = max(16, next_power_of_2(dp_size * kv_packing))
    assert min_token_size == 16
    # Ensure it satisfies the downstream power-of-two assertion
    assert (min_token_size & (min_token_size - 1) == 0)

    # Scenario 2: dp_size=5, kv_packing=8
    dp_size = 5
    kv_packing = 8
    # raw_val = 5 * 8 = 40 -> next_p2 is 64 -> max(16, 64) = 64
    min_token_size = max(16, next_power_of_2(dp_size * kv_packing))

    assert min_token_size == 64
    assert (min_token_size & (min_token_size - 1) == 0)


def test_min_req_size_alignment():
    """
    Simulates the logic in tpu_runner.py to verify that non-power-of-two 
    Data Parallel (DP) sizes are correctly aligned.
    """
    # Simulate a scenario with DP=6 (non-power-of-two)
    dp_size = 6
    MIN_NUM_SEQS = 8

    min_num_reqs = max(MIN_NUM_SEQS, next_power_of_2(dp_size))

    # Assert that 6 correctly aligns to 8
    assert min_num_reqs == 8
    assert (min_num_reqs & (min_num_reqs - 1) == 0) and min_num_reqs > 0


def test_get_padded_num_reqs_with_upper_limit():
    """Tests the get_padded_num_reqs_with_upper_limit function."""
    # From utils.py, MIN_NUM_SEQS = 8
    assert get_padded_num_reqs_with_upper_limit(4, 128) == 8
    assert get_padded_num_reqs_with_upper_limit(8, 128) == 8
    assert get_padded_num_reqs_with_upper_limit(9, 128) == 16
    assert get_padded_num_reqs_with_upper_limit(16, 128) == 16
    assert get_padded_num_reqs_with_upper_limit(17, 128) == 32
    assert get_padded_num_reqs_with_upper_limit(100, 64) == 64
    assert get_padded_num_reqs_with_upper_limit(1, 128) == 8


def test_get_paddings():
    # Bucketed padding
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    expected_paddings = [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)

    # Bucketed padding with max_token_size not a power of two.
    max_token_size = 317
    expected_paddings = [16, 32, 64, 128, 192, 256, 320]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)
    assert actual_paddings == expected_paddings

    # Exponential padding.
    max_token_size, padding_gap = 1024, 0
    expected_paddings = [16, 32, 64, 128, 256, 512, 1024]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)
    assert actual_paddings == expected_paddings
    # Exponential padding with max_token_size not a power of two.
    max_token_size = 317
    expected_paddings = [16, 32, 64, 128, 256, 512]
    actual_paddings = get_token_paddings(min_token_size, max_token_size,
                                         padding_gap)
    assert actual_paddings == expected_paddings


def test_get_padded_token_len():
    min_token_size, max_token_size, padding_gap = 16, 512, 64
    paddings = get_token_paddings(min_token_size, max_token_size, padding_gap)
    assert get_padded_token_len(paddings, 1) == 16
    assert get_padded_token_len(paddings, 16) == 16
    assert get_padded_token_len(paddings, 20) == 32
    assert get_padded_token_len(paddings, 300) == 320
    assert get_padded_token_len(paddings, 512) == 512


def test_get_req_paddings():
    assert get_req_paddings(1, 32) == [8, 16, 32]
    assert get_req_paddings(8, 32) == [8, 16, 32]
    assert get_req_paddings(8, 36) == [8, 16, 32, 36]


def test_latency_tracker(caplog):
    """Tests the LatencyTracker context manager."""
    logger_name = "vllm.tpu_inference.runner.utils"
    logger = logging.getLogger(logger_name)

    original_level = logger.level
    original_propagate = logger.propagate

    # Create an in-memory stream to capture log output
    log_capture_string = io.StringIO()
    # Create a handler that writes to our in-memory stream
    capture_handler = logging.StreamHandler(log_capture_string)

    try:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(capture_handler)

        sleep_duration = 0.01
        with LatencyTracker("test_op") as tracker:
            time.sleep(sleep_duration)

        elapsed = tracker.end_time - tracker.start_time
        assert elapsed >= sleep_duration
        log_contents = log_capture_string.getvalue()

        assert "Latency for 'test_op'" in log_contents
        assert f"{elapsed:.3f} seconds" in log_contents

    finally:
        # --- IMPORTANT: Clean up and restore the logger's original state ---
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        logger.removeHandler(capture_handler)


# Define a fixture to clear the JAX cache before each test
@pytest.fixture(autouse=True)
def clear_jax_cache():
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.fixture
def jitted_function():
    """Defines a jitted function for testing."""

    @jax.jit
    def my_jitted_func(x):
        return x * 2

    return my_jitted_func


@pytest.fixture
def jnp_array_input():
    return jnp.ones((2, 3))


@pytest.fixture
def jnp_array_input_same_shape():
    return jnp.zeros((2, 3))


@pytest.fixture
def jnp_array_input_new():
    return jnp.ones((3, 3))


def test_forbid_compile_raises_error_on_first_call(jitted_function,
                                                   jnp_array_input):
    """Test that ForbidCompile raises an error when a compilation occurs."""
    with pytest.raises(RuntimeError, match="JAX compilation occurred"):
        with ForbidCompile():
            jitted_function(jnp_array_input)


def test_forbid_compile_succeeds_on_cached_call(jitted_function,
                                                jnp_array_input):
    """Test that ForbidCompile does not raise an error on a cached call."""
    # Warm up the cache
    jitted_function(jnp_array_input)
    with ForbidCompile():
        jitted_function(jnp_array_input)


def test_forbid_compile_restores_original_function():
    """Test that ForbidCompile restores the original JAX function after exit."""
    original_func = pxla._cached_lowering_to_hlo
    with ForbidCompile():
        pass
    assert pxla._cached_lowering_to_hlo is original_func


def test_forbid_compile_with_exception():
    """Test that ForbidCompile restores the original function even if an exception occurs."""
    original_func = pxla._cached_lowering_to_hlo
    with pytest.raises(ValueError, match="Test exception"):
        with ForbidCompile():
            raise ValueError("Test exception")
    assert pxla._cached_lowering_to_hlo is original_func


def test_forbid_compile_raises_on_new_shape(jitted_function, jnp_array_input,
                                            jnp_array_input_same_shape,
                                            jnp_array_input_new):
    """
    Tests that ForbidCompile raises a RuntimeError when a jitted function
    is called with an input shape that triggers a new compilation.
    """
    # Clear cache for a clean test state.
    pxla._cached_lowering_to_hlo.cache_clear()

    # Warm up the JIT cache with the SCALAR input.
    # This causes the first compilation and cache miss.
    jitted_function(jnp_array_input)
    misses_after_warmup = pxla._cached_lowering_to_hlo.cache_info().misses
    assert misses_after_warmup == 1

    # This call uses the same shape/dtype, so it should be a cache HIT.
    # No RuntimeError expected.
    with ForbidCompile():
        jitted_function(jnp_array_input_same_shape)
    assert pxla._cached_lowering_to_hlo.cache_info(
    ).misses == misses_after_warmup  # No new misses

    # Now, call with a VECTOR input. This has a different shape,
    # forcing a NEW compilation (cache MISS).
    # This *should* raise a RuntimeError within the ForbidCompile context.
    expected_error_message = "JAX compilation occurred but was forbidden in this context."
    with pytest.raises(RuntimeError, match=expected_error_message):
        with ForbidCompile(message=expected_error_message):
            jitted_function(jnp_array_input_new)


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

    assert stats["num_prefill_tokens"] == expected_prefill
    assert stats["num_decode_tokens"] == expected_decode
    assert stats["num_reqs"] == num_reqs
    assert stats["total_num_scheduled_tokens"] == total_tokens


@pytest.mark.parametrize("prefill_tokens, total_tokens, expected_phase", [
    (90, 100, InferencePhase.PREFILL_HEAVY),
    (89, 100, InferencePhase.AMBIGUOUS),
    (15, 100, InferencePhase.DECODE_HEAVY),
    (50, 100, InferencePhase.BALANCED),
    (70, 100, InferencePhase.AMBIGUOUS),
    (30, 100, InferencePhase.AMBIGUOUS),
    (40, 100, InferencePhase.BALANCED),
    (50, 100, InferencePhase.BALANCED),
    (60, 100, InferencePhase.BALANCED),
    (100, 100, InferencePhase.PREFILL_ONLY),
    (20, 100, InferencePhase.DECODE_HEAVY),
    (21, 100, InferencePhase.AMBIGUOUS),
    (0, 100, InferencePhase.DECODE_ONLY),
])
def test_determine_phase_from_batch_composition_stats(prefill_tokens,
                                                      total_tokens,
                                                      expected_phase):
    """Tests the phase determination logic based on prefill ratios."""
    stats = {
        "num_prefill_tokens": prefill_tokens,
        "total_num_scheduled_tokens": total_tokens
    }
    phase = determine_phase_from_batch_composition_stats(stats)
    assert phase == expected_phase


@pytest.fixture
def continuous_logger_fixture(tmp_path):
    """Fixture to mock dependencies for ContinuousBatchStatsLogger."""
    target_module = "tpu_inference.runner.utils"
    with patch(f"{target_module}.datetime") as mock_datetime, \
         patch(f"{target_module}.atexit") as mock_atexit, \
         patch(f"{target_module}.subprocess.run") as mock_subprocess_run, \
         patch(f"{target_module}.tempfile.gettempdir", return_value=str(tmp_path)):

        mock_now = MagicMock()
        mock_now.strftime.return_value = "2025_01_01_12_00_00"
        mock_datetime.datetime.now.return_value = mock_now

        yield {
            "mock_datetime": mock_datetime,
            "mock_atexit": mock_atexit,
            "mock_subprocess_run": mock_subprocess_run,
            "tmp_path": tmp_path,
        }


def test_continuous_logger_initialization_local_path(continuous_logger_fixture):
    """Test logger initialization with a local directory."""
    tmp_path = continuous_logger_fixture["tmp_path"]
    profile_dir = tmp_path / "profiles"
    logger = ContinuousBatchStatsLogger(profile_dir=str(profile_dir))

    expected_filename = "all_batches_stats_2025_01_01_12_00_00.jsonl"
    expected_path = profile_dir / expected_filename

    assert logger.profile_dir == str(profile_dir)
    assert logger.local_temp_file == str(expected_path)
    assert logger.target_file == str(expected_path)
    assert profile_dir.exists()
    assert expected_path.exists()
    assert expected_path.read_text() == ""

    continuous_logger_fixture["mock_atexit"].register.assert_called_once_with(
        logger.close)


def test_continuous_logger_initialization_gcs_path(continuous_logger_fixture):
    """Test logger initialization with a GCS directory."""
    tmp_path = continuous_logger_fixture["tmp_path"]
    profile_dir = "gs://my-bucket/profiles"
    logger = ContinuousBatchStatsLogger(profile_dir=profile_dir)

    expected_filename = "all_batches_stats_2025_01_01_12_00_00.jsonl"
    expected_local_path = tmp_path / expected_filename
    expected_target_path = f"gs://my-bucket/profiles/{expected_filename}"

    assert logger.profile_dir == profile_dir
    assert logger.local_temp_file == str(expected_local_path)
    assert logger.target_file == expected_target_path
    assert expected_local_path.exists()
    assert expected_local_path.read_text() == ""

    continuous_logger_fixture["mock_atexit"].register.assert_called_once_with(
        logger.close)


def test_continuous_logger_log_single_entry(continuous_logger_fixture):
    """Test logging a single statistics dictionary."""
    tmp_path = continuous_logger_fixture["tmp_path"]
    profile_dir = tmp_path / "logs"
    logger = ContinuousBatchStatsLogger(profile_dir=str(profile_dir))

    stats = {"batch_no": 1, "tokens": 100}
    logger.log(stats)

    expected_json = '{"batch_no": 1, "tokens": 100}\n'
    local_file_path = Path(logger.local_temp_file)
    assert local_file_path.read_text() == expected_json


def test_continuous_logger_auto_flush(continuous_logger_fixture):
    """Test that flush is called automatically after flush_interval."""
    profile_dir = "gs://my-bucket/logs"
    flush_interval = 3
    logger = ContinuousBatchStatsLogger(profile_dir=profile_dir,
                                        flush_interval=flush_interval)
    mock_subprocess_run = continuous_logger_fixture["mock_subprocess_run"]

    for i in range(flush_interval - 1):
        logger.log({"step": i})
    mock_subprocess_run.assert_not_called()

    logger.log({"step": flush_interval - 1})
    mock_subprocess_run.assert_called_once()


def test_continuous_logger_close_flushes_and_cleans_up_gcs(continuous_logger_fixture):
    """Test that close() flushes and removes the local temp file for GCS."""
    profile_dir = "gs://my-bucket/logs"
    logger = ContinuousBatchStatsLogger(profile_dir=profile_dir)
    mock_subprocess_run = continuous_logger_fixture["mock_subprocess_run"]
    logger.log({"step": 1})
    local_file_path = Path(logger.local_temp_file)

    with patch("os.remove") as mock_os_remove:
        logger.close()
        mock_subprocess_run.assert_called_once()
        mock_os_remove.assert_called_once_with(str(local_file_path))


@pytest.fixture
def profiler_fixture(tmp_path):
    """Fixture to set up a PhasedBasedProfiler with mocked dependencies."""
    target_module = "tpu_inference.runner.utils"
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
        profiler.num_steps_to_profile_for = PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

        yield {
            "profiler": profiler,
            "mock_start": mock_start,
            "mock_stop": mock_stop,
            "mock_file": mock_file,
            "mock_determine_phase": mock_determine_phase,
        }


def test_phased_profiler_initializes_continuous_logger(tmp_path):
    """Tests that PhasedBasedProfiler initializes ContinuousBatchStatsLogger."""
    with patch("tpu_inference.runner.utils.envs.ENABLE_CONTINUOUS_BATCH_LOGGER", True), \
         patch("tpu_inference.runner.utils.ContinuousBatchStatsLogger") as mock_logger_cls:
        # Test with worker_rank = 0
        PhasedBasedProfiler(profile_dir=str(tmp_path),
                            worker_rank=0,
                            flush_interval=50)
        mock_logger_cls.assert_called_once_with(str(tmp_path), 50)

        # Test with worker_rank != 0
        mock_logger_cls.reset_mock()
        PhasedBasedProfiler(profile_dir=str(tmp_path), worker_rank=1)
        mock_logger_cls.assert_not_called()

    with patch("tpu_inference.runner.utils.envs.ENABLE_CONTINUOUS_BATCH_LOGGER", False), \
         patch("tpu_inference.runner.utils.ContinuousBatchStatsLogger") as mock_logger_cls:
        # Test with continuous logging disabled
        PhasedBasedProfiler(profile_dir=str(tmp_path), worker_rank=0)
        mock_logger_cls.assert_not_called()


def test_phased_profiler_step_calls_continuous_logger(profiler_fixture):
    """Tests that profiler.step() calls continuous_logger.log()."""
    profiler = profiler_fixture["profiler"]
    profiler.continuous_logger = MagicMock()
    stats = {"batch_no": 1, "tokens": 100}
    profiler.step(stats)
    profiler.continuous_logger.log.assert_called_once_with(stats)


def test_phased_profiler_full_cycle(profiler_fixture):
    """Tests a full start-step-stop profiling cycle for one phase."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_stop = profiler_fixture["mock_stop"]
    mock_file = profiler_fixture["mock_file"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}

    # 1. Start profiling on PREFILL_HEAVY phase
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    mock_start.assert_called_once()
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR
    assert profiler.current_phase == "prefill_heavy"
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]
    assert mock_file().write.call_count == 1  # Wrote stats on start

    # 2. Step profiling (N-1 steps)
    for i in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1):
        profiler.step(stats)
        assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR - 1 - i
        mock_start.assert_called_once()  # Not called again
        mock_stop.assert_not_called()

    # 3. Final step stops profiling
    profiler.step(stats)
    mock_stop.assert_called_once()
    assert profiler.profiling_n_steps_left == 0
    assert profiler.current_phase == ""
    assert mock_file(
    ).write.call_count == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR + 1


def test_phased_profiler_ignores_initial_request(profiler_fixture):
    """Tests that profiling is not triggered for initial small requests."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY

    profiler.step({"num_reqs": 1, "total_num_scheduled_tokens": 1})
    mock_start.assert_not_called()

    profiler.step({"num_reqs": 1, "total_num_scheduled_tokens": 100})
    mock_start.assert_not_called()

    profiler.step({"num_reqs": 2, "total_num_scheduled_tokens": 1})
    mock_start.assert_not_called()

    profiler.step({"num_reqs": 2, "total_num_scheduled_tokens": 2})
    mock_start.assert_called_once()


def test_phased_profiler_handles_all_phases(profiler_fixture):
    """Tests that the profiler can profile all defined phases sequentially."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_stop = profiler_fixture["mock_stop"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}
    phases_to_profile = [
        InferencePhase.PREFILL_ONLY, InferencePhase.PREFILL_HEAVY,
        InferencePhase.DECODE_ONLY, InferencePhase.DECODE_HEAVY,
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
        for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
            profiler.step(stats)

        assert mock_stop.call_count == i + 1
        assert profiler.current_phase == ""

    # After all phases seen, should not start again
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    assert mock_start.call_count == len(phases_to_profile)


def test_phased_profiler_skips_decode_steps_before_profiling(profiler_fixture):
    """Tests that the profiler skips N decode-heavy steps before profiling."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_stop = profiler_fixture["mock_stop"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    num_steps_to_skip = 3
    profiler.num_decode_steps_to_skip = num_steps_to_skip

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}
    mock_determine_phase.return_value = InferencePhase.DECODE_HEAVY

    # Each of these steps should be skipped (no profiling started)
    for i in range(num_steps_to_skip):
        profiler.step(stats)
        assert profiler.decode_steps_skipped == i + 1
        mock_start.assert_not_called()
        assert not profiler.inference_phase_seen[InferencePhase.DECODE_HEAVY]

    # The next step should actually start profiling
    profiler.step(stats)
    mock_start.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.DECODE_HEAVY]
    assert profiler.current_phase == "decode_heavy"
    assert profiler.profiling_n_steps_left == PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR

    # Complete the profiling cycle
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
        profiler.step(stats)
    mock_stop.assert_called_once()
    assert profiler.current_phase == ""


def test_phased_profiler_skip_only_affects_decode_heavy(profiler_fixture):
    """Tests that the skip logic only applies to the DECODE_HEAVY phase."""
    profiler = profiler_fixture["profiler"]
    mock_start = profiler_fixture["mock_start"]
    mock_stop = profiler_fixture["mock_stop"]
    mock_determine_phase = profiler_fixture["mock_determine_phase"]

    profiler.num_decode_steps_to_skip = 5  # Large skip count

    stats = {"num_reqs": 2, "total_num_scheduled_tokens": 100}

    # PREFILL_HEAVY should start profiling immediately (no skipping)
    mock_determine_phase.return_value = InferencePhase.PREFILL_HEAVY
    profiler.step(stats)
    mock_start.assert_called_once()
    assert profiler.inference_phase_seen[InferencePhase.PREFILL_HEAVY]
    assert profiler.decode_steps_skipped == 0  # Not incremented

    # Complete the PREFILL_HEAVY profiling cycle
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
        profiler.step(stats)
    mock_stop.assert_called_once()

    # BALANCED should also start immediately (no skipping)
    mock_determine_phase.return_value = InferencePhase.BALANCED
    profiler.step(stats)
    assert mock_start.call_count == 2
    assert profiler.inference_phase_seen[InferencePhase.BALANCED]
    assert profiler.decode_steps_skipped == 0  # Still not incremented

    # Complete the BALANCED profiling cycle
    for _ in range(PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR):
        profiler.step(stats)
    assert mock_stop.call_count == 2

    # DECODE_HEAVY should be skipped
    mock_determine_phase.return_value = InferencePhase.DECODE_HEAVY
    profiler.step(stats)
    assert mock_start.call_count == 2  # Not started yet
    assert profiler.decode_steps_skipped == 1
