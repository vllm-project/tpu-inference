"""
Benchmark script for _prepare_inputs_dp based on test_tpu_runner_dp.py structure.

This script benchmarks the DECODE PHASE with:
- 4096 requests with DP=4
- 1 token scheduled per request (decode phase)
- Varying context lengths (computed tokens) between 1024-4096 per request

This simulates the high-load decode scenario where the CPU input preparation
can become a bottleneck with many concurrent requests.

Run with: python benchmark_prepare_inputs_from_test.py
"""

import time
import numpy as np
from unittest.mock import MagicMock, patch

import jax
from tpu_inference.runner.tpu_runner import TPUModelRunner, AsyncPreResults


def create_mock_runner(num_reqs=4096, dp_size=4, tp_size=2, max_tokens=8192):
    """Create a mock runner similar to test setup."""
    runner = MagicMock()
    
    # Devices
    devices = jax.devices()
    num_devices_needed = dp_size * tp_size
    num_devices = min(len(devices), num_devices_needed)
        
    print(f"  Creating mesh with {num_devices} devices (DP={dp_size}, TP={tp_size})")
    
    # Mesh
    runner.mesh = jax.sharding.Mesh(
        np.array(devices[:num_devices]).reshape(dp_size, tp_size),
        ('data', 'model')
    )
    print(f"  Mesh shape: {runner.mesh.devices.shape}, axis names: {runner.mesh.axis_names}")
    
    # Model Runner
    runner.dp_size = dp_size
    runner.max_num_tokens = max_tokens * num_reqs
    runner.max_num_reqs = num_reqs
    runner.max_model_len = max_tokens
    runner.max_num_blocks_per_req = max_tokens // 16
    runner.num_tokens_paddings = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    runner.num_tokens_paddings_per_dp = [p // dp_size for p in runner.num_tokens_paddings]
    runner.num_reqs_paddings = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    runner.num_reqs_paddings_per_dp = [p // dp_size for p in runner.num_reqs_paddings]
    
    # Input batch
    runner.input_batch = MagicMock()
    runner.input_batch.num_reqs = num_reqs
    runner.input_batch.req_ids = [f"req{i}" for i in range(num_reqs)]
    runner.input_batch.req_id_to_index = {f"req{i}": i for i in range(num_reqs)}
    runner.input_batch.num_computed_tokens_cpu = np.zeros(num_reqs, dtype=np.int32)
    runner.input_batch.token_ids_cpu = np.random.randint(
        0, 32000, (num_reqs, max_tokens), dtype=np.int32)
    
    # Block table
    mock_block_table = MagicMock()
    mock_block_table.get_cpu_tensor.return_value = np.arange(
        num_reqs * runner.max_num_blocks_per_req).reshape(num_reqs, runner.max_num_blocks_per_req)
    runner.input_batch.block_table = [mock_block_table]
    
    # Initialize CPU arrays
    runner.input_ids_cpu = np.zeros(runner.max_num_tokens, dtype=np.int32)
    runner.positions_cpu = np.zeros(runner.max_num_tokens, dtype=np.int32)
    runner.query_start_loc_cpu = np.zeros(num_reqs + dp_size, dtype=np.int32)
    runner.seq_lens_cpu = np.zeros(num_reqs, dtype=np.int32)
    runner.logits_indices_cpu = np.zeros(num_reqs, dtype=np.int32)
    runner.block_tables_cpu = [np.zeros((num_reqs, runner.max_num_blocks_per_req), dtype=np.int32)]
    runner.arange_cpu = np.arange(runner.max_num_tokens, dtype=np.int64)
    
    # Mock kv cache config
    mock_kv_cache_config = MagicMock()
    mock_kv_cache_group = MagicMock()
    mock_kv_cache_group.layer_names = [f"layer_{i}" for i in range(32)]
    mock_kv_cache_config.kv_cache_groups = [mock_kv_cache_group]
    runner.kv_cache_config = mock_kv_cache_config
    runner.use_hybrid_kvcache = False
    
    # Mock scheduler config
    runner.scheduler_config = MagicMock()
    runner.scheduler_config.async_scheduling = True
    runner._pre_async_results = None
    
    # Additional required attributes
    runner.uses_mrope = False
    runner.phase_based_profiler = None
    runner.lora_config = None
    runner.mm_manager = MagicMock()
    runner.speculative_decoding_manager = MagicMock()
    runner.lora_utils = MagicMock()
    
    # Bind the actual methods
    runner._prepare_inputs_dp = TPUModelRunner._prepare_inputs_dp.__get__(runner)
    runner._prepare_inputs = TPUModelRunner._prepare_inputs.__get__(runner)
    runner._prepare_dp_input_metadata = TPUModelRunner._prepare_dp_input_metadata.__get__(runner)
    runner._prepare_async_token_substitution_indices_dp = TPUModelRunner._prepare_async_token_substitution_indices_dp.__get__(runner)
    
    return runner


def create_mock_scheduler_output(runner, num_reqs, dp_size, min_computed_tokens=1024, max_computed_tokens=4096):
    """Create mock scheduler output for decode phase (1 token scheduled per request).
    
    Args:
        runner: Mock runner to update with computed tokens
        num_reqs: Number of requests
        dp_size: Data parallel size
        min_computed_tokens: Minimum number of already computed tokens (context length)
        max_computed_tokens: Maximum number of already computed tokens (context length)
    """
    mock_output = MagicMock()
    
    np.random.seed(42) 
    
    num_scheduled_tokens = {}
    assigned_dp_ranks = {}
    total_tokens = 0
    
    for i in range(num_reqs):
        req_id = f"req{i}"
        # Decode phase: always 1 token scheduled per request
        num_scheduled_tokens[req_id] = 1
        # DP rank assignment
        assigned_dp_ranks[req_id] = i % dp_size
        total_tokens += 1
        
        # Set varying context lengths (computed tokens)
        computed_tokens = np.random.randint(min_computed_tokens, max_computed_tokens + 1)
        runner.input_batch.num_computed_tokens_cpu[i] = computed_tokens
    
    # Create DP rank assignment mapping for scheduled requests
    rank_to_req_ids = {dp_rank: [] for dp_rank in range(dp_size)}
    num_scheduled_tokens_per_dp_rank = {dp_rank: 0 for dp_rank in range(dp_size)}
    scheduled_tokens_per_dp_rank = {
        dp_rank: []
        for dp_rank in range(dp_size)
    }
    num_req_per_dp_rank = {dp_rank: 0 for dp_rank in range(dp_size)}

    for req_id in num_scheduled_tokens.keys():
        dp_rank = assigned_dp_ranks[req_id]
        rank_to_req_ids[dp_rank].append(req_id)
        num_scheduled_tokens_per_dp_rank[dp_rank] += num_scheduled_tokens[req_id]
        scheduled_tokens_per_dp_rank[dp_rank].append(num_scheduled_tokens[req_id])
        num_req_per_dp_rank[dp_rank] += 1
    
    mock_output.num_scheduled_tokens = num_scheduled_tokens
    mock_output.assigned_dp_rank = assigned_dp_ranks
    mock_output.total_num_scheduled_tokens = total_tokens
    mock_output.scheduled_spec_decode_tokens = {}
    mock_output.grammar_bitmask = None
    mock_output.rank_to_req_ids = rank_to_req_ids
    mock_output.num_scheduled_tokens_per_dp_rank = num_scheduled_tokens_per_dp_rank
    mock_output.scheduled_tokens_per_dp_rank = scheduled_tokens_per_dp_rank
    mock_output.num_req_per_dp_rank = num_req_per_dp_rank
    
    return mock_output


@patch('tpu_inference.runner.tpu_runner.runner_utils')
@patch('tpu_inference.runner.tpu_runner.TPUSupportedSamplingMetadata')
def benchmark_prepare_inputs_dp(mock_sampling_metadata,
                               mock_runner_utils,
                               num_reqs=4096, dp_size=4, tp_size=2,
                               min_tokens=1024, max_tokens=8192,
                               warmup_iters=3, bench_iters=10):
    """Benchmark _prepare_inputs_dp function with real JAX mesh (DP=4, TP=2)."""
    
    print("=" * 80)
    print("Benchmark Configuration (DECODE PHASE):")
    print("=" * 80)
    print(f"JAX devices available: {len(jax.devices())}")
    print(f"Number of requests: {num_reqs}")
    print(f"Scheduled tokens per request: 1 (decode)")
    print(f"Computed tokens (context) range: {min_tokens} - {max_tokens}")
    print(f"Data parallel size: {dp_size}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print("=" * 80)
    print()
    
    # Setup mocking
    def mock_get_padded_token_len(paddings_list, val):
        for padding in paddings_list:
            if val <= padding:
                return padding
        return paddings_list[-1]
    
    mock_runner_utils.get_padded_token_len.side_effect = mock_get_padded_token_len
    mock_sampling_metadata.from_input_batch.return_value = MagicMock()
    
    # Create runner and scheduler output
    print("Creating mock runner...")
    runner = create_mock_runner(num_reqs=num_reqs, dp_size=dp_size, tp_size=tp_size, max_tokens=max_tokens)
    
    print("Creating mock scheduler output (decode phase: 1 token/request)...")
    scheduler_output = create_mock_scheduler_output(
        runner=runner,
        num_reqs=num_reqs, 
        dp_size=dp_size,
        min_computed_tokens=min_tokens,
        max_computed_tokens=max_tokens
    )
    
    total_tokens = scheduler_output.total_num_scheduled_tokens
    avg_computed = np.mean(runner.input_batch.num_computed_tokens_cpu[:num_reqs])
    print(f"Total scheduled tokens: {total_tokens:,} (1 per request - decode phase)")
    print(f"Average computed tokens (context): {avg_computed:.1f}")
    print(f"Computed tokens range: {min_tokens} - {max_tokens}")
    print()
    
    # Create AsyncPreResults for async_scheduling=True mode
    print("Creating AsyncPreResults for async scheduling...")
    # Create mock previous results (simulating a previous iteration)
    prev_next_tokens = jax.numpy.zeros(num_reqs, dtype=jax.numpy.int32)
    runner._pre_async_results = AsyncPreResults(
        req_ids=runner.input_batch.req_ids,
        next_tokens=prev_next_tokens,
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        placeholder_req_id_to_index={},
        logits_indices_selector=None
    )
    print()
    
    # Warmup
    print(f"Running {warmup_iters} warmup iterations...")
    for i in range(warmup_iters):
        try:
            _ = runner._prepare_inputs_dp(scheduler_output)
            print(f"  Warmup {i+1}/{warmup_iters} complete")
        except Exception as e:
            print(f"  Warmup {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Benchmark
    print(f"\nRunning {bench_iters} benchmark iterations...")
    times = []
    
    for i in range(bench_iters):
        start = time.perf_counter()
        _ = runner._prepare_inputs_dp(scheduler_output)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        print(f"  Iteration {i+1}/{bench_iters}: {elapsed_ms:.2f} ms")
    
    # Results
    times = np.array(times)
    print()
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Mean:   {np.mean(times):.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")
    print(f"Std:    {np.std(times):.2f} ms")
    print(f"Min:    {np.min(times):.2f} ms")
    print(f"Max:    {np.max(times):.2f} ms")
    

if __name__ == "__main__":
    try:
        benchmark_prepare_inputs_dp(
            num_reqs=4096,
            dp_size=4,
            tp_size=2,
            min_tokens=4096,
            max_tokens=8192,
            warmup_iters=3,
            bench_iters=10,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

# JAX devices available: 8
# Number of requests: 4096
# Scheduled tokens per request: 1 (decode)
# Computed tokens (context) range: 4096 - 8192
# Data parallel size: 4
# Warmup iterations: 3
# Benchmark iterations: 10

# ================================================================================
# RESULTS:
# ================================================================================
# Mean:   2.41 ms
# Median: 2.42 ms
# Std:    0.05 ms
# Min:    2.29 ms
# Max:    2.47 ms