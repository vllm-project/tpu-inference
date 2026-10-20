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
"""Benchmark script comparing fused_moe_func vs fused_moe_func2.

This script benchmarks and compares two implementations of fused MoE:
1. fused_moe_func: Original single-batch implementation
2. fused_moe_func2: Batch-split implementation that processes tokens in two halves

Usage:
    # Run as a module (recommended):
    cd tpu-inference
    python -m benchmarks.benchmark_ep

    # Run directly:
    cd tpu-inference/benchmarks
    python benchmark_ep.py

    # Run with specific token count
    python -m benchmarks.benchmark_ep --num-tokens 1024

    # Run with custom model dimensions (e.g. Llama-4-Maverick-like config)
    python -m benchmarks.benchmark_ep --num-tokens 4096 --hidden-size 6144 \
        --intermediate-size 2560 --num-experts 160 --topk 8

    # Run with a smaller model for quick testing
    python -m benchmarks.benchmark_ep --num-tokens 512 --hidden-size 4096 \
        --intermediate-size 1024 --num-experts 64 --topk 4

    # Override expert parallelism size (default: auto-detect from num_devices)
    python -m benchmarks.benchmark_ep --ep-size 4

    # Run comprehensive benchmark suite (sweeps multiple token counts)
    python -m benchmarks.benchmark_ep --run-suite

    # Adjust warmup and benchmark iterations for faster/more-accurate runs
    python -m benchmarks.benchmark_ep --warmup-iters 10 --benchmark-iters 50

    # Enable SparseCore offloading for collectives (Ironwood TPU)
    python -m benchmarks.benchmark_ep --sparsecore

    # Run with JAX profiler tracing (saves trace to directory)
    python -m benchmarks.benchmark_ep --trace-dir /data/logs

    # Combine SparseCore + profiler tracing
    python -m benchmarks.benchmark_ep --sparsecore --trace-dir /data/logs

    # Dump HLO (pre- and post-optimization) for analysis
    python -m benchmarks.benchmark_ep --dump-hlo /data/hlo_dump

    # Dump HLO from XLA via LIBTPU_INIT_ARGS
    LIBTPU_INIT_ARGS="--xla_jf_dump_to=/data/hlo_dump" \
        python -m benchmarks.benchmark_ep --sparsecore --trace-dir /data/logs

    # Full example: SparseCore + profiling + HLO dump + custom config
    python -m benchmarks.benchmark_ep --sparsecore --trace-dir /data/logs \
        --dump-hlo /data/hlo_dump --num-tokens 8192 --num-experts 160 --topk 8 \
        --warmup-iters 5 --benchmark-iters 30

Requirements:
    - Must be run on a TPU VM with multiple chips (for expert parallelism)
    - JAX with TPU support
    - num_tokens must be even (fused_moe_func2 splits the batch in half)
    - num_tokens * topk must be divisible by 16 (kernel alignment requirement)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Handle both direct execution and module execution.
# We need the repo root on sys.path for "benchmarks.*" and "tpu_inference.*" imports.
if __name__ == "__main__" and __package__ is None:
    _file_path = Path(__file__).resolve()
    # .parent x2: benchmark_ep.py -> benchmarks/ -> tpu-inference/
    _repo_root = _file_path.parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

# Parse --sparsecore early, BEFORE importing JAX.
# LIBTPU_INIT_ARGS must be set before JAX/libtpu initialization.
# Define flags inline to avoid importing kernels (which triggers JAX import).
if __name__ == "__main__" and "--sparsecore" in sys.argv:
    _SC_FLAGS = (
        # Disable async collective fusion so collectives can be offloaded individually
        " --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false"
        " --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false"
        " --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false"
        # Enable SparseCore offloading for each collective type
        " --xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
        " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true"
        " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true"
        # Enable tracing for offloaded collectives (visible in profiler)
        " --xla_tpu_enable_all_gather_offload_tracing=true"
        " --xla_tpu_enable_reduce_scatter_offload_tracing=true"
        " --xla_tpu_enable_all_reduce_offload_tracing=true"
        # Disable all-reduce combiner to prevent merging multiple all-reduces into
        # a single tuple all-reduce, which is not eligible for SparseCore offloading.
        " --xla_jf_crs_combiner_threshold_in_bytes=0"
        " --xla_jf_crs_combiner_threshold_count=1"
        # SparseCore base flags
        " --xla_tpu_use_tc_device_shape_on_sc=true"
        " --xla_sc_enable_instruction_fusion=false"
        " --xla_sc_disjoint_spmem=false"
        " --xla_sc_disable_megacore_partitioning=true")
    existing = os.environ.get("LIBTPU_INIT_ARGS", "")
    os.environ["LIBTPU_INIT_ARGS"] = existing + _SC_FLAGS
    print(
        "\nSparseCore offloading ENABLED (LIBTPU_INIT_ARGS set before JAX init)"
    )

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from benchmarks.ep_gmm_pipelined_benchmark import (fused_moe_func,
                                                   fused_moe_func2,
                                                   print_xla_flags)

# Global flag to only capture trace once
TRACED = False


# ============================================================================
# Benchmark infrastructure
# ============================================================================
def create_fake_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    dtype: jnp.dtype = jnp.bfloat16,
):
    """Create fake inputs for fused_moe_func / fused_moe_func2.

    Both functions expect:
    - hidden_states: [num_tokens, hidden_size]
    - w1: [num_experts, intermediate_size * 2, hidden_size]  (gate+up fused)
    - w2: [num_experts, hidden_size, intermediate_size]       (down projection)
    - gating_output: [num_tokens, num_experts]

    w1/w2 have a padded_hidden_size leading dimension, but for simplicity
    we set padded_hidden_size == the relevant dimension (no padding needed
    when hidden_size is already aligned).
    """
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    # hidden_states: [num_tokens, hidden_size]
    hidden_states = jax.random.normal(keys[0], (num_tokens, hidden_size),
                                      dtype=dtype)

    # w1: [num_experts, padded_hidden_size, intermediate_size * 2]
    # padded_hidden_size >= hidden_size; for benchmark we use hidden_size directly
    w1 = (jax.random.normal(keys[1],
                            (num_experts, hidden_size, intermediate_size * 2),
                            dtype=dtype) * 0.01)

    # w2: [num_experts, intermediate_size, hidden_size]
    w2 = (jax.random.normal(keys[2],
                            (num_experts, intermediate_size, hidden_size),
                            dtype=dtype) * 0.01)

    # gating_output: [num_tokens, num_experts]  (raw logits before softmax)
    gating_output = jax.random.normal(keys[3], (num_tokens, num_experts),
                                      dtype=jnp.float32)

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "w1_scale": None,
        "w2_scale": None,
        "w1_bias": None,
        "w2_bias": None,
        "gating_output": gating_output,
    }


def benchmark_function(fn,
                       inputs,
                       warmup_iters=5,
                       benchmark_iters=20,
                       name=""):
    """Benchmark a function and return timing statistics."""

    # Warmup
    for _ in range(warmup_iters):
        result = fn(**inputs)
        result.block_until_ready()

    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        start = time.perf_counter()
        result = fn(**inputs)
        result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    print(f"\n{name}:")
    print(f"  Mean:   {times.mean() * 1000:.3f} ms")
    print(f"  Std:    {times.std() * 1000:.3f} ms")
    print(f"  Min:    {times.min() * 1000:.3f} ms")
    print(f"  Max:    {times.max() * 1000:.3f} ms")
    print(f"  Median: {np.median(times) * 1000:.3f} ms")

    return result, times


def run_benchmark(
    num_tokens: int = 8192,
    hidden_size: int = 6144,
    intermediate_size: int = 2560,
    num_experts: int = 160,
    topk: int = 8,
    ep_size: int = None,
    trace_dir: str = None,
    dump_hlo: str = None,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
):
    """Run the benchmark comparing fused_moe_func vs fused_moe_func2.

    Args:
        num_tokens: Number of tokens (batch size). Must be even for the split.
        hidden_size: Hidden dimension size.
        intermediate_size: Intermediate/FFN dimension size.
        num_experts: Total number of experts.
        topk: Number of experts per token.
        ep_size: Expert parallelism size (default: num_devices).
        trace_dir: Optional directory to save JAX profiler trace.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
    """
    global TRACED

    print("=" * 70)
    print("fused_moe_func vs fused_moe_func2 Benchmark")
    print("=" * 70)

    # Get devices and create mesh
    devices = jax.devices()
    num_devices = len(devices)

    if ep_size is None:
        ep_size = num_devices

    print("\nConfiguration:")
    print(f"  num_tokens:        {num_tokens}")
    print(f"  hidden_size:       {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts:       {num_experts}")
    print(f"  topk:              {topk}")
    print(f"  num_devices:       {num_devices}")
    print(f"  ep_size:           {ep_size}")

    assert (num_tokens *
            topk) % 16 == 0, "num_tokens * topk must be divisible by 16"
    assert (num_tokens %
            2 == 0), "num_tokens must be even for fused_moe_func2 batch split"

    # Create mesh for EP
    # For EP-only mode: mesh has shape (1, ep_size) with axes ('data', 'model')
    mesh_devices = np.array(devices[:ep_size]).reshape(1, ep_size)
    mesh = Mesh(mesh_devices, axis_names=("data", "model"))
    print(f"  Mesh shape:        {mesh.shape}")
    print(f"  Mesh axis names:   {mesh.axis_names}")

    # Create fake inputs
    print("\nCreating fake inputs...")
    inputs = create_fake_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )

    # Print input shapes
    print("\nInput shapes:")
    print(f"  hidden_states: {inputs['hidden_states'].shape}")
    print(f"  w1:            {inputs['w1'].shape}")
    print(f"  w2:            {inputs['w2'].shape}")
    print(f"  gating_output: {inputs['gating_output'].shape}")

    # Common static args shared by both functions
    static_kwargs = {
        "topk": topk,
        "renormalize": True,
        "mesh": mesh,
        "use_ep": True,
        "activation": "silu",
        "scoring_fn": "softmax",
    }

    fn_inputs = {**inputs, **static_kwargs}

    # ---- Warmup & compile ----
    print("\nJIT compiling & warming up fused_moe_func (original)...")
    try:
        result_orig = fused_moe_func(**fn_inputs)
        result_orig.block_until_ready()
        print("  -> fused_moe_func warmup complete")
    except Exception as e:
        print(f"  -> fused_moe_func warmup FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("JIT compiling & warming up fused_moe_func2 (batch-split)...")
    try:
        result_split = fused_moe_func2(**fn_inputs)
        result_split.block_until_ready()
        print("  -> fused_moe_func2 warmup complete")
    except Exception as e:
        print(f"  -> fused_moe_func2 warmup FAILED: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Check output shapes
    print("\nOutput shapes:")
    print(f"  fused_moe_func:  {result_orig.shape}")
    print(f"  fused_moe_func2: {result_split.shape}")

    # Check numerical closeness
    max_diff = jnp.max(jnp.abs(result_orig - result_split)).item()
    mean_diff = jnp.mean(jnp.abs(result_orig - result_split)).item()
    print("\nNumerical comparison:")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")

    # Dump HLO if requested
    if dump_hlo:
        import json

        dump_dir = Path(dump_hlo)
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDumping HLO compilation package to {dump_dir}/...")

        lowered = fused_moe_func2.lower(**fn_inputs)
        compiled = lowered.compile()

        # 1. Pre-optimization HLO (what JAX produced, before XLA optimizes)
        pre_opt_text = lowered.as_text()
        with open(dump_dir / "hlo_pre_optimization.txt", "w") as f:
            f.write(pre_opt_text)
        print(f"  -> hlo_pre_optimization.txt ({len(pre_opt_text)} bytes)")

        # 2. Post-optimization HLO (what XLA compiled to)
        post_opt_text = compiled.as_text()
        with open(dump_dir / "hlo_post_optimization.txt", "w") as f:
            f.write(post_opt_text)
        print(f"  -> hlo_post_optimization.txt ({len(post_opt_text)} bytes)")

        # 3. Serialized HLO module proto (binary, includes full module metadata)
        try:
            serialized_hlo = compiled.as_serialized_hlo()
            with open(dump_dir / "hlo_module.pb", "wb") as f:
                f.write(serialized_hlo)
            print(f"  -> hlo_module.pb ({len(serialized_hlo)} bytes)")
        except AttributeError:
            try:
                # Alternative: get HLO module from lowered IR
                hlo_module = lowered.compiler_ir(dialect="hlo")
                serialized_hlo = hlo_module.as_serialized_hlo_module_proto()
                with open(dump_dir / "hlo_module.pb", "wb") as f:
                    f.write(serialized_hlo)
                print(f"  -> hlo_module.pb ({len(serialized_hlo)} bytes)")
            except Exception:
                print(
                    "  -> hlo_module.pb SKIPPED (API not available in this JAX version)"
                )

        # 4. Compilation environment info
        env_info = {
            "jax_version": jax.__version__,
            "num_devices": len(jax.devices()),
            "device_kind": str(jax.devices()[0].device_kind),
            "device_platform": str(jax.devices()[0].platform),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
            "mesh_shape": dict(mesh.shape),
            "mesh_axis_names": list(mesh.axis_names),
            "static_kwargs": {
                "topk": topk,
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_experts": num_experts,
            },
            "input_shapes": {
                k: str(v.shape) + " " + str(v.dtype)
                for k, v in inputs.items() if hasattr(v, "shape")
            },
        }
        with open(dump_dir / "compilation_env.json", "w") as f:
            json.dump(env_info, f, indent=2)
        print("  -> compilation_env.json")

        # 5. Cost analysis (if available)
        try:
            cost = compiled.cost_analysis()
            if cost:
                with open(dump_dir / "cost_analysis.json", "w") as f:
                    json.dump([dict(c) for c in cost], f, indent=2)
                print("  -> cost_analysis.json")
        except Exception:
            pass

        # Quick check for key collective ops in post-optimization HLO
        print("\n  Collective ops in post-optimization HLO:")
        for pattern in ["reduce-scatter", "all-gather", "all-reduce"]:
            count = post_opt_text.lower().count(pattern)
            print(f"    {pattern}: {count}")

    # Capture trace if requested
    if trace_dir and not TRACED:
        print(f"\nCapturing JAX profiler trace to {trace_dir}...")
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 3
        options.device_tracer_level = 1
        options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        }
        with jax.profiler.trace(trace_dir, profiler_options=options):
            result_orig_trace = fused_moe_func(**fn_inputs)
            result_orig_trace.block_until_ready()
            result_split_trace = fused_moe_func2(**fn_inputs)
            result_split_trace.block_until_ready()
        TRACED = True
        print(f"  -> Trace captured successfully in {trace_dir}")

    # ---- Run benchmarks ----
    print("\n" + "=" * 70)
    print("Running benchmarks...")
    print("=" * 70)

    result_orig, times_orig = benchmark_function(
        fused_moe_func,
        fn_inputs,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        name="fused_moe_func (original)",
    )

    result_split, times_split = benchmark_function(
        fused_moe_func2,
        fn_inputs,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        name="fused_moe_func2 (batch-split)",
    )

    # ---- Comparison ----
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)

    speedup = times_orig.mean() / times_split.mean()
    print(f"\nSpeedup (fused_moe_func2 vs fused_moe_func): {speedup:.3f}x")

    if speedup > 1.0:
        print(f"  -> fused_moe_func2 is {(speedup - 1) * 100:.1f}% faster")
    else:
        print(f"  -> fused_moe_func2 is {(1 - speedup) * 100:.1f}% slower")

    return {
        "original": {
            "times": times_orig,
            "result": result_orig
        },
        "batch_split": {
            "times": times_split,
            "result": result_split
        },
        "speedup": speedup,
    }


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark fused_moe_func vs fused_moe_func2")
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=8192,
        help=
        "Number of tokens (must be even and num_tokens*topk divisible by 16)",
    )
    parser.add_argument("--hidden-size",
                        type=int,
                        default=6144,
                        help="Hidden size")
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=2560,
        help="Intermediate size (MLP hidden dim)",
    )
    parser.add_argument("--num-experts",
                        type=int,
                        default=160,
                        help="Number of experts")
    parser.add_argument("--topk",
                        type=int,
                        default=8,
                        help="Top-k experts per token")
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help="Expert parallelism size (default: num_devices)",
    )
    parser.add_argument("--warmup-iters",
                        type=int,
                        default=5,
                        help="Warmup iterations")
    parser.add_argument("--benchmark-iters",
                        type=int,
                        default=20,
                        help="Benchmark iterations")
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Run benchmark suite with multiple configurations",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory to save JAX profiler trace (e.g., /data/logs)",
    )
    parser.add_argument(
        "--sparsecore",
        action="store_true",
        help=
        "Enable SparseCore offloading for collectives (all-reduce, reduce-scatter, all-gather)",
    )
    parser.add_argument(
        "--dump-hlo",
        type=str,
        default=None,
        help=
        "Dump optimized HLO text to the specified file path (e.g., /data/hlo_dump.txt)",
    )

    args = parser.parse_args()

    print_xla_flags()

    print("\n" + "=" * 70)
    print("fused_moe_func vs fused_moe_func2 Benchmark")
    print("=" * 70)
    print(f"\nAvailable devices: {len(jax.devices())}")
    print(f"Device type: {jax.devices()[0].device_kind}")

    if args.run_suite:
        # (num_tokens, hidden_size, intermediate_size, num_experts, topk)
        configs = [
            (1024, 6144, 2560, 160, 8),
            (2048, 6144, 2560, 160, 8),
            (4096, 6144, 2560, 160, 8),
            (8192, 6144, 2560, 160, 8),
        ]

        results = []
        for config in configs:
            num_tokens, hidden_size, intermediate_size, num_experts, topk = config

            try:
                result = run_benchmark(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    topk=topk,
                    trace_dir=args.trace_dir,
                    warmup_iters=args.warmup_iters,
                    benchmark_iters=args.benchmark_iters,
                )
                results.append((config, result))
            except Exception as e:
                print(f"\nFailed for config {config}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Summary
        print("\n" + "=" * 70)
        print("Summary of Results")
        print("=" * 70)
        print(f"\n{'Config':<50} {'Speedup':>10}")
        print("-" * 62)
        for config, result in results:
            num_tokens, hidden_size, intermediate_size, num_experts, topk = config
            config_str = f"tokens={num_tokens}, hidden={hidden_size}, experts={num_experts}, topk={topk}"
            print(f"{config_str:<50} {result['speedup']:>10.3f}x")
    else:
        try:
            result = run_benchmark(
                num_tokens=args.num_tokens,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                num_experts=args.num_experts,
                topk=args.topk,
                ep_size=args.ep_size,
                trace_dir=args.trace_dir,
                dump_hlo=args.dump_hlo,
                warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters,
            )
        except Exception as e:
            print(f"\nBenchmark failed: {e}")
            import traceback

            traceback.print_exc()
