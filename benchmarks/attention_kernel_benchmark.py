# Copyright 2025 Google LLC
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

"""
Attention Kernel Microbenchmark

This script specifically benchmarks the attention kernels used in Qwen3VL:
1. Pallas Flash Attention (for vision encoder)
2. Pallas Ragged Paged Attention (for text decoder)
3. Reference implementations for comparison

This helps identify if the Pallas kernels are being used correctly and
provides insights into attention performance bottlenecks.

Usage:
    python benchmarks/attention_kernel_benchmark.py \
        --seq-lengths 128,256,512,1024 \
        --num-heads 28 \
        --kv-heads 4 \
        --head-dim 128 \
        --num-iterations 50
"""

import argparse
import dataclasses
import gc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Ensure JAX is configured before other imports
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@dataclass
class AttentionBenchmarkResult:
    """Result of attention benchmark."""
    name: str
    seq_length: int
    num_heads: int
    kv_heads: int
    head_dim: int
    batch_size: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    flops: float
    tflops_per_sec: float
    memory_bandwidth_gb_s: float
    is_pallas: bool = False


def create_mesh() -> Mesh:
    """Create a simple mesh for testing."""
    devices = jax.local_devices()
    return Mesh(np.array(devices).reshape((len(devices), 1, 1)),
                axis_names=("data", "attn_dp", "model"))


class Timer:
    """Context manager for timing with JAX synchronization."""

    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        jax.block_until_ready(jax.device_count())
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        jax.block_until_ready(jax.device_count())
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def benchmark_fn(fn, num_warmup: int = 3, num_iterations: int = 20) -> Tuple[float, float, float, float]:
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(num_warmup):
        result = fn()
        jax.block_until_ready(result)

    gc.collect()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        with Timer() as t:
            result = fn()
            jax.block_until_ready(result)
        times.append(t.elapsed_ms)

    times = np.array(times)
    return float(np.mean(times)), float(np.std(times)), float(np.min(times)), float(np.max(times))


def compute_attention_flops(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> float:
    """Compute FLOPs for attention computation."""
    # Q @ K^T: 2 * B * H * S * S * D (multiply-add)
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    # Softmax: ~5 * B * H * S * S
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    # Attention @ V: 2 * B * H * S * S * D
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    return qk_flops + softmax_flops + av_flops


def reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    scale: float,
) -> jax.Array:
    """Reference attention implementation using JAX."""
    # q: [B, H, S, D]
    # k, v: [B, H, S, D]
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
    return output


def benchmark_reference_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_warmup: int = 3,
    num_iterations: int = 20,
) -> AttentionBenchmarkResult:
    """Benchmark reference attention implementation."""
    rng = jax.random.PRNGKey(42)
    q = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    k = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    v = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
    scale = head_dim ** -0.5

    @jax.jit
    def attention_fn():
        return reference_attention(q, k, v, scale)

    mean_ms, std_ms, min_ms, max_ms = benchmark_fn(attention_fn, num_warmup, num_iterations)

    flops = compute_attention_flops(batch_size, seq_len, num_heads, head_dim)
    tflops_per_sec = (flops / (mean_ms / 1000)) / 1e12

    # Estimate memory bandwidth (simplified)
    bytes_accessed = (3 * batch_size * num_heads * seq_len * head_dim * 2)  # Q, K, V in bf16
    memory_bandwidth_gb_s = (bytes_accessed / (mean_ms / 1000)) / 1e9

    return AttentionBenchmarkResult(
        name="reference_attention",
        seq_length=seq_len,
        num_heads=num_heads,
        kv_heads=num_heads,
        head_dim=head_dim,
        batch_size=batch_size,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        flops=flops,
        tflops_per_sec=tflops_per_sec,
        memory_bandwidth_gb_s=memory_bandwidth_gb_s,
        is_pallas=False,
    )


def benchmark_pallas_flash_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
    num_warmup: int = 3,
    num_iterations: int = 20,
) -> Optional[AttentionBenchmarkResult]:
    """Benchmark Pallas Flash Attention kernel."""
    try:
        from tpu_inference.kernels.flash_attention.kernel import flash_attention

        rng = jax.random.PRNGKey(42)
        # Flash attention expects: [B, H, S, D]
        q = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
        k = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
        v = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
        segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        @jax.jit
        def flash_attention_fn():
            return flash_attention(
                q, k, v,
                segment_ids=segment_ids,
                causal=False,
                sm_scale=head_dim ** -0.5,
            )

        mean_ms, std_ms, min_ms, max_ms = benchmark_fn(flash_attention_fn, num_warmup, num_iterations)

        flops = compute_attention_flops(batch_size, seq_len, num_heads, head_dim)
        tflops_per_sec = (flops / (mean_ms / 1000)) / 1e12

        bytes_accessed = (3 * batch_size * num_heads * seq_len * head_dim * 2)
        memory_bandwidth_gb_s = (bytes_accessed / (mean_ms / 1000)) / 1e9

        return AttentionBenchmarkResult(
            name="pallas_flash_attention",
            seq_length=seq_len,
            num_heads=num_heads,
            kv_heads=num_heads,
            head_dim=head_dim,
            batch_size=batch_size,
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms,
            flops=flops,
            tflops_per_sec=tflops_per_sec,
            memory_bandwidth_gb_s=memory_bandwidth_gb_s,
            is_pallas=True,
        )
    except Exception as e:
        print(f"  Pallas Flash Attention benchmark failed: {e}")
        return None


def benchmark_sharded_flash_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    mesh: Mesh,
    dtype: jnp.dtype = jnp.bfloat16,
    num_warmup: int = 3,
    num_iterations: int = 20,
) -> Optional[AttentionBenchmarkResult]:
    """Benchmark Sharded Flash Attention (used in vision encoder)."""
    try:
        from tpu_inference.layers.common.attention_interface import sharded_flash_attention

        rng = jax.random.PRNGKey(42)
        # Input shape for sharded_flash_attention: [B, H, S, D]
        q = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
        k = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
        v = jax.random.normal(rng, (batch_size, num_heads, seq_len, head_dim), dtype=dtype)
        segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

        # Create sharded flash attention function
        flash_fn = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=head_dim ** -0.5,
            vmem_limit_bytes=128 * 1024 * 1024,
        )

        def sharded_flash_fn():
            return flash_fn(q, k, v, segment_ids)

        mean_ms, std_ms, min_ms, max_ms = benchmark_fn(sharded_flash_fn, num_warmup, num_iterations)

        flops = compute_attention_flops(batch_size, seq_len, num_heads, head_dim)
        tflops_per_sec = (flops / (mean_ms / 1000)) / 1e12

        bytes_accessed = (3 * batch_size * num_heads * seq_len * head_dim * 2)
        memory_bandwidth_gb_s = (bytes_accessed / (mean_ms / 1000)) / 1e9

        return AttentionBenchmarkResult(
            name="sharded_flash_attention",
            seq_length=seq_len,
            num_heads=num_heads,
            kv_heads=num_heads,
            head_dim=head_dim,
            batch_size=batch_size,
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms,
            flops=flops,
            tflops_per_sec=tflops_per_sec,
            memory_bandwidth_gb_s=memory_bandwidth_gb_s,
            is_pallas=True,
        )
    except Exception as e:
        print(f"  Sharded Flash Attention benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_ragged_paged_attention(
    total_tokens: int,
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    mesh: Mesh,
    num_blocks: int = 64,
    block_size: int = 16,
    dtype: jnp.dtype = jnp.bfloat16,
    num_warmup: int = 3,
    num_iterations: int = 20,
) -> Optional[AttentionBenchmarkResult]:
    """Benchmark Ragged Paged Attention (used in text decoder)."""
    try:
        from tpu_inference.layers.common.attention_interface import attention
        from tpu_inference.layers.common.attention_metadata import AttentionMetadata
        from tpu_inference.runner.kv_cache import create_kv_caches

        rng = jax.random.PRNGKey(42)

        # Create inputs: [T, N, H] format
        q = jax.random.normal(rng, (total_tokens, num_heads, head_dim), dtype=dtype)
        k = jax.random.normal(rng, (total_tokens, kv_heads, head_dim), dtype=dtype)
        v = jax.random.normal(rng, (total_tokens, kv_heads, head_dim), dtype=dtype)

        # Create KV cache
        kv_caches = create_kv_caches(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=kv_heads,
            head_size=head_dim,
            mesh=mesh,
            layer_names=["layer_0"],
            cache_dtype=dtype,
        )
        kv_cache = kv_caches[0]

        # Create attention metadata
        seq_len = total_tokens
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        block_tables = jnp.zeros((1, (seq_len + block_size - 1) // block_size), dtype=jnp.int32).reshape(-1)
        seq_lens = jnp.array([seq_len], dtype=jnp.int32)
        query_start_loc = jnp.array([0, seq_len], dtype=jnp.int32)
        request_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

        attn_metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        @jax.jit
        def rpa_fn():
            return attention(
                kv_cache,
                q, k, v,
                attn_metadata,
                mesh,
                head_dim,
            )

        mean_ms, std_ms, min_ms, max_ms = benchmark_fn(rpa_fn, num_warmup, num_iterations)

        # FLOPs for ragged attention (approximate, depends on actual sequence layout)
        flops = compute_attention_flops(1, total_tokens, num_heads, head_dim)
        tflops_per_sec = (flops / (mean_ms / 1000)) / 1e12

        bytes_accessed = (3 * total_tokens * num_heads * head_dim * 2)
        memory_bandwidth_gb_s = (bytes_accessed / (mean_ms / 1000)) / 1e9

        return AttentionBenchmarkResult(
            name="ragged_paged_attention",
            seq_length=total_tokens,
            num_heads=num_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            batch_size=1,
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms,
            flops=flops,
            tflops_per_sec=tflops_per_sec,
            memory_bandwidth_gb_s=memory_bandwidth_gb_s,
            is_pallas=True,
        )
    except Exception as e:
        print(f"  Ragged Paged Attention benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_results(results: List[AttentionBenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("ATTENTION KERNEL BENCHMARK RESULTS")
    print("=" * 100)

    print(f"\n{'Kernel':<30} {'SeqLen':<8} {'Heads':<8} {'Time (ms)':<15} {'TFLOPS':<10} {'Pallas':<8}")
    print("-" * 100)

    for r in results:
        time_str = f"{r.mean_time_ms:.2f} ±{r.std_time_ms:.2f}"
        print(f"{r.name:<30} {r.seq_length:<8} {r.num_heads:<8} {time_str:<15} "
              f"{r.tflops_per_sec:.2f}{'':<6} {'Yes' if r.is_pallas else 'No':<8}")


def run_benchmarks(
    seq_lengths: List[int],
    num_heads: int = 28,
    kv_heads: int = 4,
    head_dim: int = 128,
    batch_size: int = 1,
    num_warmup: int = 3,
    num_iterations: int = 20,
) -> List[AttentionBenchmarkResult]:
    """Run all attention benchmarks."""
    mesh = create_mesh()
    results = []

    print("\n" + "#" * 60)
    print("# Attention Kernel Microbenchmark")
    print(f"# Devices: {jax.device_count()}")
    print(f"# Platform: {jax.default_backend()}")
    print("#" * 60)

    for seq_len in seq_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        # Reference attention
        print("  Reference Attention...")
        ref_result = benchmark_reference_attention(
            batch_size, seq_len, num_heads, head_dim,
            num_warmup=num_warmup, num_iterations=num_iterations
        )
        results.append(ref_result)
        print(f"    Time: {ref_result.mean_time_ms:.2f} ms, TFLOPS: {ref_result.tflops_per_sec:.2f}")

        # Pallas Flash Attention
        print("  Pallas Flash Attention...")
        flash_result = benchmark_pallas_flash_attention(
            batch_size, seq_len, num_heads, head_dim,
            num_warmup=num_warmup, num_iterations=num_iterations
        )
        if flash_result:
            results.append(flash_result)
            print(f"    Time: {flash_result.mean_time_ms:.2f} ms, TFLOPS: {flash_result.tflops_per_sec:.2f}")
            if ref_result:
                speedup = ref_result.mean_time_ms / flash_result.mean_time_ms
                print(f"    Speedup vs Reference: {speedup:.2f}x")

        # Sharded Flash Attention (for vision)
        print("  Sharded Flash Attention...")
        sharded_result = benchmark_sharded_flash_attention(
            batch_size, seq_len, num_heads, head_dim, mesh,
            num_warmup=num_warmup, num_iterations=num_iterations
        )
        if sharded_result:
            results.append(sharded_result)
            print(f"    Time: {sharded_result.mean_time_ms:.2f} ms, TFLOPS: {sharded_result.tflops_per_sec:.2f}")
            if ref_result:
                speedup = ref_result.mean_time_ms / sharded_result.mean_time_ms
                print(f"    Speedup vs Reference: {speedup:.2f}x")

        # Ragged Paged Attention (for text decoder)
        print("  Ragged Paged Attention...")
        rpa_result = benchmark_ragged_paged_attention(
            seq_len, num_heads, kv_heads, head_dim, mesh,
            num_warmup=num_warmup, num_iterations=num_iterations
        )
        if rpa_result:
            results.append(rpa_result)
            print(f"    Time: {rpa_result.mean_time_ms:.2f} ms, TFLOPS: {rpa_result.tflops_per_sec:.2f}")

    return results


def diagnose_attention_performance(results: List[AttentionBenchmarkResult]):
    """Diagnose potential attention performance issues."""
    print("\n" + "=" * 60)
    print("PERFORMANCE DIAGNOSIS")
    print("=" * 60)

    # Group results by sequence length
    by_seq_len = {}
    for r in results:
        if r.seq_length not in by_seq_len:
            by_seq_len[r.seq_length] = {}
        by_seq_len[r.seq_length][r.name] = r

    issues_found = []

    for seq_len, kernels in sorted(by_seq_len.items()):
        ref = kernels.get("reference_attention")
        pallas_flash = kernels.get("pallas_flash_attention")
        sharded_flash = kernels.get("sharded_flash_attention")
        rpa = kernels.get("ragged_paged_attention")

        # Check if Pallas kernels are faster than reference
        if ref and pallas_flash:
            speedup = ref.mean_time_ms / pallas_flash.mean_time_ms
            if speedup < 1.2:
                issues_found.append(
                    f"[seq_len={seq_len}] Pallas Flash Attention is only {speedup:.2f}x faster "
                    f"than reference (expected >2x). Kernel may not be optimized."
                )

        if ref and sharded_flash:
            speedup = ref.mean_time_ms / sharded_flash.mean_time_ms
            if speedup < 1.0:
                issues_found.append(
                    f"[seq_len={seq_len}] Sharded Flash Attention is SLOWER than reference "
                    f"({speedup:.2f}x). Sharding overhead may be too high."
                )

        # Check if sharded version has too much overhead vs non-sharded
        if pallas_flash and sharded_flash:
            overhead = sharded_flash.mean_time_ms / pallas_flash.mean_time_ms
            if overhead > 1.5:
                issues_found.append(
                    f"[seq_len={seq_len}] Sharded Flash Attention has {overhead:.2f}x overhead "
                    f"vs non-sharded. Consider reviewing sharding strategy."
                )

    if issues_found:
        print("\nPotential Issues Found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo obvious performance issues detected. Pallas kernels appear to be working correctly.")

    # Recommendations
    print("\nRecommendations:")
    print("  1. If Pallas kernels are not faster than reference, check:")
    print("     - Head dimension alignment (should be 64 or 128)")
    print("     - Sequence length alignment (block sizes)")
    print("     - Sharding configuration matches mesh")
    print("  2. Use JAX profiler to verify Pallas kernels are being called")
    print("  3. Check for excessive recompilation with VLLM_XLA_CHECK_RECOMPILATION=1")


def parse_args():
    parser = argparse.ArgumentParser(description="Attention Kernel Microbenchmark")
    parser.add_argument("--seq-lengths", type=str, default="128,256,512,1024",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--num-heads", type=int, default=28, help="Number of attention heads")
    parser.add_argument("--kv-heads", type=int, default=4, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=20, help="Benchmark iterations")
    return parser.parse_args()


def main():
    args = parse_args()

    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]

    results = run_benchmarks(
        seq_lengths=seq_lengths,
        num_heads=args.num_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        batch_size=args.batch_size,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )

    print_results(results)
    diagnose_attention_performance(results)


if __name__ == "__main__":
    main()
