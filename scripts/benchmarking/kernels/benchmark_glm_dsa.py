# Copyright 2026 Gianluigi Vitale
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
"""Profiler-free TPU microbenchmarks for the GLM-5.x DSA decode path.

The default shape is one live decode row at 256K context with DSA top-k 2048.
Each sample is individually synchronized, so the reported distribution is
steady host wall latency rather than queued dispatch time.

Run on a TPU host:

    python scripts/benchmarking/kernels/benchmark_glm_dsa.py --mode all
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.deepseek_v32.indexer_cache import \
    insert_indexer_k_cache
from tpu_inference.kernels.experimental.deepseek_v32.sparse_mla import (
    gather_paged_kv, sparse_mla_decode)

_PAGE_SIZE = 128
_PACKING = 32
_INDEX_RECORD_WIDTH = 256
_LATENT_WIDTH = 640


@dataclass(frozen=True)
class BenchmarkConfig:
    """Static GLM DSA benchmark geometry and sampling controls."""

    seq_len: int = 262_144
    topk: int = 2_048
    pages_per_block: int = 4
    mla_block_size: int = 512
    warmup: int = 5
    samples: int = 20


def _percentile(values_ms: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values_ms), q))


def _checksum(output: Any) -> int:
    array = np.asarray(output)
    return int(array.view(np.uint8).sum(dtype=np.uint64))


def _measure(
    name: str,
    fn: Callable[[], Any],
    config: BenchmarkConfig,
    *,
    checksum_fn: Callable[[Any], Any] = lambda output: output,
) -> dict[str, object]:
    for _ in range(config.warmup):
        jax.block_until_ready(fn())

    values_ms: list[float] = []
    checksum = 0
    for _ in range(config.samples):
        start_ns = time.perf_counter_ns()
        output = fn()
        jax.block_until_ready(output)
        values_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
        checksum = _checksum(checksum_fn(output))

    return {
        "name": name,
        "warmup": config.warmup,
        "samples": config.samples,
        "min_ms": min(values_ms),
        "mean_ms": statistics.fmean(values_ms),
        "p50_ms": _percentile(values_ms, 50),
        "p90_ms": _percentile(values_ms, 90),
        "p95_ms": _percentile(values_ms, 95),
        "p99_ms": _percentile(values_ms, 99),
        "max_ms": max(values_ms),
        "checksum": checksum,
        "samples_ms": values_ms,
    }


def _index_cache(config: BenchmarkConfig) -> jax.Array:
    num_pages = (config.seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    host_cache = np.zeros(
        (num_pages, _PAGE_SIZE // _PACKING, _PACKING, _INDEX_RECORD_WIDTH),
        dtype=np.uint8,
    )
    host_cache[..., 128:132] = np.asarray([1.0], np.float32).view(np.uint8)
    return jnp.asarray(host_cache)


def _latent_cache(config: BenchmarkConfig) -> jax.Array:
    num_pages = (config.seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    return jnp.ones(
        (num_pages, _PAGE_SIZE // _PACKING, _PACKING, _LATENT_WIDTH),
        dtype=jnp.bfloat16,
    )


def _block_table(config: BenchmarkConfig) -> jax.Array:
    num_pages = (config.seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE
    return jnp.roll(jnp.arange(num_pages, dtype=jnp.int32), 17)[None, :]


def _streamindex(
    config: BenchmarkConfig,
    index_cache: jax.Array,
    block_table: jax.Array,
) -> jax.Array:
    return streamindex_topk(
        q=jnp.zeros((1, 32, 128), dtype=jnp.float8_e4m3fn),
        indexer_weights=jnp.ones((1, 32), dtype=jnp.float32),
        cache_kv=index_cache,
        seq_lens=jnp.asarray([config.seq_len], dtype=jnp.int32),
        page_indices=block_table.reshape(-1),
        cu_q_lens=jnp.asarray([0, 1], dtype=jnp.int32),
        distribution=jnp.asarray([1, 1, 1], dtype=jnp.int32),
        k=config.topk,
        compression_ratio=1,
        scale_storage="float32",
        exact_topk=True,
        num_kv_pages_per_block=config.pages_per_block,
        num_queries_per_block=1,
        decode_req_batch_size=1,
    )


def benchmark_topk(config: BenchmarkConfig) -> dict[str, object]:
    """Measure the JAX top-k baseline at the protected decode shape."""
    scores = jnp.zeros((1, config.seq_len), dtype=jnp.float32)

    @jax.jit
    def run() -> jax.Array:
        return jax.lax.top_k(scores, config.topk)[1]

    return _measure(f"lax_top_k_n{config.seq_len}_k{config.topk}", run, config)


def benchmark_scorer(config: BenchmarkConfig) -> dict[str, object]:
    """Measure exact StreamIndex scoring and selection."""
    index_cache = _index_cache(config)
    block_table = _block_table(config)

    @jax.jit
    def run() -> jax.Array:
        return _streamindex(config, index_cache, block_table)

    return _measure(
        f"streamindex_n{config.seq_len}_k{config.topk}_"
        f"bkvp{config.pages_per_block}",
        run,
        config,
    )


def benchmark_index_cache_insert(
    config: BenchmarkConfig, ) -> dict[str, object]:
    """Measure one indexer-key insertion into the 256K paged cache."""
    cache = _index_cache(config)
    key = jnp.ones((1, 128), dtype=jnp.bfloat16)
    slot = jnp.asarray([config.seq_len - 1], dtype=jnp.int32)

    @jax.jit
    def run(cache: jax.Array, key: jax.Array, slot: jax.Array) -> jax.Array:
        return insert_indexer_k_cache(cache, key, slot)

    return _measure(
        f"index_cache_insert_n{config.seq_len}",
        lambda: run(cache, key, slot),
        config,
        checksum_fn=lambda output: output.reshape(-1, output.shape[-1])[
            config.seq_len - 1],
    )


def benchmark_sparse_mla(config: BenchmarkConfig) -> dict[str, object]:
    """Measure sparse MLA compute after selected KV is resident."""
    q_nope = jnp.ones((1, 32, 512), dtype=jnp.bfloat16)
    q_pe = jnp.ones((1, 32, 64), dtype=jnp.bfloat16)
    selected = jnp.ones((1, config.topk, _LATENT_WIDTH), dtype=jnp.bfloat16)
    selected_lens = jnp.asarray([config.topk], dtype=jnp.int32)

    @jax.jit
    def run() -> jax.Array:
        return sparse_mla_decode(
            q_nope,
            q_pe,
            selected,
            selected_lens,
            sm_scale=576**-0.5,
            block_size=config.mla_block_size,
        )

    return _measure(f"sparse_mla_k{config.topk}_block{config.mla_block_size}",
                    run, config)


def benchmark_gather_sparse_mla(
    config: BenchmarkConfig, ) -> dict[str, object]:
    """Measure selected-KV gather followed by sparse MLA."""
    latent_cache = _latent_cache(config)
    block_table = _block_table(config)
    topk_indices = jnp.linspace(0,
                                config.seq_len - 1,
                                config.topk,
                                dtype=jnp.int32)[None, :]
    q_nope = jnp.ones((1, 32, 512), dtype=jnp.bfloat16)
    q_pe = jnp.ones((1, 32, 64), dtype=jnp.bfloat16)

    @jax.jit
    def run(
        latent_cache: jax.Array,
        block_table: jax.Array,
        topk_indices: jax.Array,
        q_nope: jax.Array,
        q_pe: jax.Array,
    ) -> jax.Array:
        selected, selected_lens = gather_paged_kv(latent_cache, block_table,
                                                  topk_indices)
        return sparse_mla_decode(
            q_nope,
            q_pe,
            selected,
            selected_lens,
            sm_scale=576**-0.5,
            block_size=config.mla_block_size,
        )

    return _measure(
        f"gather_sparse_mla_n{config.seq_len}_k{config.topk}",
        lambda: run(latent_cache, block_table, topk_indices, q_nope, q_pe),
        config,
    )


def benchmark_decode_chain(config: BenchmarkConfig) -> dict[str, object]:
    """Measure exact scorer, selected-KV gather, and sparse MLA as one JIT."""
    index_cache = _index_cache(config)
    latent_cache = _latent_cache(config)
    block_table = _block_table(config)
    q_nope = jnp.ones((1, 32, 512), dtype=jnp.bfloat16)
    q_pe = jnp.ones((1, 32, 64), dtype=jnp.bfloat16)

    @jax.jit
    def run(
        index_cache: jax.Array,
        latent_cache: jax.Array,
        block_table: jax.Array,
        q_nope: jax.Array,
        q_pe: jax.Array,
    ) -> jax.Array:
        topk_indices = _streamindex(config, index_cache, block_table)
        selected, selected_lens = gather_paged_kv(latent_cache, block_table,
                                                  topk_indices)
        return sparse_mla_decode(
            q_nope,
            q_pe,
            selected,
            selected_lens,
            sm_scale=576**-0.5,
            block_size=config.mla_block_size,
        )

    return _measure(
        f"dsa_decode_chain_n{config.seq_len}_k{config.topk}",
        lambda: run(index_cache, latent_cache, block_table, q_nope, q_pe),
        config,
    )


_BENCHMARKS: dict[str, Callable[[BenchmarkConfig], dict[str, object]]] = {
    "topk": benchmark_topk,
    "scorer": benchmark_scorer,
    "index_cache_insert": benchmark_index_cache_insert,
    "sparse_mla": benchmark_sparse_mla,
    "gather_sparse_mla": benchmark_gather_sparse_mla,
    "decode_chain": benchmark_decode_chain,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("all", *_BENCHMARKS), default="all")
    parser.add_argument("--seq-len", type=int, default=262_144)
    parser.add_argument("--topk", type=int, default=2_048)
    parser.add_argument("--pages-per-block", type=int, default=4)
    parser.add_argument("--mla-block-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    config = BenchmarkConfig(
        seq_len=args.seq_len,
        topk=args.topk,
        pages_per_block=args.pages_per_block,
        mla_block_size=args.mla_block_size,
        warmup=args.warmup,
        samples=args.samples,
    )
    if jax.default_backend() != "tpu":
        parser.error("this performance benchmark requires a TPU backend")

    names = list(_BENCHMARKS) if args.mode == "all" else [args.mode]
    result = {
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "config": asdict(config),
        "measurements": [_BENCHMARKS[name](config) for name in names],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
