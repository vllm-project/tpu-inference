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

import collections
import functools
import itertools
import os
import time
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference import utils as tpu_utils
from tpu_inference.kernels.ragged_paged_attention.v3 import kernel, kernel_hd64
from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes import \
    get_lookup_keys
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv
from tpu_inference.tools.autotune import benchmarks, utils

console = utils.console


class RpaKey(NamedTuple):
    max_model_len: int
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int


class RpaBlock(NamedTuple):
    num_kv_pages_per_block: int
    num_q_per_block: int


def get_kernel_module(head_dim):
    if head_dim == 64:
        return kernel_hd64
    return kernel


def get_qkv_lens_example(max_num_tokens, max_model_len, actual_num_seqs):
    assert max_num_tokens >= actual_num_seqs
    decode_end = actual_num_seqs - 1
    cu_q_lens = list(range(actual_num_seqs + 1))
    cu_q_lens[-1] = min(max_num_tokens, max_model_len)
    kv_lens = [max_model_len for _ in range(actual_num_seqs)]
    return cu_q_lens, kv_lens, decode_end


def make_rpa_configs(
    page_sizes,
    q_dtypes,
    kv_dtypes,
    num_q_heads_list,
    num_kv_heads_list,
    head_dims,
    max_model_lens,
    bkv_p_lst,
    bq_sz_lst,
) -> List[Tuple[RpaKey, RpaBlock]]:
    """Generates a flat list of all configuration and block size combinations."""

    # All high-level params
    input_combos = itertools.product(
        page_sizes,
        q_dtypes,
        kv_dtypes,
        num_q_heads_list,
        num_kv_heads_list,
        head_dims,
        max_model_lens,
    )

    configs = []

    for ps, q_dt, kv_dt, n_q, n_kv, h_dim, m_len in input_combos:
        # Validate input config
        if n_q % n_kv != 0:
            continue

        key = RpaKey(m_len, q_dt, kv_dt, n_q, n_kv, h_dim, ps)
        pages_per_seq = cdiv(m_len, ps)

        # All block params for this input
        for kv_blk in bkv_p_lst:
            # Filter invalid block params
            if kv_blk > pages_per_seq:
                continue
            if ps * kv_blk > 4096:
                continue

            for q_blk in bq_sz_lst:
                block = RpaBlock(kv_blk, q_blk)
                configs.append((key, block))

    return configs


def benchmark_kernel(
    rpa_key: RpaKey,
    rpa_block: RpaBlock,
    total_num_pages=1000,
    num_iterations=100,
    num_repeats=5,
    benchmarking_method: benchmarks.BenchmarkMethod = benchmarks.
    BenchmarkMethod.AMORTIZED,
    vmem_limit_bytes=60 * 1024 * 1024,
    smem_limit_bytes=0.9 * 1024 * 1024,
    dry_run=False,
    num_sequences=35,
) -> benchmarks.BenchmarkResult:
    """Benchmarks a single configuration (Key + Block)."""

    # Unpack Key
    (
        max_model_len,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
    ) = rpa_key

    # Unpack Block
    num_kv_pages_per_block, num_q_per_block = rpa_block

    if dry_run:
        return benchmarks.BenchmarkResult(mean_time_ns=1.0,
                                          std_time_ns=0.0,
                                          min_time_ns=1.0,
                                          max_time_ns=1.0,
                                          samples_ns=[1.0],
                                          metadata={
                                              "compile_time_s": 0.0,
                                              "lower_time_s": 0.0
                                          })

    # Handle string dtypes
    q_dtype = jnp.dtype(q_dtype_name)
    kv_dtype = jnp.dtype(kv_dtype_name)

    pages_per_seq = cdiv(max_model_len, page_size)

    # Setup Data
    actual_num_seqs = num_sequences
    max_num_seqs = max(128, actual_num_seqs)
    max_num_tokens = max(max_model_len, actual_num_seqs)

    example = get_qkv_lens_example(max_num_tokens, max_model_len,
                                   actual_num_seqs)
    cu_q_lens, kv_lens, decode_end = example

    # Resolve kernel
    mod = get_kernel_module(head_dim)
    rpa_fn = (mod.ragged_paged_attention_hd64
              if head_dim == 64 else mod.ragged_paged_attention)

    # Validation
    kwargs = {
        "num_kv_pages_per_block": num_kv_pages_per_block,
        "num_queries_per_block": num_q_per_block,
        "vmem_limit_bytes": vmem_limit_bytes,
    }

    # Prepare Inputs
    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
    kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
    kv_lens = jnp.pad(kv_lens, (0, max_num_seqs - kv_lens.shape[0]))

    q_shape = (max_num_tokens, num_q_heads, head_dim)
    kv_shape = (max_num_tokens, num_kv_heads, head_dim)
    kv_cache_shape = mod.get_kv_cache_shape(total_num_pages, page_size,
                                            num_kv_heads, head_dim, kv_dtype)

    # Random Data
    rng_key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    q = jax.random.uniform(k1, q_shape, dtype=q_dtype)
    k = jax.random.uniform(k2, kv_shape, dtype=kv_dtype)
    v = jax.random.uniform(k3, kv_shape, dtype=kv_dtype)
    kv_cache = jax.random.uniform(k4, kv_cache_shape, dtype=kv_dtype)

    page_indices = np.random.randint(0,
                                     total_num_pages,
                                     size=(max_num_seqs * pages_per_seq, ),
                                     dtype=np.int32)
    page_indices = jnp.array(page_indices, dtype=jnp.int32)
    distribution = jnp.array([decode_end, decode_end, actual_num_seqs],
                             dtype=jnp.int32)

    args = [q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution]

    # Pre-checks
    vmem = mod.get_vmem_estimate_bytes(
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_q_per_block,
        num_kv_pages_per_block,
        q_dtype,
        kv_dtype,
    )
    if vmem > vmem_limit_bytes:
        return benchmarks.BenchmarkResult(float("inf"), 0.0, 0.0, 0.0, [], {})

    smem = mod.get_smem_estimate_bytes(max_num_seqs, pages_per_seq)
    if smem > smem_limit_bytes:
        return benchmarks.BenchmarkResult(float("inf"), 0.0, 0.0, 0.0, [], {})

    # Validate
    try:
        mod.dynamic_validate_inputs(*args, **kwargs)
    except Exception as e:
        console.print(f"[red]Error in validation: {e}[/red]")
        return benchmarks.BenchmarkResult(float("inf"), 0.0, 0.0, 0.0, [], {})

    # Execution
    try:
        # Explicit Compilation for Stats
        t0 = time.perf_counter()
        lowered = rpa_fn.lower(*args, **kwargs)
        t1 = time.perf_counter()
        lower_time = t1 - t0

        t2 = time.perf_counter()
        _ = lowered.compile()
        t3 = time.perf_counter()
        compile_time = t3 - t2

        # Prepare function for benchmarking
        # NOTE: benchmark_kernel manages JIT and amortization depending on method
        fn_to_benchmark = functools.partial(rpa_fn, **kwargs)

        result = benchmarks.benchmark_kernel(
            benchmark_fn=fn_to_benchmark,
            args=tuple(args),
            num_iterations=num_iterations,
            num_repeats=num_repeats,
            method=benchmarking_method,
        )

        # Append compile stats to result metadata
        result.metadata["compile_time_s"] = compile_time
        result.metadata["lower_time_s"] = lower_time

        return result

    except Exception as e:
        console.print(f"[red]Error in execution: {e}[/red]")
        return benchmarks.BenchmarkResult(float("inf"), 0.0, 0.0, 0.0, [], {})


def tune_rpa(
    page_sizes: List[int],
    q_dtypes: List[str],
    kv_dtypes: List[str],
    num_q_heads_list: List[int],
    num_kv_heads_list: List[int],
    head_dims: List[int],
    max_model_lens: List[int],
    kv_block_sizes: List[int],
    q_block_sizes: List[int],
    num_iterations: int = 100,
    num_repeats: int = 5,
    benchmarking_method: str = "amortized",
    dry_run: bool = False,
    num_sequences: int = 35,
    update_registry: bool = False,
    tp_size: int = 1,
    run_name: Optional[str] = None,
    output_dir: str = "tuning_runs",
    no_save: bool = False,
):
    """Main entry point for tuning RPA kernels."""

    # Resolve benchmarking method
    try:
        method = benchmarks.BenchmarkMethod(benchmarking_method)
    except ValueError:
        raise ValueError(
            f"Invalid benchmarking method: {benchmarking_method}. "
            f"Choose from {[m.value for m in benchmarks.BenchmarkMethod]}")

    if tp_size > 1:
        console.print(
            f"[bold cyan]Applying TP Scaling (TP={tp_size})[/bold cyan]")

        # Scale Q Heads
        num_q_heads_list = [
            utils.apply_tp_scaling(x, tp_size, "num_q_heads")
            for x in num_q_heads_list
        ]

        # Scale KV Heads
        num_kv_heads_list = [
            utils.apply_tp_scaling(x, tp_size, "num_kv_heads")
            for x in num_kv_heads_list
        ]

    # Setup CSV with context manager
    fieldnames = [
        "page_size",
        "q_dtype",
        "kv_dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "max_model_len",
        "num_kv_pages_per_block",
        "num_q_per_block",
        "time_ns",
        "time_std_ns",
        "compile_time_s",
        "lower_time_s",
        "is_best",
        "benchmarking_method",
    ]

    # Setup Experiment Run
    run_ctx = utils.RunContext(run_name, output_dir, no_save)

    # Save Metadata
    run_ctx.save_metadata({
        "kernel": "rpa_v3",
        "cli_args": {
            "page_sizes": page_sizes,
            "q_dtypes": q_dtypes,
            "kv_dtypes": kv_dtypes,
            "num_q_heads_list": num_q_heads_list,
            "num_kv_heads_list": num_kv_heads_list,
            "head_dims": head_dims,
            "max_model_lens": max_model_lens,
            "num_iterations": num_iterations,
            "num_repeats": num_repeats,
            "benchmarking_method": benchmarking_method,
            "tp_size": tp_size
        }
    })

    # Setup CSV with context manager
    fieldnames = [
        "page_size",
        "q_dtype",
        "kv_dtype",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "max_model_len",
        "num_kv_pages_per_block",
        "num_q_per_block",
        "time_ns",
        "time_std_ns",
        "compile_time_s",
        "lower_time_s",
        "is_best",
        "benchmarking_method",
    ]

    with run_ctx.open_csv(fieldnames) as csv_writer:

        # Flatten Configs
        configs = make_rpa_configs(
            page_sizes,
            q_dtypes,
            kv_dtypes,
            num_q_heads_list,
            num_kv_heads_list,
            head_dims,
            max_model_lens,
            kv_block_sizes,
            q_block_sizes,
        )

        console.print(
            f"Generated {len(configs)} configuration-block combinations to tune "
            f"using '{method.value}' method.")

        # Results Aggregation
        aggregated_results = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(
                dict)))

        best_results = {}  # Map RpaKey -> (BenchmarkResult, is_best)

        with utils.setup_progress() as progress:
            task = progress.add_task("[green]Tuning...", total=len(configs))

            for rpa_key, rpa_block in configs:
                desc_str = f"Tuning {rpa_block.num_kv_pages_per_block}x{rpa_block.num_q_per_block} for Len={rpa_key.max_model_len}"
                progress.update(task, description=f"[green]{desc_str}")

                result = benchmark_kernel(
                    rpa_key,
                    rpa_block,
                    num_iterations=num_iterations,
                    num_repeats=num_repeats,
                    benchmarking_method=method,
                    dry_run=dry_run,
                    num_sequences=num_sequences,
                )

                if result.mean_time_ns == float("inf"):
                    progress.update(task, advance=1)
                    continue

                # Update Best
                is_best = False
                current_best = best_results.get(rpa_key)
                # Compare mean latency
                if current_best is None or result.mean_time_ns < current_best[
                        0].mean_time_ns:
                    best_results[rpa_key] = (result, rpa_block)
                    is_best = True

                # CSV Logging
                if csv_writer:
                    row = {
                        "page_size":
                        rpa_key.page_size,
                        "q_dtype":
                        rpa_key.q_dtype,
                        "kv_dtype":
                        rpa_key.kv_dtype,
                        "num_q_heads":
                        rpa_key.num_q_heads,
                        "num_kv_heads":
                        rpa_key.num_kv_heads,
                        "head_dim":
                        rpa_key.head_dim,
                        "max_model_len":
                        rpa_key.max_model_len,
                        "num_kv_pages_per_block":
                        rpa_block.num_kv_pages_per_block,
                        "num_q_per_block":
                        rpa_block.num_q_per_block,
                        "time_ns":
                        result.mean_time_ns,
                        "time_std_ns":
                        result.std_time_ns,
                        "compile_time_s":
                        result.metadata.get("compile_time_s", 0.0),
                        "lower_time_s":
                        result.metadata.get("lower_time_s", 0.0),
                        "is_best":
                        is_best,
                        "benchmarking_method":
                        method.value,
                    }
                    csv_writer.writerow(row)
                    # We might want to persist samples in another file or extended CSV if debugging

                progress.update(task, advance=1)

    # Populate Aggregated Results for Log/Printing
    for rpa_key, (result, best_block) in best_results.items():
        (m_len, q_dt, kv_dt, n_q, n_kv, h_dim, ps) = rpa_key

        d_name, ps_key, dtype_key, config_key, len_key = get_lookup_keys(
            ps,
            q_dt,
            kv_dt,
            n_q,
            n_kv,
            h_dim,
            m_len,
            None,  # sliding_window
        )

        aggregated_results[ps_key][dtype_key][config_key][len_key] = {
            "config": {
                "num_kv_pages_per_block": best_block.num_kv_pages_per_block,
                "num_q_per_block": best_block.num_q_per_block,
            },
            "stats": {
                "latency_avg_ns": result.mean_time_ns,
                "latency_std_ns": result.std_time_ns,
                "compile_time_s": result.metadata.get("compile_time_s", 0.0),
                "lower_time_s": result.metadata.get("lower_time_s", 0.0),
            },
            "metadata": {
                "benchmarking_method": method.value,
                "num_repeats": num_repeats,
                "samples_ns": result.samples_ns
            }
        }

    # Print JSON output
    # Ensure top-level keys (page_size) are strings for JSON
    json_results = {str(k): v for k, v in aggregated_results.items()}

    # Save Results
    run_ctx.save_results(json_results)

    # Flatten for Display (Rich Table expects Key -> Entry)
    display_results = {}
    for ps_key, v1 in aggregated_results.items():
        for dtype_key, v2 in v1.items():
            for config_key, v3 in v2.items():
                for len_key, entry in v3.items():
                    # Create a readable key for the TABLE
                    flat_key = f"{ps_key} | {len_key}"
                    display_results[flat_key] = entry

    run_ctx.print_summary_table(display_results)

    if update_registry:
        tpu_version = tpu_utils.get_tpu_name_slug()
        norm_name = utils.get_registry_file_name(tpu_version)

        base_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))  # tpu_inference/
        data_dir = os.path.join(
            base_dir, "kernels/tuned_data/ragged_paged_attention/v3")
        target_file = os.path.join(data_dir, f"{norm_name}.json")

        console.print(f"[bold]Updating registry at {target_file}...[/bold]")
        utils.update_json_registry(target_file, json_results)
