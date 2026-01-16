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
import contextlib
import csv
import functools
import itertools
import json
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
from tpu_inference.tools.autotune import utils

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
    vmem_limit_bytes=1024 * 1024 * 1024,
    smem_limit_bytes=0.9 * 1024 * 1024,
    dry_run=False,
    num_sequences=35,
):
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
        return 1.0, 0.0, 0.0, 0.0  # Mock latency

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
        return float("inf"), 0.0, 0.0, 0.0

    smem = mod.get_smem_estimate_bytes(max_num_seqs, pages_per_seq)
    if smem > smem_limit_bytes:
        return float("inf"), 0.0, 0.0, 0.0

    # Validate
    try:
        mod.dynamic_validate_inputs(*args, **kwargs)
    except Exception:
        return float("inf"), 0.0, 0.0, 0.0

    # Execution
    try:
        # Explicit Compilation
        t0 = time.perf_counter()
        lowered = rpa_fn.lower(*args, **kwargs)
        t1 = time.perf_counter()
        lower_time = t1 - t0

        t2 = time.perf_counter()
        _ = lowered.compile()
        t3 = time.perf_counter()
        compile_time = t3 - t2

        # Compile amortized function (Use source function rpa_fn)
        # We ignore return values/donation inside the loop for benchmarking stability
        # The latency of N repetitions of the same op is what we measure.
        fn_to_wrap = functools.partial(rpa_fn, **kwargs)
        amortized_fn = utils.amortized_wrapper(fn_to_wrap,
                                               n_iter=num_iterations)
        amortized_jit = jax.jit(amortized_fn)

        # Warmup (Only pass args, kwargs are frozen)
        outputs = amortized_jit(*args)
        utils.block_until_ready(outputs)

        # Measure (One call = num_iterations runs)
        start = time.perf_counter_ns()
        outputs = amortized_jit(*args)
        utils.block_until_ready(outputs)
        end = time.perf_counter_ns()

        total_time_ns = end - start
        avg_time_ns = total_time_ns / num_iterations
        std_time_ns = 0.0

        return avg_time_ns, std_time_ns, compile_time, lower_time

    except Exception:
        return float("inf"), 0.0, 0.0, 0.0


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
    dry_run: bool = False,
    num_sequences: int = 35,
    csv_file: Optional[str] = None,
    update_registry: bool = False,
    tp_size: int = 1,
):
    """Main entry point for tuning RPA kernels."""
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
    csv_context = open(csv_file, 'w',
                       newline='') if csv_file else contextlib.nullcontext()

    with csv_context as csv_f:
        csv_writer = None
        if csv_f:
            try:
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
                    "is_best",
                ]
                csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
                csv_writer.writeheader()
                console.print(f"Streaming results to {csv_file}")
            except IOError as e:
                console.print(f"Error opening CSV file {csv_file}: {e}")
                return

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
            f"Generated {len(configs)} configuration-block combinations to tune."
        )

        # Results Aggregation
        aggregated_results = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(
                dict)))

        best_results = {}  # Map RpaKey -> (min_latency, best_block)

        with utils.setup_progress() as progress:
            task = progress.add_task("[green]Tuning...", total=len(configs))

            for rpa_key, rpa_block in configs:
                desc_str = f"Tuning {rpa_block.num_kv_pages_per_block}x{rpa_block.num_q_per_block} for Len={rpa_key.max_model_len}"
                progress.update(task, description=f"[green]{desc_str}")

                t_mean, t_std, compile_time, lower_time = benchmark_kernel(
                    rpa_key,
                    rpa_block,
                    num_iterations=num_iterations,
                    dry_run=dry_run,
                    num_sequences=num_sequences,
                )

                if t_mean == float("inf"):
                    progress.update(task, advance=1)
                    continue

                # Update Best
                is_best = False
                current = best_results.get(rpa_key)
                if current is None or t_mean < current[0]:
                    # Store (latency, block, std, compile_time)
                    best_results[rpa_key] = (t_mean, rpa_block, t_std,
                                             compile_time, lower_time)
                    is_best = True

                # CSV Logging
                if csv_writer:
                    row = {
                        "page_size": rpa_key.page_size,
                        "q_dtype": rpa_key.q_dtype,
                        "kv_dtype": rpa_key.kv_dtype,
                        "num_q_heads": rpa_key.num_q_heads,
                        "num_kv_heads": rpa_key.num_kv_heads,
                        "head_dim": rpa_key.head_dim,
                        "max_model_len": rpa_key.max_model_len,
                        "num_kv_pages_per_block":
                        rpa_block.num_kv_pages_per_block,
                        "num_q_per_block": rpa_block.num_q_per_block,
                        "time_ns": t_mean,
                        "time_std_ns": t_std,
                        "compile_time_s": compile_time,
                        "lower_time_s": lower_time,
                        "is_best": is_best,
                    }
                    csv_writer.writerow(row)
                    csv_f.flush()

                progress.update(task, advance=1)

    # Populate Aggregated Results for Log/Printing
    # Populate Aggregated Results for Log/Printing

    for rpa_key, (t_mean, best_block, t_std, compile_time,
                  lower_time) in best_results.items():
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
        # device_name = d_name

        # Format block as tuple (kv, q)
        # Format as Rich Object Schema
        aggregated_results[ps_key][dtype_key][config_key][len_key] = {
            "config": {
                "num_kv_pages_per_block": best_block.num_kv_pages_per_block,
                "num_q_per_block": best_block.num_q_per_block,
            },
            "stats": {
                "latency_avg_ns": t_mean,
                "latency_std_ns": t_std,
                "compile_time_s": compile_time,
                "lower_time_s": lower_time,
                # "throughput_tok_s": ... # Add later if needed
            }
        }

    # Print JSON output
    tpu_version = tpu_utils.get_tpu_name_slug()
    norm_name = utils.get_registry_file_name(tpu_version)

    print(
        f"\n[Output for tpu-inference/kernels/tuned_data/rpa/{norm_name}.json]:"
    )

    # Ensure top-level keys (page_size) are strings for JSON
    json_results = {str(k): v for k, v in aggregated_results.items()}
    print(json.dumps(json_results, indent=2, sort_keys=True))

    if update_registry:
        base_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))  # tpu_inference/
        data_dir = os.path.join(base_dir, "kernels/tuned_data/rpa")
        target_file = os.path.join(data_dir, f"{norm_name}.json")

        console.print(f"[bold]Updating registry at {target_file}...[/bold]")
        utils.update_json_registry(target_file, json_results)

    if csv_file:
        console.print(f"\nResults written to {csv_file}")
