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
import itertools
import time
from typing import List, NamedTuple, Tuple

import click
import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from tpu_inference.kernels.ragged_paged_attention.v3 import kernel, kernel_hd64

console = Console()


# Define NamedTuples for clearer structure
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


def cdiv(a, b):
    return (a + b - 1) // b


def get_qkv_lens_example(max_num_tokens, max_model_len, actual_num_seqs):
    assert max_num_tokens >= actual_num_seqs
    decode_end = actual_num_seqs - 1
    cu_q_lens = list(range(actual_num_seqs + 1))
    cu_q_lens[-1] = min(max_num_tokens, max_model_len)
    kv_lens = [max_model_len for _ in range(actual_num_seqs)]
    return cu_q_lens, kv_lens, decode_end


def make_rpa_configs(page_sizes, q_dtypes, kv_dtypes, num_q_heads_list,
                     num_kv_heads_list, head_dims, max_model_lens, bkv_p_lst,
                     bq_sz_lst) -> List[Tuple[RpaKey, RpaBlock]]:
    """Generates a flat list of all configuration and block size combinations."""

    # All high-level params
    input_combos = itertools.product(page_sizes, q_dtypes, kv_dtypes,
                                     num_q_heads_list, num_kv_heads_list,
                                     head_dims, max_model_lens)

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


def benchmark_kernel(rpa_key: RpaKey,
                     rpa_block: RpaBlock,
                     total_num_pages=1000,
                     num_iterations=100,
                     vmem_limit_bytes=60 * 1024 * 1024,
                     smem_limit_bytes=0.9 * 1024 * 1024,
                     dry_run=False,
                     num_sequences=35):
    """Benchmarks a single configuration (Key + Block)."""

    # Unpack Key
    (max_model_len, q_dtype_name, kv_dtype_name, num_q_heads, num_kv_heads,
     head_dim, page_size) = rpa_key

    # Unpack Block
    num_kv_pages_per_block, num_q_per_block = rpa_block

    if dry_run:
        return 1.0, 0.0  # Mock latency

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
    rpa_fn = mod.ragged_paged_attention_hd64 if head_dim == 64 else mod.ragged_paged_attention

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
    # rng_key = jax.random.PRNGKey(0)  # Reproducible - unused but kept for reference if needed, or remove completely. 
    # Ruff says it's unused.

    q = jnp.zeros(q_shape, dtype=q_dtype)
    k = jnp.zeros(kv_shape, dtype=kv_dtype)
    v = jnp.zeros(kv_shape, dtype=kv_dtype)
    kv_cache = jnp.zeros(kv_cache_shape, dtype=kv_dtype)

    page_indices = np.random.randint(0,
                                     total_num_pages,
                                     size=(max_num_seqs * pages_per_seq, ),
                                     dtype=np.int32)
    page_indices = jnp.array(page_indices, dtype=jnp.int32)
    distribution = jnp.array([decode_end, decode_end, actual_num_seqs],
                             dtype=jnp.int32)

    args = [q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution]

    # Pre-checks
    vmem = mod.get_vmem_estimate_bytes(num_q_heads, num_kv_heads, head_dim,
                                       num_q_per_block, num_kv_pages_per_block,
                                       q_dtype, kv_dtype)
    if vmem > vmem_limit_bytes:
        return float('inf'), 0.0

    smem = mod.get_smem_estimate_bytes(max_num_seqs, pages_per_seq)
    if smem > smem_limit_bytes:
        return float('inf'), 0.0

    # Validate
    try:
        mod.dynamic_validate_inputs(*args, **kwargs)
    except Exception:
        return float('inf'), 0.0

    # Execution
    try:
        # Warmup
        ret = jax.block_until_ready(rpa_fn(*args, **kwargs))
        if isinstance(ret, (list, tuple)) and len(ret) >= 2:
            args[3] = ret[1]  # Handle donation

        if isinstance(ret, (list, tuple)) and len(ret) >= 2:
            args[3] = ret[1]  # Handle donation

        timings = []
        for _ in range(num_iterations):
            start = time.perf_counter_ns()
            ret = jax.block_until_ready(rpa_fn(*args, **kwargs))
            end = time.perf_counter_ns()
            timings.append(end - start)
            if isinstance(ret, (list, tuple)) and len(ret) >= 2:
                args[3] = ret[1]

        t_arr = np.array(timings)
        return np.mean(t_arr), np.std(t_arr)

    except Exception:
        # console.print(f"Error: {e}")
        return float('inf'), 0.0


@click.command()
@click.option('--page-size',
              default='128',
              help="Comma separated list of page sizes",
              show_default=True)
@click.option('--q-dtype',
              default='bfloat16',
              help="Comma separated list of q dtypes",
              show_default=True)
@click.option('--kv-dtype',
              default='bfloat16',
              help="Comma separated list of kv dtypes",
              show_default=True)
@click.option('--num-q-heads',
              default='128',
              help="Comma separated list of num q heads",
              show_default=True)
@click.option('--num-kv-heads',
              default='1',
              help="Comma separated list of num kv heads",
              show_default=True)
@click.option('--head-dim',
              default='128',
              help="Comma separated list of head dims",
              show_default=True)
@click.option('--max-model-len',
              default='1024',
              help="Comma separated list of max model lengths",
              show_default=True)
@click.option('--num-iterations',
              default=100,
              help="Number of iterations for benchmarking",
              show_default=True)
@click.option('--dry-run',
              is_flag=True,
              help="Run without actual kernel calls")
@click.option('--num-sequences',
              default=35,
              help="Number of sequences for autotuning example",
              show_default=True)
@click.option('--csv-file',
              default=None,
              help="Optional path to output results to a CSV file")
@click.option('--kv-block-sizes',
              default='1,2,4,8,16,32,64,128',
              help="Comma separated list of KV pages per block to test",
              show_default=True)
@click.option('--q-block-sizes',
              default='8,16,32,64,128,256',
              help="Comma separated list of Q queries per block to test",
              show_default=True)
def main(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim,
         max_model_len, num_iterations, dry_run, num_sequences, csv_file,
         kv_block_sizes, q_block_sizes):

    def parse_arg(arg, type_fn=str):
        if isinstance(arg, str):
            res = [type_fn(x.strip()) for x in arg.split(',')]
            return res
        return [type_fn(arg)]

    # Parse args
    page_sizes = parse_arg(page_size, int)
    q_dtypes = parse_arg(q_dtype, str)
    kv_dtypes = parse_arg(kv_dtype, str)
    num_q_heads_list = parse_arg(num_q_heads, int)
    num_kv_heads_list = parse_arg(num_kv_heads, int)
    head_dims = parse_arg(head_dim, int)
    max_model_lens = parse_arg(max_model_len, int)

    bkv_p_lst = parse_arg(kv_block_sizes, int)
    bq_sz_lst = parse_arg(q_block_sizes, int)

    # Setup CSV with context manager (replaces try/finally)
    csv_context = open(csv_file, 'w',
                       newline='') if csv_file else contextlib.nullcontext()

    with csv_context as csv_f:
        csv_writer = None
        if csv_f:
            try:
                fieldnames = [
                    "page_size", "q_dtype", "kv_dtype", "num_q_heads",
                    "num_kv_heads", "head_dim", "max_model_len",
                    "num_kv_pages_per_block", "num_q_per_block", "time_ns",
                    "time_std_ns", "is_best"
                ]
                csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
                csv_writer.writeheader()
                console.print(f"Streaming results to {csv_file}")
            except IOError as e:
                console.print(f"Error opening CSV file {csv_file}: {e}")
                return

        # Flatten Configs
        configs = make_rpa_configs(page_sizes, q_dtypes, kv_dtypes,
                                   num_q_heads_list, num_kv_heads_list,
                                   head_dims, max_model_lens, bkv_p_lst,
                                   bq_sz_lst)

        console.print(
            f"Generated {len(configs)} configuration-block combinations to tune."
        )

        # Results Aggregation
        aggregated_results = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(
                dict)))

        best_results = {}  # Map RpaKey -> (min_latency, best_block)

        from rich.progress import (MofNCompleteColumn, TaskProgressColumn,
                                   TimeElapsedColumn)

        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(),
                      TaskProgressColumn(),
                      MofNCompleteColumn(),
                      TimeElapsedColumn(),
                      TimeRemainingColumn(),
                      console=console) as progress:
            task = progress.add_task("[green]Tuning...", total=len(configs))

            for rpa_key, rpa_block in configs:
                desc_str = f"Tuning {rpa_block.num_kv_pages_per_block}x{rpa_block.num_q_per_block} for Len={rpa_key.max_model_len}"
                progress.update(task, description=f"[green]{desc_str}")

                t_mean, t_std = benchmark_kernel(
                    rpa_key,
                    rpa_block,
                    num_iterations=num_iterations,
                    dry_run=dry_run,
                    num_sequences=num_sequences,
                )

                if t_mean == float('inf'):
                    progress.update(task, advance=1)
                    continue

                # Update Best
                is_best = False
                current = best_results.get(rpa_key)
                if current is None or t_mean < current[0]:
                    best_results[rpa_key] = (t_mean, rpa_block)
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
                        "is_best": is_best
                    }
                    csv_writer.writerow(row)
                    csv_f.flush()

                progress.update(task, advance=1)

    # Populate Aggregated Results for Print
    for rpa_key, (t_mean, best_block) in best_results.items():
        (m_len, q_dt, kv_dt, n_q, n_kv, h_dim, ps) = rpa_key

        len_key = f"max_model_len-{m_len}-sw-None"
        dtype_key = f"q_{q_dt}_kv_{kv_dt}"
        config_key = f"q_head-{n_q}_kv_head-{n_kv}_head-{h_dim}"

        # Format block as tuple (kv, q)
        aggregated_results[ps][dtype_key][config_key][len_key] = (
            best_block.num_kv_pages_per_block, best_block.num_q_per_block)

    # Print Final Output
    print("\n[Output for tpu-inference/tuned_block_sizes.py]:")
    tpu_device_key = "TPU v6e (Place Inside Correct Device Key)"

    print(f"'{tpu_device_key}': {{")
    for ps in sorted(aggregated_results.keys()):
        print(f"    {ps}: {{")
        for dtype_key in sorted(aggregated_results[ps].keys()):
            print(f"        '{dtype_key}': {{")
            for config_key in sorted(aggregated_results[ps][dtype_key].keys()):
                print(f"            '{config_key}': {{")
                len_dict = aggregated_results[ps][dtype_key][config_key]
                for len_key in sorted(len_dict.keys()):
                    print(f"                '{len_key}': {len_dict[len_key]},")
                print("            },")
            print("        },")
        print("    },")
    print("},")

    if csv_file:
        console.print(f"\nResults written to {csv_file}")


if __name__ == "__main__":
    main()
