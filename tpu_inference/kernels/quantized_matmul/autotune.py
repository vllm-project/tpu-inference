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
import csv
import itertools
import time
from typing import List, NamedTuple

import click
import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from tpu_inference.kernels.quantized_matmul import tuned_block_sizes, util


class TestResult(NamedTuple):
    tuned_value: tuned_block_sizes.TunedValue
    latency: float


def factors_of_n(target: int, n: int) -> List[int]:
    target = util.next_multiple(target, n)
    i = n
    factors = []
    while target >= i:
        if target % i == 0:
            factors.append(i)
        i += n
    return factors


def make_configs(batch_sizes, out_in_features, x_q_dtype, w_q_dtype):
    configs = set()
    for batch_size in batch_sizes:
        if batch_size < 128:
            batch_block_sizes = [batch_size]
        else:
            batch_block_sizes = factors_of_n(batch_size, 128)

        for out_feature, in_feature in out_in_features:
            tuned_key = tuned_block_sizes.get_key(batch_size, out_feature,
                                                  in_feature, x_q_dtype,
                                                  w_q_dtype)
            out_block_sizes = factors_of_n(out_feature, 128)
            in_block_sizes = factors_of_n(in_feature, 128)

            values = itertools.product(batch_block_sizes, out_block_sizes,
                                       in_block_sizes)

            for value in values:
                configs.add((tuned_key, tuned_block_sizes.TunedValue(*value)))

    return list(configs)


def autotune_kernel(
    tuned_key,
    tuned_value,
    dtype=jnp.bfloat16,
    num_iter=10,
    dry_run=False,
    console=None,
):
    if dry_run:
        return 1.0, 0.0  # Mock latency (mean, std)

    msg_print = console.print if console else print

    from tpu_inference.kernels.quantized_matmul import kernel

    # Generate inputs
    key = jax.random.key(1234)
    x = jax.random.uniform(key, (tuned_key.n_batch, tuned_key.n_in),
                           dtype=dtype)

    # Quantize weight
    w_q_dtype_obj = jnp.dtype(tuned_key.w_q_dtype)
    w_fp = jax.random.uniform(key, (tuned_key.n_out, tuned_key.n_in),
                              dtype=dtype)
    w_q, w_scale = util.quantize_tensor(w_fp, w_q_dtype_obj, dim=1)
    w_scale = jnp.squeeze(w_scale)

    x_q_dtype_obj = jnp.dtype(tuned_key.x_q_dtype)

    kernel_name = util.get_kernel_name(tuned_value)

    try:
        # Compile kernel
        fn = kernel.quantized_matmul_kernel.lower(
            x,
            w_q,
            w_scale,
            x_q_dtype=x_q_dtype_obj,
            tuned_value=tuned_value,
        ).compile()
    except Exception as e:
        msg_print(f"Failed to compile {kernel_name}: {e}")
        return float('inf'), 0.0

    try:
        # Warmup
        fn(x, w_q, w_scale).block_until_ready()

        # Timing loop
        timings = []
        for _ in range(num_iter):
            start = time.perf_counter_ns()
            fn(x, w_q, w_scale).block_until_ready()
            end = time.perf_counter_ns()
            timings.append(end - start)

        timings = np.array(timings)
        avg_time_ns = np.mean(timings)
        std_time_ns = np.std(timings)
        return avg_time_ns, std_time_ns

    except Exception as e:
        msg_print(f"Execution failed for {kernel_name}: {e}")
        return float('inf'), 0.0


@click.command()
@click.option('--batch-sizes',
              required=True,
              help="Comma separated batch sizes")
@click.option('--out-in-features',
              required=True,
              help="Comma separated out/in features ex: 2048/4096")
@click.option('--x-q-dtype', default='int8')
@click.option('--w-q-dtype', default='int8')
@click.option('--dry-run',
              is_flag=True,
              help="Run without actual kernel calls")
@click.option('--num-iterations',
              default=10,
              help="Number of iterations for benchmarking")
@click.option('--csv-file',
              default=None,
              help="Optional path to output results to a CSV file")
def main(batch_sizes, out_in_features, x_q_dtype, w_q_dtype, dry_run,
         num_iterations, csv_file):
    batch_sizes_list = [int(x) for x in batch_sizes.split(',')]
    out_in_features_list = [
        tuple(int(x) for x in feature.split('/'))
        for feature in out_in_features.split(',')
    ]

    configs = make_configs(batch_sizes_list, out_in_features_list, x_q_dtype,
                           w_q_dtype)
    print(f"Generated {len(configs)} configurations to tune.")

    # Setup CSV writing
    csv_f = None
    csv_writer = None
    if csv_file:
        try:
            csv_f = open(csv_file, 'w', newline='')
            fieldnames = [
                "batch_size", "out_feature", "in_feature", "x_q_dtype",
                "w_q_dtype", "batch_block_size", "out_block_size",
                "in_block_size", "time_ns", "time_std_ns", "is_best"
            ]
            csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_f.flush()
            print(f"Streaming results to {csv_file}")
        except IOError as e:
            print(f"Error opening CSV file {csv_file}: {e}")
            return

    results = collections.defaultdict(list)
    results_best = {}  # Track best per key for is_best flag

    try:
        console = Console()
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

            for tuned_key, tuned_value in configs:
                # Update description to show current config
                progress.update(
                    task,
                    description=
                    f"[green]Tuning {tuned_value.batch_block_size}x{tuned_value.out_block_size}x{tuned_value.in_block_size}..."
                )

                latency_ns, std_ns = autotune_kernel(tuned_key,
                                                     tuned_value,
                                                     dry_run=dry_run,
                                                     console=console,
                                                     num_iter=num_iterations)

                # Check if best so far
                is_best = False
                current_best = results_best.get(tuned_key)
                if current_best is None or latency_ns < current_best:
                    results_best[tuned_key] = latency_ns
                    is_best = True

                results[tuned_key].append(TestResult(tuned_value, latency_ns))

                if is_best:
                    # console.print(f"New Best: ... -> {latency_ns/1e9:.6f} s")
                    pass

                if csv_writer:
                    row = {
                        "batch_size": tuned_key.n_batch,
                        "out_feature": tuned_key.n_out,
                        "in_feature": tuned_key.n_in,
                        "x_q_dtype": tuned_key.x_q_dtype,
                        "w_q_dtype": tuned_key.w_q_dtype,
                        "batch_block_size": tuned_value.batch_block_size,
                        "out_block_size": tuned_value.out_block_size,
                        "in_block_size": tuned_value.in_block_size,
                        "time_ns": latency_ns,
                        "time_std_ns": std_ns,
                        "is_best": is_best
                    }
                    csv_writer.writerow(row)
                    csv_f.flush()

                progress.update(task, advance=1)
    finally:
        if csv_f:
            csv_f.close()

    # Find best
    print("\n[Output for tpu-inference/tuned_block_sizes.py]:")
    print("TUNED_BLOCK_SIZES_RAW = {")
    sorted_keys = sorted(results.keys(), key=lambda k: (k.n_batch, k.n_out))

    for key in sorted_keys:
        values = results[key]
        best = min(values, key=lambda x: x.latency)
        # Format key as tuple
        # Key: (tpu_version, n_batch, n_out, n_in, x_q_dtype, w_q_dtype)
        key_tuple = (key.tpu_version, key.n_batch, key.n_out, key.n_in,
                     key.x_q_dtype, key.w_q_dtype)
        val_tuple = (best.tuned_value.batch_block_size,
                     best.tuned_value.out_block_size,
                     best.tuned_value.in_block_size)
        print(f"    {key_tuple}: {val_tuple},")
    print("}")


if __name__ == "__main__":
    main()
