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
import itertools
import os
import time
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from tpu_inference import utils as tpu_utils
from tpu_inference.kernels.quantized_matmul import tuned_block_sizes, util
from tpu_inference.tools.autotune import benchmarks, utils

console = utils.console


class BenchmarkResult(NamedTuple):
    tuned_value: tuned_block_sizes.TunedValue
    latency: float
    std: float
    compile_time: float
    lower_time: float
    metadata: dict


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
    tpu_version = tpu_utils.get_tpu_generation()
    configs = set()
    for batch_size in batch_sizes:
        if batch_size < 128:
            batch_block_sizes = [batch_size]
        else:
            batch_block_sizes = factors_of_n(batch_size, 128)

        for out_feature, in_feature in out_in_features:
            tuned_key = tuned_block_sizes.TunedKey(tpu_version, batch_size,
                                                   out_feature, in_feature,
                                                   x_q_dtype, w_q_dtype)
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
    num_repeats=5,
    benchmarking_method: benchmarks.BenchmarkMethod = benchmarks.
    BenchmarkMethod.AMORTIZED,
    dry_run=False,
) -> benchmarks.BenchmarkResult:
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

    # kernel_name = util.get_kernel_name(tuned_value)

    try:
        # Compile kernel
        t0 = time.perf_counter()
        lowered = kernel.quantized_matmul_kernel.lower(
            x,
            w_q,
            w_scale,
            x_q_dtype=x_q_dtype_obj,
            tuned_value=tuned_value,
        )
        t1 = time.perf_counter()
        lower_time = t1 - t0

        t2 = time.perf_counter()
        _ = lowered.compile()
        t3 = time.perf_counter()
        compile_time = t3 - t2
    except Exception:
        # console.print(f"Failed to compile {kernel_name}: {e}")
        return benchmarks.BenchmarkResult(float("inf"), 0.0, 0.0, 0.0, [], {})

    try:
        # Prepare function for benchmarking
        import functools
        pallas_fn = functools.partial(kernel.quantized_matmul_kernel,
                                      x_q_dtype=x_q_dtype_obj,
                                      tuned_value=tuned_value)

        result = benchmarks.benchmark_kernel(
            benchmark_fn=pallas_fn,
            args=(x, w_q, w_scale),
            num_iterations=num_iter,
            num_repeats=num_repeats,
            method=benchmarking_method,
        )

        result.metadata["compile_time_s"] = compile_time
        result.metadata["lower_time_s"] = lower_time

        return result

    except Exception:
        # console.print(f"Execution failed for {kernel_name}: {e}")
        return benchmarks.BenchmarkResult(float("inf"), 0.0, 0.0, 0.0, [], {})


def tune_matmul(
    batch_sizes: List[int],
    out_in_features: List[Tuple[int, int]],
    x_q_dtype: str = "int8",
    w_q_dtype: str = "int8",
    dry_run: bool = False,
    num_iterations: int = 10,
    num_repeats: int = 5,
    benchmarking_method: str = "amortized",
    update_registry: bool = False,
    tp_size: int = 1,
    tp_split_dim: str = "out",
    run_name: Optional[str] = None,
    output_dir: str = "tuning_runs",
    no_save: bool = False,
):

    # Resolve benchmarking method
    try:
        method = benchmarks.BenchmarkMethod(benchmarking_method)
    except ValueError:
        raise ValueError(
            f"Invalid benchmarking method: {benchmarking_method}. "
            f"Choose from {[m.value for m in benchmarks.BenchmarkMethod]}")

    # Apply sharding (TP) if requested
    if tp_size > 1:
        console.print(
            f"[bold cyan]Applying TP Scaling (TP={tp_size} on {tp_split_dim})[/bold cyan]"
        )
        new_features = []
        for out_f, in_f in out_in_features:
            if tp_split_dim == "out":
                scaled_out = utils.apply_tp_scaling(out_f, tp_size,
                                                    "out_features")
                new_features.append((scaled_out, in_f))
            elif tp_split_dim == "in":
                scaled_in = utils.apply_tp_scaling(in_f, tp_size,
                                                   "in_features")
                new_features.append((out_f, scaled_in))
        out_in_features = new_features

    configs = make_configs(batch_sizes, out_in_features, x_q_dtype, w_q_dtype)
    console.print(f"Generated {len(configs)} configurations to tune "
                  f"using '{method.value}' method.")

    # Setup Experiment Run
    run_ctx = utils.RunContext(run_name, output_dir, no_save)

    # Save Metadata
    run_ctx.save_metadata({
        "kernel": "quantized_matmul",
        "cli_args": {
            "batch_sizes": batch_sizes,
            "out_in_features": out_in_features,
            "x_q_dtype": x_q_dtype,
            "w_q_dtype": w_q_dtype,
            "num_iterations": num_iterations,
            "num_repeats": num_repeats,
            "benchmarking_method": benchmarking_method,
            "tp_size": tp_size,
            "tp_split_dim": tp_split_dim
        }
    })

    # Setup CSV with context manager
    fieldnames = [
        "batch_size",
        "out_feature",
        "in_feature",
        "x_q_dtype",
        "w_q_dtype",
        "batch_block_size",
        "out_block_size",
        "in_block_size",
        "time_ns",
        "time_std_ns",
        "compile_time_s",
        "lower_time_s",
        "is_best",
        "benchmarking_method",
    ]

    with run_ctx.open_csv(fieldnames) as csv_writer:

        results = collections.defaultdict(list)
        results_best = {}  # Track best per key for is_best flag

        with utils.setup_progress() as progress:
            task = progress.add_task("[green]Tuning...", total=len(configs))

            for tuned_key, tuned_value in configs:
                # Update description to show current config
                progress.update(
                    task,
                    description=
                    f"[green]Tuning {tuned_value.batch_block_size}x{tuned_value.out_block_size}x{tuned_value.in_block_size}...",
                )

                result = autotune_kernel(
                    tuned_key,
                    tuned_value,
                    dry_run=dry_run,
                    num_iter=num_iterations,
                    num_repeats=num_repeats,
                    benchmarking_method=method,
                )

                if result.mean_time_ns == float("inf"):
                    progress.update(task, advance=1)
                    continue

                # Check if best so far
                is_best = False
                current_best = results_best.get(tuned_key)
                if current_best is None or result.mean_time_ns < current_best:
                    results_best[tuned_key] = result.mean_time_ns
                    is_best = True

                results[tuned_key].append(
                    BenchmarkResult(tuned_value,
                                    result.mean_time_ns,
                                    result.std_time_ns,
                                    result.metadata.get("compile_time_s", 0.0),
                                    result.metadata.get("lower_time_s", 0.0),
                                    metadata={
                                        "benchmarking_method": method.value,
                                        "num_repeats": num_repeats,
                                        "samples_ns": result.samples_ns
                                    }))

                if csv_writer:
                    row = {
                        "batch_size":
                        tuned_key.n_batch,
                        "out_feature":
                        tuned_key.n_out,
                        "in_feature":
                        tuned_key.n_in,
                        "x_q_dtype":
                        tuned_key.x_q_dtype,
                        "w_q_dtype":
                        tuned_key.w_q_dtype,
                        "batch_block_size":
                        tuned_value.batch_block_size,
                        "out_block_size":
                        tuned_value.out_block_size,
                        "in_block_size":
                        tuned_value.in_block_size,
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

                progress.update(task, advance=1)

    # Find best and Aggregate
    aggregated_results = {}
    sorted_keys = sorted(results.keys(), key=lambda k: (k.n_batch, k.n_out))

    for key in sorted_keys:
        values = results[key]
        if not values:
            continue
        best = min(values, key=lambda x: x.latency)

        # Format key as string for JSON
        json_key = f"{key.n_batch},{key.n_out},{key.n_in},{key.x_q_dtype},{key.w_q_dtype}"

        aggregated_results[json_key] = {
            "config": {
                "batch_block_size": best.tuned_value.batch_block_size,
                "out_block_size": best.tuned_value.out_block_size,
                "in_block_size": best.tuned_value.in_block_size,
            },
            "stats": {
                "latency_avg_ns": best.latency,
                "latency_std_ns": best.std,
                "compile_time_s": best.compile_time,
                "lower_time_s": best.lower_time,
            },
            "metadata": best.metadata
        }

    # Save Results
    run_ctx.save_results(aggregated_results)
    run_ctx.print_summary_table(aggregated_results)

    if update_registry:
        tpu_version = tpu_utils.get_tpu_name_slug()
        norm_name = utils.get_registry_file_name(tpu_version)

        base_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))  # tpu_inference/
        data_dir = os.path.join(base_dir,
                                "kernels/tuned_data/quantized_matmul")
        target_file = os.path.join(data_dir, f"{norm_name}.json")

        console.print(f"[bold]Updating registry at {target_file}...[/bold]")
        utils.update_json_registry(target_file, aggregated_results)
