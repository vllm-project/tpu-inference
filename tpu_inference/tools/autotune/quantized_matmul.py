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
import json
import os
import time
from typing import List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from tpu_inference import utils as tpu_utils
from tpu_inference.kernels.quantized_matmul import tuned_block_sizes, util
from tpu_inference.tools.autotune import utils

console = utils.console


class TestResult(NamedTuple):
    tuned_value: tuned_block_sizes.TunedValue
    latency: float
    std: float
    compile_time: float
    lower_time: float


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
):
    if dry_run:
        return 1.0, 0.0, 0.0, 0.0  # Mock latency (mean, std, compile_time, lower_time)

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
        return float("inf"), 0.0, 0.0, 0.0

    try:
        # Compile amortized function (Use source kernel, not compiled binary)
        import functools
        pallas_fn = functools.partial(kernel.quantized_matmul_kernel,
                                      x_q_dtype=x_q_dtype_obj,
                                      tuned_value=tuned_value)
        amortized_fn = utils.amortized_wrapper(pallas_fn, n_iter=num_iter)
        amortized_jit = jax.jit(amortized_fn)

        # Warmup the amortized function
        outputs = amortized_jit(x, w_q, w_scale)
        utils.block_until_ready(outputs)

        # Measure
        start = time.perf_counter_ns()
        outputs = amortized_jit(x, w_q, w_scale)
        utils.block_until_ready(outputs)
        end = time.perf_counter_ns()

        total_time_ns = end - start
        avg_time_ns = total_time_ns / num_iter

        return avg_time_ns, 0.0, compile_time, lower_time

    except Exception:
        # console.print(f"Execution failed for {kernel_name}: {e}")
        return float("inf"), 0.0, 0.0, 0.0


def tune_matmul(
    batch_sizes: List[int],
    out_in_features: List[Tuple[int, int]],
    x_q_dtype: str = "int8",
    w_q_dtype: str = "int8",
    dry_run: bool = False,
    num_iterations: int = 10,
    csv_file: Optional[str] = None,
    update_registry: bool = False,
    tp_size: int = 1,
    tp_split_dim: str = "out",
):
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
    console.print(f"Generated {len(configs)} configurations to tune.")

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
    ]

    with utils.CsvResultLogger(csv_file, fieldnames) as csv_logger:
        if not csv_logger and csv_file:
            return

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

                latency_ns, std_ns, compile_time_s, lower_time_s = autotune_kernel(
                    tuned_key,
                    tuned_value,
                    dry_run=dry_run,
                    num_iter=num_iterations,
                )

                if latency_ns == float("inf"):
                    progress.update(task, advance=1)
                    continue

                # Check if best so far
                is_best = False
                current_best = results_best.get(tuned_key)
                if current_best is None or latency_ns < current_best:
                    results_best[tuned_key] = latency_ns
                    is_best = True

                results[tuned_key].append(
                    TestResult(tuned_value, latency_ns, std_ns, compile_time_s,
                               lower_time_s))

                if csv_logger:
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
                        "compile_time_s": compile_time_s,
                        "lower_time_s": lower_time_s,
                        "is_best": is_best,
                    }
                    csv_logger.writer.writerow(row)
                    csv_logger.flush()

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
        }

    # Print JSON output
    tpu_version = tpu_utils.get_tpu_name_slug()
    norm_name = utils.get_registry_file_name(tpu_version)

    print(
        f"\n[Output for tpu-inference/kernels/tuned_data/quantized_matmul/{norm_name}.json]:"
    )
    print(json.dumps(aggregated_results, indent=2, sort_keys=True))

    if update_registry:
        base_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))  # tpu_inference/
        data_dir = os.path.join(base_dir,
                                "kernels/tuned_data/quantized_matmul")
        target_file = os.path.join(data_dir, f"{norm_name}.json")

        console.print(f"[bold]Updating registry at {target_file}...[/bold]")
        utils.update_json_registry(target_file, aggregated_results)

    if csv_file:
        console.print(f"\nResults written to {csv_file}")
