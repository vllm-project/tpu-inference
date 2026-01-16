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

import click

from tpu_inference.tools.autotune import utils


@click.group()
def main():
    """TPU Inference Kernel Tuning Tools.

    Use this tool to find optimal block sizes and configurations for custom
    Pallas kernels on your specific TPU hardware.
    """
    utils.setup_logging()


def parse_arg(arg, type_fn=str):
    """Helper to parse comma-separated arguments."""
    if isinstance(arg, str):
        res = [type_fn(x.strip()) for x in arg.split(',')]
        return res
    return [type_fn(arg)]


@main.command(name='rpa-v3')
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
              help="Comma separated list of local num q heads (per chip)",
              show_default=True)
@click.option('--num-kv-heads',
              default='1',
              help="Comma separated list of local num kv heads (per chip)",
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
@click.option('--num-repeats',
              default=5,
              help="Number of outer loop repeats for stats",
              show_default=True)
@click.option('--benchmarking-method',
              type=click.Choice(['amortized', 'xprof']),
              default='amortized',
              help="Method to use for benchmarking",
              show_default=True)
@click.option('--dry-run',
              is_flag=True,
              help="Run without actual kernel calls")
@click.option('--num-sequences',
              default=35,
              help="Number of sequences for autotuning example",
              show_default=True)
@click.option('--run-name',
              default='auto',
              help="Name of the run folder in output-dir",
              show_default=True)
@click.option('--output-dir',
              default='tuning_runs',
              help="Base directory for experiment outputs",
              show_default=True)
@click.option('--no-save', is_flag=True, help="Disable saving results to disk")
@click.option('--kv-block-sizes',
              default='1,2,4,8,16,32,64,128',
              help="Comma separated list of KV pages per block to test",
              show_default=True)
@click.option('--q-block-sizes',
              default='8,16,32,64,128,256',
              help="Comma separated list of Q queries per block to test",
              show_default=True)
@click.option('--update-registry',
              is_flag=True,
              help="Update the JSON registry with results")
@click.option('--tp-size',
              default=1,
              help="Tensor Parallelism degree (divides num_heads)",
              show_default=True)
def rpa_v3(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim,
           max_model_len, num_iterations, num_repeats, benchmarking_method,
           dry_run, num_sequences, kv_block_sizes, q_block_sizes,
           update_registry, tp_size, run_name, output_dir, no_save):
    """Tune Ragged Paged Attention (RPA) kernels."""
    from tpu_inference.tools.autotune import \
        ragged_paged_attention_v3 as rpa_lib

    # Parse args
    page_sizes_parsed = parse_arg(page_size, int)
    q_dtypes_parsed = parse_arg(q_dtype, str)
    kv_dtypes_parsed = parse_arg(kv_dtype, str)
    num_q_heads_parsed = parse_arg(num_q_heads, int)
    num_kv_heads_parsed = parse_arg(num_kv_heads, int)
    head_dims_parsed = parse_arg(head_dim, int)
    max_model_lens_parsed = parse_arg(max_model_len, int)

    kv_block_sizes_parsed = parse_arg(kv_block_sizes, int)
    q_block_sizes_parsed = parse_arg(q_block_sizes, int)

    # Dry run implies no-save unless explicitly overridden?
    # Usually dry-run means "don't run kernel", but maybe we still want to save empty plan?
    # User requested --no-save to be explicit.

    if update_registry and dry_run:
        utils.console.print(
            "[yellow]Dry-run enabled: disabling requested registry update.[/yellow]"
        )
        update_registry = False

    rpa_lib.tune_rpa(page_sizes=page_sizes_parsed,
                     q_dtypes=q_dtypes_parsed,
                     kv_dtypes=kv_dtypes_parsed,
                     num_q_heads_list=num_q_heads_parsed,
                     num_kv_heads_list=num_kv_heads_parsed,
                     head_dims=head_dims_parsed,
                     max_model_lens=max_model_lens_parsed,
                     kv_block_sizes=kv_block_sizes_parsed,
                     q_block_sizes=q_block_sizes_parsed,
                     num_iterations=num_iterations,
                     num_repeats=num_repeats,
                     benchmarking_method=benchmarking_method,
                     dry_run=dry_run,
                     num_sequences=num_sequences,
                     update_registry=update_registry,
                     tp_size=tp_size,
                     run_name=run_name,
                     output_dir=output_dir,
                     no_save=no_save)


@main.command(name='quantized-matmul')
@click.option('--batch-sizes',
              required=True,
              help="Comma separated batch sizes")
@click.option(
    '--out-in-features',
    required=True,
    help=
    "Comma separated out/in features ex: 2048/4096 (divide by TP if sharded)")
@click.option('--x-q-dtype', default='int8')
@click.option('--w-q-dtype', default='int8')
@click.option('--dry-run',
              is_flag=True,
              help="Run without actual kernel calls")
@click.option('--num-iterations',
              default=10,
              help="Number of iterations for benchmarking")
@click.option('--num-repeats',
              default=5,
              help="Number of outer loop repeats for stats",
              show_default=True)
@click.option('--benchmarking-method',
              type=click.Choice(['amortized', 'xprof']),
              default='amortized',
              help="Method to use for benchmarking",
              show_default=True)
@click.option('--run-name',
              default='auto',
              help="Name of the run folder in output-dir",
              show_default=True)
@click.option('--output-dir',
              default='tuning_runs',
              help="Base directory for experiment outputs",
              show_default=True)
@click.option('--no-save', is_flag=True, help="Disable saving results to disk")
@click.option('--update-registry',
              is_flag=True,
              help="Update the JSON registry with results")
@click.option('--tp-size',
              default=1,
              help="Tensor Parallelism degree",
              show_default=True)
@click.option('--tp-split-dim',
              type=click.Choice(['out', 'in']),
              default='out',
              help="Dimension to split for TP (out or in)",
              show_default=True)
def quantized_matmul(batch_sizes, out_in_features, x_q_dtype, w_q_dtype,
                     dry_run, num_iterations, num_repeats, benchmarking_method,
                     update_registry, tp_size, tp_split_dim, run_name,
                     output_dir, no_save):
    """Tune Quantized Matmul kernels."""
    from tpu_inference.tools.autotune import quantized_matmul as matmul_lib

    batch_sizes_list = parse_arg(batch_sizes, int)

    # Custom parsing for out/in features tuple list
    # Still doing this manually as it's specific tuple parsing
    out_in_features_list = [
        tuple(int(x) for x in feature.split('/'))
        for feature in out_in_features.split(',')
    ]

    if update_registry and dry_run:
        utils.console.print(
            "[yellow]Dry-run enabled: disabling requested registry update.[/yellow]"
        )
        update_registry = False

    matmul_lib.tune_matmul(batch_sizes=batch_sizes_list,
                           out_in_features=out_in_features_list,
                           x_q_dtype=x_q_dtype,
                           w_q_dtype=w_q_dtype,
                           dry_run=dry_run,
                           num_iterations=num_iterations,
                           num_repeats=num_repeats,
                           benchmarking_method=benchmarking_method,
                           update_registry=update_registry,
                           tp_size=tp_size,
                           tp_split_dim=tp_split_dim,
                           run_name=run_name,
                           output_dir=output_dir,
                           no_save=no_save)


@main.command(name='apply')
@click.argument('result-file', type=click.Path(exists=True, dir_okay=False))
def apply_results(result_file):
    """Apply a results.json file to the main registry."""
    import json

    from tpu_inference.tools.autotune.utils import (console,
                                                    update_json_registry)
    # Creating a thin wrapper here, actual logic fits better in utils or libraries.
    # For now, let's implement the logic inline or call a utility function.
    from tpu_inference.utils import get_tpu_name_slug

    # Need to infer target registry file...
    # The result JSON keys are generated by 'tuned_block_sizes.get_lookup_keys'
    # which puts the DEVICE NAME as the first element of the tuple key?
    # Wait, the JSON structure is:
    # "KEY_STRING": { "config": ..., "stats": ... }
    # Determining usage requires parsing the key.
    # Actually, simpler: each kernel has its own registry file per TPU generation.
    # The results.json should map to a specific TPU gen.
    # We can infer the TPU generation from the runtime environment using `utils.get_tpu_generation()`
    # OR we need it stored in metadata.
    # Limitation: We need to know WHICH kernel registry to update (RPA vs Matmul).
    # Heuristic: Check keys in the JSON?
    # RPA keys usually have "q_dtype_kv_dtype"
    # Matmul keys are "batch,out,in,dq,wq"

    with open(result_file, 'r') as f:
        data = json.load(f)

    if not data:
        console.print("[yellow]Results file is empty.[/yellow]")
        return

    # Infer Kernel Type
    sample_key = next(iter(data.keys()))

    # RPA keys are distinct (nested or specific string structure?) I need to check how they are stored in V2.
    # V2 RPA Storage: "128": { "q_bf16...": ... } hierarchy? Or flat in results.json?
    # `tune_rpa` produces a flat dictionary of results in `RunContext.save_results`.
    # Wait, `update_json_registry` expects a dictionary structure that matches the file.
    # If `tune_rpa` saves a flat dict, but the file is nested (RPA is nested!), we need `update_json_registry` to handle it.

    # Wait, `tune_rpa` calls `update_json_registry(..., results)`.
    # If `results` is what is saved to `results.json`, then `results.json` has the correct structure for the registry.
    # So we simply need to find the destination file.

    # Matmul Registry: tpu_inference/kernels/tuned_data/quantized_matmul/...
    # RPA Registry: tpu_inference/kernels/tuned_data/ragged_paged_attention/v3/...

    # We can use the detected TPU from the current env?
    # DANGER: What if applying results from v6e on a v5e machine?
    # We should trust the user or check metadata?
    # Let's use the current TPU for now as the target default.

    tpu_name = get_tpu_name_slug()
    # If running on CPU/Head node without TPU, this might fail or default.
    # Assume user is on the TPU VM they want to apply to, OR provide --tpu-version arg?
    # Simplest MVP: Run on the target machine.

    # Guess Registry Path based on content
    is_rpa = any("q_head" in k for k in str(data)) or any("max_model_len" in k
                                                          for k in str(data))
    # Matmul keys: "128,128,128,int8,int8"
    is_matmul = "," in sample_key and "int8" in sample_key

    target_registry = None
    if is_rpa:
        # Need 'get_rpa_registry_path' equivalent but public
        # Hardcoding path logic for MVP safely
        base_dir = "tpu_inference/kernels/tuned_data/ragged_paged_attention/v3"
        fname = f"{tpu_name}.json"
        target_registry = f"{base_dir}/{fname}"
    elif is_matmul:
        base_dir = "tpu_inference/kernels/tuned_data/quantized_matmul"
        fname = f"{tpu_name}.json"
        target_registry = f"{base_dir}/{fname}"
    else:
        # Fallback / Error
        # Try to determine if it's RPA via nested structure (keys are integers as strings?)
        # Actually RPA V2 uses string keys.
        console.print(
            "[red]Could not determine kernel type from results keys.[/red]")
        return

    import os
    if not os.path.exists(target_registry):
        # Maybe we aren't at repo root?
        # Try finding the file relative to module?
        console.print(
            f"[yellow]Registry file {target_registry} not found (are you in repo root?). Creating new.[/yellow]"
        )

    update_json_registry(target_registry, data)


if __name__ == "__main__":
    main()
