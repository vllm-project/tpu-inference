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


@main.command(name='rpa')
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
@click.option('--update-registry',
              is_flag=True,
              help="Update the JSON registry with results")
@click.option('--tp-size',
              default=1,
              help="Tensor Parallelism degree (divides num_heads)",
              show_default=True)
def rpa(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim,
        max_model_len, num_iterations, dry_run, num_sequences, csv_file,
        kv_block_sizes, q_block_sizes, update_registry, tp_size):
    """Tune Ragged Paged Attention (RPA) kernels."""
    from tpu_inference.tools.autotune import rpa as rpa_lib

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
                     dry_run=dry_run,
                     num_sequences=num_sequences,
                     csv_file=csv_file,
                     update_registry=update_registry,
                     tp_size=tp_size)


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
@click.option('--csv-file',
              default=None,
              help="Optional path to output results to a CSV file")
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
                     dry_run, num_iterations, csv_file, update_registry,
                     tp_size, tp_split_dim):
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
                           csv_file=csv_file,
                           update_registry=update_registry,
                           tp_size=tp_size,
                           tp_split_dim=tp_split_dim)


def legacy_rpa_cli():
    """Shim for legacy tpu-tune-rpa command."""
    utils.setup_logging()
    rpa()


def legacy_matmul_cli():
    """Shim for legacy tpu-tune-quantized-matmul command."""
    utils.setup_logging()
    quantized_matmul()


if __name__ == "__main__":
    main()
