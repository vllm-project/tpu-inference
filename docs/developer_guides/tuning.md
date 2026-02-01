# Kernel Tuning Guide

> [!CAUTION]
> **Alpha Status**: These tools are in active development.

This guide explains how to optimize block sizes for Pallas kernels (RPA and Quantized Matmul) on TPU. Correct tuning is essential for maximizing serving throughput.

## Installation
The autotuning tools require additional dependencies (`click`, `rich`). Install them using the `tuning` extra:

```bash
# Standard pip
pip install -e ".[tuning]"

# Or using uv (recommended)
uv pip install -e ".[tuning]"
```

## Quick Start
If you see a generic warning in your logs like `WARNING: Couldn't find tuned sizes for key...`, it means the kernel is falling back to a default, unoptimized configuration. You can resolve this by running the autotuner for that specific shape.

For example, to tune a quantized matmul layer:

```bash
# 1. Activate Environment
source .venv/bin/activate

# 2. Run Tuner
tpu-tune quantized-matmul \
    --batch-sizes 16 \
    --out-in-features 2048/4096 \
    --update-registry
```

This command benchmarks various block sizes on your TPU and automatically saves the fastest configuration to the registry (`tpu_inference/kernels/tuned_data/`). The next time you run your model, it will pick up these optimized values.

## Why Tuning Matters
Pallas kernels split large matrix operations into smaller "blocks" or tiles to fit within the TPU's high-speed memory (VMEM). The optimal block size is a trade-off: small blocks incur high scheduling overhead, while large blocks can cause memory spilling or out-of-memory errors.

Because this balance depends heavily on the specific hardware generation (e.g., TPU v5e vs v6e) and the exact tensor shapes, we cannot use a single static configuration. Autotuning empirically finds the best fit for your specific deployment.

## Usage Reference

### Ragged Paged Attention (RPA)
Use this tuner if you are changing sequence lengths, head dimensions, or the number of KV heads (e.g., Grouped Query Attention).

```bash
tpu-tune rpa-v3 \
    --page-size 128 \
    --head-dim 128 \
    --num-q-heads 32 \
    --num-kv-heads 8 \
    --max-model-len 8192 \
    --run-name my_llama_tune \
    --update-registry
```

**Common Parameters:**

| Option | Default | Description |
| :--- | :--- | :--- |
| `--page-size` | `128` | Size of the KV cache page. |
| `--head-dim` | `128` | Dimension of each attention head. |
| `--num-q-heads` | `128` | **Global** number of Query heads. |
| `--num-kv-heads` | `1` | **Global** number of Key/Value heads. |
| `--max-model-len` | `1024` | Maximum sequence length to tune for. |
| `--tp-size` | `1` | Tensor Parallelism degree. Scales global heads to local heads automatically. |
| `--benchmarking-method` | `amortized` | `amortized` (recommended) or `xprof`. |
| `--run-name` | `auto` | Name of the experiment folder in `tuning_runs/`. |
| `--update-registry` | `False` | Algorithmically update the JSON registry with best results. |

**Note on Sharding:** If your model uses Tensor Parallelism (TP), you must specify the `--tp-size` flag (e.g., `--tp-size 4`). The tool will automatically calculate the local head count for each chip. Forgetting this flag will result in tuning for the wrong shape.

### Quantized Matmul
Use this tuner for linear layers, such as MLP blocks or input/output projections, particularly when changing batch sizes.

```bash
tpu-tune quantized-matmul \
    --batch-sizes 16,32,64 \
    --out-in-features 4096/12288 \
    --tp-size 4 \
    --tp-split-dim out \
    --update-registry
```

**Common Parameters:**

| Option | Default | Description |
| :--- | :--- | :--- |
| `--batch-sizes` | *(Required)* | Comma-separated list of batch sizes (e.g., `16,32`). |
| `--out-in-features` | *(Required)* | Dimensions as `OUT/IN` pairs (e.g., `4096/12288`). |
| `--tp-size` | `1` | Tensor Parallelism degree. |
| `--tp-split-dim` | `out` | Dimension to split: `out` (Column Parallel) or `in` (Row Parallel). |
| `--x-q-dtype` | `int8` | Quantization type for input activation. |
| `--w-q-dtype` | `int8` | Quantization type for weights. |
| `--benchmarking-method` | `amortized` | `amortized` (recommended) or `xprof`. |
| `--update-registry` | `False` | Update JSON registry with best results. |

For the `--out-in-features` argument, format pairs as `OUT/IN`. When using Tensor Parallelism, specify which dimension is split using `--tp-split-dim` (use `out` for Column Parallel and `in` for Row Parallel).

## Benchmarking Methods
The tools provide two methods for measuring performance.

We generally recommend the **Amortized** default. This method runs the kernel in a loop on the device (usually 100+ iterations) and measures the average time. It effectively mimics a high-throughput serving scenario by hiding the Python host overhead.

The **XProf** method is available for deeper debugging. It captures a full XPlane trace to measure the exact device-side execution time of a single kernel call. While precise, this method often includes tracing overhead and is slower to run, making it less suitable for broad optimization sweeps.

**Experiment Safety:** The tooling includes a regression check. If you experiment with different settings or benchmarking methods, `tpu-tune` will not overwrite an existing registry entry unless the new result is strictly faster.

## Experiment Tracking
All tuning runs work as experiments. Output artifacts are saved to `tuning_runs/<run_name>/`:

* **results.json**: The best configurations found.
* **trials.csv**: A complete log of every block size tested, useful for analyzing performance trends.
* **run_metadata.json**: Reproducibility information.

If you run an experiment without the `--update-registry` flag, you can apply the results later:

```bash
tpu-tune apply tuning_runs/my_run/results.json
```

## Contributing Tuned Values
If you identify optimized configurations for common models (e.g., Llama 3, DeepSeek) that are significantly faster than the defaults or missing from the registry, please contribute them back!

1. Run the tuner with `--update-registry`.
2. Verify the changes in `tpu_inference/kernels/tuned_data/*.json`.
3. Commit the updated JSON files.
4. Submit a Pull Request to the `tpu-inference` repository.

Sharing these results helps the entire community serve models faster on TPU.

## Troubleshooting

* **Invalid Block Sizes:** If the tuner suggests unexpected block sizes, verify your `--tp-size` setting. Tuning for the global shape instead of the local sharded shape is a common error.
* **Skipped Updates:** If the logs say updates were skipped, it means the new results were slower than the existing registry entries. You can inspect `trials.csv` to see the comparison.
* **Inf Latency:** A latency of `inf` typically indicates the configuration failed to run, often due to an Out of Memory (OOM) error or an invalid block size combination.
