# Kernel Tuning Guide

> [!CAUTION]
> **Experimental Feature**: These autotuning tools are currently in **alpha**. APIs, CLI arguments, and output formats may change significantly in future releases. Use with caution.

This guide explains how to use the built-in tuning tools in `tpu-inference` to optimize kernel performance for your specific model and TPU hardware.

## What is Kernel Tuning?

Kernel tuning finds optimal block sizes for custom Pallas kernels (RPA and Quantized Matmul). These settings are critical for performance and vary by TPU generation (v5e/v6e) and model architecture.

## How do I know if I need to tune?

The most reliable way to check if your model needs specific tuning is to **run the model and inspect the logs**.

If the system cannot find a tuned configuration for a specific shape, it will emit a warning with the *exact* parameters you need to tune for:

```text
WARNING ... [tuned_block_sizes.py] Couldn't find tuned sizes for ... TunedKey(tpu_version=6, n_batch=16, n_out=2048, n_in=4096, ...)
```

In this example, the log tells you exactly what to run:
* Batch Size: `16`
* Out Features: `2048`
* In Features: `4096`

## Important: Sharding and Local Shapes
The autotuners must tune for the **local (per-chip) shape**. You can do this automatically using `--tp-size` or manually.

* **Automatic:** Pass `--tp-size N`. The tool will automatically divide `num_heads` (RPA) or features (Matmul) by N.
* **Manual:** Calculate the local shape yourself and pass it directly.

> [!TIP]
> **Example:** 32 Heads with TP=8.
> * **Option A (Auto):** `--num-q-heads 32 --tp-size 8` -> Tool tunes for 4 heads.
> * **Option B (Manual):** `--num-q-heads 4`

## Installation

These tools are included with `tpu-inference`. Please follow the main [Installation Guide](../getting_started/installation.md) to set up your environment (supports `uv` and `conda`).

Once installed, you can access the tools directly from your CLI.

### Verification

You can verify your installation using the built-in help commands:

```bash
tpu-tune --help
tpu-tune rpa --help
tpu-tune quantized-matmul --help
```

## Tuning Ragged Paged Attention (RPA)

### Usage

```bash
tpu-tune rpa [OPTIONS]
```

### Key Options
* `--page-size`: Page blocks (e.g., `128`).
* `--head-dim`: Head dimension (e.g., `128`).
* `--num-q-heads`: Global query heads (will be scaled if `--tp-size > 1`).
* `--num-kv-heads`: Global KV heads (will be scaled if `--tp-size > 1`).
* `--tp-size`: Tensor Parallelism degree (default: 1).
* `--max-model-len`: Local sequence length.
* `--csv-file`: Output file for results.

### Example
Tuning Llama-3-8B (Head Dim 128, GQA 32:8) on TPU v6e with **TP=1** (No Sharding):

```bash
tpu-tune rpa \
    --page-size 128 \
    --head-dim 128 \
    --num-q-heads 32 \
    --num-kv-heads 8 \
    --max-model-len 8192 \
    --csv-file rpa_results.csv
```

## Tuning Quantized Matmul

### Usage

```bash
tpu-tune quantized-matmul [OPTIONS]
```

### Key Options
* `--batch-sizes`: List of batch sizes to tune (e.g., `16,32,64`).
* `--out-in-features`: List of `out/in` pairs.
* `--tp-size`: Tensor Parallelism degree (default: 1).
* `--tp-split-dim`: Dimension to split (`out` or `in`, default: `out`).
* `--csv-file`: Output file for results.

### Example
Tuning a specific layer found in logs (e.g., Batch 16, 2048x4096):

```bash
tpu-tune quantized-matmul \
    --batch-sizes 16 \
    --out-in-features 2048/4096 \
    --csv-file matmul_results.csv
```

## Applying Results

The easiest way to apply your tuned config is to use the `--update-registry` flag. This will automatically update the JSON registry files tailored to your TPU generation (e.g., `tpu_v6e.json`).

```bash
tpu-tune rpa \
    ... \
    --update-registry
```

### Manual Update
If you prefer to review changes first, you can omit the flag. The tool will print a JSON snippet that you can check.

The registry files are located in `tpu_inference/kernels/tuned_data/`. The system automatically loads the correct file at runtime based on the detected TPU version.

## Sharding Cheat Sheet

| Model | Weight Sharding | Tuning Strategy | Example (TP=4) |
| :--- | :--- | :--- | :--- |
| **Llama 3 70B** | **Q / KV Heads** | `--num-q-heads 64 --tp-size 4` | Tunes for 16 heads |
| **DeepSeek V3** | **Experts** | `--out-in-features ... --tp-size 4` | Tunes for `Experts/4` |

> [!NOTE]
> For **Quantized Matmul**, use `--tp-split-dim` (default `out`) to specify which dimension is sharded.
> * **Column Parallel** split output features (`out`).
> * **Row Parallel** split input features (`in`).
