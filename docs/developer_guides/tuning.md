# Kernel Tuning Guide

> [!CAUTION]
> **Development Note**: These autotuning tools are currently in **alpha**. APIs and output formats may change in future releases.

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

## Benchmarking Strategies

The tools support two benchmarking methods:

### 1. Amortized (Default)
**Flag:** `--benchmarking-method amortized`

### Benchmarking Methods

* **`amortized` (Default)**: Runs the kernel in a `lax.fori_loop` on the device (e.g., 100 iterations) and measures the average time. **Recommended for Tuning.** This metric best represents the "serving" throughput of the kernel (hiding dispatch overhead).
* **`xprof`**: Captures a full XPlane trace of a single kernel execution. **Recommended for Debugging.** This gives precise device-level ops duration but often includes trace overhead, resulting in higher reported latencies (e.g., ~1ms vs 50µs).

> [!NOTE]
> **Safety Check**: The `tpu-tune apply` command includes a safety check. It will **skip** updating the registry if the new configuration is slower than the existing entry. This prevents experimental runs (like XProf) from overwriting optimized results.

## Registry vs. Logs

* **Registry (`tpu_inference/kernels/tuned_data/*.json`)**: Stores only the **best** configuration found for a given shape. This file is loaded at runtime.
* **Logs (`tuning_runs/<run>/trials.csv`)**: Stores **every** trial run during tuning. Use this for debugging or analyzing the optimization landscape.

### Registry Schema

The registry uses a structured format designed for provenance and debugging:

```json
"key_params": {
  "config": {
    "block_size_1": 128,
    "block_size_2": 32
  },
  "stats": {
    "latency_avg_ns": 4500.0,
    "latency_std_ns": 120.0,
    "compile_time_s": 0.5
  },
  "metadata": {
    "benchmarking_method": "xprof",
    "samples_ns": [4600.0, 4400.0, ...]
  }
}
```

## Tuning Ragged Paged Attention (RPA)

### Usage

```bash
tpu-tune rpa-v3 [OPTIONS]
```

### Key Options
### Full CLI Reference

| Option | Default | Description |
| :--- | :--- | :--- |
| `--page-size` | `128` | Comma-separated list of page sizes to tune. |
| `--head-dim` | `128` | Comma-separated list of head dimensions. |
| `--num-q-heads` | `128` | Global query heads. |
| `--num-kv-heads` | `1` | Global KV heads. |
| `--max-model-len` | `1024` | sequence lengths to tune. |
| `--tp-size` | `1` | Tensor Parallelism degree. Scales num_heads automatically. |
| `--benchmarking-method` | `amortized` | Strategy: `amortized` (fast) or `xprof` (precise). |
| `--num-repeats` | `5` | Number of outer loop repeats for stats variance. |
| `--num-iterations` | `100` | Number of inner loop iterations provided to JAX. |
| `--run-name` | `auto` | Name of the experiment run (folder name). |
| `--output-dir` | `tuning_runs` | Base directory for experiments. |
| `--no-save` | `False` | Disable saving results to disk. |
| `--update-registry` | `False` | Algorithmically update the JSON registry with best results. |
| `--dry-run` | `False` | Print configuration plan without running kernels. |
| `--num-sequences` | `35` | Number of sequences for the synthetic benchmark data. |
| `--kv-block-sizes` | `1,2...128` | Search space for KV pages per block. |
| `--q-block-sizes` | `8,16...256` | Search space for Query tokens per block. |

### Example
Tuning Llama-3-8B (Head Dim 128, GQA 32:8) on TPU v6e with **TP=1** (No Sharding):

```bash
tpu-tune rpa-v3 \
    --page-size 128 \
    --head-dim 128 \
    --num-q-heads 32 \
    --num-kv-heads 8 \
    --max-model-len 8192 \
    --run-name rpa_llama3_v6e
```

### Experiment Tracking
By default, every run is saved to `tuning_runs/<run_name>/`.
* `results.json`: Best configurations ready to be applied.
* `trials.csv`: Detailed log of every trial.
* `run_metadata.json`: Reproducibility info (args, machine info).

### Applying Results
To update your registry with the best results from a run:

```bash
tpu-tune apply tuning_runs/tune_20240101_120000/results.json
```

## Tuning Quantized Matmul

### Usage

```bash
tpu-tune quantized-matmul [OPTIONS]
```

### Key Options
### Full CLI Reference

| Option | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `--batch-sizes` | Yes | - | Comma-separated batch sizes (e.g., `16,32`). |
| `--out-in-features` | Yes | - | Comma-separated pairs (e.g., `2048/4096`). |
| `--tp-size` | No | `1` | Tensor Parallelism degree. |
| `--tp-split-dim` | No | `out` | Dimension sharded: `out` (Column) or `in` (Row). |
| `--x-q-dtype` | No | `int8` | Input quantization dtype. |
| `--w-q-dtype` | No | `int8` | Weight quantization dtype. |
| `--benchmarking-method` | No | `amortized` | Strategy: `amortized` or `xprof`. |
| `--num-repeats` | No | `5` | Outer loop repeats. |
| `--num-iterations` | No | `10` | Inner loop iterations. |
| `--run-name` | No | `auto` | Name of the experiment run (folder name). |
| `--output-dir` | No | `tuning_runs` | Base directory for experiments. |
| `--no-save` | No | `False` | Disable saving results to disk. |
| `--update-registry` | No | `False` | Update JSON registry. |
| `--dry-run` | No | `False` | Print plan only. |

### Example
Tuning a specific layer found in logs (e.g., Batch 16, 2048x4096):

```bash
tpu-tune quantized-matmul \
    --batch-sizes 16 \
    --out-in-features 2048/4096 \
    --run-name matmul_test
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
