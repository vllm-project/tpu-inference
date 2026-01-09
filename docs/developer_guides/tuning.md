# Kernel Tuning Guide

This guide explains how to use the built-in tuning tools in `tpu-inference` to optimize kernel performance on your specific TPU hardware.

## Overview

Kernel tuning helps find the optimal block sizes and configurations for custom Pallas kernels. These configurations can vary significantly across different TPU generations (v5e, v6e) and model architectures.

## Prerequisites

The tuning tools require a source installation of `tpu-inference` and `vllm`. We recommend using `uv` for fast and reliable dependency management.

### Installation Steps

1. **Install `uv` (if not present):**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   ```

2. **Clone Repositories:**

   ```bash
   git clone https://github.com/vllm-project/vllm.git
   git clone https://github.com/vllm-project/tpu-inference.git
   ```

3. **Create Virtual Environment:**

   ```bash
   uv venv .venv --python 3.12
   source .venv/bin/activate
   ```

4. **Install `vllm` from Source:**

   ```bash
   cd vllm
   uv pip install -r requirements/tpu.txt
   VLLM_TARGET_DEVICE="tpu" uv pip install -e .
    cd ..
    ```

5. **Install `tpu-inference` with Tuning Extras:**

    ```bash
    cd tpu-inference
    uv pip install -e .[tuning]
    cd ..
    ```

## Tuning Ragged Paged Attention (RPA)

The RPA kernel is critical for attention performance. You can tune it for specific head dimensions and page sizes.

### Usage

```bash
tpu-tune-rpa [OPTIONS]
```

### Key Options

- `--page-size`: Comma-separated list of page sizes to tune (e.g., `128` or `16,32`).
- `--head-dim`: Head dimension (e.g., `128` for Llama-3-8B).
- `--num-q-heads`, `--num-kv-heads`: Number of query and KV heads.
- `--max-model-len`: Maximum sequence length to tune for.
- `--num-iterations`: Number of benchmark iterations per config (default: 10).
- `--csv-file`: Path to save results in CSV format.
- `--dry-run`: Run without executing actual kernels (for testing pipeline).
- `--kv-block-sizes`: (Advanced) Comma-separated list of KV block sizes to search (default: `1,2,4,8,16,32,64,128`).
- `--q-block-sizes`: (Advanced) Comma-separated list of Query block sizes to search (default: `8,16,32,64,128,256`).

### Example

Tune for Llama-3-8B (Head Dim 128, GQA 32:8 => 4:1 ratio) on TPU v6e:

```bash
tpu-tune-rpa \
    --page-size 128 \
    --head-dim 128 \
    --num-q-heads 32 \
    --num-kv-heads 8 \
    --max-model-len 8192 \
    --csv-file rpa_results.csv
```

## Tuning Quantized Matmul

The quantized matrix multiplication kernel supports INT8/INT8 activation/weight quantization and requires tuning for optimal block sizes.

### Usage

```bash
tpu-tune-quantized-matmul [OPTIONS]
```

### Key Options

- `--batch-sizes`: Comma-separated list of batch sizes (e.g., `16,32,64`).
- `--out-in-features`: Comma-separated list of `out_features/in_features` pairs (e.g., `2048/4096`).
- `--num-iterations`: Number of benchmark iterations per config (default: 10).
- `--csv-file`: Path to save results in CSV format.

### Example

Tune for a specific layer shape:

```bash
tpu-tune-quantized-matmul \
    --batch-sizes 16,32,64,128 \
    --out-in-features 4096/4096,11008/4096 \
    --csv-file matmul_results.csv
```

## Applying Results

The tuning scripts output a Python dictionary snippet at the end of execution. You can copy this snippet into the respective `tuned_block_sizes.py` file in the source code to persist the optimized parameters.

Example output:

```python
TUNED_BLOCK_SIZES_RAW = {
    (6, 128, 128, 128, 'int8', 'int8'): (128, 128, 128),
}
```
