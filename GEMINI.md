# tpu-inference Project Overview

## Project Overview
`tpu-inference` is a Python-based hardware plugin that provides a unified backend for the **vLLM** project on Google TPUs. It enables both JAX and PyTorch models to be run performantly on TPUs with a single lowering path. The project aims to push the limits of TPU hardware performance in open source while retaining standard vLLM user experience, telemetry, and interfaces.

**Key Technologies:**
- Python (>= 3.10)
- vLLM
- JAX & PyTorch
- Google Cloud TPUs (v3, v4, v5e, v5p, v6e, v7x)
- `xpk` (Google Cloud TPU orchestration)

**Architecture & Directory Structure:**
The codebase follows a directory structure similar to vLLM, categorizing components based on their compatibility:
- `tpu_inference/layers/` & `tpu_inference/models/`:
  - `common/`: Implementations common to both vLLM (PyTorch) and JAX.
  - `jax/`: Implementations specific to JAX models.
  - `vllm/`: Implementations specific to vLLM models.

## Building and Running

### Installation
You can install the package in editable mode alongside `vllm`:
```bash
pip install -e .
```
*(Note: As seen in internal scripts, you may also need to install `vllm` with `VLLM_TARGET_DEVICE="tpu" pip install -e ./vllm`)*

### Running Inference
Inference is typically run using the standard `vllm serve` command with specific environment variables and arguments for TPU targeting.

Example running via pathways (from `run_pathways.sh`):
```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export MODEL_IMPL_TYPE=vllm

JAX_PLATFORMS=proxy,cpu JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
vllm serve <model_path> \
    --tensor-parallel-size <size> \
    --data-parallel-size <size> \
    --max-model-len <len>
```

### Running on Google Cloud (xpk)
The project utilizes `xpk` for managing workloads on TPUs. Example command:
```bash
xpk workload create-pathways --workload <name> --base-docker-image vllm/vllm-tpu:nightly --script-dir <dir> --cluster <cluster> --tpu-type=<type> --num-slices=<slices> --command "bash run_pathways.sh"
```

## Development Conventions

- **Linting & Formatting:** The project relies on `pre-commit` for linting, formatting, and static type checking. 
  - Setup: `pip install pre-commit && pre-commit install --hook-type pre-commit --hook-type commit-msg`
  - Manual run: `pre-commit run --all-files`
- **Testing:** The project uses `pytest`. All new features must include relevant unit tests and CI tests. CI configuration can be found in `.buildkite/`.
- **Contribution Guidelines:** Ensure new layers or model implementations are placed in the correct sub-directory (`common`, `jax`, or `vllm`) depending on their framework compatibility. Filter issues by the "Good First Issue" tag for beginner-friendly tasks.
