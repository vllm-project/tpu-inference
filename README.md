## Setup development environment

### Install `vLLM-TPU`:

Follow this [guide](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator.html#set-up-using-python) to install vLLM from source.

### Install `tpu_commons`:

```
cd ~
git clone https://github.com/vllm-project/tpu_commons.git
cd tpu_commons
pip install -e .
```

### Setup pre-commit hooks

```
pip install pre-commit

# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg

# You can manually run pre-commit with
pre-commit run --all-files
```

## Run JAX path examples

**NOTE**: This is under development so the run may fail.

Run `Llama 3.1 8B` offline inference on 4 TPU chips:

```
TPU_BACKEND_TYPE=jax
python vllm/examples/offline_inference/basic/generate.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024 \
    --max_num_seqs=1
```

## Run vLLM Pytorch models on the JAX path

Run the vLLM's implementation of `Llama 3.1 8B`, which is in Pytorch:

```
MODEL_IMPL_TYPE=vllm
TPU_BACKEND_TYPE=jax
python vllm/examples/offline_inference/basic/generate.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=1 \
    --task=generate \
    --max_model_len=1024 \
    --max_num_seqs=1
```

Currently only single chip is supported, so `--tensor_parallel_size=1`

## Relevant env

To enable JAX path:

```
TPU_BACKEND_TYPE=jax
```

To use vLLM model implementations, set both:

```
MODEL_IMPL_TYPE=vllm
TPU_BACKEND_TYPE=jax
```

To enable experimental scheduler:

```
EXP_SCHEDULER=1
```

To inspect model weights sharding:

```
INSPECT_MODEL=1
```

## Torchax Guide

**NOTE**: This is under development so the run may fail.

### Install dependencies

#### Install `vLLM`

Follow the above [step](#install-vllm-tpu) to install vllm for TPU backend.

#### Install `tpu_commons`

Follow the above step to install [tpu_commons](#install-tpu_commons)

### Run example script

```
cd vllm
TPU_BACKEND_TYPE=torchax VLLM_TORCHAX_ENABLED=1 VLLM_USE_V1=1 python examples/offline_inference/tpu.py
```

## How to test kernel?

Install dependencies:

```
pip install -r requirements.txt
```

Make sure TPU device is accessible:

```
tpu-info
```

Run the test:

```
pytest -v ./tests/ragged_paged_attention_test.py
```
