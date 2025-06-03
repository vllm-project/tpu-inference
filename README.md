## Setup development environment

### Install `tpu_commons`:

```
cd ~
git clone https://github.com/vllm-project/tpu_commons.git
cd tpu_commons
pip install -e .
```

### Install `vLLM-TPU`:

Follow this [guide](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator.html#set-up-using-python) to install vLLM from source with the additional step:

Right after the `git clone` step and before the `pip install -e .` step, run the following command:

```
sed -i 's|return "vllm.platforms.tpu.TpuPlatform" if is_tpu else None|return "tpu_commons.platforms.TpuPlatform" if is_tpu else None|g' vllm/platforms/__init__.py
```

Then continue the installation steps until pip install succeeds.

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

Run `Llama 3.2 1B` offline inference on 1 TPU chip:

```
export TPU_BACKEND_TYPE=jax
python vllm/examples/offline_inference/basic/generate.py \
    --model=meta-llama/Llama-3.2-1B \
    --tensor_parallel_size=1 \
    --task=generate \
    --max_model_len=1024
```

Run `Llama 3.1 8B` offline inference on 4 TPU chips:

```
export TPU_BACKEND_TYPE=jax
python vllm/examples/offline_inference/basic/generate.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024
```

## Relevant env

To enable JAX path:

```
export TPU_BACKEND_TYPE=jax
```

To enable experimental scheduler:

```
export EXP_SCHEDULER=1
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
