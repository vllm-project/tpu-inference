## Structure

Current structure, please add, modify, remove etc.

```
tpu_commons/
│── tpu_commons/
│   ├── __init__.py
|   ├── worker/
│   |   ├── __init__.py
│   |   ├── tpu_worker.py         # Moved and adapted from vllm/v1/worker/
│   |   └── tpu_model_runner.py   # Moved and adapted from vllm/v1/worker/
|   ├── kernels/
│   |   ├── __init__.py
│   |   └── ragged_paged_attention
│   |       ├── __init__.py
│   |       ├── kernel.py
│   |       └── tuned_block_sizes.py
|   │── sample/
│       ├── __init__.py
│       └── tpu/                  # <<< MOVED from vllm/v1/sample/tpu/
│          ├── __init__.py
│          ├── metadata.py
│          └── sampler.py
├── setup.py
├── tests
│   ├── __init__.py
│   └── ragged_paged_attention_test.py
├── pyproject.toml
└── .buildkite/
    └── pipeline.yml
```

## How to setup development environment?

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
cd vllm
sed -i 's|return "vllm.platforms.tpu.TpuPlatform" if is_tpu else None|return "tpu_commons.platforms.TpuPlatform" if is_tpu else None|g' vllm/platforms/__init__.py
```

Then continue the installation steps until pip install succeeds.

## How to test the JAX path?

**NOTE**: This is under development so the run may fail.

```
TPU_BACKEND_TYPE=jax python vllm/examples/offline_inference/basic/generate.py --task=generate --max_model_len=1024
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

## How to format the code?

```
pip install pre-commit

# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg

# You can manually run pre-commit with
pre-commit run --all-files
```
