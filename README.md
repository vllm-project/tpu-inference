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
