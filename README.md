Current structure, please add, modify, remove etc.

tpu_commons/
│── __init__.py
│── worker/
│   ├── __init__.py
│   ├── tpu_worker.py         # Moved and adapted from vllm/v1/worker/
│   └── tpu_model_runner.py   # Moved and adapted from vllm/v1/worker/
│── attention/
│   ├── __init__.py
│   └── backends/
│       ├── __init__.py
│       └── pallas/           # <<< MOVED from vllm/v1/attention/backends/pallas/
│           ├── __init__.py
│           ├── attention.py  # (or whatever files are in original pallas dir)
│           └── metadata.py   # (e.g., pallas_attention.py, pallas_metadata.py)
│── sample/
│   ├── __init__.py
│   └── tpu/                  # <<< MOVED from vllm/v1/sample/tpu/
│       ├── __init__.py
│       ├── metadata.py
│       └── sampler.py
├── setup.py
├── pyproject.toml
└── .buildkite/
    └── pipeline.yml

