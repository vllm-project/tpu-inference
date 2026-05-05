# Buildkite

https://buildkite.com/tpu-commons

The GitHub webhook is configured to trigger the Buildkite pipeline. The current step configuration of the pipeline:

```
steps:
  - label: ":pipeline: Upload Pipeline"
    agents:
      queue: cpu
    priority: 200
    command: "bash .buildkite/scripts/bootstrap.sh"
```

# Support Matrices
Besides continuous integration and continuous delivery, a major goal of our pipeline is to generate support matrices for our users for each release:
- model support matrix (intended to replace [this](https://github.com/vllm-project/vllm/blob/f552d5e578077574276aa9d83139b91e1d5ae163/docs/models/hardware_supported_models/tpu.md) from the vllm upstream)
- feature support matrix (intended to replace [this](https://github.com/vllm-project/vllm/blob/f552d5e578077574276aa9d83139b91e1d5ae163/docs/features/README.md) from the vllm upstream)
- kernel support matrix
- kernel support matrix (microbenchmarks)
- parallelism support matrix
- quantization support matrix
- rl support matrix

To support this requirement, each model and feature will go through a series of stages of testing, and the test results will be used to generate the support matrices automatically.

# Adding a new model to CI
To add a new model to CI, model owners can use the prepared [add_model_to_ci.py](pipeline_generation/add_model_to_ci.py) script. The script will populate a buildkite yaml config file in the `.buildkite/models` directory; config files under this directory will be integrated to our pipeline automatically.

## Interactive Mode
The script will prompt you for inputs and generate a YAML in the appropriate directory based on your inputs.

```bash
python .buildkite/pipeline_generation/add_model_to_ci.py
```

# Adding a new feature to CI
To add a new feature to CI, feature owners can use the prepared [`add_feature_to_ci.py`](pipeline_generation/add_feature_to_ci.py) script. The script will populate a buildkite yaml config file in the appropriate subdirectory (e.g., `.buildkite/features`, `.buildkite/parallelism`, etc.). These files will be integrated into our pipeline automatically.

## Interactive Mode
The script will prompt you for inputs and generate a YAML in the appropriate directory based on your inputs.

```bash
python .buildkite/pipeline_generation/add_feature_to_ci.py
```
