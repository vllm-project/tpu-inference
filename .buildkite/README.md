# Buildkite

https://buildkite.com/tpu-commons

The GitHub webhook is configured to trigger the Buildkite pipeline. The current step configuration of the pipeline:

```
steps:
  - label: ":pipeline: Upload Pipeline"
    agents:
      queue: tpu_v6e_queue
    command: "bash .buildkite/scripts/bootstrap.sh"
```

# Support Matrices
Besides continuous integration and continuous delivery, a major goal of our pipeline is to generate support matrices for our users for each release:
- model support matrix (intended to replace [this](https://github.com/vllm-project/vllm/blob/f552d5e578077574276aa9d83139b91e1d5ae163/docs/models/hardware_supported_models/tpu.md) from the vllm upstream)
- feature support matrix (intended to replace [this](https://github.com/vllm-project/vllm/blob/f552d5e578077574276aa9d83139b91e1d5ae163/docs/features/README.md) from the vllm upstream)

To support this requirement, each model and feature will go through a series of stages of testing, and the test results will be used to generate the support matrices automatically.

# Adding a new model to CI
## Adding a TPU-optimized model
TPU-optimized models are models we rewrite the model definition as opposed to using the model definition from the vLLM upstream. These models will go through benchmark on top of unit and integration (accuracy) tests. To add a TPU-optimized model to CI, model owners can use the prepared [add_model_to_ci.py](pipeline_generation/add_model_to_ci.py) script. The script will populate a buildkite yaml config file in the `.buildkite/models` directory; config files under this directory will be integrated to our pipeline automatically. The python script takes 2 arguments:
- **--model-name**: this is the **full name** of your model on Hugging Face. Please ensure to use the **full name** (ex: `meta-llama/Llama-3.1-8B` instead of `Llama-3.1-8B`) or else we won't be able to find your model.
- **--queue**: this is the queue you want to run on (ex: `tpu_v6e_queue`)
- **--category**: this parameter allows you to set the model category, with the following options available: "text-only" or "multimodal". (default: "text-only")

```bash
python add_model_to_ci.py --model-name <MODEL_NAME> --queue <QUEUE_NAME>
```

In the generated yml file, there are three TODOs that will need your input:
1. The test command for the unit tests of your model
2. The accuracy target for your model
3. The performance benchmark target for your model

## Adding a vLLM-native model
vLLM-native models are models using the model definition from the vLLM upstream. These models will not go through benchmark on our pipeline. To add a vLLM-native model to CI, model owners can use the prepared [add_model_to_ci.py](pipeline_generation/add_model_to_ci.py) script. The script will populate a buildkite yaml config file in the `.buildkite/models` directory; config files under this directory will be integrated to our pipeline automatically. The python script takes 3 arguments:
- **--model-name**: this is the **full name** of your model on Hugging Face. Please ensure to use the **full name** (ex: `meta-llama/Llama-3.1-8B` instead of `Llama-3.1-8B`) or else we won't be able to find your model.
- **--queue**: this is the queue you want to run on (ex: `tpu_v6e_queue`)
- **--category**: this parameter allows you to set the model category, with the following options available: "text-only" or "multimodal". (default: "text-only")

```bash
python add_model_to_ci.py --model-name <MODEL_NAME> --queue <QUEUE_NAME> --type vllm-native
```

In the generated yml file, there are two TODOs that will need your input:
1. The test command for the unit tests of your model
2. The accuracy target for your model

# Adding a new feature to CI
To add a new feature to CI, feature owners can use the prepared [add_feature_to_ci.py](pipeline_generation/add_feature_to_ci.py) script. The script will populate a buildkite yaml config file in the `.buildkite/features` directory; config files under this directory will be integrated to our pipeline automatically. The python script takes 2 arguments:
- **--feature-name**: this is the name of your feature
- **--queue**: this is the queue you want to run on (ex: `tpu_v6e_queue`)
- **--category**: this parameter allows you to set the feature category, with the following options available: "feature support matrix" or "kernel support matrix". (default: "feature support matrix")

```bash
python add_feature_to_ci.py --feature-name <FEATURE_NAME> --queue <QUEUE_NAME>

# If your feature name contains spaces, please wrap it in quotes
# ex: python add_feature_to_ci.py --feature-name 'my feature name' --queue <QUEUE_NAME>
```

In the generated yml file, there are two TODOs that will need your input:
1. The test command for the correctness tests of your feature
2. The test command for the performance tests of your feature
