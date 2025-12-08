# Contributing to TPU Inference

Thank you for your interest in contributing to TPU Inference! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

* Identify and report any issues or bugs.
* Request or add support for a new model.
* Suggest or implement new features.
* Improve documentation or contribute a how-to guide.

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about TPU Inference. Talk about it in your blog posts and highlight how it's driving your incredible projects. Express your support on social media if you're using TPU Inference, or simply offer your appreciation by starring our repository!

## Getting Started
We recommend filtering on the “Good First Issue” tag in the [Issues](https://github.com/vllm-project/tpu-inference/issues) section of Github if it's your first time contributing!

## Issues
If you encounter a bug or have a feature request, please search [existing issues](https://github.com/vllm-project/tpu-inference/issues) first to see if it has already been reported. If not, please [file a new issue](https://github.com/vllm-project/tpu-inference/issues/new/choose), providing as much relevant information as possible.

## Directory Structure
We choose to follow a similar directory structure as vLLM:
* `tpu_inference/layers/`:
  * `common` contains layers that are common to both vLLM and JAX
  * `jax` contains layers that are only used by JAX models
  * `vllm` contains layers that are only used by vLLM models
* `tpu_inference/models/`
  * `common` contains model implementations/functionalities that are used by both vLLM and JAX
  * `jax` contains model implementations/functionalities that are only used by JAX models
  * `vllm` contains model implementations/functionalities that are only used by vLLM models

## Testing
When checking in a new feature, we expect that you you add relevant unit tests as well as CI tests.  You can read more about the latter [here](https://github.com/vllm-project/tpu-inference/tree/main/.buildkite#adding-a-new-feature-to-ci).

## Setting up linting, formatting, and static type checking

```
pip install pre-commit

# Linting, formatting and static type checking
pre-commit install --hook-type pre-commit --hook-type commit-msg

# You can manually run pre-commit with
pre-commit run --all-files
```

## Thank You!
We wanted to thank you for taking the time to read these guidelines and for your interest in contributing to TPU Inference. All of your contributions help make TPU Infernece a great tool and community for everyone!
