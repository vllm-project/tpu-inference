# Contributing to TPU Commons

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
