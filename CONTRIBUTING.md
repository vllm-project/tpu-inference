# Contributing to TPU Commons

## Directory Structure
We choose to follow a similar directory structure as vLLM:
* `tpu_commons/layers/`:
  * `common` contains layers that are common to both TorchAX and JAX
  * `jax` contains layers that are only used by JAX models
  * `vllm` contains layers that are only used by TorchAX models
* `tpu_commons/models/`
  * `common` contains model implementations/functionalities that are used by both TorchAX and JAX
  * `jax` contains model implementations/functionalities that are only used by JAX models
  * `vllm` contains model implementations/functionalities that are only used by TorchAX models
