# 🔬 **IMPORTANT: EXPERIMENTAL AND NOT SUPPORTED** 🔬

This is an exploratory repository provided for informational and learning purposes only.
The code is **not feature-complete** and **may not be stable**.

> ⚠️ **DO NOT USE IN A PRODUCTION ENVIRONMENT.**

## Develop on a TPU VM

### Install `vLLM-TPU`:

Follow this [guide](https://docs.vllm.ai/en/latest/getting_started/installation/google_tpu.html#set-up-using-python) to install vLLM from source.

**NOTE**: Right after `git clone` vLLM repo and before running any `pip install` commands, run the following command to pin the version:

```
git checkout 0f199f197b4e7a835ccc5b4d15363f8faa7824c8
```

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

### Run JAX path examples

Run `Llama 3.1 8B` offline inference on 4 TPU chips:

```
export TPU_BACKEND_TYPE=jax
python tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024
```

### Run JAX path examples with disaggregated serving

Run `Llama 3.1 8B Instruct` offline inference on 4 TPU chips in disaggregated mode:

```
PREFILL_SLICES=2 \
DECODE_SLICES=2 \
TPU_BACKEND_TYPE=jax \
python tpu_commons/examples/offline_inference.py \
    --task=generate \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --max_model_len=1024 \
    --max_num_seqs=8
```

### Run vLLM Pytorch models on the JAX path

Run the vLLM's implementation of `Llama 3.1 8B`, which is in Pytorch. It is the same command as above with the extra env var `MODEL_IMPL_TYPE=vllm`:

```
export MODEL_IMPL_TYPE=vllm
export TPU_BACKEND_TYPE=jax
python tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024
```

Run the vLLM Pytorch `Qwen3-30B-A3B` MoE model, use `--enable-expert-parallel` for expert parallelism, otherwise it defaults to tensor parallelism:

```
export MODEL_IMPL_TYPE=vllm
export TPU_BACKEND_TYPE=jax
python vllm/examples/offline_inference/basic/generate.py \
    --model=Qwen/Qwen3-30B-A3B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024 \
    --enable-expert-parallel
```

### Relevant env

To switch different backends:

```
TPU_BACKEND_TYPE=jax
TPU_BACKEND_TYPE=torchax
TPU_BACKEND_TYPE=pytorch_xla
```

To switch different model implementations:

```
MODEL_IMPL_TYPE=flax_nnx
MODEL_IMPL_TYPE=vllm
```

To enable profiling:

```
VLLM_TORCH_PROFILER_DIR=$PWD
```

To run JAX path without precompiling the model:

```
SKIP_JAX_PRECOMPILE=1
```

To run JAX path without loading real model weights:

```
JAX_RANDOM_WEIGHTS=1
```

To enable experimental scheduler:

```
EXP_SCHEDULER=1
```

## Develop on a CPU VM and run docker on a TPU VM

### On the CPU VM

Build docker image:

```
cd ~
git clone https://github.com/vllm-project/tpu_commons.git
cd tpu_commons

DOCKER_URI=<Specify a GCR URI>
# example:
# DOCKER_URI=gcr.io/cloud-nas-260507/ullm:$USER-test

docker build -f docker/Dockerfile -t $DOCKER_URI .
docker push $DOCKER_URI
```

### On the TPU-VM side:

Pull the docker image and run it:

```
DOCKER_URI=<the same URI used in docker build>
docker pull $DOCKER_URI
docker run \
  --rm \
  -e TPU_BACKEND_TYPE=jax \
  $DOCKER_URI \
  python /workspace/tpu_commons/examples/offline_inference.py \
  --model=meta-llama/Llama-3.1-8B \
  --tensor_parallel_size=4 \
  --task=generate \
  --max_model_len=1024 \
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

## How to run an End-To-End (E2E) benchmarking run?
In order to run an [E2E benchmark test](https://github.com/vllm-project/tpu_commons/blob/main/scripts/vllm/benchmarking/README.md), which will spin up a vLLM server with Llama 3.1 8B and run a single request from the MLPerf dataset against it, you can run the
following command locally:

```
BUILDKITE_COMMIT=0f199f1 .buildkite/scripts/run_in_docker.sh bash /workspace/tpu_commons/tests/e2e/benchmarking/mlperf.sh
```

While this will run the code in a Docker image, you can also run the bare `tests/e2e/benchmarking/mlperf.sh` script itself,
being sure to pass the proper args for your machine.

You might need to run the benchmark client *twice* to make sure all compilations are cached server-side.

## Quantization
### Overview
Currently, we support overall model weight/activation quantization through the [Qwix](https://github.com/google/qwix?tab=readme-ov-file#quantization-config) framework.

To enable quantization, you can specify a quantization config filename found inside the quantization config directory (`tpu_commons/models/jax/utils/quantization/configs/`), for example:

```
... --additional_config='{"quantization": "int8_default.yaml"}'
```

### Creating your own quantization config
To create your own quantization:

1. Add a new file to the quantization config directory (`tpu_commons/models/jax/utils/quantization/configs/`)
2. For Qwix quantization, add a new entry to the file as follows:

```
qwix:
  rules:
    # NOTE: each entry corresponds to a qwix.QuantizationRule
    - module_path: '.*'
      weight_qtype: 'int8'
      act_qtype: 'int8'
```

where each entry under `rules` corresponds to a `qwix.QuantizationRule`.  To learn more about Qwix and defining Qwix rules, please see the relevant docs [here](https://github.com/google/qwix?tab=readme-ov-file#quantization-config).
