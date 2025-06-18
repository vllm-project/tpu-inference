## Develop on a TPU VM

### Install `vLLM-TPU`:

Follow this [guide](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator.html#set-up-using-python) to install vLLM from source.

**NOTE**: Right after `git clone` vLLM repo and before running any `pip install` commands, run the following command to pin the version:

```
git checkout 53a5a0ce30dd623808ebd02947e5183f918b6c2f
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
python vllm/examples/offline_inference/basic/generate.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024 \
    --max_num_seqs=1
```

### Run vLLM Pytorch models on the JAX path

Run the vLLM's implementation of `Llama 3.1 8B`, which is in Pytorch. It is the same command as above with the extra env var `MODEL_IMPL_TYPE=vllm`:

```
export MODEL_IMPL_TYPE=vllm
export TPU_BACKEND_TYPE=jax
python vllm/examples/offline_inference/basic/generate.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --task=generate \
    --max_model_len=1024 \
    --max_num_seqs=1
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
    --max_num_seqs=1 \
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
MODEL_IMPL_TYPE=flax_nn
MODEL_IMPL_TYPE=flax_nnx
MODEL_IMPL_TYPE=vllm
```

To enable experimental scheduler:

```
EXP_SCHEDULER=1
```

To inspect model weights sharding:

```
INSPECT_MODEL=1
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
  python /workspace/vllm/examples/offline_inference/basic/generate.py \
  --model=meta-llama/Llama-3.1-8B \
  --tensor_parallel_size=4 \
  --task=generate \
  --max_model_len=1024 \
  --max_num_seqs=1
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
BUILDKITE_COMMIT=3843efc .buildkite/scripts/run_in_docker.sh bash /workspace/tpu_commons/tests/e2e/benchmarking/llama3.1_8b_mmlu.sh
```

While this will run the code in a Docker image, you can also run the bare `tests/e2e/benchmarking/llama3.1_8b_mmlu.sh` script itself,
being sure to pass the proper args for your machine.
