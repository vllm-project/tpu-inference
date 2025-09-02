# 🔬 **IMPORTANT: EXPERIMENTAL AND NOT SUPPORTED** 🔬

This is an exploratory repository provided for informational and learning purposes only.
The code is **not feature-complete** and **may not be stable**.

> ⚠️ **DO NOT USE IN A PRODUCTION ENVIRONMENT.**

## Develop on a TPU VM

### Install `vLLM-TPU`:

Follow this [guide](https://docs.vllm.ai/en/latest/getting_started/installation/google_tpu.html#set-up-using-python) to install vLLM from source.

### Install `tpu_commons`:

```
cd ~
git clone https://github.com/vllm-project/tpu_commons.git
cd tpu_commons
pip install -r requirements.txt
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

## Run examples

### Run JAX models

Run `Llama 3.1 8B` offline inference on 4 TPU chips:

```
HF_TOKEN=<huggingface_token> python tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --max_model_len=1024
```

### Run JAX models with local disaggregated serving

Run `Llama 3.1 8B Instruct` offline inference on 4 TPU chips in disaggregated mode:

```
PREFILL_SLICES=2 DECODE_SLICES=2 HF_TOKEN=<huggingface_token> \
python tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --max_model_len=1024 \
    --max_num_seqs=8
```

### Run JAX models with llm-d disaggregated serving

We simulate the llm-d scenario using a single TPU VM.

```
bash examples/disagg/run_disagg_servers.sh
```

Then follow the instructions output by the command to send requests.

### Run JAX model with Ray-based multi-host serving

Run `Llama 3.1 70B Instruct` offline inference on 4 hosts (v6e-16) in interleaved mode:

1. Deploy Ray cluster and containers:

```
~/tpu_commons/scripts/multihost/deploy_cluster.sh \
    -s ~/tpu_commons/scripts/multihost/run_cluster.sh \
    -d "<your_docker_image>" \
    -c "<path_on_remote_hosts_for_hf_cache>" \
    -t "<your_hugging_face_token>" \
    -H "<head_node_public_ip>" \
    -i "<head_node_private_ip>" \
    -W "<worker1_public_ip>,<worker2_public_ip>,<etc...>"
```

1. On the head node, use `sudo docker exec -it node /bin/bash` to enter the container. And then execute:

```
HF_TOKEN=<huggingface_token> python /workspace/tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-70B  \
    --tensor_parallel_size=16  \
    --max_model_len=1024
```

### Run vLLM Pytorch models on the JAX path

Run the vLLM's implementation of `Llama 3.1 8B`, which is in Pytorch. It is the same command as above with the extra env var `MODEL_IMPL_TYPE=vllm`:

```
export MODEL_IMPL_TYPE=vllm
export HF_TOKEN=<huggingface_token>
python tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --max_model_len=1024
```

Run the vLLM Pytorch `Qwen3-30B-A3B` MoE model, use `--enable-expert-parallel` for expert parallelism, otherwise it defaults to tensor parallelism:

```
export MODEL_IMPL_TYPE=vllm
export HF_TOKEN=<huggingface_token>
python vllm/examples/offline_inference/basic/generate.py \
    --model=Qwen/Qwen3-30B-A3B \
    --tensor_parallel_size=4 \
    --max_model_len=1024 \
    --enable-expert-parallel
```

## Run docker containers

### Build and push docker image

This can be run on a CPU VM.

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

### Download docker image and run

Pull the docker image and run it:

```
DOCKER_URI=<the same URI used in docker build>
docker pull $DOCKER_URI
docker run \
  --rm \
  $DOCKER_URI \
  python /workspace/tpu_commons/examples/offline_inference.py \
  --model=meta-llama/Llama-3.1-8B \
  --tensor_parallel_size=4 \
  --max_model_len=1024 \
```

### Relevant env

To switch different model implementations (default is flax_nnx):

```
MODEL_IMPL_TYPE=flax_nnx
MODEL_IMPL_TYPE=vllm
```

To run JAX models without precompiling:

```
SKIP_JAX_PRECOMPILE=1
```

To run JAX models with random initialized weights:

```
JAX_RANDOM_WEIGHTS=1
```

To run workloads on multi-host:

```
TPU_MULTIHOST_BACKEND=ray

```

## Profiling

There are two ways to profile your workload:

### Using `PHASED_PROFILING_DIR`
If you set the following environment variable:

```

PHASED_PROFILING_DIR=<DESIRED PROFILING OUTPUT DIR>

```

we will automatically capture profiles during three phases of your workload (assuming they are encountered):
1. Prefill-heavy (the quotient of prefill / total scheduled tokens for the given batch is => 0.9)
2. Decode-heavy (the quotient of prefill / total scheduled tokens for the given batch is <= 0.2)
3. Mixed (the quotient of prefill / total scheduled tokens for the given batch is between 0.4 and 0.6)

To aid in your analysis, we will also log the batch composition for the profiled batches.

#### Using `USE_JAX_PROFILER_SERVER`
If you set the following environment variable:

```

USE_JAX_PROFILER_SERVER=True

```

you can instead manually decide when to capture a profile and for how long, which can helpful if your workload (e.g. E2E benchmarking) is
large and taking a profile of the entire workload (i.e. using the above method) will generate a massive tracing file.

You can additionally set the desired profiling port (default is `9999`):

```

JAX_PROFILER_SERVER_PORT=XXXX

```

In order to use this approach, you can do the following:

1. Run your typical `vllm serve` or `offline_inference` command (making sure to set `USE_JAX_PROFILER_SERVER=True`)
2. Run your benchmarking command (`python benchmark_serving.py...`)
3. Once the warmup has completed and your benchmark is running, start a new tensorboard instance with your `logdir` set to the desired output location of your profiles (e.g. `tensorboard --logdir=profiles/llama3-mmlu/`)
4. Open the tensorboard instance and navigate to the `profile` page (e.g. `http://localhost:6006/#profile`)
5. Click `Capture Profile` and, in the `Profile Service URL(s) or TPU name` box, enter `localhost:XXXX` where `XXXX` is your `JAX_PROFILER_SERVER_PORT` (default is `9999`)

6. Enter the desired amount of time (in ms) you'd like to capture the profile for and then click `Capture`.   If everything goes smoothly, you should see a success message, and your `logdir` should be populated.

## How to run an End-To-End (E2E) benchmark?
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

To enable quantization, you can do one of the following:

#### Using a quantization config YAML
Simply pass the name of a quantization config found inside the quantization config directory (`tpu_commons/models/jax/utils/quantization/configs/`), for example:

```

... --additional_config='{"quantization": "int8_default.yaml"}'

```

#### Using a quantization config JSON
Alternatively, you can pass the explicit quantization configuration as JSON string, where each entry in `rules` corresponds to a Qwix rule (see below):

```

{ "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "int8", "act_qtype": "int8" }]}}

```

### Creating your own quantization config YAML
To create your own quantization config YAML file:

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

1. To use the config, simply pass the name of the file you created in the `--additional_config`, e.g.:

```

... --additional_config='{"quantization": "YOUR_FILE_NAME_HERE.yaml"}'

```
