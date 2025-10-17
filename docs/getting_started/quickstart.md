# Get started with vLLM TPU

Google Cloud TPUs (Tensor Processing Units) accelerate machine learning workloads. vLLM supports TPU v6e and v5e. For architecture, supported topologies, and more, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) and specific TPU version pages ([v5e](https://cloud.google.com/tpu/docs/v5e) and [v6e](https://cloud.google.com/tpu/docs/v6e)).

---

## Requirements

* **Google Cloud TPU VM:** Access to a TPU VM. For more information, see [Manage TPU resources](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm).
* **TPU versions:** v6e, v5e
* **Python:** 3.11 or newer (3.12 used in examples).

---

## Install using pre-built wheels

To install vLLM TPU, you can either install using `pip` (see section [Install using pip](#install-using-pip)) or run `vllm-tpu` as a Docker image (see section [Run `vllm-tpu` as a Docker image](#run-vllm-tpu-as-a-docker-image)).

### Install using pip

1. Create a working directory:

    ```shell
    mkdir ~/work-dir
    cd ~/work-dir
    ```

1. Set up a Python virtual environment:

    ```shell
    python3.12 -m venv vllm_env --symlinks
    source vllm_env/bin/activate
    ```

    Note: You don’t need to clone the `vllm` and `tpu-inference` repos to execute any of the commands in this guide

1. Use the following command to install vllm-tpu using `pip`

    ```shell
    pip install vllm-tpu
    ```

1. Get access to the ["meta-llama/Llama-3.1-8B" model](https://huggingface.co/meta-llama/Llama-3.1-8B) on Hugging Face. This is the model used in the following example, but you can use any supported model.

1. Generate a new [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) if you don't already have one:

    1. Go to **Your Profile \> Settings \> Access Tokens**.
    2. Select **Create new token**.
    3. Specify a name of your choice and a role with at least **Read** permissions.
    4. Select **Generate a token**.
    5. Copy the generated token to your clipboard, set it as an environment variable, and authenticate with the `huggingface-cli`. Replace `YOUR_TOKEN` with the generated token:

        ```shell
        export TOKEN=YOUR_TOKEN
        git config --global credential.helper store
        huggingface-cli login --token $TOKEN
        ```

1. Launch the vLLM server:

    The following command downloads the model weights from [Hugging Face Model Hub](https://huggingface.co/docs/hub/en/models-the-hub) to the TPU VM's `/tmp` directory, pre-compiles a range of input shapes, and writes the model compilation to `~/.cache/vllm/xla_cache`.

    ```shell
    cd ~/work-dir/vllm
    vllm serve "meta-llama/Llama-3.1-8B" --download_dir /tmp --disable-log-requests --tensor_parallel_size=1 --max-model-len=2048 &> serve.log &
    ```

1. Run the vLLM benchmarking script:

    ```shell
    export MODEL="meta-llama/Llama-3.1-8B"
    pip install pandas
    pip install datasets
    vllm bench serve \
        --backend vllm \
        --model $MODEL  \
        --dataset-name random \
        --random-input-len 1820 \
        --random-output-len 128 \
        --random-prefix-len 0
    ```

### Run vllm-tpu as a Docker image

1. Include the `--privileged`, `--net=host`, and `--shm-size=150gb` options to enable TPU interaction and shared memory.

    ```shell
    export DOCKER_URI=vllm/vllm-tpu:latest
    sudo docker run -it --rm --name $USER-vllm --privileged --net=host \
        -v /dev/shm:/dev/shm \
        --shm-size 150gb \
        -p 8000:8000 \
        --entrypoint /bin/bash ${DOCKER_URI}
    ```

1. Start the vLLM OpenAI API server (inside the container):

    ```shell
    export HF_HOME=/dev/shm/vllm
    export HF_TOKEN=<your-token>
    export MAX_MODEL_LEN=4096
    export TP=1 # number of chips
    vllm serve meta-llama/Meta-Llama-3.1-8B \
        --seed 42 \
        --disable-log-requests \
        --gpu-memory-utilization 0.98 \
        --max-num-batched-tokens 2048 \
        --max-num-seqs 256 \
        --tensor-parallel-size $TP \
        --max-model-len $MAX_MODEL_LEN
    ```

    Note: Adjust `--model` if you’re using a different model and `--tensor-parallel-size` if you want to use a different number of tensor parallel replicas.

1. Send a client request from your host or another terminal. For example:

    First, let's try to get into the running docker:
    ```shell
    sudo docker exec -it $USER-vllm bash
    ```

    Now, we can send the request to the vllm server:
    ```shell
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Llama-3.1-8B",
            "prompt": "Hello, my name is",
            "max_tokens": 20,
            "temperature": 0.7
        }'
    ```

## For further reading

* [Documentation](https://github.com/vllm-project/tpu-inference/tree/main/docs)
* [Examples](https://github.com/vllm-project/tpu-inference/tree/main/examples)
* [Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)
* [GKE serving with vLLM TPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)
