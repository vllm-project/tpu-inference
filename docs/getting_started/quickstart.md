# Get started with vLLM TPU

Google Cloud TPUs (Tensor Processing Units) accelerate machine learning workloads. vLLM supports TPU v6e and v5e. For architecture, supported topologies, and more, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) and specific TPU version pages ([v5e](https://cloud.google.com/tpu/docs/v5e) and [v6e](https://cloud.google.com/tpu/docs/v6e)).

---

## Requirements

* **Google Cloud TPU VM:** Access to a TPU VM. For more information, see the [Cloud TPU Setup guide](tpu_setup.md).
* **TPU versions:** v6e, v5e
* **Python:** 3.11 or newer (3.12 used in examples).

---

## Installation

First, follow the [Installation guide](installation.md) to install `vllm-tpu`.

## Running the server

After installation, you can launch the vLLM server.

1. Get access to the ["meta-llama/Llama-3.1-8B" model](https://huggingface.co/meta-llama/Llama-3.1-8B) on Hugging Face. This is the model used in the following example, but you can use any supported model.

2. Generate a new [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) if you don't already have one:

    1. Go to **Your Profile > Settings > Access Tokens**.
    2. Select **Create new token**.
    3. Specify a name of your choice and a role with at least **Read** permissions.
    4. Select **Generate a token**.
    5. Copy the generated token to your clipboard, set it as an environment variable, and authenticate with the `huggingface-cli`. Replace `YOUR_TOKEN` with the generated token:

        ```shell
        export TOKEN=YOUR_TOKEN
        git config --global credential.helper store
        huggingface-cli login --token $TOKEN
        ```

3. Launch the vLLM server:

    The following command downloads the model weights from [Hugging Face Model Hub](https://huggingface.co/docs/hub/en/models-the-hub) to the TPU VM's `/tmp` directory, pre-compiles a range of input shapes, and writes the model compilation to `~/.cache/vllm/xla_cache`.

    ```shell
    vllm serve "meta-llama/Llama-3.1-8B" --download_dir /tmp --disable-log-requests --tensor_parallel_size=1 --max-model-len=2048 &> serve.log &
    ```

4. Confirm vLLM server is running:

    ```shell
    tail -f serve.log
    ```

    When you see the following, the server is ready to recieve requests:

    ```
    INFO:     Started server process []
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    ```

5. Run the vLLM benchmarking script:

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

## For further reading

* [Documentation](https://github.com/vllm-project/tpu-inference/tree/main/docs)
* [Examples](https://github.com/vllm-project/tpu-inference/tree/main/examples)
* [Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)
* [GKE serving with vLLM TPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)
