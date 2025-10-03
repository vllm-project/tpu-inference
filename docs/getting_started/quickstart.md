# Get started with vLLM TPU and `tpu-inference`

This guide outlines the steps to install vLLM and `tpu-inference`. You can [install vLLM TPU within a Docker container](#docker-based-installation), or you can [build from source](#build-from-source). 

> [!NOTE]
> A Colab for the install guide is available here: [`tpu_commons -
README.ipynb`](https://colab.research.google.com/drive/1VezoeLmK3UUBgJ5MJdmtR3oOm-U4vh1l?resourcekey=0-dkk83hE-cPF8o7fc6UCyXA#scrollTo=d3mM9Wt6B2o5)

**Important considerations**:

- **Python 3.12**: This setup targets Python 3.12. If your environment uses an
  older version of Python, the installation step will upgrade it, requiring a **runtime
  restart**.

- **XLA compilation**: The first model run will incur significant XLA compilation
time (20-30+ minutes).

## Docker-based installation

The following sections describe how to install vLLM TPU within a Docker container. 
This installation method requires Docker to be installed and configured for TPU access.

### Prerequisites

Before you begin, you need to set up the following components:

- **Docker**: [Install Docker Engine](https://docs.docker.com/engine/install/), and [make sure the Docker daemon is running](https://docs.docker.com/engine/daemon/start/).
- **Docker repository**: [Create a Docker repository in Artifact Registry](https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images) or use an existing one.
- **TPU access configuration**: You must configure your Docker daemon to access
  TPUs. For more information, see [Run TPU workloads in a Docker container](https://cloud.google.com/tpu/docs/run-in-container)
- **Hugging Face access token**: [Create a HuggingFace User Access Token](https://huggingface.co/docs/hub/en/security-tokens).

### 1. Build the Docker image

1. Clone the `tpu-inference` repository and change to the `tpu-inference` directory:

    ```bash
    cd ~
    git clone https://github.com/vllm-project/tpu-inference.git
    cd tpu-inference
    ```
    
1. Set the full name and destination for your Docker container image.
   Replace `<YOUR-ARTIFACT-REGISTRY-URI>` with the path where your container image will be stored.
   For example, `us-central1-docker.pkg.dev/my-gcp-project/my-app-repo/vllm-tpu:v1`. 

    ```bash
    DOCKER_URI=<YOUR-ARTIFACT-REGISTRY-URI>
    ```

1. Build and push the Docker image:

    ```bash
    docker build -f docker/Dockerfile -t $DOCKER_URI .
    docker push $DOCKER_URI
    ```

### 2. Run the Docker container for inference

> [!IMPORTANT]
> Ensure your Docker environment is configured for TPU access.

The following command runs an offline inference task using a pre-built Docker image (`vllm-tpu-tpu_commons`). 
The command loads the TinyLlama model from HuggingFace and uses the example [`offline_inference.py`](https://github.com/vllm-project/tpu-inference/blob/main/examples/offline_inference.py) script to run inference.

Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your HuggingFace access token. For more information, see [User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens).

```bash
# Basic run command (ensure your Docker setup provides TPU access)
docker run -it --rm \
    -e HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN" \
    -e MODEL_IMPL_TYPE="vllm" \
    # Add persistent caching options  
    -v $(pwd)/vllm_tpu_data/xla_cache:/app/xla_cache \
    -v $(pwd)/vllm_tpu_data/hf_cache:/app/hf_cache \
    # Add your TPU runtime/device flags here (e.g., --runtime=cloud-tpu)
    vllm-tpu-tpu_commons \
    /app/tpu-inference/examples/offline_inference.py \
    --model=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tensor_parallel_size=1 \
    --max_model_len=1024
```

## Build from source

The following sections describe how to build vLLM-TPU from source.

### Prerequisites

Before you begin, you need to set up the following components:

- **Hugging Face access token**: [Create a HuggingFace User Access Token](https://huggingface.co/docs/hub/en/security-tokens).

### 1. Create Python virtual environment and clone repositories

Create a virtual environment and clone `vllm` and `tpu-inference`.

1. Create a virtual environment:

    ```bash
    # Create venv using python3.12 (or default python3 if 3.11+)
    python3.12 -m venv vllm_env --symlinks
    export VENV_PYTHON="/content/vllm_env/bin/python" # Adjust path if not in /content
    ```

1. Make sure `pip` is installed and up-to-date in the virtual environment:

    ```bash
    # Ensure pip in venv
    $VENV_PYTHON -m ensurepip --upgrade --default-pip
    ```

1. Clone the `vllm` and `tpu-inference` repositories.

   Replace `${GITHUB_PAT}` with your GitHub Personal Access Token (PAT).

    ```bash
    # Clone repositories
    git clone https://github.com/vllm-project/vllm.git
    git clone https://${GITHUB_PAT}@github.com/vllm-project/tpu-inference.git
    ```

### 2. Install Dependencies

Install `vLLM-TPU` from source and `tpu-inference` with its requirements.

1. Install vLLM-TPU:

    ```bash
    echo "Installing vLLM-TPU..."
    cd vllm
    $VENV_PYTHON -m pip uninstall torch torch-xla -y || echo "No conflicting torch/torch-xla packages found."
    sudo apt-get update
    sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev
    VLLM_TARGET_DEVICE="tpu" $VENV_PYTHON -m pip install -e .
    cd ..
    ```

1. Install `tpu-inference`:

    ```bash
    echo "Installing tpu-inference..."
    cd tpu-inference
    $VENV_PYTHON -m pip install -r requirements.txt
    $VENV_PYTHON -m pip install -e . numba
    cd ..
    ```

### 3. Verify Installation

Check if the key packages are installed in your virtual environment:

```bash
echo "Verifying installed packages..."
$VENV_PYTHON -m pip list | grep vllm
$VENV_PYTHON -m pip list | grep torch-xla
$VENV_PYTHON -m pip list | grep tpu-commons
```

## Examples

The following sections provide example commands for various scenarios, including offline inference with Llama 3.1 models on single or multiple TPU chips, disaggregated serving, and multi-host serving with Ray. These examples will help you get started with running vLLM on TPUs.

### Run Llama 3.1 8B offline inference on 4 TPU chips

This example demonstrates a standard offline inference task. The command runs the Llama 3.1 8B model across four TPU chips, using a tensor parallel size of 4 to distribute the workload.

Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
HF_TOKEN=hf_YOUR_HUGGING_FACE_TOKEN \ 
python tpu-inference/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=4 \
    --max_model_len=1024
```

### Run Llama 3.1 8B Instruct offline inference on 4 TPU chips in disaggregated mode

This example shows how to run inference in disaggregated mode, which separates the prefill and decode stages of inference onto different TPU chips. The `PREFILL_SLICES` and `DECODE_SLICES` environment variables are used to configure this setup for the Llama 3.1 8B Instruct model 

Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
PREFILL_SLICES=2 \
DECODE_SLICES=2 \ 
HF_TOKEN=hf_YOUR_HUGGING_FACE_TOKEN \
python tpu-inference/examples/offline_inference.py \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --max_model_len=1024 \
    --max_num_seqs=8
```

### Run JAX models with llm-d disaggregated serving

This example simulates a production-like disaggregated serving scenario known as llm-d on a single TPU VM. Executing the `run_disagg_single_host.sh` script starts the necessary servers to mimic a disaggregated environment.

```bash
bash examples/disagg/run_disagg_single_host.sh
```

Follow the instructions in the command output to send requests.

### Run JAX model with Ray-based multi-host serving

This example demonstrates how to run inference on a large model, Llama 3.1 70B, across a multi-host environment using Ray for orchestration. The example runs on 4 hosts (v6e-16) in interleaved mode.

1. Deploy Ray cluster and containers:

   ```bash
    ~/tpu-inference/scripts/multihost/deploy_cluster.sh \
        -s ~/tpu-inference/scripts/multihost/run_cluster.sh \
        -d "<your_docker_image>" \
        -c "<path_on_remote_hosts_for_hf_cache>" \
        -t "<your_hugging_face_token>" \
        -H "<head_node_public_ip>" \
        -i "<head_node_private_ip>" \
        -W "<worker1_public_ip>,<worker2_public_ip>,<etc...>"
   ```
    
1. On the head node, use `sudo docker exec -it node /bin/bash` to enter the
   container:

    ```bash
    sudo docker exec -it node /bin/bash
    ```

1. Run the inference script on the head node. Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

    ```bash
    HF_TOKEN=hf_YOUR_HUGGING_FACE_TOKEN python /workspace/tpu-inference/examples/offline_inference.py \
        --model=meta-llama/Llama-3.1-70B  \
        --tensor_parallel_size=16  \
        --max_model_len=1024
    ```

### Run vLLM PyTorch offline inference

This example shows how to run an offline inference task using the vLLM PyTorch implementation. The example explicitly sets the `MODEL_IMPL_TYPE` environment variable to `vllm`, which instructs the program to use the PyTorch-based model path for the TinyLlama model.

Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
export HF_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN"
export MODEL_IMPL_TYPE="vllm"
$VENV_PYTHON /content/tpu-inference/examples/offline_inference.py \
    --model=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tensor_parallel_size=1 \
    --max_model_len=1024
```
