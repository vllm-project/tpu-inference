# Installation

This guide provides instructions for installing and running `tpu-inference`.

There are three ways to install `tpu-inference`:

1. **[Install with pip](#install-using-pip)**
2. **[Run with Docker](#run-with-docker)**
3. **[Install from source](#install-from-source)**

## Install using pip

1. Create a working directory:

    ```shell
    mkdir ~/work-dir
    cd ~/work-dir
    ```

2. Set up a Python virtual environment:

    ```shell
    python3.12 -m venv vllm_env --symlinks
    source vllm_env/bin/activate
    ```

3. Use the following command to install vllm-tpu using `pip`

    ```shell
    # v6e and prior
    pip install vllm-tpu --version=0.13.2.post6

    # v7x
    pip install vllm-tpu
    ```

    !!! note
        Until jax=0.8.3 is released, v7x will have different build requirements from previous TPU generations (v6e and prior). As a result, please use the following until consolidation is complete.

## Run with Docker

Include the `--privileged`, `--net=host`, and `--shm-size=150gb` options to enable TPU interaction and shared memory.

```shell
export DOCKER_URI=vllm/vllm-tpu:latest
sudo docker run -it --rm --name $USER-vllm --privileged --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 150gb \
    -p 8000:8000 \
    --entrypoint /bin/bash ${DOCKER_URI}
```

## Install from source

For debugging or development purposes, you can install `tpu-inference` from source. `tpu-inference` is a plugin for `vllm`, so you need to install both from source.

1. Install system dependencies:

    ```shell
    sudo apt-get update && sudo apt-get install -y libopenblas-base libopenmpi-dev libomp-dev
    ```

1. Clone the `vllm` and `tpu-inference` repositories:

    ```shell
    git clone https://github.com/vllm-project/vllm.git
    git clone https://github.com/vllm-project/tpu-inference.git
    ```

1. Set up a Python virtual environment:

    ```shell
    python3.12 -m venv vllm_env --symlinks
    source vllm_env/bin/activate
    ```

1. Install `vllm` from source, targeting the TPU device:

    ```shell
    cd vllm
    pip install -r requirements/tpu.txt
    VLLM_TARGET_DEVICE="tpu" pip install -e .
    cd ..
    ```

1. Install `tpu-inference` from source:

    ```shell
    cd tpu-inference
    pip install -e .
    cd ..
    ```
