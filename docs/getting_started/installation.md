# Installation

This guide provides instructions for installing and running `tpu-inference`.

There are three ways to install `tpu-inference`:

1. **[Install using pip via uv](#install-using-pip-via-uv)**
2. **[Run with Docker](#run-with-docker)**
3. **[Install from source](#install-from-source)**

## Install using pip via uv

We recommend using [uv](https://docs.astral.sh/uv/) (`uv pip install`) instead of standard `pip` as it improves installation speed.

1. Create a working directory:

    ```shell
    mkdir ~/work-dir
    cd ~/work-dir
    ```

1. Install `uv` and set up a Python virtual environment:

    ```shell
    # If you prefer standard pip, simply use `python3.12 -m venv vllm_env`
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv venv vllm_env --python 3.12
    source vllm_env/bin/activate
    ```

1. Use the following command to install vllm-tpu using `uv` or `pip`:

    ```shell
    uv pip install vllm-tpu
    # Or instead: pip install vllm-tpu
    ```

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
    git clone https://github.com/vllm-project/tpu-inference.git
    export VLLM_COMMIT_HASH="$(cat tpu-inference/.buildkite/vllm_lkg.version)"
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout "${VLLM_COMMIT_HASH}"
    cd ..
    ```

1. Install `uv` and set up a Python virtual environment:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv venv vllm_env --python 3.12
    source vllm_env/bin/activate
    ```

1. Install `vllm` from source, targeting the TPU device:

    NOTE: `tpu-inference` repo pins `vllm` revision in `vllm_lkg.version` file,
    make sure to checkout proper revision beforehand.

    ```shell
    cd vllm
    uv pip install -r requirements/tpu.txt
    VLLM_TARGET_DEVICE="tpu" uv pip install -e .
    cd ..
    ```

1. Install `tpu-inference` from source:

    ```shell
    cd tpu-inference
    uv pip install -e .
    cd ..
    ```

## Verify Installation

To quickly verify that the installation was successful under any of the above methods and `vllm-tpu` is correctly configured:

```shell
python -c '
import jax
import vllm
import importlib.metadata
from vllm.platforms import current_platform

tpu_version = importlib.metadata.version("tpu_inference")
print(f"vllm version: {vllm.__version__}")
print(f"tpu_inference version: {tpu_version}")
print(f"vllm platform: {current_platform.get_device_name()}")
print(f"jax backends: {jax.devices()}")
'
# Expected output:
# vllm version: 0.x.x
# tpu_inference version: 0.x.x
# vllm platform: TPU V6E (or your specific TPU architecture)
# jax backends: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), ...]
```
