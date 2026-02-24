# Get started with vLLM TPU

Google Cloud TPUs (Tensor Processing Units) accelerate machine learning workloads. vLLM supports TPU v6e and v5e. For architecture, supported topologies, and more, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) and specific TPU version pages ([v5e](https://cloud.google.com/tpu/docs/v5e) and [v6e](https://cloud.google.com/tpu/docs/v6e)).

---

## Requirements

* **Google Cloud TPU VM:** Access to a TPU VM. For setup instructions, see the [Cloud TPU Setup guide](tpu_setup.md).
* **TPU versions:** v7x, v6e, v5e
* **Python:** 3.11 or newer (3.12 used in examples).

---

## Installation

For detailed steps on installing `vllm-tpu` with `pip` or running it as a Docker image, please see the [**Installation Guide**](installation.md).

We recommend using [uv](https://docs.astral.sh/uv/) (`uv pip install`) instead of standard `pip` as it improves installation speed.

```shell
pip install vllm-tpu
```

## Verify Installation

To quickly verify that the installation was successful and `vllm-tpu` is correctly configured:

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

## Run the vLLM Server

After installing `vllm-tpu`, you can start the API server.

1. **Log in to Hugging Face:**
   You'll need a Hugging Face token to download models.

   ```shell
   export TOKEN=YOUR_TOKEN
   git config --global credential.helper store
   huggingface-cli login --token $TOKEN
   ```

2. **Launch the Server:**
   The following command starts the server with the Llama-3.1-8B model.

   ```shell
   vllm serve "meta-llama/Llama-3.1-8B" \
       --download_dir /tmp \
       --disable-log-requests \
       --tensor_parallel_size=1 \
       --max-model-len=2048
   ```

3. **Send a Request:**

Once the server is running, you can send it a request using `curl`:

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

## Next steps:

Check out complete, end-to-end example recipes in the [tpu-recipes repository](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)

## For further reading

* [Examples](https://github.com/vllm-project/tpu-inference/tree/main/examples)
* [v7x (Ironwood) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM)
* [v6e (Trillium) Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)
* [GKE serving with vLLM TPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)
