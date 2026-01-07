# Custom JAX Model Onboarding as a Plugin

This guide walks you through the steps to implement a basic JAX model to TPU Inference.

## 1. Bring your model code

This guide assumes that your model is written for JAX.

## 2. Make your code compatible with vLLM

To ensure compatibility with vLLM, your model must meet the following requirements:

**Initialization Code**

All vLLM modules within the model must include a `vllm_config` argument in their constructor. This holds all vllm-related configuration as well as model configuration.

The initialization code should look like this:

```python
class LlamaForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = LlamaModel(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )
```

**Computation Code**

The forward pass of the model should be in `__call__` which must have at least these arguments:

```python
def __call__(
    self,
    kv_caches: List[jax.Array],
    input_ids: jax.Array,
    attention_metadata: AttentionMetadata,
) -> Tuple[List[jax.Array], jax.Array]:
    …
```

For reference, check out [our Llama implementation](https://github.com/vllm-project/tpu-inference/blob/aad6cc2a36a2cf0de681f76055ce632d5abeca5f/tpu_inference/models/jax/llama3.py).

## 3. Implement the weight loading logic

You now need to implement the `load_weights` method in your `*ForCausalLM` class. This method should load the weights from the HuggingFace's checkpoint file (or a compatible local checkpoint) and assign them to the corresponding layers in your model.

## 4. Register your model

TPU Inference relies on a model registry to determine how to run each model. A list of pre-registered architectures can be found [here](https://github.com/vllm-project/tpu-inference/blob/aad6cc2a36a2cf0de681f76055ce632d5abeca5f/tpu_inference/models/jax/model_loader.py#L22).

If your model is not on this list, you must register it to TPU Inference. You can load an external model using a plugin (similar to [vLLM’s plugins](https://docs.vllm.ai/en/latest/contributing/model/registration.html)) without modifying the TPU Inference codebase.

Structure your plugin as following:

```shell
├── setup.py
├── your_code
│   ├── your_code.py
│   └── __init__.py
```

The `setup.py` build script should follow the [same guidance as for vLLM plugins](https://docs.vllm.ai/en/latest/design/plugin_system.html#how-vllm-discovers-plugins).

To register the model, use the following code in `your_code/__init__.py`:

```python
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model

logger = init_logger(__name__)

def register():
    from .your_code import YourModelForCausalLM
    register_model("YourModelForCausalLM", YourModelForCausalLM)
```

## 5. Install and run your model

Ensure that you `pip install .` your model from within the same Python environment as vllm/tpu inference. Then to run your model:

```shell
HF_TOKEN=token TPU_BACKEND_TYPE=jax \
  python -m vllm.entrypoints.cli.main serve \
  /path/to/hf_compatible/weights/ \
  --max-model-len=1024 \
  --tensor-parallel-size 8 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs=1 \
```
