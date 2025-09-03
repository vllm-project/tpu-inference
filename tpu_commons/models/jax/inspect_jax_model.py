import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig, VllmConfig

from tpu_commons.models.jax.llama_guard_4 import LlamaGuard4ForCausalLM


def print_keys_recursive(state, parent_key=""):
    """Recursively prints all keys in a nested dictionary-like object."""
    if isinstance(state, (dict, nnx.State)):
        for key, value in state.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, nnx.State)):
                print_keys_recursive(value, current_key)
            else:
                print(current_key)
    else:
        print(parent_key)


vllm_config = VllmConfig(model_config=ModelConfig(
    model="meta-llama/Llama-Guard-4-12B",
    max_model_len=1024,
))
rng = jax.random.PRNGKey(0)
mesh_devices = jax.devices()
devices_array = np.array(mesh_devices).reshape(1, 1, 1, 8)
mesh = Mesh(devices_array, ('data', 'expert', 'seq', 'model'))

print("Initializing model...")
model = LlamaGuard4ForCausalLM(vllm_config, rng, mesh)

print("\n--- JAX Model Parameter Paths ---")
model_params = nnx.state(model)
print_keys_recursive(model_params)
