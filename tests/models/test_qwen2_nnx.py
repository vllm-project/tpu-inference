import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import Qwen2Config
# Import the PyTorch version for comparison
from vllm.model_executor.models.qwen import QWenMLP as PytorchQWenMLP

from tpu_commons.models.jax.qwen_nnx import Qwen2MLP


def test_qwen2_mlp():
    hidden_size = 16
    intermediate_size = 32

    config = Qwen2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key)
    dtype = jnp.bfloat16
    torch_dtype = torch.bfloat16

    batch_size = 2
    seq_len = 4
    x_jax = jnp.ones((batch_size, seq_len, config.hidden_size), dtype=dtype)

    # Initialize JAX Qwen2MLP
    jax_mlp = Qwen2MLP(config=config, dtype=dtype, rng=rngs)

    # Manually create and set weights for JAX MLP for reproducibility
    key_gate, key_up, key_down = jax.random.split(key, 3)
    w_gate_jax = jax.random.normal(
        key_gate, (config.hidden_size, config.intermediate_size), dtype=dtype)
    w_up_jax = jax.random.normal(
        key_up, (config.hidden_size, config.intermediate_size), dtype=dtype)
    w_down_jax = jax.random.normal(
        key_down, (config.intermediate_size, config.hidden_size), dtype=dtype)

    jax_mlp.gate_proj.kernel.value = w_gate_jax
    jax_mlp.up_proj.kernel.value = w_up_jax
    jax_mlp.down_proj.kernel.value = w_down_jax

    # Run JAX MLP
    output_jax = jax_mlp(x_jax)

    # Check JAX output shape and dtype
    assert output_jax.shape == (batch_size, seq_len, config.hidden_size)
    assert output_jax.dtype == dtype

    # Initialize PyTorch QWenMLP
    # Note: PyTorch QWenMLP's intermediate_size directly corresponds to Qwen2Config's intermediate_size here.
    torch_mlp = PytorchQWenMLP(hidden_size=config.hidden_size,
                               intermediate_size=config.intermediate_size)
    torch_mlp.to(torch_dtype)

    # Set weights for PyTorch MLP from JAX weights
    # PyTorch Linear layers expect (out_features, in_features)
    # JAX nnx.Linear kernel is (in_features, out_features)
    # MergedColumnParallelLinear stacks weights for gate and up
    torch_mlp.gate_up_proj.weight.data = torch.cat([
        torch.from_numpy(np.array(w_gate_jax.T)),
        torch.from_numpy(np.array(w_up_jax.T))
    ],
                                                   dim=0).to(torch_dtype)
    torch_mlp.c_proj.weight.data = torch.from_numpy(np.array(
        w_down_jax.T)).to(torch_dtype)

    # Prepare PyTorch input and run
    x_torch = torch.from_numpy(np.array(x_jax)).to(torch_dtype)
    output_torch = torch_mlp(x_torch)
    output_torch_jax = jnp.asarray(output_torch.detach().numpy(), dtype=dtype)

    # Compare outputs
    assert jnp.allclose(output_jax, output_torch_jax, atol=1e-2, rtol=1e-2), \
        f"JAX and PyTorch MLP outputs differ.\nJAX:\n{output_jax}\nPyTorch:\n{output_torch_jax}"

    print(
        "Successfully tested Qwen2MLP (JAX) against QWenMLP (PyTorch). Outputs match."
    )


if __name__ == "__main__":
    test_qwen2_mlp()
