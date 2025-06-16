import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.experimental import mesh_utils
# JAX utilities
from jax.sharding import Mesh
from transformers import Qwen2Config as HfQwen2Config
from vllm.config import CacheConfig  # For VllmQwen2Attention initialization
# PyTorch/vLLM components
from vllm.model_executor.models.qwen2 import \
    Qwen2Attention as VllmQwen2Attention
from vllm.model_executor.models.qwen2 import Qwen2MLP as VllmQwen2MLP

from tpu_commons.models.jax.layers.rope import apply_rope
# JAX/Flax components from tpu_commons
from tpu_commons.models.jax.qwen_nnx import Qwen2Attention as JaxQwen2Attention
from tpu_commons.models.jax.qwen_nnx import Qwen2MLP as JaxQwen2MLP


# Helper to convert torch tensor to JAX array
def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    """Converts a PyTorch tensor to a JAX array."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    return jnp.asarray(tensor.cpu().numpy())


# Helper to convert JAX array to torch tensor
def jax_to_torch(array: jax.Array) -> torch.Tensor:
    """Converts a JAX array to a PyTorch tensor."""
    return torch.from_numpy(np.asarray(array))


class TestQwen2NnxParity(unittest.TestCase):

    def setUp(self):
        self.hf_config = HfQwen2Config(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA
            rms_norm_eps=1e-6,
            max_position_embeddings=128,  # Used by RoPE and vLLM Attention
            rope_theta=10000.0,
            hidden_act="silu",
            # num_hidden_layers=1 # Not directly used by component tests
        )
        self.dtype_jax = jnp.float32
        self.dtype_torch = torch.float32
        self.rng_key = jax.random.PRNGKey(0)

        # Setup JAX mesh (required for Qwen2Attention JAX init due to partitioning)
        try:
            if jax.local_device_count() > 1:
                # Create a 2D mesh if multiple devices, to support "model" axis sharding
                # JAX modules use "model" for partitioning specs.
                device_mesh_shape = (1, jax.local_device_count())
                axis_names = ("data", "model")
            else:  # Single device
                device_mesh_shape = (1, )
                axis_names = (
                    "data",
                )  # "model" sharding will be a no-op or map to this single device

            device_mesh = mesh_utils.create_device_mesh(device_mesh_shape)
            self.mesh = Mesh(devices=device_mesh, axis_names=axis_names)
        except RuntimeError:
            # Fallback for CPU-only environments or if jax.devices() is problematic
            # This ensures tests can run in minimal environments.
            device_mesh = mesh_utils.create_device_mesh((1, ))
            self.mesh = Mesh(devices=device_mesh, axis_names=("data", ))

        # Common test parameters
        self.batch_size = 2
        self.seq_len = 10
        self.block_size = 16  # For vLLM CacheConfig, not directly used in these specific tests but good for consistency

    def test_qwen2_mlp_parity(self):
        # Initialize JAX model
        key_params, _ = jax.random.split(self.rng_key)

        jax_mlp_module = JaxQwen2MLP(config=self.hf_config,
                                     dtype=self.dtype_jax,
                                     rng=nnx.Rngs(params=key_params))
        jax_mlp_params, jax_mlp_static = nnx.split(jax_mlp_module, nnx.Param)

        # Initialize PyTorch model
        torch_mlp = VllmQwen2MLP(
            hidden_size=self.hf_config.hidden_size,
            intermediate_size=self.hf_config.intermediate_size,
            hidden_act=self.hf_config.hidden_act).to(self.dtype_torch).eval()

        torch_gate_up_weight = torch.randn_like(
            torch_mlp.gate_up_proj.weight.data)
        torch_mlp.gate_up_proj.weight.data = torch_gate_up_weight
        torch_down_weight = torch.randn_like(torch_mlp.down_proj.weight.data)
        torch_mlp.down_proj.weight.data = torch_down_weight

        torch_gate_w = torch_gate_up_weight[:self.hf_config.
                                            intermediate_size, :]
        jax_mlp_params.gate_proj.kernel = torch_to_jax(torch_gate_w.T)
        torch_up_w = torch_gate_up_weight[self.hf_config.intermediate_size:, :]
        jax_mlp_params.up_proj.kernel = torch_to_jax(torch_up_w.T)
        jax_mlp_params.down_proj.kernel = torch_to_jax(torch_down_weight.T)

        jax_mlp_model_runnable = nnx.merge(jax_mlp_static, jax_mlp_params)

        input_np = np.random.rand(self.batch_size, self.seq_len,
                                  self.hf_config.hidden_size).astype(
                                      np.float32)
        jax_input = jnp.asarray(input_np)
        torch_input = torch.from_numpy(input_np).to(self.dtype_torch)

        jax_output = jax_mlp_model_runnable(jax_input)
        torch_output = torch_mlp(torch_input)

        np.testing.assert_allclose(np.asarray(jax_output),
                                   torch_output.cpu().detach().numpy(),
                                   rtol=1e-5,
                                   atol=1e-5,
                                   err_msg="Qwen2MLP outputs do not match.")
        print("Qwen2MLP parity test passed!")

    def test_qwen2_attention_qkv_projection_and_rope_parity(self):
        hidden_size = self.hf_config.hidden_size
        num_heads = self.hf_config.num_attention_heads
        num_kv_heads = self.hf_config.num_key_value_heads
        head_dim = hidden_size // num_heads
        rope_theta = self.hf_config.rope_theta
        max_position_embeddings = self.hf_config.max_position_embeddings

        key_params, _ = jax.random.split(self.rng_key)
        # JAX Qwen2Attention expects rng: nnx.Rngs and mesh
        jax_attn_module = JaxQwen2Attention(config=self.hf_config,
                                            dtype=self.dtype_jax,
                                            rng=nnx.Rngs(params=key_params),
                                            mesh=self.mesh)
        jax_attn_params, jax_attn_static = nnx.split(jax_attn_module,
                                                     nnx.Param)

        # Initialize PyTorch model
        # VllmQwen2Attention requires CacheConfig
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=0.9,  # Dummy value
            swap_space=0,  # Dummy value
            cache_dtype="auto")
        torch_attn = VllmQwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_theta=rope_theta,
            max_position=max_position_embeddings,
            cache_config=cache_config,
            quant_config=None,  # No quantization for this test
            rope_scaling=getattr(self.hf_config, "rope_scaling",
                                 None)).to(self.dtype_torch).eval()

        # --- Create and assign identical QKV weights and biases ---
        # vLLM qkv_proj.weight: ((N+2K)H, D_model), qkv_proj.bias: ((N+2K)H)
        torch_qkv_weight = torch.randn_like(torch_attn.qkv_proj.weight.data)
        torch_qkv_bias = torch.randn_like(torch_attn.qkv_proj.bias.data)
        torch_attn.qkv_proj.weight.data = torch_qkv_weight
        torch_attn.qkv_proj.bias.data = torch_qkv_bias

        # Split vLLM weights/biases for JAX
        q_size_torch = num_heads * head_dim
        k_size_torch = num_kv_heads * head_dim
        # v_size_torch is same as k_size_torch for Qwen2

        torch_Wq = torch_qkv_weight[0:q_size_torch, :]
        torch_Wk = torch_qkv_weight[q_size_torch:q_size_torch +
                                    k_size_torch, :]
        torch_Wv = torch_qkv_weight[q_size_torch + k_size_torch:, :]

        torch_bq = torch_qkv_bias[0:q_size_torch]
        torch_bk = torch_qkv_bias[q_size_torch:q_size_torch + k_size_torch]
        torch_bv = torch_qkv_bias[q_size_torch + k_size_torch:]

        # Assign to JAX model (kernels are (N_or_K, D_model, H_dim), biases are (N_or_K, H_dim))
        jax_attn_params.q_proj.kernel = torch_to_jax(
            torch_Wq.reshape(num_heads, head_dim,
                             hidden_size).permute(0, 2, 1))
        jax_attn_params.q_proj.bias = torch_to_jax(
            torch_bq.reshape(num_heads, head_dim))

        jax_attn_params.k_proj.kernel = torch_to_jax(
            torch_Wk.reshape(num_kv_heads, head_dim,
                             hidden_size).permute(0, 2, 1))
        jax_attn_params.k_proj.bias = torch_to_jax(
            torch_bk.reshape(num_kv_heads, head_dim))

        jax_attn_params.v_proj.kernel = torch_to_jax(
            torch_Wv.reshape(num_kv_heads, head_dim,
                             hidden_size).permute(0, 2, 1))
        jax_attn_params.v_proj.bias = torch_to_jax(
            torch_bv.reshape(num_kv_heads, head_dim))

        jax_attn_model_runnable = nnx.merge(jax_attn_static, jax_attn_params)

        # --- Create random inputs ---
        total_tokens = self.batch_size * self.seq_len
        # JAX input: (Batch, SeqLen, HiddenSize)
        input_np_jax = np.random.rand(self.batch_size, self.seq_len,
                                      hidden_size).astype(np.float32)
        jax_x = jnp.array(input_np_jax)
        # PyTorch input: (TotalTokens, HiddenSize)
        torch_hidden_states = torch.from_numpy(
            input_np_jax.reshape(total_tokens,
                                 hidden_size)).to(self.dtype_torch)

        # Positions for RoPE
        # JAX apply_rope expects (Batch, SeqLen)
        jax_input_positions_reshaped = jnp.tile(
            jnp.arange(self.seq_len, dtype=jnp.int32), (self.batch_size, 1))
        # PyTorch rotary_emb expects (TotalTokens)
        torch_positions = torch.from_numpy(
            np.asarray(jax_input_positions_reshaped.flatten())).long()

        # --- JAX forward pass (QKV projection + RoPE) ---
        # q_raw: (B, N, T, H), k_raw: (B, K, T, H), v_raw: (B, K, T, H)
        q_jax_raw = jax_attn_model_runnable.q_proj(jax_x)
        k_jax_raw = jax_attn_model_runnable.k_proj(jax_x)
        v_jax_raw = jax_attn_model_runnable.v_proj(jax_x)

        q_jax_rope = apply_rope(q_jax_raw, jax_input_positions_reshaped,
                                head_dim, rope_theta,
                                getattr(self.hf_config, "rope_scaling", None))
        k_jax_rope = apply_rope(k_jax_raw, jax_input_positions_reshaped,
                                head_dim, rope_theta,
                                getattr(self.hf_config, "rope_scaling", None))

        # --- PyTorch forward pass (QKV projection + RoPE) ---
        torch_qkv_out, _ = torch_attn.qkv_proj(
            torch_hidden_states)  # Output shape: (TotalTokens, (N+2K)H)
        torch_q_raw, torch_k_raw, torch_v_raw = torch_qkv_out.split(
            [q_size_torch, k_size_torch, k_size_torch
             ],  # K and V have same size (num_kv_heads * head_dim)
            dim=-1)

        # Reshape for RoPE: (TotalTokens, NumHeads_or_KVHeads, HeadDim)
        torch_q_raw_reshaped = torch_q_raw.view(total_tokens, num_heads,
                                                head_dim)
        torch_k_raw_reshaped = torch_k_raw.view(total_tokens, num_kv_heads,
                                                head_dim)
        torch_v_raw_reshaped = torch_v_raw.view(
            total_tokens, num_kv_heads, head_dim)  # For direct comparison

        q_torch_rope, k_torch_rope = torch_attn.rotary_emb(
            torch_positions, torch_q_raw_reshaped, torch_k_raw_reshaped)

        # --- Compare outputs ---
        # Reshape JAX outputs for comparison: (TotalTokens, NumHeads_or_KVHeads, HeadDim)
        # JAX q_rope is (B, N, T, H) -> (B, T, N, H) -> (B*T, N, H)
        q_jax_rope_cmp = q_jax_rope.transpose(0, 2, 1, 3).reshape(
            total_tokens, num_heads, head_dim)
        k_jax_rope_cmp = k_jax_rope.transpose(0, 2, 1, 3).reshape(
            total_tokens, num_kv_heads, head_dim)
        v_jax_raw_cmp = v_jax_raw.transpose(0, 2, 1,
                                            3).reshape(total_tokens,
                                                       num_kv_heads, head_dim)

        np.testing.assert_allclose(
            np.asarray(q_jax_rope_cmp),
            torch_to_jax(q_torch_rope),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Q (post-RoPE) outputs do not match.")
        np.testing.assert_allclose(
            np.asarray(k_jax_rope_cmp),
            torch_to_jax(k_torch_rope),
            rtol=1e-5,
            atol=1e-5,
            err_msg="K (post-RoPE) outputs do not match.")
        np.testing.assert_allclose(np.asarray(v_jax_raw_cmp),
                                   torch_to_jax(torch_v_raw_reshaped),
                                   rtol=1e-5,
                                   atol=1e-5,
                                   err_msg="V (raw) outputs do not match.")
        print("Qwen2Attention QKV projection and RoPE parity test passed!")

    def test_qwen2_attention_o_projection_parity(self):
        hidden_size = self.hf_config.hidden_size
        num_heads = self.hf_config.num_attention_heads
        head_dim = hidden_size // num_heads

        key_params, _ = jax.random.split(self.rng_key)
        jax_attn_module = JaxQwen2Attention(config=self.hf_config,
                                            dtype=self.dtype_jax,
                                            rng=nnx.Rngs(params=key_params),
                                            mesh=self.mesh)
        # Only need o_proj params, but split will give all params and static parts
        jax_attn_params, jax_attn_static = nnx.split(jax_attn_module,
                                                     nnx.Param)

        cache_config = CacheConfig(block_size=self.block_size,
                                   gpu_memory_utilization=0.9,
                                   swap_space=0,
                                   cache_dtype="auto")
        torch_attn = VllmQwen2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=self.hf_config.num_key_value_heads,
            max_position=self.hf_config.max_position_embeddings,
            rope_theta=self.hf_config.rope_theta,
            cache_config=cache_config).to(self.dtype_torch).eval()

        # --- Create and assign identical O-projection weights ---
        # vLLM o_proj.weight: (D_model, N*H)
        torch_o_weight = torch.randn_like(torch_attn.o_proj.weight.data)
        torch_attn.o_proj.weight.data = torch_o_weight

        # JAX o_proj.kernel: (N, H, D_model)
        jax_attn_params.o_proj.kernel = torch_to_jax(
            torch_o_weight.T.reshape(num_heads, head_dim, hidden_size))
        # JAX Qwen2Attention o_proj does not have bias by default in qwen_nnx.py

        jax_attn_model_runnable = nnx.merge(jax_attn_static, jax_attn_params)

        # --- Create random input for o_proj ---
        # This simulates the output of the attention mechanism (scaled dot-product attention * V)
        # JAX o_proj expects (B, N, T, H)
        total_tokens = self.batch_size * self.seq_len
        attn_output_np_bnth = np.random.rand(self.batch_size, num_heads,
                                             self.seq_len,
                                             head_dim).astype(np.float32)

        jax_input_to_o_proj = jnp.array(attn_output_np_bnth)
        # PyTorch o_proj expects (TotalTokens, N*H)
        # Transpose (B,N,T,H) to (B,T,N,H) then reshape
        torch_input_to_o_proj = torch.from_numpy(
            attn_output_np_bnth.transpose(0, 2, 1, 3).reshape(
                total_tokens, num_heads * head_dim)).to(self.dtype_torch)

        # --- JAX forward pass (O-projection only) ---
        # Output: (B, T, D_model)
        jax_output = jax_attn_model_runnable.o_proj(jax_input_to_o_proj)

        # --- PyTorch forward pass (O-projection only) ---
        # Output: (TotalTokens, D_model)
        torch_output_raw, _ = torch_attn.o_proj(torch_input_to_o_proj)
        # Reshape for comparison: (B, T, D_model)
        torch_output_cmp = torch_output_raw.reshape(self.batch_size,
                                                    self.seq_len, hidden_size)

        np.testing.assert_allclose(
            np.asarray(jax_output),
            torch_to_jax(torch_output_cmp),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Qwen2Attention o_projection outputs do not match.")
        print("Qwen2Attention o_projection parity test passed!")


if __name__ == '__main__':
    unittest.main()
