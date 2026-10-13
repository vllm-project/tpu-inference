# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness and sanity tests for Mamba Mixer 2 custom operator on TPU."""

from unittest.mock import MagicMock
import jax
import jax.numpy as jnp
import numpy as np
import torch
import unittest
from jax.sharding import Mesh
import torchax

from tpu_inference.layers.vllm.custom_ops.mamba_mixer2 import mamba_mixer2_core_tpu
from tpu_inference.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
from vllm.forward_context import ForwardContext, override_forward_context


class MambaMixer2OpTest(unittest.TestCase):

  def test_mamba_mixer2_core_tpu(self):
    # 1. Initialize local TPU mesh (2x1x1 layout to test DP sharding)
    dp_size = 2
    attn_dp_size = 1
    tp_size = 1

    devices = jax.local_devices()
    self.assertGreaterEqual(len(devices), 2, "Test requires at least 2 TPU devices")
    
    mesh = Mesh(
        np.array(devices[0:2]).reshape(dp_size, attn_dp_size, tp_size),
        ("data", "attn_dp", "model"),
    )

    # 2. Dimensions matching test setup
    num_tokens = 128
    num_seqs = 4
    num_heads = 16
    head_dim = 64
    n_groups = 4
    ssm_state_size = 16
    conv_kernel_size = 4

    intermediate_size = num_heads * head_dim  # 1024
    groups_ssm_state_size_tped = n_groups * ssm_state_size  # 64
    tped_intermediate_size = intermediate_size // tp_size  # 1024
    tped_conv_size = (
        intermediate_size + 2 * groups_ssm_state_size_tped
    ) // tp_size  # 1152
    tped_dt_size = num_heads // tp_size  # 16

    total_proj_size = (
        tped_intermediate_size + tped_conv_size + tped_dt_size
    )  # 2192

    # 3. Create mock layer module
    layer_module = MagicMock()
    layer_module.num_heads = num_heads
    layer_module.head_dim = head_dim
    layer_module.n_groups = n_groups
    layer_module.ssm_state_size = ssm_state_size
    layer_module.conv_kernel_size = conv_kernel_size
    layer_module.tp_size = tp_size

    layer_module.intermediate_size = intermediate_size
    layer_module.conv_dim = tped_conv_size * tp_size
    layer_module.tped_dt_size = tped_dt_size

    # Helper to convert torch tensors to torchax compatible tensors
    from torchax.ops.mappings import t2j
    from torchax.interop import torch_view

    def to_torchax(t: torch.Tensor) -> torch.Tensor:
      return torch_view(t2j(t, use_dlpack=False))

    # Initialize mock weights (PyTorch Tensors)
    torch.manual_seed(0)
    layer_module.conv1d = MagicMock()

    layer_module.conv1d.weight = to_torchax(
        torch.randn(
            tped_conv_size,
            1,
            conv_kernel_size,
            dtype=torch.bfloat16,
            device="cpu",
        )
    )
    layer_module.conv1d.bias = to_torchax(
        torch.randn(tped_conv_size, dtype=torch.bfloat16, device="cpu")
    )

    layer_module.A = to_torchax(
        -torch.rand(num_heads // tp_size, dtype=torch.float32, device="cpu")
        * 5.0
    )
    layer_module.D = to_torchax(
        torch.rand(num_heads // tp_size, dtype=torch.float32, device="cpu")
    )
    layer_module.dt_bias = to_torchax(
        torch.rand(num_heads // tp_size, dtype=torch.float32, device="cpu")
    )

    # Mock norm layer
    layer_module.norm = MagicMock()
    layer_module.norm.weight = to_torchax(
        torch.randn(tped_intermediate_size, dtype=torch.bfloat16, device="cpu")
    )
    layer_module.norm.variance_epsilon = 1e-5

    # 4. Create mock inputs
    projected_states = to_torchax(
        torch.randn(
            num_tokens, total_proj_size, dtype=torch.bfloat16, device="cpu"
        )
    )
    ssm_output = torch.zeros(
        num_tokens, tped_intermediate_size, dtype=torch.bfloat16, device="cpu"
    )
    layer_name = "decoder.layers.0.mixer"

    # 5. Create mock attention metadata
    attn_metadata = MagicMock()
    # Simple chunk setup: 4 sequences, each length 32, padded to 6 (num_seqs + dp_size)
    seq_lens = np.array([32, 32, 32, 32, 0, 0], dtype=np.int32)
    query_start_loc = np.cumsum(np.concatenate([[0], seq_lens])).astype(
        np.int32
    )

    attn_metadata.mamba_state_indices = np.array(
        [0, 1, 2, 3, 0, 0], dtype=np.int32
    )
    attn_metadata.padded_num_reqs = num_seqs
    attn_metadata.query_start_loc = query_start_loc
    attn_metadata.seq_lens = seq_lens
    attn_metadata.request_distribution = np.array(
        [32, 32, num_seqs], dtype=np.int32
    )

    # 6. Initialize mock KV cache in JAX
    max_seqs = 8

    # Initialize cache arrays using JAX
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    conv_cache_shape = (max_seqs, conv_kernel_size - 1, tped_conv_size)
    recurrent_cache_shape = (
        max_seqs,
        num_heads // tp_size,
        head_dim,
        ssm_state_size,
    )

    j_conv_cache = jax.random.normal(k1, conv_cache_shape, dtype=jnp.bfloat16)
    j_recurrent_cache = jax.random.normal(
        k2, recurrent_cache_shape, dtype=jnp.bfloat16
    )

    kv_caches = [(j_conv_cache, j_recurrent_cache)]
    layer_name_to_kvcache_index = {layer_name: 0}

    # 7. Set up contexts and execute custom op
    fc_no_compile_layers = {layer_name: layer_module}
    fc_attn_metadata = {layer_name: attn_metadata}

    fc = ForwardContext(
        no_compile_layers=fc_no_compile_layers,
        attn_metadata=fc_attn_metadata,
        slot_mapping={},
    )

    with torchax.default_env():
      with set_vllm_model_wrapper_context(
          kv_caches=kv_caches,
          mesh=mesh,
          layer_name_to_kvcache_index=layer_name_to_kvcache_index,
      ):
        with override_forward_context(fc):
          mamba_mixer2_core_tpu(
              projected_states,
              ssm_output,
              layer_name,
          )

    # Basic output checks
    self.assertEqual(ssm_output.shape, (num_tokens, tped_intermediate_size))
    self.assertFalse(torch.isnan(ssm_output).any(), "Output contains NaNs!")

    # Check that cache states were updated
    updated_conv_cache, updated_recurrent_cache = kv_caches[0]

    conv_diff = jnp.abs(updated_conv_cache - j_conv_cache).mean()
    recurrent_diff = jnp.abs(updated_recurrent_cache - j_recurrent_cache).mean()

    self.assertGreater(conv_diff, 0, "Conv Cache was not updated!")
    self.assertGreater(recurrent_diff, 0, "Recurrent Cache was not updated!")


if __name__ == "__main__":
  unittest.main()
