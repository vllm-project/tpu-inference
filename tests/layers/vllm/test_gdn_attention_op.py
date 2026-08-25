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

from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


class TestVllmGatedDeltaNetAttention:

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.gdn_attention_core_tpu"
    )
    def test_forward_cuda_lora(self, mock_gdn_attention_core_tpu, mesh):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"

        # Mocks for LoRA path (uses in_proj_qkv and in_proj_z)
        attn.in_proj_qkv = MagicMock()
        attn.in_proj_z = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        attn.in_proj_qkv.return_value = (torch.randn(num_tokens, 96), None)
        attn.in_proj_z.return_value = (torch.randn(num_tokens, 64), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        with set_vllm_model_wrapper_context(kv_caches=[],
                                            mesh=mesh,
                                            layer_name_to_kvcache_index={}):
            attn.forward(hidden_states, output)

        attn.in_proj_qkv.assert_called_once_with(hidden_states)
        attn.in_proj_z.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert mock_gdn_attention_core_tpu.call_count == 1
        core_args = mock_gdn_attention_core_tpu.call_args[0]
        core_kwargs = mock_gdn_attention_core_tpu.call_args[1]

        assert core_args[0].shape == (num_tokens, 96)  # mixed_qkv
        assert core_args[1].shape == (num_tokens, 16)  # b
        assert core_args[2].shape == (num_tokens, 16)  # a
        assert core_args[3].shape == (num_tokens, 4, 16)  # core_attn_out
        assert core_args[3].dtype == hidden_states.dtype
        assert core_args[4] == "test_layer"
        assert core_kwargs["mesh"] == mesh

        attn.norm.assert_called_once()
        # Verify z was correctly reshaped: [num_tokens, -1, head_v_dim]
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        # Verify reshaped output from norm went to out_proj
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        # Check that output buffer was updated only up to num_tokens
        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.gdn_attention_core_tpu"
    )
    def test_forward_cuda_non_lora_no_gqa(self, mock_gdn_attention_core_tpu,
                                          mesh):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = False
        attn.key_dim = 32
        attn.value_dim = 64

        # Mocks for non-LoRA no GQA path
        attn.in_proj_qkvz = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size  # 128
        z_size = attn.value_dim // attn.tp_size  # 64
        mixed_qkvz = torch.randn(num_tokens, qkv_size + z_size)

        attn.in_proj_qkvz.return_value = (mixed_qkvz, None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        with set_vllm_model_wrapper_context(kv_caches=[],
                                            mesh=mesh,
                                            layer_name_to_kvcache_index={}):
            attn.forward(hidden_states, output)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)

        assert mock_gdn_attention_core_tpu.call_count == 1
        core_args = mock_gdn_attention_core_tpu.call_args[0]
        core_kwargs = mock_gdn_attention_core_tpu.call_args[1]

        # mixed_qkv should be separated accurately
        assert core_args[0].shape == (num_tokens, 128)
        assert core_args[1].shape == (num_tokens, 16)
        assert core_args[2].shape == (num_tokens, 16)
        assert core_args[3].shape == (num_tokens, 4, 16)
        assert core_args[4] == "test_layer"
        assert core_kwargs["mesh"] == mesh

        attn.norm.assert_called_once()
        # Verify z was split and reshaped correctly
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.gdn_attention_core_tpu"
    )
    def test_forward_cuda_non_lora_gqa(self, mock_gdn_attention_core_tpu,
                                       mesh):
        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.head_v_dim = 16
        attn.num_v_heads = 4
        attn.tp_size = 1
        attn.prefix = "test_layer"
        attn.gqa_interleaved_layout = True

        # Mocks for non-LoRA GQA path
        attn.in_proj_qkvz = MagicMock()
        attn.in_proj_ba = MagicMock()
        attn.fix_query_key_value_ordering = MagicMock()
        attn.norm = MagicMock()
        attn.out_proj = MagicMock()

        num_tokens = 2
        hidden_states = torch.randn(num_tokens, 64)
        output = torch.zeros(5, 64)

        attn.in_proj_qkvz.return_value = (torch.randn(num_tokens, 192), None)
        attn.in_proj_ba.return_value = (torch.randn(num_tokens, 32), None)

        query = torch.randn(num_tokens, 4, 8)
        key = torch.randn(num_tokens, 4, 8)
        value = torch.randn(num_tokens, 4, 8)
        z = torch.randn(num_tokens, 4, 16)
        b = torch.randn(num_tokens, 16)
        a = torch.randn(num_tokens, 16)

        attn.fix_query_key_value_ordering.return_value = (query, key, value, z,
                                                          b, a)

        norm_out = torch.randn(num_tokens, 4, 16)
        attn.norm.return_value = norm_out
        attn.out_proj.return_value = (torch.ones(num_tokens, 64) * 5, None)

        with set_vllm_model_wrapper_context(kv_caches=[],
                                            mesh=mesh,
                                            layer_name_to_kvcache_index={}):
            attn.forward(hidden_states, output)

        attn.in_proj_qkvz.assert_called_once_with(hidden_states)
        attn.in_proj_ba.assert_called_once_with(hidden_states)
        attn.fix_query_key_value_ordering.assert_called_once()

        assert mock_gdn_attention_core_tpu.call_count == 1
        core_args = mock_gdn_attention_core_tpu.call_args[0]
        core_kwargs = mock_gdn_attention_core_tpu.call_args[1]

        # mixed_qkv should be cat of rearranged query, key, value
        # rearranged from "l p d -> l (p d)", e.g. 2x(4*8) = 2x32 -> cat into 2x96
        assert core_args[0].shape == (num_tokens, 96)
        assert core_args[1].shape == (num_tokens, 16)
        assert core_args[2].shape == (num_tokens, 16)
        assert core_args[3].shape == (num_tokens, 4, 16)
        assert core_args[4] == "test_layer"
        assert core_kwargs["mesh"] == mesh

        attn.norm.assert_called_once()
        # Verify unpacked z is natively used
        assert attn.norm.call_args[0][1].shape == (num_tokens, 4, 16)

        attn.out_proj.assert_called_once()
        assert attn.out_proj.call_args[0][0].shape == (num_tokens, 64)

        assert torch.all(output[:num_tokens] == 5)
        assert torch.all(output[num_tokens:] == 0)

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.torch_view",
        side_effect=lambda t: torch.as_tensor(np.array(t)),
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.jax_view",
        side_effect=lambda t: jax.numpy.array(t.detach().cpu().numpy())
        if isinstance(t, torch.Tensor) else t,
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.run_jax_gdn_attention"
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_gdn_attention_core_tpu_align_mode(self, mock_get_fc, mock_run_jax,
                                               mock_jax_v, mock_torch_v, mesh):
        from tpu_inference.layers.common.attention_metadata import \
            AttentionMetadata
        from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
            gdn_attention_core_tpu

        layer_name = "test_layer"
        num_tokens = 4
        n_kq = 2
        n_v = 4
        d_k = 16
        d_v = 16
        kernel_size = 4
        mamba_block_size = 16

        # Layer mock
        layer_module = MagicMock()
        layer_module.num_k_heads = n_kq
        layer_module.num_v_heads = n_v
        layer_module.head_k_dim = d_k
        layer_module.head_v_dim = d_v
        layer_module.conv_kernel_size = kernel_size
        layer_module.conv1d.weight = torch.randn((n_kq * d_k) * 2 + n_v * d_v,
                                                 1, kernel_size)
        layer_module.conv1d.bias = None
        layer_module.A_log = torch.randn(n_v)
        layer_module.dt_bias = torch.randn(n_v)
        layer_module.cache_config.mamba_cache_mode = "align"
        layer_module.cache_config.mamba_block_size = mamba_block_size

        # AttentionMetadata: 2 requests
        # Request 0: computed 16 tokens, scheduled 16 tokens -> crosses from block 0 to block 1
        # Request 1: computed 0 tokens, scheduled 16 tokens -> first block 0
        block_tables = jax.numpy.array([[10, 11], [20, 21]],
                                       dtype=jax.numpy.int32)
        seq_lens = jax.numpy.array([32, 16], dtype=jax.numpy.int32)
        query_start_loc = jax.numpy.array([0, 16, 32], dtype=jax.numpy.int32)
        request_distribution = jax.numpy.array([0, 2, 2],
                                               dtype=jax.numpy.int32)

        attn_metadata = AttentionMetadata(
            input_positions=jax.numpy.zeros(32, dtype=jax.numpy.int32),
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            padded_num_reqs=2,
        )

        fc = MagicMock()
        fc.attn_metadata = {layer_name: attn_metadata}
        fc.no_compile_layers = {layer_name: layer_module}
        mock_get_fc.return_value = fc

        # KV cache tensors
        conv_dim = (n_kq * d_k) * 2 + n_v * d_v
        conv_state = jax.numpy.zeros((30, kernel_size - 1, conv_dim))
        recurrent_state = jax.numpy.zeros((30, n_v, d_k, d_v))

        mock_run_jax.return_value = (
            (conv_state[:2], recurrent_state[:2]),
            jax.numpy.zeros((num_tokens, n_v * d_v)),
        )

        mixed_qkv = torch.randn(num_tokens, conv_dim)
        b = torch.randn(num_tokens, n_v)
        a = torch.randn(num_tokens, n_v)
        core_attn_out = torch.zeros(num_tokens, n_v, d_v)

        vllm_config = MagicMock()
        vllm_config.cache_config.mamba_cache_mode = "align"
        vllm_config.cache_config.block_size = mamba_block_size

        with set_vllm_model_wrapper_context(
                kv_caches=[(conv_state, recurrent_state)],
                mesh=mesh,
                layer_name_to_kvcache_index={layer_name: 0},
                vllm_config=vllm_config,
        ):
            gdn_attention_core_tpu(mixed_qkv, b, a, core_attn_out, layer_name,
                                   mesh)

        assert mock_run_jax.call_count == 1
        call_kwargs = mock_run_jax.call_args[1]
        call_args = mock_run_jax.call_args[0]

        # state_indices (write slots) should be [11, 20]
        # read_state_indices (read slots) should be [10, 20]
        actual_state_indices = call_args[9]  # 10th positional arg
        actual_read_state_indices = call_kwargs["read_state_indices"]

        np.testing.assert_array_equal(np.array(actual_state_indices), [11, 20])
        np.testing.assert_array_equal(np.array(actual_read_state_indices),
                                      [10, 20])

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.torch_view",
        side_effect=lambda t: torch.as_tensor(np.array(t)),
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.jax_view",
        side_effect=lambda t: jax.numpy.array(t.detach().cpu().numpy())
        if isinstance(t, torch.Tensor) else t,
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.run_jax_gdn_attention"
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_gdn_attention_core_tpu_non_align_mode(self, mock_get_fc,
                                                   mock_run_jax, mock_jax_v,
                                                   mock_torch_v, mesh):
        from tpu_inference.layers.common.attention_metadata import \
            AttentionMetadata
        from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
            gdn_attention_core_tpu

        layer_name = "test_layer"
        num_tokens = 4
        n_kq = 2
        n_v = 4
        d_k = 16
        d_v = 16
        kernel_size = 4

        # Layer mock (mamba_cache_mode is none)
        layer_module = MagicMock()
        layer_module.num_k_heads = n_kq
        layer_module.num_v_heads = n_v
        layer_module.head_k_dim = d_k
        layer_module.head_v_dim = d_v
        layer_module.conv_kernel_size = kernel_size
        layer_module.conv1d.weight = torch.randn((n_kq * d_k) * 2 + n_v * d_v,
                                                 1, kernel_size)
        layer_module.conv1d.bias = None
        layer_module.A_log = torch.randn(n_v)
        layer_module.dt_bias = torch.randn(n_v)
        layer_module.cache_config.mamba_cache_mode = "none"

        # AttentionMetadata: compact slot pool indices
        mamba_state_indices = jax.numpy.array([3, 7], dtype=jax.numpy.int32)
        seq_lens = jax.numpy.array([32, 16], dtype=jax.numpy.int32)
        query_start_loc = jax.numpy.array([0, 16, 32], dtype=jax.numpy.int32)
        request_distribution = jax.numpy.array([0, 2, 2],
                                               dtype=jax.numpy.int32)

        attn_metadata = AttentionMetadata(
            input_positions=jax.numpy.zeros(32, dtype=jax.numpy.int32),
            block_tables=None,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            padded_num_reqs=2,
        )

        fc = MagicMock()
        fc.attn_metadata = {layer_name: attn_metadata}
        fc.no_compile_layers = {layer_name: layer_module}
        mock_get_fc.return_value = fc

        # KV cache tensors
        conv_dim = (n_kq * d_k) * 2 + n_v * d_v
        conv_state = jax.numpy.zeros((30, kernel_size - 1, conv_dim))
        recurrent_state = jax.numpy.zeros((30, n_v, d_k, d_v))

        mock_run_jax.return_value = (
            (conv_state[:2], recurrent_state[:2]),
            jax.numpy.zeros((num_tokens, n_v * d_v)),
        )

        mixed_qkv = torch.randn(num_tokens, conv_dim)
        b = torch.randn(num_tokens, n_v)
        a = torch.randn(num_tokens, n_v)
        core_attn_out = torch.zeros(num_tokens, n_v, d_v)

        vllm_config = MagicMock()
        vllm_config.cache_config.mamba_cache_mode = "none"

        with set_vllm_model_wrapper_context(
                kv_caches=[(conv_state, recurrent_state)],
                mesh=mesh,
                layer_name_to_kvcache_index={layer_name: 0},
                vllm_config=vllm_config,
        ):
            gdn_attention_core_tpu(mixed_qkv, b, a, core_attn_out, layer_name,
                                   mesh)

        assert mock_run_jax.call_count == 1
        call_kwargs = mock_run_jax.call_args[1]
        call_args = mock_run_jax.call_args[0]

        # state_indices and read_state_indices should both be [3, 7]
        actual_state_indices = call_args[9]  # 10th positional arg
        actual_read_state_indices = call_kwargs["read_state_indices"]

        np.testing.assert_array_equal(np.array(actual_state_indices), [3, 7])
        np.testing.assert_array_equal(np.array(actual_read_state_indices),
                                      [3, 7])
