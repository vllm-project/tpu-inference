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
        vllm_config.cache_config.mamba_block_size = mamba_block_size

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
        from tpu_inference.layers.common.attention_metadata import (
            AttentionMetadata, MambaMetadata)
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
            mamba=MambaMetadata(state_indices=mamba_state_indices),
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
        assert "read_offsets" not in call_kwargs
        assert "num_spec_tokens" not in call_kwargs

    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.torch_view",
        side_effect=lambda x: torch.from_numpy(np.array(x)),
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.jax_view",
        side_effect=lambda x: jax.numpy.array(x.detach().numpy())
        if isinstance(x, torch.Tensor) else x,
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.run_jax_gdn_attention"
    )
    @patch(
        "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.get_forward_context"
    )
    def test_gdn_attention_core_tpu_spec_decode_forwards_read_offsets(
            self, mock_get_fc, mock_run_jax, mock_jax_v, mock_torch_v, mesh):
        from tpu_inference.layers.common.attention_metadata import (
            AttentionMetadata, MambaMetadata)
        from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
            gdn_attention_core_tpu

        layer_name = "test_layer"
        num_tokens = 8
        n_kq = 2
        n_v = 4
        d_k = 16
        d_v = 16
        kernel_size = 4

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

        mamba_state_indices = jax.numpy.array([4, 8], dtype=jax.numpy.int32)
        read_offsets = jax.numpy.array([2, 1], dtype=jax.numpy.int32)
        seq_lens = jax.numpy.array([12, 10], dtype=jax.numpy.int32)
        query_start_loc = jax.numpy.array([0, 4, 8], dtype=jax.numpy.int32)
        request_distribution = jax.numpy.array([2, 2, 2],
                                               dtype=jax.numpy.int32)

        attn_metadata = AttentionMetadata(
            input_positions=jax.numpy.zeros(num_tokens, dtype=jax.numpy.int32),
            block_tables=None,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba=MambaMetadata(state_indices=mamba_state_indices,
                                read_offsets=read_offsets),
            padded_num_reqs=2,
        )

        fc = MagicMock()
        fc.attn_metadata = {layer_name: attn_metadata}
        fc.no_compile_layers = {layer_name: layer_module}
        mock_get_fc.return_value = fc

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
        vllm_config.speculative_config.num_speculative_tokens = 3

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
        np.testing.assert_array_equal(np.array(call_kwargs["read_offsets"]),
                                      [2, 1])
        assert call_kwargs["num_spec_tokens"] == 3

    @pytest.mark.parametrize("enabled", [True, False])
    def test_fp32_4d_conv_state_shape_and_dtype(self, enabled):
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
            QwenGatedDeltaNetAttention

        import tpu_inference.envs as envs
        from tpu_inference.layers.common.attention_backend import \
            GDNLinearAttentionBackend

        attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
        attn.linear_attention_backend = GDNLinearAttentionBackend()

        with (
                patch.object(QwenGatedDeltaNetAttention,
                             "get_state_shape",
                             return_value=((3, 64), (8, 16, 16))),
                patch.object(QwenGatedDeltaNetAttention,
                             "get_state_dtype",
                             return_value=(torch.bfloat16, torch.float32)),
                patch.object(envs, "GDN_FP32_4D_CONV_STATE", enabled),
                patch(
                    "tpu_inference.layers.vllm.custom_ops.gdn_attention_op.is_conv_state_dim_first",
                    return_value=False),
        ):
            conv_shape, _ = attn.get_state_shape()
            conv_dtype, _ = attn.get_state_dtype()

        assert conv_shape == ((3, 1, 64) if enabled else (3, 64))
        assert conv_dtype == (torch.float32 if enabled else torch.bfloat16)

    def test_kv_cache_manager_conv_state_sharding(self, mesh):
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec
        from vllm.config import set_current_vllm_config
        from vllm.v1.kv_cache_interface import (KVCacheConfig,
                                                KVCacheGroupSpec,
                                                KVCacheTensor, MambaSpec)

        from tpu_inference.layers.common.sharding import ShardingAxisName
        from tpu_inference.runner.kv_cache_manager import KVCacheManager

        for conv_shape, conv_dtype, expected_spec in (
            (
                (3, 1, 64),
                torch.float32,
                PartitionSpec(
                    ShardingAxisName.ATTN_DATA,
                    None,
                    None,
                    ShardingAxisName.ATTN_HEAD,
                ),
            ),
            (
                (3, 64),
                torch.bfloat16,
                PartitionSpec(
                    ShardingAxisName.ATTN_DATA,
                    None,
                    ShardingAxisName.ATTN_HEAD,
                ),
            ),
        ):
            runner = MagicMock()
            runner.mesh = mesh
            runner.model_config.use_mla = False
            runner.model_config.architectures = ["Qwen3_5MoeForCausalLM"]
            runner.vllm_config.model_config = runner.model_config
            runner.vllm_config.kv_transfer_config = None
            runner.vllm_config.speculative_config = None
            runner.vllm_config.offload_config.uva.cpu_offload_gb = 0
            runner.cache_config.block_size = 16
            runner.cache_config.num_gpu_blocks_override = None
            runner.cache_config.mamba_cache_mode = "none"
            runner.max_num_reqs = 4
            runner.kv_caches = []
            runner.layer_name_to_kvcache_index = {}

            mgr = KVCacheManager(runner)
            mgr._mamba_num_blocks = 5
            spec = MambaSpec(
                shapes=(conv_shape, (4, 8, 8)),
                dtypes=(conv_dtype, torch.float32),
                block_size=16,
            )
            page_bytes = spec.page_size_bytes
            kv_cfg = KVCacheConfig(
                num_blocks=8,
                kv_cache_tensors=[
                    KVCacheTensor(
                        size=8 * page_bytes,
                        layers=["layer.0"],
                        layer_stride=8 * page_bytes,
                        block_stride=page_bytes,
                    )
                ],
                kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
            )
            with (
                    set_current_vllm_config(runner.vllm_config),
                    patch(
                        "tpu_inference.runner.kv_cache_manager.utils.hbm_usage_gb",
                        return_value=[(0.0, 0.0)],
                    ),
            ):
                mgr.initialize_kv_cache(kv_cfg)
            conv_arr, ssm_arr = runner.kv_caches[0]
            assert conv_arr.shape == (5, *conv_shape)
            assert conv_arr.dtype == (jnp.float32 if conv_dtype
                                      == torch.float32 else jnp.bfloat16)
            assert conv_arr.sharding.spec == expected_spec
            assert ssm_arr.shape == (5, 4, 8, 8)

    def test_fused_conv1d_gdn_donates_4d_fp32_conv_state(self):
        import functools

        import jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu

        from tpu_inference.kernels.gdn.v3 import wrapper as gdn_v3_wrapper

        chip = getattr(pltpu.ChipVersion, "TPU_7X", pltpu.ChipVersion.TPU_V6E)
        tpu_info = pltpu.get_tpu_info_for_chip(chip, 1)
        n_kq, n_v, d_k, d_v, kernel_size = 2, 4, 128, 128, 4
        dim = n_kq * d_k * 2 + n_v * d_v
        batch_size = 4
        num_seqs = 4
        num_blocks = 8

        qkv = jax.ShapeDtypeStruct((batch_size, dim), jnp.bfloat16)
        b = jax.ShapeDtypeStruct((batch_size, n_v), jnp.bfloat16)
        a = jax.ShapeDtypeStruct((batch_size, n_v), jnp.bfloat16)
        rec_state = jax.ShapeDtypeStruct((num_blocks, n_v, d_k, d_v),
                                         jnp.float32)
        conv_weight = jax.ShapeDtypeStruct((dim, 1, kernel_size), jnp.bfloat16)
        conv_bias = jax.ShapeDtypeStruct((dim, ), jnp.bfloat16)
        a_log = jax.ShapeDtypeStruct((n_v, ), jnp.float32)
        dt_bias = jax.ShapeDtypeStruct((n_v, ), jnp.float32)
        q_loc = jax.ShapeDtypeStruct((num_seqs + 1, ), jnp.int32)
        state_indices = jax.ShapeDtypeStruct((num_seqs, ), jnp.int32)
        distribution = jax.ShapeDtypeStruct((3, ), jnp.int32)
        seq_lens = jax.ShapeDtypeStruct((num_seqs, ), jnp.int32)
        read_state_indices = jax.ShapeDtypeStruct((num_seqs, ), jnp.int32)

        fn_unwrapped = functools.partial(
            gdn_v3_wrapper.fused_conv1d_gdn.__wrapped__,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )
        fn_jitted = functools.partial(
            gdn_v3_wrapper.fused_conv1d_gdn,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

        with (
                patch.object(pltpu, "get_tpu_info", return_value=tpu_info),
                patch("jax._src.tpu_info.get_tpu_info", return_value=tpu_info),
        ):
            conv_4d_f32 = jax.ShapeDtypeStruct(
                (num_blocks, kernel_size - 1, 1, dim), jnp.float32)
            args_4d = (
                qkv,
                b,
                a,
                conv_4d_f32,
                rec_state,
                conv_weight,
                conv_bias,
                a_log,
                dt_bias,
                q_loc,
                state_indices,
                distribution,
                seq_lens,
                read_state_indices,
            )
            outer_jaxpr = jax.make_jaxpr(fn_jitted)(*args_4d)
            jit_eqns = [
                e for e in outer_jaxpr.jaxpr.eqns if e.primitive.name == "jit"
            ]
            assert len(jit_eqns) == 1
            assert jit_eqns[0].params["donated_invars"][3] is True
            assert jit_eqns[0].params["donated_invars"][4] is True

            closed_4d = jax.make_jaxpr(fn_unwrapped)(*args_4d)
            conv_invar = closed_4d.jaxpr.invars[3]
            pallas_eqns = [
                e for e in closed_4d.jaxpr.eqns
                if e.primitive.name == "pallas_call"
            ]
            assert len(pallas_eqns) == 2

            idx0 = pallas_eqns[0].invars.index(conv_invar)
            aliases0 = dict(pallas_eqns[0].params["input_output_aliases"])
            assert aliases0.get(idx0) == 1

            mid_conv_var = pallas_eqns[0].outvars[1]
            idx1 = pallas_eqns[1].invars.index(mid_conv_var)
            aliases1 = dict(pallas_eqns[1].params["input_output_aliases"])
            assert aliases1.get(idx1) == 1

            final_conv_var = pallas_eqns[1].outvars[1]
            assert closed_4d.jaxpr.outvars[0] is final_conv_var

            conv_chain_vars = {conv_invar, mid_conv_var, final_conv_var}
            bad_eqns = [
                e for e in closed_4d.jaxpr.eqns
                if e.primitive.name in ("convert_element_type",
                                        "reshape") and any(
                                            v in conv_chain_vars
                                            for v in (*e.invars, *e.outvars))
            ]
            assert bad_eqns == []
