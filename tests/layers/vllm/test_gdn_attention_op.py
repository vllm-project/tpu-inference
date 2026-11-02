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

    def test_compute_align_spec_state_indices(self):
        from tpu_inference.layers.common.gdn_attention import \
            compute_align_spec_state_indices

        # 4 requests, block_size=16, 64 rows in the shard:
        #   req 0: verify window (q=4, seq=18 -> n=14), crosses block 0 boundary
        #          (16) at window pos 1, resumes from scratch offset 2.
        #   req 1: verify window (q=4, seq=8 -> n=4), does NOT reach boundary
        #          16, resumes from block table (-1 offset, right after prefill).
        #   req 2: prefill chunk (q=16, seq=32 -> n=16), not a window.
        #   req 3: padded dummy (q=0, seq=0).
        block_tables = jax.numpy.array(
            [[3, 4], [5, 0], [6, 7], [0, 0]],
            dtype=jax.numpy.int32,
        )
        query_lens = jax.numpy.array([4, 4, 16, 0], dtype=jax.numpy.int32)
        seq_lens = jax.numpy.array([18, 8, 32, 0], dtype=jax.numpy.int32)
        is_window = jax.numpy.array([True, True, False, False])
        scratch_bases = jax.numpy.array([40, 44, 48, 0], dtype=jax.numpy.int32)
        read_offsets = jax.numpy.array([2, -1, -1, 0], dtype=jax.numpy.int32)

        (state_indices, read_state_indices, copy_src, copy_dst,
         copy_mask) = compute_align_spec_state_indices(
             block_tables,
             query_lens,
             seq_lens,
             is_window,
             scratch_bases,
             read_offsets,
             mamba_block_size=16,
             num_local_rows=64,
         )

        # Windows write to their scratch base; prefill writes to block 1 (id 7).
        np.testing.assert_array_equal(np.asarray(state_indices),
                                      [40, 44, 7, 0])
        # req 0 reads scratch 40 + 2 = 42; req 1 reads block 0 (id 5);
        # req 2 reads block 0 (id 6).
        np.testing.assert_array_equal(np.asarray(read_state_indices),
                                      [42, 5, 6, 0])
        # Only req 0 crosses a block boundary (16 = 14 + 2 -> scratch pos 1 = 41
        # copied into block 0 = id 3).
        np.testing.assert_array_equal(np.asarray(copy_mask),
                                      [True, False, False, False])
        assert int(copy_src[0]) == 41
        assert int(copy_dst[0]) == 3

    def test_copy_gdn_state_slots(self, mesh):
        from tpu_inference.layers.common.gdn_attention import \
            copy_gdn_state_slots

        conv = jax.numpy.arange(8 * 3 * 4,
                                dtype=jax.numpy.float32).reshape(8, 3, 4)
        rec = jax.numpy.arange(8 * 2 * 4 * 4,
                               dtype=jax.numpy.float32).reshape(8, 2, 4, 4)
        src = jax.numpy.array([5, 6, 0, 0], dtype=jax.numpy.int32)
        dst = jax.numpy.array([1, 2, 3, 4], dtype=jax.numpy.int32)
        mask = jax.numpy.array([False, True, False, False])

        new_conv, new_rec = copy_gdn_state_slots(conv, rec, src, dst, mask,
                                                 mesh)
        # Slot 6 copied to slot 2; slot 1 (masked off) untouched.
        np.testing.assert_array_equal(np.asarray(new_conv[2]),
                                      np.asarray(conv[6]))
        np.testing.assert_array_equal(np.asarray(new_rec[2]),
                                      np.asarray(rec[6]))
        np.testing.assert_array_equal(np.asarray(new_conv[1]),
                                      np.asarray(conv[1]))

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
    def test_gdn_attention_core_tpu_align_mode_with_spec_decode(
            self, mock_get_fc, mock_run_jax, mock_jax_v, mock_torch_v, mesh):
        """Align mode + spec decode checkpoints verify windows into scratch
        slots and copies block-boundary states into the scheduler's mamba
        blocks after the kernel runs."""
        from tpu_inference.layers.common.attention_metadata import \
            AttentionMetadata
        from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
            gdn_attention_core_tpu

        layer_name = "test_layer"
        num_tokens = 8
        n_kq, n_v, d_k, d_v, kernel_size = 2, 4, 16, 16, 4
        conv_dim = (n_kq * d_k) * 2 + n_v * d_v

        layer_module = MagicMock()
        layer_module.num_k_heads = n_kq
        layer_module.num_v_heads = n_v
        layer_module.head_k_dim = d_k
        layer_module.head_v_dim = d_v
        layer_module.conv_kernel_size = kernel_size
        layer_module.conv1d.weight = torch.randn(conv_dim, 1, kernel_size)
        layer_module.conv1d.bias = None
        layer_module.A_log = torch.randn(n_v)
        layer_module.dt_bias = torch.randn(n_v)

        # 2 verify windows of 4 tokens (1 target + 3 drafts):
        #   req 0: computed 14 -> seq_len 18, crosses block 0 boundary (16) at
        #          window position 1 (scratch slot 20 + 1 = 21 -> block id 3),
        #          resumes from scratch checkpoint 20 + 2 = 22.
        #   req 1: computed 16 -> seq_len 20, resumes from block 0 (id 5) via
        #          gdn_read_offsets = -1, does not cross block 1 boundary (32).
        attn_metadata = AttentionMetadata(
            input_positions=jax.numpy.zeros(num_tokens, dtype=jax.numpy.int32),
            block_tables=jax.numpy.array([[3, 4], [5, 6]],
                                         dtype=jax.numpy.int32),
            seq_lens=jax.numpy.array([18, 20], dtype=jax.numpy.int32),
            query_start_loc=jax.numpy.array([0, 4, 8], dtype=jax.numpy.int32),
            request_distribution=jax.numpy.array([2, 2, 2],
                                                 dtype=jax.numpy.int32),
            mamba_state_indices=jax.numpy.array([20, 24],
                                                dtype=jax.numpy.int32),
            gdn_read_offsets=jax.numpy.array([2, -1], dtype=jax.numpy.int32),
            num_spec_tokens=3,
            padded_num_reqs=2,
        )

        fc = MagicMock()
        fc.attn_metadata = {layer_name: attn_metadata}
        fc.no_compile_layers = {layer_name: layer_module}
        mock_get_fc.return_value = fc

        conv_state = jax.numpy.zeros((30, kernel_size - 1, conv_dim))
        recurrent_state = jax.numpy.zeros((30, n_v, d_k, d_v))
        # Kernel writes checkpoints into scratch slots; put a distinctive value
        # at req 0's boundary checkpoint (slot 21) and verify it lands in
        # scheduler block 3.
        kernel_conv = conv_state.at[21].set(7.0)
        kernel_rec = recurrent_state.at[21].set(9.0)
        mock_run_jax.return_value = (
            (kernel_conv, kernel_rec),
            jax.numpy.zeros((num_tokens, n_v * d_v)),
        )

        vllm_config = MagicMock()
        vllm_config.cache_config.mamba_cache_mode = "align"
        vllm_config.cache_config.mamba_block_size = 16

        kv_caches = [(conv_state, recurrent_state)]
        with set_vllm_model_wrapper_context(
                kv_caches=kv_caches,
                mesh=mesh,
                layer_name_to_kvcache_index={layer_name: 0},
                vllm_config=vllm_config,
        ):
            gdn_attention_core_tpu(torch.randn(num_tokens, conv_dim),
                                   torch.randn(num_tokens, n_v),
                                   torch.randn(num_tokens, n_v),
                                   torch.zeros(num_tokens, n_v, d_v),
                                   layer_name, mesh)

        call_args, call_kwargs = mock_run_jax.call_args
        np.testing.assert_array_equal(np.asarray(call_args[9]), [20, 24])
        np.testing.assert_array_equal(
            np.asarray(call_kwargs["read_state_indices"]), [22, 5])
        np.testing.assert_array_equal(np.asarray(call_kwargs["read_offsets"]),
                                      [0, 0])
        assert call_kwargs["num_spec_tokens"] == 3

        updated_conv, updated_rec = kv_caches[0]
        assert np.all(np.asarray(updated_conv[3]) == 7.0)
        assert np.all(np.asarray(updated_rec[3]) == 9.0)
