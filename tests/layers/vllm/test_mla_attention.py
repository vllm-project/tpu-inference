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
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.v1.attention.backend import AttentionType

from tpu_inference.layers.vllm.mla_attention import (
    TPUMLAAttention, VllmTPUMultiHeadLatentAttentionWrapper)
from tpu_inference.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


class TestTPUMLAAttention:

    @patch(
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
        autospec=True)
    def test_init_auto_dtype(self, mock_super_init):
        # Emulate how vLLM initializes the cache type
        def side_effect(self, *args, **kwargs):
            self.kv_cache_dtype = "auto"

        mock_super_init.side_effect = side_effect

        kv_b_proj = MagicMock()
        attn = TPUMLAAttention(
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            kv_b_proj=kv_b_proj,
            prefix="test",
        )

        # Validate that the fallback attributes are set correctly
        assert attn.kv_sharing_target_layer_name is None
        assert attn.attn_type == AttentionType.DECODER
        assert attn.sliding_window is None
        assert attn.kv_cache_quantized_dtype is None

    @patch(
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
        autospec=True)
    def test_init_fp8_dtype(self, mock_super_init):

        def side_effect(self, *args, **kwargs):
            self.kv_cache_dtype = "fp8"

        mock_super_init.side_effect = side_effect

        kv_b_proj = MagicMock()
        attn = TPUMLAAttention(
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            kv_b_proj=kv_b_proj,
            prefix="test",
        )

        assert attn.kv_cache_quantized_dtype == jnp.float8_e4m3fn

    @patch(
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
        autospec=True)
    def test_process_weights_after_loading(self, mock_super_init):

        def side_effect(self, *args, **kwargs):
            self.kv_cache_dtype = "auto"

        mock_super_init.side_effect = side_effect

        # Mock a linear layer (to simulate column parallel linear)
        kv_b_proj = torch.nn.Linear(10, 10)
        attn = TPUMLAAttention(
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            kv_b_proj=kv_b_proj,
            prefix="test",
        )

        attn.W_UK_T = torch.rand(10, 10)
        attn.W_UV = torch.rand(10, 10)
        attn.kv_b_proj = MagicMock()

        with patch(
                "vllm.model_executor.layers.attention.mla_attention.MLAAttention.process_weights_after_loading"
        ) as mock_super_process:
            attn.process_weights_after_loading(torch.float32)

            mock_super_process.assert_called_once_with(torch.float32)

            # W_UK_T and W_UV should now be un-grad Parameters
            assert isinstance(attn.W_UK_T, torch.nn.Parameter)
            assert not attn.W_UK_T.requires_grad
            assert isinstance(attn.W_UV, torch.nn.Parameter)
            assert not attn.W_UV.requires_grad

    @patch("tpu_inference.layers.vllm.mla_attention.get_attention_context")
    @patch(
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
        autospec=True)
    def test_forward(self, mock_super_init, mock_get_attention_context, mesh):

        def side_effect(self, *args, **kwargs):
            self.kv_cache_dtype = "auto"

        mock_super_init.side_effect = side_effect

        attn = TPUMLAAttention(
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            kv_b_proj=MagicMock(),
            prefix="test_layer",
        )

        attn.calculate_kv_scales = False
        attn.layer_name = "test_layer"
        attn.impl = MagicMock()

        mock_outputs = jnp.zeros((1, 10))
        mock_new_kv_cache = jnp.zeros((10, 10))
        attn.impl.forward.return_value = (mock_outputs, mock_new_kv_cache)

        mock_attn_metadata = MagicMock()
        mock_get_attention_context.return_value = (mock_attn_metadata, None,
                                                   None, None)

        q = torch.rand(1, 10)
        kv_c_normed = torch.rand(1, 10)
        k_pe = torch.rand(1, 10)

        kv_cache = jnp.zeros((10, 10))

        with set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={"test_layer": 0}):
            outputs = attn.forward(q, kv_c_normed, k_pe)

            mock_get_attention_context.assert_called_once_with("test_layer")
            attn.impl.forward.assert_called_once_with(q, kv_c_normed, k_pe,
                                                      kv_cache,
                                                      mock_attn_metadata, mesh,
                                                      attn)

            assert isinstance(outputs, torch.Tensor)
            context = get_vllm_model_wrapper_context()
            assert context.kv_caches[0] is mock_new_kv_cache

    @patch("torch.ops.vllm.maybe_calc_kv_scales", create=True)
    @patch("tpu_inference.layers.vllm.mla_attention.get_attention_context")
    @patch(
        "vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
        autospec=True)
    def test_forward_calculates_kv_scales(self, mock_super_init,
                                          mock_get_attention_context,
                                          mock_maybe_calc_kv_scales, mesh):

        def side_effect(self, *args, **kwargs):
            self.kv_cache_dtype = "auto"

        mock_super_init.side_effect = side_effect

        attn = TPUMLAAttention(
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            kv_b_proj=MagicMock(),
            prefix="test_layer",
        )

        attn.calculate_kv_scales = True
        attn.layer_name = "test_layer"
        attn.impl = MagicMock()
        attn.impl.forward.return_value = (jnp.zeros(1), jnp.zeros(1))
        mock_get_attention_context.return_value = (MagicMock(), None, None,
                                                   None)

        q = torch.rand(1, 10)
        kv_c_normed = torch.rand(1, 10)
        k_pe = torch.rand(1, 10)

        with set_vllm_model_wrapper_context(
                kv_caches=[jnp.zeros(1)],
                mesh=mesh,
                layer_name_to_kvcache_index={"test_layer": 0}):
            attn.forward(q, kv_c_normed, k_pe)

            mock_maybe_calc_kv_scales.assert_called_once_with(
                q, kv_c_normed, k_pe, "test_layer")


class TestVllmTPUMultiHeadLatentAttentionWrapper:

    @patch("tpu_inference.layers.vllm.mla_attention.TPUMLAAttention",
           autospec=True)
    def test_init(self, mock_tpu_mla_attn):
        mla_modules = MagicMock()
        mla_modules.fused_qkv_a_proj = "fused_qkv_a_proj"
        mla_modules.kv_a_proj_with_mqa = "kv_a_proj_with_mqa"
        mla_modules.q_a_layernorm = "q_a_layernorm"
        mla_modules.q_b_proj = "q_b_proj"
        mla_modules.q_proj = "q_proj"
        mla_modules.kv_a_layernorm = "kv_a_layernorm"
        mla_modules.kv_b_proj = MagicMock()
        mla_modules.rotary_emb = "rotary_emb"
        mla_modules.o_proj = "o_proj"
        mla_modules.indexer = None
        mla_modules.indexer_rotary_emb = "indexer_rotary_emb"
        mla_modules.is_sparse = False

        wrapper = VllmTPUMultiHeadLatentAttentionWrapper(
            hidden_size=128,
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            mla_modules=mla_modules,
            cache_config=None,
            quant_config=None,
            prefix="test_wrapper",
        )

        assert wrapper.hidden_size == 128
        assert wrapper.qk_head_dim == 96
        assert wrapper.fused_qkv_a_proj == "fused_qkv_a_proj"
        assert wrapper.kv_a_proj_with_mqa == "kv_a_proj_with_mqa"
        assert wrapper.q_a_layernorm == "q_a_layernorm"
        assert wrapper.q_b_proj == "q_b_proj"
        assert wrapper.q_proj == "q_proj"
        assert wrapper.kv_a_layernorm == "kv_a_layernorm"
        assert wrapper.kv_b_proj == mla_modules.kv_b_proj
        assert wrapper.rotary_emb == "rotary_emb"
        assert wrapper.o_proj == "o_proj"
        assert wrapper.indexer is None
        assert wrapper.is_sparse is False

        mock_tpu_mla_attn.assert_called_once_with(
            num_heads=8,
            scale=1.0,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            q_lora_rank=None,
            kv_lora_rank=32,
            cache_config=None,
            quant_config=None,
            prefix="test_wrapper.attn",
            kv_b_proj=mla_modules.kv_b_proj,
            use_sparse=False,
            indexer=None,
        )
        assert wrapper.mla_attn == mock_tpu_mla_attn.return_value
