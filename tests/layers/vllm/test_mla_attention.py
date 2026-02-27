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
import torchax
from jax.sharding import Mesh
from torchax.interop import torch_view
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import set_forward_context

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.vllm.mla_attention import TPUMLAAttention
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context

TOTAL_TOKENS = 10
NUM_SEQS = 2
MAX_NUM_SEQS = 4
NUM_HEADS = 4
KV_LORA_RANK = 64
QK_NOPE_HEAD_DIM = 32
QK_ROPE_HEAD_DIM = 16
V_HEAD_DIM = 32
SCALE = 0.1
PREFIX = "model.layers.0.attn"


@pytest.fixture
def mesh():
    """Provides a mock 1D JAX mesh for testing."""
    devices = np.array(jax.local_devices())[0:1]
    if not devices.any():
        devices = np.array([jax.devices("cpu")[0]])
    return Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))


@pytest.fixture
def mock_vllm_config():
    """Mocks the global vLLM config required by TPUMLAAttention."""
    # Patch the function in the module where it is used (mla_attention.py),
    # because it was imported using 'from ... import ...'
    with patch(
            "tpu_inference.layers.vllm.mla_attention.get_current_vllm_config"
    ) as mock_get_cfg:
        mock_config = MagicMock(spec=VllmConfig)
        mock_config.compilation_config.static_forward_context = {}
        mock_config.parallel_config.pipeline_parallel_size = 1
        mock_config.parallel_config.data_parallel_size = 1
        mock_get_cfg.return_value = mock_config
        yield mock_config


def create_mla_inputs(
    mesh: Mesh,
    num_heads: int = NUM_HEADS,
    qk_nope_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_dim: int = QK_ROPE_HEAD_DIM,
    kv_lora_rank: int = KV_LORA_RANK,
    dtype: jnp.dtype = jnp.bfloat16,
):
    key = jax.random.key(0)

    # Inputs to forward
    # q: (batch, num_heads, qk_nope + qk_rope)
    q = jax.random.uniform(
        key, (TOTAL_TOKENS, num_heads, qk_nope_dim + qk_rope_dim), dtype=dtype)
    # kv_c_normed: (batch, 1, kv_lora_rank) - usually expanded in VLLM before calling forward
    kv_c_normed = jax.random.uniform(key, (TOTAL_TOKENS, 1, kv_lora_rank),
                                     dtype=dtype)
    # k_pe: (batch, 1, qk_rope_dim)
    k_pe = jax.random.uniform(key, (TOTAL_TOKENS, 1, qk_rope_dim), dtype=dtype)

    # Convert to torch
    q = torch_view(q)
    kv_c_normed = torch_view(kv_c_normed)
    k_pe = torch_view(k_pe)

    # Dummy KV Cache (Shape depends on backend implementation, creating generic placeholder)
    # Assuming standard KV cache structure for tests
    kv_cache = jax.random.normal(key, (1, 100, 1024), dtype=dtype)

    # Metadata
    positions = jnp.ones((TOTAL_TOKENS, ), dtype=jnp.int32)
    block_tables = jnp.zeros((MAX_NUM_SEQS * 8), dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.array([5, 5, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 5, 10, 10, 10], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, NUM_SEQS], dtype=jnp.int32)

    metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )

    return q, kv_c_normed, k_pe, kv_cache, metadata


class TestTPUMLAAttention:

    def test_init_valid_params(self, mock_vllm_config):
        mock_kv_b_proj = MagicMock()

        attn = TPUMLAAttention(
            num_heads=NUM_HEADS,
            scale=SCALE,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            kv_b_proj=mock_kv_b_proj,
            prefix=PREFIX,
        )

        assert attn.num_heads == NUM_HEADS
        assert attn.qk_nope_head_dim == QK_NOPE_HEAD_DIM
        assert attn.qk_rope_head_dim == QK_ROPE_HEAD_DIM
        assert attn.v_head_dim == V_HEAD_DIM
        assert attn.kv_lora_rank == KV_LORA_RANK
        assert attn.layer_name == PREFIX
        assert attn.num_kv_heads == 1
        # Check registration in config
        assert mock_vllm_config.compilation_config.static_forward_context[
            PREFIX] == attn

    def test_init_duplicate_prefix_raises_error(self, mock_vllm_config):
        # Pre-populate context to trigger error
        mock_vllm_config.compilation_config.static_forward_context[
            PREFIX] = "existing"
        mock_kv_b_proj = MagicMock()

        with pytest.raises(ValueError, match="Duplicate layer name"):
            TPUMLAAttention(
                num_heads=NUM_HEADS,
                scale=SCALE,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                q_lora_rank=None,
                kv_lora_rank=KV_LORA_RANK,
                kv_b_proj=mock_kv_b_proj,
                prefix=PREFIX,
            )

    def test_process_weights_after_loading(self, mock_vllm_config):
        mock_kv_b_proj = MagicMock()
        mock_kv_b_proj.quant_method = None
        out_features = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        in_features = KV_LORA_RANK

        # Create a real parameter so .shape, .to(), and .T work correctly during the assertion
        mock_kv_b_proj.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features))
        mock_kv_b_proj.named_parameters.return_value = {
            "weight": mock_kv_b_proj.weight
        }.items()

        attn = TPUMLAAttention(
            num_heads=NUM_HEADS,
            scale=SCALE,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            kv_b_proj=mock_kv_b_proj,
            prefix=PREFIX,
        )

        with torchax.default_env():
            attn.process_weights_after_loading(torch.float32)

        # Check that kv_b_proj attributes were deleted
        assert not hasattr(attn.kv_b_proj, "weight")
        # Check that weights are frozen
        assert not attn.W_UK_T.requires_grad
        assert not attn.W_UV.requires_grad

    @patch("vllm.config.get_current_vllm_config")
    @patch("tpu_inference.layers.vllm.mla_attention.get_attention_context")
    def test_forward(self, mock_get_attn_context, mock_get_current_vllm_config,
                     mesh, mock_vllm_config):
        out_features = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        in_features = KV_LORA_RANK

        mock_kv_b_proj = MagicMock()
        mock_kv_b_proj.quant_method = None
        mock_kv_b_proj.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features))
        mock_kv_b_proj.named_parameters.return_value = {
            "weight": mock_kv_b_proj.weight
        }.items()

        with torchax.default_env():
            attn = TPUMLAAttention(
                num_heads=NUM_HEADS,
                scale=SCALE,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                q_lora_rank=None,
                kv_lora_rank=KV_LORA_RANK,
                kv_b_proj=mock_kv_b_proj,
                prefix=PREFIX,
            )

            attn.process_weights_after_loading(torch.float32)

            # Inputs
            q, kv_c_normed, k_pe, kv_cache, metadata = create_mla_inputs(mesh)

            mock_get_attn_context.return_value = (metadata, None, None, None)
            mock_get_current_vllm_config.return_value = mock_vllm_config

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={PREFIX: 0}), set_forward_context(
                    attn_metadata=metadata, vllm_config=mock_vllm_config):

            output = attn.forward(q, kv_c_normed, k_pe)

        # Verify output shape
        # Output should be (TOTAL_TOKENS, NUM_HEADS * V_HEAD_DIM) after W_UV projection
        expected_shape = (TOTAL_TOKENS, NUM_HEADS * V_HEAD_DIM)
        assert output.shape == expected_shape
        assert isinstance(output, torch.Tensor)

    @patch(
        "vllm.model_executor.layers.attention.attention.get_attention_context")
    def test_forward_with_quantization(self, mock_get_attn_context, mesh,
                                       mock_vllm_config):
        mock_kv_b_proj = MagicMock()
        cache_config = MagicMock(spec=CacheConfig)
        cache_config.cache_dtype = "fp8"
        cache_config.calculate_kv_scales = False

        attn = TPUMLAAttention(
            num_heads=NUM_HEADS,
            scale=SCALE,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            kv_b_proj=mock_kv_b_proj,
            cache_config=cache_config,
            prefix=PREFIX,
        )

        # Setup weights
        target_proj_dim = 64
        attn.W_UK_T = torch.nn.Parameter(
            torch.randn(NUM_HEADS, QK_NOPE_HEAD_DIM, target_proj_dim))
        attn.W_UV = torch.nn.Parameter(
            torch.randn(NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM))
        attn._q_scale_float = 1.0
        attn._k_scale_float = 1.0
        attn._v_scale_float = 1.0

        q, kv_c_normed, k_pe, kv_cache, metadata = create_mla_inputs(
            mesh, dtype=jnp.float32)

        # mock_get_attn_context.return_value = (metadata, None, None, None)

        with torchax.default_env(), set_vllm_model_wrapper_context(
                kv_caches=[kv_cache],
                mesh=mesh,
                layer_name_to_kvcache_index={PREFIX: 0}), set_forward_context(
                    attn_metadata=metadata, vllm_config=mock_vllm_config):

            output = attn.forward(q, kv_c_normed, k_pe)

        assert output.shape == (TOTAL_TOKENS, NUM_HEADS * V_HEAD_DIM)
