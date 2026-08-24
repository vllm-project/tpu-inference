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

import torch
import torch.nn.functional as F

from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import (
    LARGE_ATTN_ELEMENT_THRESHOLD, _flash_attn_sdpa, apply_qwen3_vl_patches)


def test_large_attn_threshold_constant():
    assert LARGE_ATTN_ELEMENT_THRESHOLD == 40 * 1024 * 1024


def test_flash_attn_sdpa_small_shape_fallback():
    # Small shapes (<40M elements) should route directly to _orig_sdpa without error
    batch_size = 1
    num_heads = 16
    seq_len = 100
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Test with standard arguments
    out = _flash_attn_sdpa(q, k, v)
    assert out.shape == (batch_size, num_heads, seq_len, head_dim)

    # Test with *args, **kwargs (e.g. enable_gqa) forwarded without TypeError
    out_kwargs = _flash_attn_sdpa(q,
                                  k,
                                  v,
                                  is_causal=False,
                                  scale=0.125,
                                  enable_gqa=False)
    assert out_kwargs.shape == (batch_size, num_heads, seq_len, head_dim)


def test_flash_attn_sdpa_args_kwargs_forwarding():
    # Verify that arbitrary kwargs (e.g. future vLLM additions) are cleanly accepted and forwarded
    mock_orig = MagicMock(return_value=torch.zeros((1, 2, 4, 8)))
    with patch(
            "tpu_inference.models.vllm.experimental.qwen3_vl_patcher._orig_sdpa",
            mock_orig):
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)

        _flash_attn_sdpa(q,
                         k,
                         v,
                         attn_mask=None,
                         dropout_p=0.0,
                         is_causal=False,
                         scale=0.5,
                         custom_kwarg=123)
        mock_orig.assert_called_once()
        _, kwargs = mock_orig.call_args
        assert kwargs.get("custom_kwarg") == 123
        assert kwargs.get("scale") == 0.5


def test_scoped_vit_patching():
    # Mock model
    mock_model = MagicMock()
    mock_model.language_model.model.do_not_compile = False

    # Apply patches
    apply_qwen3_vl_patches(mock_model)

    # Global F.scaled_dot_product_attention should remain untouched
    assert F.scaled_dot_product_attention != _flash_attn_sdpa
