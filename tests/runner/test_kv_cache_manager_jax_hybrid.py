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
"""Tests for hybrid attention plus linear attention (GDN) kv cache specs on
the JAX-native model path, where no vLLM attention modules are registered in
the compilation config and specs are derived from the HF config."""

import dataclasses
from unittest.mock import MagicMock

import jax
import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from tpu_inference.runner.kv_cache_manager import KVCacheManager

QWEN3_NEXT_LINEAR = "linear_attention"
QWEN3_NEXT_FULL = "full_attention"


def _mock_runner(layer_types):
    runner = MagicMock()
    devices = np.asarray(jax.devices()[:1])
    runner.mesh = jax.sharding.Mesh(devices.reshape((1, 1, 1, 1)),
                                    ('data', 'attn_dp', 'model', 'expert'))
    runner.cache_config.block_size = 16
    runner.cache_config.num_gpu_blocks_override = None
    runner.cache_config.mamba_block_size = 1024
    runner.cache_config.mamba_cache_dtype = "auto"
    runner.cache_config.mamba_ssm_cache_dtype = "auto"
    runner.cache_config.mamba_cache_mode = "none"
    runner.cache_config.cache_dtype = "auto"
    runner.kv_cache_dtype = torch.bfloat16
    runner.speculative_config = None
    runner.vllm_config.parallel_config.decode_context_parallel_size = 1
    runner.vllm_config.compilation_config.static_forward_context = {}
    runner.vllm_config.speculative_config = None

    model_config = runner.model_config
    model_config.use_mla = False
    model_config.dtype = torch.bfloat16
    model_config.get_num_layers.return_value = len(layer_types)
    model_config.get_total_num_kv_heads.return_value = 2
    model_config.get_head_size.return_value = 256

    text_config = MagicMock()
    text_config.layer_types = list(layer_types)
    text_config.num_key_value_heads = 2
    text_config.head_dim = 256
    text_config.linear_num_key_heads = 16
    text_config.linear_num_value_heads = 32
    text_config.linear_key_head_dim = 128
    text_config.linear_value_head_dim = 128
    text_config.linear_conv_kernel_dim = 4
    # Attributes probed with getattr(...) that must not exist on a real
    # Qwen3-Next config.
    del text_config.num_global_key_value_heads
    del text_config.global_head_dim
    del text_config.kv_sharing_target_layer_names
    del text_config.mtp_kv_share
    model_config.hf_text_config = text_config
    model_config.hf_config = text_config
    return runner


def _conv_dim():
    return 2 * 16 * 128 + 32 * 128


class TestJaxHybridKVCacheSpec:

    def test_mamba_spec_emitted_for_linear_attention_layers(self):
        layer_types = [QWEN3_NEXT_LINEAR] * 3 + [QWEN3_NEXT_FULL]
        manager = KVCacheManager(_mock_runner(layer_types))
        spec = manager.get_kv_cache_spec()

        assert set(spec.keys()) == {f"layer.{i}" for i in range(4)}
        for i in range(3):
            assert isinstance(spec[f"layer.{i}"], MambaSpec)
        assert isinstance(spec["layer.3"], FullAttentionSpec)

        mamba_spec = spec["layer.0"]
        assert mamba_spec.shapes == ((3, _conv_dim()), (32, 128, 128))
        assert mamba_spec.dtypes == (torch.bfloat16, torch.bfloat16)

    def test_hybrid_uniform_page_size_applied_to_all_specs(self):
        layer_types = [QWEN3_NEXT_LINEAR] * 3 + [QWEN3_NEXT_FULL]
        manager = KVCacheManager(_mock_runner(layer_types))
        spec = manager.get_kv_cache_spec()

        mamba_spec = spec["layer.0"]
        attn_spec = spec["layer.3"]
        assert attn_spec.page_size_padded is not None
        assert attn_spec.page_size_padded == mamba_spec.page_size_padded
        unpadded_mamba = dataclasses.replace(
            mamba_spec, page_size_padded=None).page_size_bytes
        assert attn_spec.page_size_padded >= unpadded_mamba

    def test_non_hybrid_model_unaffected(self):
        layer_types = [QWEN3_NEXT_FULL] * 4
        manager = KVCacheManager(_mock_runner(layer_types))
        spec = manager.get_kv_cache_spec()

        assert all(isinstance(s, FullAttentionSpec) for s in spec.values())
        assert manager._hybrid_uniform_page_size_bytes is None

    def test_model_without_layer_types_unaffected(self):
        runner = _mock_runner([QWEN3_NEXT_FULL] * 2)
        del runner.model_config.hf_text_config.layer_types
        manager = KVCacheManager(runner)
        spec = manager.get_kv_cache_spec()

        assert all(isinstance(s, FullAttentionSpec) for s in spec.values())
