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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from tpu_inference.models.common import hybrid, model_loader

_ARCH = "FakeHybridForCausalLM"
_VERIFIER = ("vllm.model_executor.models.config."
             "HybridAttentionMambaModelConfig.verify_and_update_config")


class _NotOptedIn:
    """A JAX model class without the hybrid hooks."""


class _HybridModel:
    """A JAX model class that is hybrid iff `hf_text_config.hybrid`."""

    @classmethod
    def is_hybrid_config(cls, hf_text_config):
        return getattr(hf_text_config, "hybrid", False)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return ((3, 64), (8, 64, 16))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return (torch.bfloat16, torch.float32)


class _HybridModelWithLayerTypes(_HybridModel):

    @classmethod
    def get_hybrid_layer_types(cls, vllm_config):
        return ["mamba", "mamba", "mamba", "attention"] * 2


def _make_vllm_config(*,
                      hybrid_config=True,
                      layer_types=None,
                      is_hybrid=False,
                      enable_prefix_caching=True,
                      architecture=_ARCH):
    hf_text_config = SimpleNamespace(hybrid=hybrid_config)
    if layer_types is not None:
        hf_text_config.layer_types = layer_types
    model_config = SimpleNamespace(architecture=architecture,
                                   model="fake/model",
                                   is_hybrid=is_hybrid,
                                   hf_text_config=hf_text_config,
                                   supports_mamba_prefix_caching=False,
                                   max_model_len=4096)
    cache_config = CacheConfig(block_size=16,
                               enable_prefix_caching=enable_prefix_caching)
    scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    return SimpleNamespace(model_config=model_config,
                           cache_config=cache_config,
                           scheduler_config=scheduler_config)


@pytest.fixture
def register(monkeypatch):

    def _register(model_cls, arch=_ARCH):
        monkeypatch.setitem(model_loader._MODEL_REGISTRY, arch, model_cls)

    return _register


def test_capability_flag():
    assert hybrid.SUPPORTS_JAX_HYBRID_MODELS is True


def test_get_jax_model_class(register):
    register(_HybridModel)
    assert hybrid.get_jax_model_class(_make_vllm_config()) is _HybridModel
    assert hybrid.get_jax_model_class(
        _make_vllm_config(architecture="UnknownForCausalLM")) is None
    assert hybrid.get_jax_model_class(
        SimpleNamespace(model_config=None)) is None


def test_get_jax_model_class_without_model_loader_import(register):
    register(_HybridModel)
    # Nothing can have called `register_model` if `model_loader` was never
    # imported, and the lookup must not import it (it pulls in JAX and model
    # code from the platform's config checks).
    with patch.dict("sys.modules", {hybrid._MODEL_LOADER_MODULE: None}):
        assert hybrid.get_jax_model_class(_make_vllm_config()) is None


@pytest.mark.parametrize("model_cls,hybrid_config,expected", [
    (_HybridModel, True, True),
    (_HybridModel, False, False),
    (_NotOptedIn, True, False),
    (None, True, False),
])
def test_jax_model_is_hybrid(register, model_cls, hybrid_config, expected):
    if model_cls is not None:
        register(model_cls)
    vllm_config = _make_vllm_config(hybrid_config=hybrid_config)
    assert hybrid.jax_model_is_hybrid(vllm_config) is expected


def test_maybe_apply_jax_hybrid_config_sets_align_mode(register):
    register(_HybridModel)
    vllm_config = _make_vllm_config()
    assert vllm_config.cache_config.mamba_cache_mode == "none"

    hybrid.maybe_apply_jax_hybrid_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert vllm_config.cache_config.mamba_block_size == 16

    # Idempotent: a second run (e.g. after another verifier) keeps the
    # values.
    hybrid.maybe_apply_jax_hybrid_config(vllm_config)
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert vllm_config.cache_config.mamba_block_size == 16


def test_maybe_apply_jax_hybrid_config_without_prefix_caching(register):
    register(_HybridModel)
    vllm_config = _make_vllm_config(enable_prefix_caching=False)

    hybrid.maybe_apply_jax_hybrid_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_mode == "none"
    assert vllm_config.cache_config.mamba_block_size == 4096


@pytest.mark.parametrize("model_cls,hybrid_config,is_hybrid", [
    (_HybridModel, True, True),
    (_HybridModel, False, False),
    (_NotOptedIn, True, False),
    (None, True, False),
])
def test_maybe_apply_jax_hybrid_config_noop(register, model_cls, hybrid_config,
                                            is_hybrid):
    if model_cls is not None:
        register(model_cls)
    vllm_config = _make_vllm_config(hybrid_config=hybrid_config,
                                    is_hybrid=is_hybrid)

    with patch(_VERIFIER) as mock_verify:
        hybrid.maybe_apply_jax_hybrid_config(vllm_config)

    mock_verify.assert_not_called()
    assert vllm_config.cache_config.mamba_cache_mode == "none"


def test_maybe_apply_jax_hybrid_config_no_model_config():
    with patch(_VERIFIER) as mock_verify:
        hybrid.maybe_apply_jax_hybrid_config(
            SimpleNamespace(model_config=None))
    mock_verify.assert_not_called()


def test_get_jax_hybrid_layer_kinds_from_model_class(register):
    register(_HybridModelWithLayerTypes)
    # The class hook takes precedence over HF layer_types.
    vllm_config = _make_vllm_config(layer_types=["full_attention"] * 8)

    kinds = hybrid.get_jax_hybrid_layer_kinds(vllm_config, num_layers=8)

    assert kinds == ["mamba", "mamba", "mamba", "attention"] * 2


def test_get_jax_hybrid_layer_kinds_from_hf_layer_types(register):
    register(_HybridModel)
    # Sparse-attention spellings (vLLM's `_ATTENTION_LAYER_TYPES`) and
    # spellings this module has never seen are attention, not mamba.
    layer_types = [
        "linear_attention", "full_attention", "sliding_attention",
        "deepseek_sparse_attention", "indexed_attention",
        "qwen_sparse_attention", "mamba", "future_attention"
    ]
    vllm_config = _make_vllm_config(layer_types=layer_types)

    kinds = hybrid.get_jax_hybrid_layer_kinds(vllm_config, num_layers=8)

    assert kinds == [
        "mamba", "attention", "attention", "attention", "attention",
        "attention", "mamba", "attention"
    ]


def test_hf_recurrent_layer_types_follow_vllm_layer_block_type():
    assert hybrid._hf_recurrent_layer_types() == frozenset(
        {"linear_attention", "mamba"})


def test_get_jax_hybrid_layer_kinds_truncates_to_num_layers(register):
    register(_HybridModelWithLayerTypes)
    kinds = hybrid.get_jax_hybrid_layer_kinds(_make_vllm_config(),
                                              num_layers=4)
    assert kinds == ["mamba", "mamba", "mamba", "attention"]


@pytest.mark.parametrize("model_cls,hybrid_config", [
    (_HybridModel, False),
    (_NotOptedIn, True),
    (None, True),
])
def test_get_jax_hybrid_layer_kinds_not_hybrid(register, model_cls,
                                               hybrid_config):
    if model_cls is not None:
        register(model_cls)
    vllm_config = _make_vllm_config(hybrid_config=hybrid_config,
                                    layer_types=["linear_attention"] * 4)
    assert hybrid.get_jax_hybrid_layer_kinds(vllm_config, 4) is None


def test_get_jax_hybrid_layer_kinds_errors(register):
    register(_HybridModelWithLayerTypes)
    with pytest.raises(ValueError, match="cover 8 layers"):
        hybrid.get_jax_hybrid_layer_kinds(_make_vllm_config(), num_layers=9)

    class _BadKinds(_HybridModel):

        @classmethod
        def get_hybrid_layer_types(cls, vllm_config):
            return ["attention", "linear_attention"]

    register(_BadKinds)
    with pytest.raises(ValueError, match="invalid layer kinds"):
        hybrid.get_jax_hybrid_layer_kinds(_make_vllm_config(), num_layers=2)

    register(_HybridModel)
    with pytest.raises(ValueError, match="layer_types"):
        hybrid.get_jax_hybrid_layer_kinds(_make_vllm_config(), num_layers=2)


def test_get_jax_hybrid_layer_kinds_pipeline_parallel_slice(register):

    class _PPHybridModel(_HybridModel):

        @classmethod
        def get_hybrid_layer_types(cls, vllm_config):
            return [
                "attention", "attention", "mamba", "mamba", "mamba",
                "attention", "mamba", "attention"
            ]

    register(_PPHybridModel)
    vllm_config = _make_vllm_config()
    vllm_config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    vllm_config.model_config.get_layers_start_end_indices = (
        lambda parallel_config: (4, 8))

    kinds = hybrid.get_jax_hybrid_layer_kinds(vllm_config, num_layers=4)

    assert kinds == ["mamba", "attention", "mamba", "attention"]


def test_get_jax_mamba_type_defaults(register):
    register(_HybridModel)
    # Without linear_conv_kernel_dim (e.g. GraniteMoeHybrid / Jamba), even
    # "linear_attention" layer_types stays on MAMBA2.
    mamba2_cfg = _make_vllm_config(
        layer_types=["linear_attention", "full_attention"])
    assert (hybrid.get_jax_mamba_type(mamba2_cfg) ==
            MambaAttentionBackendEnum.MAMBA2)

    # Presence of linear_conv_kernel_dim selects GDN_ATTN.
    gdn_cfg = _make_vllm_config(
        layer_types=["linear_attention", "full_attention"])
    gdn_cfg.model_config.hf_text_config.linear_conv_kernel_dim = 4
    assert (hybrid.get_jax_mamba_type(gdn_cfg) ==
            MambaAttentionBackendEnum.GDN_ATTN)


def test_get_jax_mamba_type_hook_and_errors(register):

    class _EnumMambaModel(_HybridModel):

        @classmethod
        def get_mamba_type_from_config(cls, vllm_config):
            return MambaAttentionBackendEnum.LINEAR

    register(_EnumMambaModel)
    assert (hybrid.get_jax_mamba_type(
        _make_vllm_config()) == MambaAttentionBackendEnum.LINEAR)

    class _StrMambaModel(_HybridModel):

        @classmethod
        def get_mamba_type_from_config(cls, vllm_config):
            return "gdn_attn"

    register(_StrMambaModel)
    assert (hybrid.get_jax_mamba_type(
        _make_vllm_config()) == MambaAttentionBackendEnum.GDN_ATTN)

    class _BadMambaModel(_HybridModel):

        @classmethod
        def get_mamba_type_from_config(cls, vllm_config):
            return "unknown_mamba"

    register(_BadMambaModel)
    with pytest.raises(ValueError, match="invalid mamba_type"):
        hybrid.get_jax_mamba_type(_make_vllm_config())
