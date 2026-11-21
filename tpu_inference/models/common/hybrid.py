# Copyright 2025 Google LLC
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
"""Opt-in hooks for hybrid attention + mamba (recurrent) JAX models.

vLLM decides whether a model is hybrid from the model *class* (the
`IsHybrid` interface), so one class cannot be hybrid for some HF configs and
not for others. That is fine for vLLM's PyTorch models but not for JAX model
classes that serve many architectures from one wrapper (for example, a
single class registered via `model_loader.register_model` that dispatches on
the HF config). Such classes can opt in to hybrid KV cache handling by
defining these classmethods:

* `is_hybrid_config(hf_text_config) -> bool`: whether this HF text config
  describes a hybrid attention + mamba model. Required to opt in.
* `get_mamba_state_shape_from_config(vllm_config)
  -> tuple[tuple[int, ...], ...]`: per-request recurrent state shapes of one
  mamba layer (vLLM `IsHybrid` naming).
* `get_mamba_state_dtype_from_config(vllm_config)
  -> tuple[torch.dtype, ...]`: the dtypes of those state tensors.
* `get_hybrid_layer_types(vllm_config) -> list[str]` (optional): per
  decoder layer, "attention" or "mamba". Defaults to deriving the kinds from
  `hf_text_config.layer_types`.
* `get_mamba_type_from_config(vllm_config) -> MambaAttentionBackendEnum | str`
  (optional): recurrent backend enum (e.g. `GDN_ATTN` or `MAMBA2`). Defaults to
  `GDN_ATTN` when `hf_text_config` defines `linear_conv_kernel_dim`, else
  `MAMBA2`.

Models that do not define `is_hybrid_config` (or return False from it) are
unaffected.
"""

import functools
import sys
import typing
from typing import TYPE_CHECKING, Any, Optional

from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

logger = init_logger(__name__)

_MODEL_LOADER_MODULE = "tpu_inference.models.common.model_loader"

# Capability flag: external model code (e.g. MaxText) can feature-detect
# these hooks with `getattr(hybrid, "SUPPORTS_JAX_HYBRID_MODELS", False)`.
SUPPORTS_JAX_HYBRID_MODELS = True

ATTENTION_LAYER_KIND = "attention"
MAMBA_LAYER_KIND = "mamba"

# Fallback HF `layer_types` spellings of recurrent (mamba-like) layers, used
# only if vLLM's `LayerBlockType` cannot be inspected. See
# `_hf_recurrent_layer_types`.
_FALLBACK_HF_RECURRENT_LAYER_TYPES = frozenset({"linear_attention", "mamba"})


@functools.lru_cache(maxsize=1)
def _hf_recurrent_layer_types() -> frozenset[str]:
    """HF `layer_types` spellings that denote recurrent (mamba) layers.

    These are the non-attention kinds of vLLM's `LayerBlockType`. Recurrent
    types are matched inclusively, and every other spelling is treated as
    attention: attention spellings keep growing (`sliding_attention`,
    `indexed_attention`, `deepseek_sparse_attention`,
    `qwen_sparse_attention`, ...), and an unknown one must not turn an
    attention layer into a mamba layer. An unknown recurrent spelling instead
    leaves the model with no mamba layers, which `get_kv_cache_layer_configs` rejects.
    """
    try:
        from vllm.config.model import LayerBlockType
        recurrent = frozenset(typing.get_args(LayerBlockType)) - {"attention"}
    except (ImportError, AttributeError):
        recurrent = frozenset()
    return recurrent or _FALLBACK_HF_RECURRENT_LAYER_TYPES


def get_jax_model_class(vllm_config: "VllmConfig") -> Optional[Any]:
    """Returns the JAX model class registered for the model's architecture.

    Only classes in `model_loader._MODEL_REGISTRY` are considered, which
    includes everything registered via `model_loader.register_model`
    (typically from a vLLM general plugin or before the engine is created,
    so before `VllmConfig` runs `TpuPlatform.check_and_update_config`).
    Built-in JAX models are added to the registry lazily on first model load,
    but none of them opt in to the hybrid hooks, so missing them here is
    harmless.

    `model_loader` is not imported here if nothing imported it yet: it pulls
    in JAX, flax and model code, and if it was never imported then
    `register_model` was never called, so no class can have opted in.

    Returns:
        The registered class, or None if there is no model config, no
        architecture, or no JAX class registered for it.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return None
    architecture = getattr(model_config, "architecture", None)
    if not architecture:
        return None
    model_loader = sys.modules.get(_MODEL_LOADER_MODULE)
    if model_loader is None:
        return None
    return model_loader._MODEL_REGISTRY.get(architecture)


def jax_model_is_hybrid(vllm_config: "VllmConfig") -> bool:
    """True iff the registered JAX model class opts in to hybrid handling.

    That is, the class defines `is_hybrid_config(hf_text_config)` and it
    returns True for `vllm_config.model_config.hf_text_config`.
    """
    model_cls = get_jax_model_class(vllm_config)
    is_hybrid_config = getattr(model_cls, "is_hybrid_config", None)
    if model_cls is None or not callable(is_hybrid_config):
        return False
    hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    if hf_text_config is None:
        return False
    return bool(is_hybrid_config(hf_text_config))


def maybe_apply_jax_hybrid_config(vllm_config: "VllmConfig") -> None:
    """Runs vLLM's hybrid config verifier for opted-in JAX hybrid models.

    vLLM only runs `HybridAttentionMambaModelConfig.verify_and_update_config`
    (which, e.g., sets `mamba_cache_mode="align"` and `mamba_block_size`
    under prefix caching) when `model_config.is_hybrid`, which is derived
    from the model class and is False for JAX model classes. This runs it for
    JAX models whose `is_hybrid_config` returns True. It is a no-op when vLLM
    already treats the model as hybrid, and running it more than once is
    safe (the verifier only fills in unset values).
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or getattr(model_config, "is_hybrid", False):
        return
    if not jax_model_is_hybrid(vllm_config):
        return
    from vllm.model_executor.models.config import \
        HybridAttentionMambaModelConfig
    logger.info(
        "JAX model class for %s declares a hybrid attention+mamba config; "
        "applying vLLM's hybrid model config defaults.",
        model_config.architecture)
    HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)


def get_jax_hybrid_layer_kinds(vllm_config: "VllmConfig",
                               num_layers: int) -> Optional[list[str]]:
    """Returns "attention" or "mamba" for each decoder layer.

    Uses the model class's `get_hybrid_layer_types(vllm_config)` if defined,
    otherwise `hf_text_config.layer_types` (recurrent spellings, see
    `_hf_recurrent_layer_types`, are mamba; everything else is attention).

    Args:
        vllm_config: The vLLM config.
        num_layers: Number of decoder layers that get a KV cache spec
            (`model_config.get_num_layers(parallel_config)`).

    Returns:
        A list of `num_layers` kinds, or None if the model does not opt in to
        hybrid handling.

    Raises:
        ValueError: if the model opts in but the layer kinds cannot be
            determined or do not cover `num_layers` layers.
    """
    if not jax_model_is_hybrid(vllm_config):
        return None
    model_cls = get_jax_model_class(vllm_config)
    get_layer_types = getattr(model_cls, "get_hybrid_layer_types", None)
    if callable(get_layer_types):
        kinds = list(get_layer_types(vllm_config))
        invalid = sorted(set(kinds) - {ATTENTION_LAYER_KIND, MAMBA_LAYER_KIND})
        if invalid:
            raise ValueError(
                f"{model_cls.__name__}.get_hybrid_layer_types returned "
                f"invalid layer kinds {invalid}; expected "
                f"'{ATTENTION_LAYER_KIND}' or '{MAMBA_LAYER_KIND}'.")
    else:
        layer_types = getattr(vllm_config.model_config.hf_text_config,
                              "layer_types", None)
        if layer_types is None:
            raise ValueError(
                f"{model_cls.__name__} declares a hybrid config but defines "
                "neither `get_hybrid_layer_types` nor "
                "`hf_text_config.layer_types`.")
        recurrent_types = _hf_recurrent_layer_types()
        kinds = [
            MAMBA_LAYER_KIND
            if layer_type in recurrent_types else ATTENTION_LAYER_KIND
            for layer_type in layer_types
        ]
    if len(kinds) < num_layers:
        raise ValueError(
            f"Hybrid layer kinds cover {len(kinds)} layers but the model has "
            f"{num_layers} decoder layers.")
    if len(kinds) > num_layers:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        get_start_end = getattr(vllm_config.model_config,
                                "get_layers_start_end_indices", None)
        if parallel_config is not None and callable(get_start_end):
            indices = get_start_end(parallel_config)
            if (isinstance(indices, tuple) and len(indices) == 2
                    and isinstance(indices[0], int)
                    and isinstance(indices[1], int)):
                start, end = indices
                if 0 <= start and end <= len(
                        kinds) and end - start == num_layers:
                    return kinds[start:end]
    return kinds[:num_layers]


def get_jax_mamba_type(
        vllm_config: "VllmConfig") -> "MambaAttentionBackendEnum":
    """Returns the recurrent backend type for an opted-in JAX hybrid model.

    Uses `model_cls.get_mamba_type_from_config(vllm_config)` if defined;
    otherwise returns `GDN_ATTN` when `hf_text_config` defines
    `linear_conv_kernel_dim` (e.g. Qwen3-Next / Qwen3.5 / Olmo-Hybrid) and
    `MAMBA2` otherwise.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    model_cls = get_jax_model_class(vllm_config)
    get_mamba_type = getattr(model_cls, "get_mamba_type_from_config", None)
    if callable(get_mamba_type):
        mamba_type = get_mamba_type(vllm_config)
        if isinstance(mamba_type, MambaAttentionBackendEnum):
            return mamba_type
        if (isinstance(mamba_type, str) and mamba_type.upper()
                in MambaAttentionBackendEnum.__members__):
            return MambaAttentionBackendEnum[mamba_type.upper()]
        raise ValueError(
            f"{model_cls.__name__}.get_mamba_type_from_config returned "
            f"invalid mamba_type {mamba_type!r}; expected "
            "MambaAttentionBackendEnum.")

    model_config = getattr(vllm_config, "model_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    if getattr(hf_text_config, "linear_conv_kernel_dim", None) is not None:
        return MambaAttentionBackendEnum.GDN_ATTN
    return MambaAttentionBackendEnum.MAMBA2
