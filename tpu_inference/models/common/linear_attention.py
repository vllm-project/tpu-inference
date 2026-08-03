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
"""Recurrent-state layout for JAX-native hybrid (linear-attention) models.

The torchax path gets this from vLLM: every linear-attention layer is a
`MambaBase` torch module that reports its own `MambaSpec`. JAX-native models
have no such runtime module -- the layer list and the state shapes are only
knowable from the HF text config -- so the KV-cache manager derives them here,
the same way it derives cross-layer KV sharing in `kv_share.py`.

Returns `None` for every config that has no linear-attention layers, which is
every JAX-native model except the Kimi family today, so the non-hybrid path is
untouched.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Optional, Tuple

import torch
from vllm.utils.torch_utils import get_dtype_size

__all__ = [
    "LinearAttentionStateLayout",
    "linear_attn_config",
    "compute_linear_attention_layers",
    "compute_linear_attention_layout",
]


@dataclass(frozen=True)
class LinearAttentionStateLayout:
    """Which layers are recurrent, and the per-slot state they own.

    `shapes` / `dtypes` are exactly the `MambaSpec` fields, in the order the
    model's state NamedTuple declares them, so the arrays allocated from this
    layout can be handed to the layer positionally. Shapes are *global* (not
    divided by any parallelism degree): the JAX KV-cache arrays carry a
    `NamedSharding` that splits them across the mesh, matching how
    `get_attention_page_size_bytes` reports global per-block bytes for the
    attention layers.
    """
    layer_indices: frozenset[int]
    shapes: Tuple[Tuple[int, ...], ...]
    dtypes: Tuple[torch.dtype, ...]

    @property
    def page_size_bytes(self) -> int:
        """Unpadded bytes for one request's state in one layer."""
        return sum(
            prod(shape) * get_dtype_size(dtype)
            for shape, dtype in zip(self.shapes, self.dtypes))


def linear_attn_config(text_config: Any) -> Dict[str, Any]:
    """The `linear_attn_config` sub-config as a plain dict ({} when absent).

    HF ships it as a dict in `config.json`, but a config class that declares
    it as an attribute hands back an object instead.
    """
    cfg = getattr(text_config, "linear_attn_config", None)
    if not cfg:
        return {}
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    return cfg


def compute_linear_attention_layers(text_config: Any) -> frozenset:
    """0-indexed indices of the layers that carry recurrent state.

    Kimi (`Kimi-Linear-48B`, `Kimi-K3`) marks its KDA layers with a
    **1-indexed** `linear_attn_config.kda_layers` list; every other layer is
    full attention. Empty for models with no linear-attention layers.
    """
    kda_layers = linear_attn_config(text_config).get("kda_layers") or ()
    return frozenset(i - 1 for i in kda_layers)


def compute_linear_attention_layout(
        text_config: Any,
        model_dtype: torch.dtype) -> Optional[LinearAttentionStateLayout]:
    """Derive the recurrent-state layout, or None if the model has none.

    The state is four buffers per layer: three depthwise-conv windows (one
    each for q/k/v, `short_conv_kernel_size - 1` positions deep) plus the
    `[heads, head_dim, head_dim]` recurrent matrix — exactly what the KDA op
    reads at the start of a step and writes back at the end.

    The conv windows follow the model dtype; the recurrent state is always
    fp32 (it accumulates over the whole sequence). Same split vLLM applies in
    `MambaStateDtypeCalculator.kda_state_dtype`, which pairs a
    `mamba_cache_dtype`-driven conv dtype with an fp32 recurrent state --
    except that we keep the conv windows at the model dtype rather than
    exposing a separate cache dtype knob, because the KDA op reads them as
    activations.
    """
    layer_indices = compute_linear_attention_layers(text_config)
    if not layer_indices:
        return None

    cfg = linear_attn_config(text_config)
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    conv_kernel_size = cfg.get("short_conv_kernel_size", 4)

    conv_shape = (conv_kernel_size - 1, num_heads * head_dim)
    recurrent_shape = (num_heads, head_dim, head_dim)
    return LinearAttentionStateLayout(
        layer_indices=layer_indices,
        shapes=(conv_shape, conv_shape, conv_shape, recurrent_shape),
        dtypes=(model_dtype, model_dtype, model_dtype, torch.float32),
    )
