# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Configuration shared by stacked RPA decode and prefill kernels."""

import dataclasses
import enum
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.stacked_rpa import utils


def accum_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Return the internal attention accumulator dtype for an output dtype."""
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.floating) and dtype.itemsize == 1:
        return jnp.bfloat16
    return dtype


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
    """Model config that will always stay constant."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    mask_value: float
    sm_scale: float = 1.0
    soft_cap: float | None = None
    sliding_window: int | None = None

    @property
    def num_q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
    """Serving config that can change depending on use cases."""

    num_seqs: int
    page_size: int
    total_q_tokens: int
    num_page_indices: int
    dtype_q: jnp.dtype
    dtype_kv: jnp.dtype
    dtype_out: jnp.dtype
    scale_q: int | None = None
    scale_k: int | None = None
    scale_v: int | None = None

    @property
    def pages_per_seq(self) -> int:
        return self.num_page_indices // self.num_seqs

    @property
    def page_size_log2(self) -> int:
        return (self.page_size - 1).bit_length()

    @property
    def page_size_mask(self) -> int:
        return self.page_size - 1

    @property
    def int_ty(self) -> jnp.dtype:
        if utils.get_dtype_packing(self.dtype_q) == 1:
            return jnp.int32

        # Absolute positions use signed arithmetic. Avoid int16 once a context
        # can reach 32768, where the attention mask would otherwise overflow.
        max_model_len = self.pages_per_seq * self.page_size
        if max_model_len > 32767:
            return jnp.int32

        match pltpu.get_tpu_info().generation:
            case 6 | 7:
                return jnp.int16
            case _:
                return jnp.int32

    @property
    def packing_q(self) -> int:
        return utils.get_dtype_packing(self.dtype_q)

    @property
    def packing_kv(self) -> int:
        return utils.get_dtype_packing(self.dtype_kv)


class RpaCase(enum.StrEnum):
    """Represents the different cases for Ragged Paged Attention.

    - DECODE: Sequences share a fixed decode q length, including spec decode.
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = enum.auto()
    PREFILL = enum.auto()
    MIXED = enum.auto()

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(
        self, distribution: jax.Array
    ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
        assert distribution.shape == (3, )
        match self:
            case RpaCase.DECODE:
                return 0, distribution[0]
            case RpaCase.PREFILL:
                return distribution[0], distribution[1]
            case RpaCase.MIXED:
                return distribution[1], distribution[2]


class AttentionConfig(Protocol):
    """Minimal configuration consumed by the shared FlashAttention math."""

    model: ModelConfigs
    serve: ServingConfigs
    aligned_num_q_heads_per_kv_head: int
