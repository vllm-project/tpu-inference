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

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfigs:
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    dtypes: Dtypes
    # TODO(kyuyeunk): Find good default value.
    head_tile_size: int = 16
    buffer_size: int = 2

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def v_dim_size(self) -> int:
        return self.num_v_heads * self.v_head_dim

    @property
    def kq_dim_size(self) -> int:
        return self.num_kq_heads * self.kq_head_dim

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def act_dtype(self) -> jnp.dtype:
        return self.dtypes.act

    @property
    def act_packing(self) -> int:
        return 4 // self.dtypes.act.itemsize

    @property
    def padded_batch_size(self) -> int:
        return pl.cdiv(self.batch_size, self.tile_size) * self.tile_size

    def get_out_shape(self) -> tuple[int, ...]:
        return (self.padded_batch_size, self.num_v_heads, self.v_head_dim)

    def get_scratch_shape_dict(self) -> dict[str, Any]:
        buffer_shape = (self.buffer_size, self.tile_size)
        buffer_act_shape = (
            self.buffer_size,
            self.tile_size // self.act_packing,
            self.act_packing,
        )
        buffer_out_shape = (
            self.buffer_size,
            self.tile_size,
            self.num_v_heads,
            self.v_head_dim,
        )
        scratch_conv_shape = (self.prev_kernel_size, 1, self.dim_size)
        buffer_conv_shape = buffer_shape + scratch_conv_shape
        scratch_recurrent_shape = (
            self.num_v_heads,
            self.kq_head_dim,
            self.v_head_dim,
        )
        buffer_recurrent_shape = buffer_shape + scratch_recurrent_shape

        return dict(
            qkv_scratch_ref=pltpu.VMEM(buffer_act_shape + (self.dim_size, ),
                                       self.act_dtype),
            b_scratch_ref=pltpu.VMEM(buffer_act_shape + (self.num_v_heads, ),
                                     self.act_dtype),
            a_scratch_ref=pltpu.VMEM(buffer_act_shape + (self.num_v_heads, ),
                                     self.act_dtype),
            out_scratch_ref=pltpu.VMEM(buffer_out_shape, self.act_dtype),
            conv_state_scratch_ref=pltpu.VMEM(buffer_conv_shape, jnp.float32),
            recurrent_state_scratch_ref=pltpu.VMEM(
                buffer_recurrent_shape, self.dtypes.recurrent_state),
            prev_qkv_scratch_ref=pltpu.VMEM(scratch_conv_shape, jnp.float32),
            prev_recurrent_state_scratch_ref=pltpu.VMEM(
                scratch_recurrent_shape, self.dtypes.compute),
            sem_ref=pltpu.SemaphoreType.DMA((self.buffer_size, )),
        )
