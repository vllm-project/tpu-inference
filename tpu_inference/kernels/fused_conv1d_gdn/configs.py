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
import enum
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class GDNMode(enum.StrEnum):
    DECODE = enum.auto()
    MIXED = enum.auto()


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
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    num_buffers: int = 2

    @property
    def tok_tile_size(self) -> int:
        if self.mode == GDNMode.DECODE:
            return 1
        return self.tile_size

    @property
    def seq_tile_size(self) -> int:
        if self.mode == GDNMode.DECODE:
            return self.tile_size
        return 1

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
    def aligned_num_v_heads(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

    def get_kernel_name(self) -> str:
        prefix = f"fused_conv1d_gdn_{self.mode.value}"
        return prefix

    def get_metadata(self) -> dict[str, str | int | float]:
        cfgs_dict = dataclasses.asdict(self)
        ret = {}
        for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if not isinstance(val, str | int | float):
                val = str(val)
            ret[key] = val
        return ret

    def get_out_shape(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (self.batch_size, self.num_v_heads, self.v_head_dim),
            self.dtypes.act,
        )

    def get_scratch_shape_dict(self) -> dict[str, Any]:
        act_prefix = (self.num_buffers, self.seq_tile_size, self.tok_tile_size,
                      1)
        state_prefix = (self.num_buffers, self.seq_tile_size)

        out_shape = (
            self.num_buffers,
            self.seq_tile_size,
            self.tok_tile_size,
            self.num_v_heads,
            self.v_head_dim,
        )
        conv_shape = (self.prev_kernel_size, 1, self.dim_size)
        recurrent_shape = (
            self.num_v_heads,
            self.kq_head_dim,
            self.v_head_dim,
        )

        qkv_scratch = pltpu.VMEM(act_prefix + (self.dim_size, ),
                                 self.dtypes.act)
        b_scratch = a_scratch = pltpu.VMEM(
            act_prefix + (self.aligned_num_v_heads, ), self.dtypes.act)
        out_scratch = pltpu.VMEM(out_shape, self.dtypes.act)
        conv_state_scratch = pltpu.VMEM(state_prefix + conv_shape,
                                        self.dtypes.conv_state)
        recurrent_state_scratch = pltpu.VMEM(state_prefix + recurrent_shape,
                                             self.dtypes.recurrent_state)

        # On decode, a token has no dependency with previous tile.
        prev_qkv_scratch = prev_recurrent_state_scratch = None
        if self.mode != GDNMode.DECODE:
            prev_qkv_scratch = pltpu.VMEM((self.seq_tile_size, ) + conv_shape,
                                          jnp.float32)
            prev_recurrent_state_scratch = pltpu.VMEM(
                (self.seq_tile_size, ) + recurrent_shape, self.dtypes.compute)

        return dict(
            qkv_scratch_ref=qkv_scratch,
            b_scratch_ref=b_scratch,
            a_scratch_ref=a_scratch,
            out_scratch_ref=out_scratch,
            conv_state_scratch_ref=conv_state_scratch,
            recurrent_state_scratch_ref=recurrent_state_scratch,
            prev_qkv_scratch_ref=prev_qkv_scratch,
            prev_recurrent_state_scratch_ref=prev_recurrent_state_scratch,
            sem_ref=pltpu.SemaphoreType.DMA((self.num_buffers, )),
        )
