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
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_conv1d_gdn import configs


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvWeightsRef:
    weight: Any
    bias: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNWeightsRef:
    a_log: Any
    dt_bias: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightRefs:
    conv: ConvWeightsRef
    gdn: GDNWeightsRef


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OptionalSmemWrapper:
    """Wraps either dynamic or static int value to allow constant folding."""

    data: Any | None = None
    data_fn: Any = dataclasses.field(default=None, metadata=dict(static=True))

    def __post_init__(self):
        assert (self.data is not None) ^ (
            self.data_fn
            is not None), "Must only provide either data or data_fn"

    def __getitem__(self, indices):
        if self.data is None:
            return self.data_fn(indices)
        return self.data[indices]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    num_tiles: Any
    start_st_idx: OptionalSmemWrapper
    st_idx_to_t_idx: OptionalSmemWrapper
    st_idx_to_s_idx: OptionalSmemWrapper
    st_idx_to_b_idx: OptionalSmemWrapper
    st_idx_to_b_size: OptionalSmemWrapper
    st_idx_is_first_tile: Any
    st_idx_is_last_tile: Any
    s_idx_to_num_tiles: Any
    s_idx_has_initial_state: Any
    s_idx_to_state_indices: Any

    @staticmethod
    def _wrap_optional_smem(x: Any) -> OptionalSmemWrapper:
        match x:
            case Callable():
                return OptionalSmemWrapper(data_fn=x)
            case jax.Array():
                return OptionalSmemWrapper(data=x)
            case _:
                raise TypeError(
                    f"Expected Callable or jax.Array, got {type(x)}")

    @classmethod
    def create(
        cls,
        num_tiles: jax.Array,
        start_st_idx: Callable[..., Any] | jax.Array,
        st_idx_to_t_idx: Callable[..., Any] | jax.Array,
        st_idx_to_s_idx: Callable[..., Any] | jax.Array,
        st_idx_to_b_idx: Callable[..., Any] | jax.Array,
        st_idx_to_b_size: Callable[..., Any] | jax.Array,
        st_idx_is_first_tile: jax.Array,
        st_idx_is_last_tile: jax.Array,
        s_idx_to_num_tiles: jax.Array,
        s_idx_has_initial_state: jax.Array,
        s_idx_to_state_indices: jax.Array,
    ):
        return cls(
            num_tiles=num_tiles,
            start_st_idx=cls._wrap_optional_smem(start_st_idx),
            st_idx_to_t_idx=cls._wrap_optional_smem(st_idx_to_t_idx),
            st_idx_to_s_idx=cls._wrap_optional_smem(st_idx_to_s_idx),
            st_idx_to_b_idx=cls._wrap_optional_smem(st_idx_to_b_idx),
            st_idx_to_b_size=cls._wrap_optional_smem(st_idx_to_b_size),
            st_idx_is_first_tile=st_idx_is_first_tile,
            st_idx_is_last_tile=st_idx_is_last_tile,
            s_idx_to_num_tiles=s_idx_to_num_tiles,
            s_idx_has_initial_state=s_idx_has_initial_state,
            s_idx_to_state_indices=s_idx_to_state_indices,
        )

    def __len__(self) -> int:
        return len(jax.tree_util.tree_leaves(self))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef(ABC):

    @abstractmethod
    def copy_in(self, st_idx, slot, sem):
        ...

    @abstractmethod
    def wait_in(self, st_idx, slot, sem):
        ...

    @abstractmethod
    def copy_out(self, st_idx, slot, sem):
        ...

    @abstractmethod
    def wait_out(self, st_idx, slot, sem):
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRefWrapper(BufferedRef):
    hbm_ref: Any
    vmem_ref: Any
    metadata_ref: MetadataRef
    cfgs: configs.GDNConfigs

    def get_slot_vmem(self, slot):
        return self.vmem_ref.at[slot]

    def is_lower_oob(self, st_idx) -> jax.Array:
        start_st_idx = self.metadata_ref.start_st_idx[...]
        return st_idx < start_st_idx

    def is_upper_oob(self, st_idx) -> jax.Array:
        num_tiles = self.metadata_ref.num_tiles[...]
        return st_idx >= num_tiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InBufferedRef(BufferedRefWrapper):

    def copy_in(self, st_idx, slot, sem):
        is_no_op = self.is_upper_oob(st_idx)
        st_idx = jnp.where(is_no_op, 0, st_idx)

        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            b_idx = self.metadata_ref.st_idx_to_b_idx[curr_st_idx]
            dma_size = self.metadata_ref.st_idx_to_b_size[curr_st_idx]
            dma_size = jnp.where(is_no_op, 0, dma_size)
            pltpu.make_async_copy(
                self.hbm_ref.at[pl.ds(b_idx, dma_size)],
                self.get_slot_vmem(slot).at[idx, pl.ds(0, dma_size)],
                sem,
            ).start()

    def wait_in(self, st_idx, slot, sem):
        dma_size = 0
        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            dma_size += self.metadata_ref.st_idx_to_b_size[curr_st_idx]

        pltpu.make_async_copy(
            self.vmem_ref.at[0, 0, pl.ds(0, dma_size)],
            self.vmem_ref.at[0, 0, pl.ds(0, dma_size)],
            sem,
        ).wait()

    def copy_out(self, st_idx, slot, sem):
        raise NotImplementedError()

    def wait_out(self, st_idx, slot, sem):
        raise NotImplementedError()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OutBufferedRef(BufferedRefWrapper):

    def copy_in(self, st_idx, slot, sem):
        raise NotImplementedError()

    def wait_in(self, st_idx, slot, sem):
        raise NotImplementedError()

    def copy_out(self, st_idx, slot, sem):
        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            b_idx = self.metadata_ref.st_idx_to_b_idx[curr_st_idx]
            dma_size = self.metadata_ref.st_idx_to_b_size[curr_st_idx]
            pltpu.make_async_copy(
                self.get_slot_vmem(slot).at[idx, pl.ds(0, dma_size)],
                self.hbm_ref.at[pl.ds(b_idx, dma_size)],
                sem,
            ).start()

    def wait_out(self, st_idx, slot, sem):
        is_no_op = self.is_lower_oob(st_idx)
        st_idx = jnp.where(is_no_op, 0, st_idx)

        dma_size = 0
        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            dma_size += self.metadata_ref.st_idx_to_b_size[curr_st_idx]
        dma_size = jnp.where(is_no_op, 0, dma_size)

        pltpu.make_async_copy(
            self.vmem_ref.at[0, 0, pl.ds(0, dma_size)],
            self.vmem_ref.at[0, 0, pl.ds(0, dma_size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ActivationBufferedRefs(BufferedRef):
    qkv: InBufferedRef
    b: InBufferedRef
    a: InBufferedRef
    out: OutBufferedRef

    def copy_in(self, st_idx, slot, sem):
        self.qkv.copy_in(st_idx, slot, sem)
        self.b.copy_in(st_idx, slot, sem)
        self.a.copy_in(st_idx, slot, sem)

    def wait_in(self, st_idx, slot, sem):
        self.qkv.wait_in(st_idx, slot, sem)
        self.b.wait_in(st_idx, slot, sem)
        self.a.wait_in(st_idx, slot, sem)

    def copy_out(self, st_idx, slot, sem):
        self.out.copy_out(st_idx, slot, sem)

    def wait_out(self, st_idx, slot, sem):
        self.out.wait_out(st_idx, slot, sem)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SharedStateBufferedRef(BufferedRefWrapper):

    def copy_in(self, st_idx, slot, sem):
        is_no_op = self.is_upper_oob(st_idx)
        st_idx = jnp.where(is_no_op, 0, st_idx)

        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            s_idx = self.metadata_ref.st_idx_to_s_idx[curr_st_idx]
            state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]

            is_first_tile = self.metadata_ref.st_idx_is_first_tile[curr_st_idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[
                s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            should_read = jnp.logical_and(should_read,
                                          jnp.logical_not(is_no_op))
            dma_size = jnp.where(should_read, 1, 0)
            pltpu.make_async_copy(
                self.hbm_ref.at[pl.ds(state_idx, dma_size)],
                self.get_slot_vmem(slot).at[pl.ds(idx, dma_size)],
                sem,
            ).start()

    def wait_in(self, st_idx, slot, sem):
        dma_size = 0
        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            s_idx = self.metadata_ref.st_idx_to_s_idx[curr_st_idx]

            is_first_tile = self.metadata_ref.st_idx_is_first_tile[curr_st_idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[
                s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            dma_size += jnp.where(should_read, 1, 0)

        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            sem,
        ).wait()

    def copy_out(self, st_idx, slot, sem):

        for idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + idx
            s_idx = self.metadata_ref.st_idx_to_s_idx[curr_st_idx]

            is_last_tile = self.metadata_ref.st_idx_is_last_tile[curr_st_idx]
            state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
            dma_size = jnp.where(is_last_tile, 1, 0)

            pltpu.make_async_copy(
                self.get_slot_vmem(slot).at[pl.ds(idx, dma_size)],
                self.hbm_ref.at[pl.ds(state_idx, dma_size)],
                sem,
            ).start()

    def wait_out(self, st_idx, slot, sem):
        is_no_op = self.is_lower_oob(st_idx)
        st_idx = jnp.where(is_no_op, 0, st_idx)

        dma_size = 0
        for s_idx in range(self.cfgs.seq_tile_size):
            curr_st_idx = st_idx * self.cfgs.seq_tile_size + s_idx
            is_last_tile = self.metadata_ref.st_idx_is_last_tile[curr_st_idx]
            dma_size += jnp.where(is_last_tile, 1, 0)
        dma_size = jnp.where(is_no_op, 0, dma_size)

        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StateBufferedRefs(BufferedRef):
    conv: SharedStateBufferedRef
    recurrent: SharedStateBufferedRef

    def copy_in(self, st_idx, slot, sem):
        self.conv.copy_in(st_idx, slot, sem)
        self.recurrent.copy_in(st_idx, slot, sem)

    def wait_in(self, st_idx, slot, sem):
        self.conv.wait_in(st_idx, slot, sem)
        self.recurrent.wait_in(st_idx, slot, sem)

    def copy_out(self, st_idx, slot, sem):
        self.conv.copy_out(st_idx, slot, sem)
        self.recurrent.copy_out(st_idx, slot, sem)

    def wait_out(self, st_idx, slot, sem):
        self.conv.wait_out(st_idx, slot, sem)
        self.recurrent.wait_out(st_idx, slot, sem)
