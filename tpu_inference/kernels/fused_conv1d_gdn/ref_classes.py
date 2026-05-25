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
class MetadataRef:
    num_tiles: Any
    b_idx_to_s_idx: Any
    b_idx_to_sz_from_old: Any
    b_idx_should_write: Any
    s_idx_to_state_idx: Any
    s_idx_has_initial_state: Any

    def __len__(self):
        return len(dataclasses.fields(self))

    def get_spec(self):
        return jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.SMEM),
                            self)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferedRef(ABC):

    @abstractmethod
    def copy_in(self, b_start, slot, sem):
        ...

    @abstractmethod
    def wait_in(self, b_start, slot, sem):
        ...

    @abstractmethod
    def copy_out(self, b_start, slot, sem):
        ...

    @abstractmethod
    def wait_out(self, b_start, slot, sem):
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QKVBufferedRef(BufferedRefWrapper):

    def copy_in(self, b_start, slot, sem):
        num_tiles = self.metadata_ref.num_tiles[...]
        last_b_start = (num_tiles - 1) * self.cfgs.tile_size
        is_no_op = jnp.where(b_start > last_b_start, True, False)

        packing = self.cfgs.act_packing
        packed_dma_size = jnp.where(is_no_op, 0,
                                    self.cfgs.tile_size // packing)
        packed_b_start = b_start // packing

        pltpu.make_async_copy(
            self.hbm_ref.at[pl.ds(packed_b_start, packed_dma_size)],
            self.get_slot_vmem(slot).at[pl.ds(0, packed_dma_size)],
            sem,
        ).start()

    def wait_in(self, b_start, slot, sem):
        pltpu.make_async_copy(
            self.vmem_ref.at[0],
            self.vmem_ref.at[0],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem):
        raise NotImplementedError()

    def wait_out(self, b_start, slot, sem):
        raise NotImplementedError()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BBufferedRef(BufferedRefWrapper):

    def copy_in(self, b_start, slot, sem):
        num_tiles = self.metadata_ref.num_tiles[...]
        last_b_start = (num_tiles - 1) * self.cfgs.tile_size
        is_no_op = jnp.where(b_start > last_b_start, True, False)

        packing = self.cfgs.act_packing
        packed_dma_size = jnp.where(is_no_op, 0,
                                    self.cfgs.tile_size // packing)
        packed_b_start = b_start // packing

        pltpu.make_async_copy(
            self.hbm_ref.at[pl.ds(packed_b_start, packed_dma_size)],
            self.get_slot_vmem(slot).at[pl.ds(0, packed_dma_size)],
            sem,
        ).start()

    def wait_in(self, b_start, slot, sem):
        pltpu.make_async_copy(
            self.vmem_ref.at[0],
            self.vmem_ref.at[0],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem):
        raise NotImplementedError()

    def wait_out(self, b_start, slot, sem):
        raise NotImplementedError()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ABufferedRef(BufferedRefWrapper):

    def copy_in(self, b_start, slot, sem):
        num_tiles = self.metadata_ref.num_tiles[...]
        last_b_start = (num_tiles - 1) * self.cfgs.tile_size
        is_no_op = jnp.where(b_start > last_b_start, True, False)

        packing = self.cfgs.act_packing
        packed_dma_size = jnp.where(is_no_op, 0,
                                    self.cfgs.tile_size // packing)
        packed_b_start = b_start // packing

        pltpu.make_async_copy(
            self.hbm_ref.at[pl.ds(packed_b_start, packed_dma_size)],
            self.get_slot_vmem(slot).at[pl.ds(0, packed_dma_size)],
            sem,
        ).start()

    def wait_in(self, b_start, slot, sem):
        pltpu.make_async_copy(
            self.vmem_ref.at[0],
            self.vmem_ref.at[0],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem):
        raise NotImplementedError()

    def wait_out(self, b_start, slot, sem):
        raise NotImplementedError()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OutBufferedRef(BufferedRefWrapper):

    def copy_in(self, b_start, slot, sem):
        raise NotImplementedError()

    def wait_in(self, b_start, slot, sem):
        raise NotImplementedError()

    def copy_out(self, b_start, slot, sem):

        pltpu.make_async_copy(
            self.get_slot_vmem(slot),
            self.hbm_ref.at[pl.ds(b_start, self.cfgs.tile_size)],
            sem,
        ).start(1)

    def wait_out(self, b_start, slot, sem):
        is_no_op = jnp.where(b_start < 0, True, False)
        dma_size = jnp.where(is_no_op, 0, self.cfgs.tile_size)

        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ActivationBufferedRefs(BufferedRef):
    qkv: QKVBufferedRef
    b: BBufferedRef
    a: ABufferedRef
    out: OutBufferedRef

    def copy_in(self, b_start, slot, sem):
        self.qkv.copy_in(b_start, slot, sem)
        self.b.copy_in(b_start, slot, sem)
        self.a.copy_in(b_start, slot, sem)

    def wait_in(self, b_start, slot, sem):
        self.qkv.wait_in(b_start, slot, sem)
        self.b.wait_in(b_start, slot, sem)
        self.a.wait_in(b_start, slot, sem)

    def copy_out(self, b_start, slot, sem):
        self.out.copy_out(b_start, slot, sem)

    def wait_out(self, b_start, slot, sem):
        self.out.wait_out(b_start, slot, sem)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvStateBufferedRef(BufferedRefWrapper):

    def copy_in(self, b_start, slot, sem):
        num_tiles = self.metadata_ref.num_tiles[...]
        last_b_start = (num_tiles - 1) * self.cfgs.tile_size
        is_no_op = jnp.where(b_start > last_b_start, True, False)

        b_start = jnp.where(is_no_op, 0, b_start)

        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
            state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
            sz_from_old = self.metadata_ref.b_idx_to_sz_from_old[b_idx]
            start_from_old = self.cfgs.prev_kernel_size - sz_from_old
            sz_from_old = jnp.where(is_no_op, 0, sz_from_old)

            pltpu.make_async_copy(
                self.hbm_ref.at[state_idx,
                                pl.ds(start_from_old, sz_from_old)],
                self.get_slot_vmem(slot).at[idx, pl.ds(0, sz_from_old)],
                sem,
            ).start()

    def wait_in(self, b_start, slot, sem):
        all_sz_from_old = 0
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            all_sz_from_old += self.metadata_ref.b_idx_to_sz_from_old[b_idx]

        pltpu.make_async_copy(
            self.vmem_ref.at[0, 0, pl.ds(0, all_sz_from_old)],
            self.vmem_ref.at[0, 0, pl.ds(0, all_sz_from_old)],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem):
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
            state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
            should_write = self.metadata_ref.b_idx_should_write[b_idx]

            pltpu.make_async_copy(
                self.get_slot_vmem(slot).at[pl.ds(idx, should_write)],
                self.hbm_ref.at[pl.ds(state_idx, should_write)],
                sem,
            ).start()

    def wait_out(self, b_start, slot, sem):
        is_no_op = jnp.where(b_start < 0, True, False)
        b_start = jnp.where(is_no_op, 0, b_start)

        all_writes = 0
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            all_writes += self.metadata_ref.b_idx_should_write[b_idx]

        all_writes = jnp.where(is_no_op, 0, all_writes)
        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, all_writes)],
            self.vmem_ref.at[0, pl.ds(0, all_writes)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RecurrentStateBufferedRef(BufferedRefWrapper):

    def copy_in(self, b_start, slot, sem):
        num_tiles = self.metadata_ref.num_tiles[...]
        last_b_start = (num_tiles - 1) * self.cfgs.tile_size
        is_no_op = jnp.where(b_start > last_b_start, True, False)

        b_start = jnp.where(is_no_op, 0, b_start)

        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
            state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
            sz_from_old = self.metadata_ref.b_idx_to_sz_from_old[b_idx]
            should_not_read = jnp.where(
                sz_from_old != self.cfgs.prev_kernel_size, True, is_no_op)
            dma_size = jnp.where(should_not_read, 0, 1)

            pltpu.make_async_copy(
                self.hbm_ref.at[pl.ds(state_idx, dma_size)],
                self.get_slot_vmem(slot).at[pl.ds(idx, dma_size)],
                sem,
            ).start()

    def wait_in(self, b_start, slot, sem):
        all_reads = 0
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            sz_from_old = self.metadata_ref.b_idx_to_sz_from_old[b_idx]
            dma_sz = jnp.where(sz_from_old != self.cfgs.prev_kernel_size, 0, 1)
            all_reads += dma_sz

        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, all_reads)],
            self.vmem_ref.at[0, pl.ds(0, all_reads)],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem):
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
            state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
            should_write = self.metadata_ref.b_idx_should_write[b_idx]

            pltpu.make_async_copy(
                self.get_slot_vmem(slot).at[pl.ds(idx, should_write)],
                self.hbm_ref.at[pl.ds(state_idx, should_write)],
                sem,
            ).start()

    def wait_out(self, b_start, slot, sem):
        is_no_op = jnp.where(b_start < 0, True, False)
        b_start = jnp.where(is_no_op, 0, b_start)

        all_writes = 0
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            all_writes += self.metadata_ref.b_idx_should_write[b_idx]

        all_writes = jnp.where(is_no_op, 0, all_writes)
        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, all_writes)],
            self.vmem_ref.at[0, pl.ds(0, all_writes)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StateBufferedRefs(BufferedRef):
    conv: ConvStateBufferedRef
    recurrent: RecurrentStateBufferedRef

    def copy_in(self, b_start, slot, sem):
        self.conv.copy_in(b_start, slot, sem)
        self.recurrent.copy_in(b_start, slot, sem)

    def wait_in(self, b_start, slot, sem):
        self.conv.wait_in(b_start, slot, sem)
        self.recurrent.wait_in(b_start, slot, sem)

    def copy_out(self, b_start, slot, sem):
        self.conv.copy_out(b_start, slot, sem)
        self.recurrent.copy_out(b_start, slot, sem)

    def wait_out(self, b_start, slot, sem):
        self.conv.wait_out(b_start, slot, sem)
        self.recurrent.wait_out(b_start, slot, sem)
