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
import functools
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvConfigs:
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int

    @property
    def padding_size(self) -> int:
        return self.kernel_size - 1

    @property
    def padded_tile_size(self) -> int:
        return self.tile_size + self.padding_size

    @property
    def num_tiles(self) -> int:
        return pl.cdiv(self.batch_size, self.tile_size)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvRhsRef:
    weight: Any
    bias: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    b_idx_to_s_idx: Any
    b_idx_to_sz_from_old: Any
    b_idx_should_write: Any
    s_idx_to_state_idx: Any
    s_idx_has_initial_state: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferWrapper(ABC):
    hbm_ref: Any
    vmem_ref: Any
    metadata_ref: MetadataRef
    cfgs: ConvConfigs

    def get_slot_vmem(self, slot):
        return self.vmem_ref.at[slot]

    @abstractmethod
    def copy_in(self, b_start, slot, sem, is_first=False):
        ...

    @abstractmethod
    def wait_in(self, b_start, slot, sem, is_first=False):
        ...

    @abstractmethod
    def copy_out(self, b_start, slot, sem, is_last=False):
        ...

    @abstractmethod
    def wait_out(self, b_start, slot, sem, is_last=False):
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class XBuffer(BufferWrapper):

    def copy_in(self, b_start, slot, sem, is_first=False):
        if is_first:
            dma = pltpu.make_async_copy(
                self.hbm_ref.at[:self.cfgs.tile_size],
                self.get_slot_vmem(slot).at[self.cfgs.padding_size:],
                sem,
            )
        else:
            next_dma_start = b_start - self.cfgs.padding_size
            dma = pltpu.make_async_copy(
                self.hbm_ref.at[pl.ds(next_dma_start,
                                      self.cfgs.padded_tile_size)],
                self.get_slot_vmem(slot),
                sem,
            )
        dma.start()
        return [dma]

    def wait_in(self, b_start, slot, sem, is_first=False):
        pltpu.make_async_copy(
            self.vmem_ref.at[0],
            self.vmem_ref.at[0],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem, is_last=False):
        dma = pltpu.make_async_copy(
            self.get_slot_vmem(slot).at[:self.cfgs.tile_size],
            self.hbm_ref.at[pl.ds(b_start, self.cfgs.tile_size)],
            sem,
        )
        dma.start()
        return [dma]

    def wait_out(self, b_start, slot, sem, is_last=False):
        pltpu.make_async_copy(
            self.vmem_ref.at[0, :self.cfgs.tile_size],
            self.vmem_ref.at[0, :self.cfgs.tile_size],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvStateBuffer(BufferWrapper):

    def copy_in(self, b_start, slot, sem, is_first=False):
        dma_list = []
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
            state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
            sz_from_old = self.metadata_ref.b_idx_to_sz_from_old[b_idx]
            start_from_old = (self.cfgs.kernel_size - 1) - sz_from_old

            dma = pltpu.make_async_copy(
                self.hbm_ref.at[state_idx,
                                pl.ds(start_from_old, sz_from_old)],
                self.get_slot_vmem(slot).at[idx, pl.ds(0, sz_from_old)],
                sem,
            )
            dma_list.append(dma)
        jax.tree.map(lambda dma: dma.start(), dma_list)
        return dma_list

    def wait_in(self, b_start, slot, sem, is_first=False):
        all_sz_from_old = 0
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            all_sz_from_old += self.metadata_ref.b_idx_to_sz_from_old[b_idx]

        pltpu.make_async_copy(
            self.vmem_ref.at[0, 0, pl.ds(0, all_sz_from_old)],
            self.vmem_ref.at[0, 0, pl.ds(0, all_sz_from_old)],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem, is_last=False):
        dma_list = []
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            s_idx = self.metadata_ref.b_idx_to_s_idx[b_idx]
            state_idx = self.metadata_ref.s_idx_to_state_idx[s_idx]
            should_write = self.metadata_ref.b_idx_should_write[b_idx]

            dma = pltpu.make_async_copy(
                self.get_slot_vmem(slot).at[pl.ds(idx, should_write)],
                self.hbm_ref.at[pl.ds(state_idx, should_write)],
                sem,
            )
            dma_list.append(dma)
        jax.tree.map(lambda dma: dma.start(), dma_list)
        return dma_list

    def wait_out(self, b_start, slot, sem, is_last=False):
        all_writes = 0
        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            all_writes += self.metadata_ref.b_idx_should_write[b_idx]

        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, all_writes)],
            self.vmem_ref.at[0, pl.ds(0, all_writes)],
            sem,
        ).wait()


def inner_kernel(
    p_id: jax.Array,
    *,
    x_buffer: XBuffer,
    conv_state_buffer: ConvStateBuffer,
    sem_ref: jax.Array,
    metadata_ref: MetadataRef,
    conv_rhs_ref: ConvRhsRef,
    cfgs: ConvConfigs,
):
    b_start = p_id * cfgs.tile_size
    prev_b_start = b_start - cfgs.tile_size
    next_b_start = b_start + cfgs.tile_size

    recv_sem = sem_ref.at[0]
    send_sem = sem_ref.at[1]

    slot = p_id % 2
    other_slot = (slot + 1) % 2

    x_slot_ref = x_buffer.get_slot_vmem(slot)
    conv_state_slot_ref = conv_state_buffer.get_slot_vmem(slot)

    def _conv1d(is_first: bool, is_last: bool):

        # Prologue.
        if is_first:
            dma_list = []
            dma_list += x_buffer.copy_in(
                b_start,
                slot,
                recv_sem,
                is_first=True,
            )
            dma_list += conv_state_buffer.copy_in(
                b_start,
                slot,
                recv_sem,
                is_first=True,
            )
            jax.tree.map(lambda dma: dma.wait(), dma_list)
        else:
            x_buffer.wait_in(b_start, slot, recv_sem)
            conv_state_buffer.wait_in(b_start, slot, recv_sem)

        if not is_first:
            x_buffer.wait_out(prev_b_start, other_slot, send_sem)
            conv_state_buffer.wait_out(prev_b_start, other_slot, send_sem)

        if not is_last:
            x_buffer.copy_in(next_b_start, other_slot, recv_sem)
            conv_state_buffer.copy_in(next_b_start, other_slot, recv_sem)

        # Body.
        out_list = []
        for idx in range(cfgs.tile_size):
            b_idx = b_start + idx

            s_idx = metadata_ref.b_idx_to_s_idx[b_idx]
            sz_from_old = metadata_ref.b_idx_to_sz_from_old[b_idx]
            has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

            out = jnp.zeros((1, cfgs.dim_size), jnp.float32)

            end_idx = idx + cfgs.padding_size
            start_idx = 1 + end_idx - cfgs.kernel_size
            for k in range(cfgs.kernel_size):
                lhs = x_slot_ref[start_idx + k]

                if k < cfgs.kernel_size - 1:
                    conv_state = conv_state_slot_ref[idx, k]
                    conv_state = jnp.where(has_initial_state, conv_state, 0)
                    lhs = jnp.where(k < sz_from_old, conv_state, lhs)

                if k > 0:
                    conv_state_slot_ref[idx, k - 1] = lhs

                rhs = conv_rhs_ref.weight[k]
                out += lhs * rhs

            if conv_rhs_ref.bias is not None:
                bias = conv_rhs_ref.bias[...].reshape(1, -1)
                out += bias
            out_list.append(out)

        for idx in range(cfgs.tile_size):
            x_slot_ref[idx] = out_list[idx]

        # Epilogue.
        dma_list = []
        dma_list += x_buffer.copy_out(b_start, slot, send_sem)
        dma_list += conv_state_buffer.copy_out(b_start, slot, send_sem)

        if is_last:
            jax.tree.map(lambda dma: dma.wait(), dma_list)

    @jax.named_scope("conv1d_first_last")
    def conv1d_first_last():
        _conv1d(is_first=True, is_last=True)

    @jax.named_scope("conv1d_first")
    def conv1d_first():
        _conv1d(is_first=True, is_last=False)

    @jax.named_scope("conv1d_last")
    def conv1d_last():
        _conv1d(is_first=False, is_last=True)

    @jax.named_scope("conv1d")
    def conv1d():
        _conv1d(is_first=False, is_last=False)

    is_first = p_id == 0
    is_last = p_id == (cfgs.num_tiles - 1)

    jax.lax.cond(
        is_first,
        lambda: jax.lax.cond(is_last, conv1d_first_last, conv1d_first),
        lambda: jax.lax.cond(is_last, conv1d_last, conv1d),
    )


def main_kernel(
    # Inputs.
    metadata_ref: MetadataRef,
    x_ref: jax.Array,
    conv_state_ref: jax.Array,
    conv_rhs_ref: ConvRhsRef,
    # Outputs.
    x_out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    # Scratch
    x_scratch_ref: jax.Array,
    conv_state_scratch_ref: jax.Array,
    sem_ref: jax.Array,
    *,
    cfgs: ConvConfigs,
):
    del x_out_ref, conv_state_out_ref

    x_buffer = XBuffer(
        hbm_ref=x_ref,
        vmem_ref=x_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )
    conv_state_buffer = ConvStateBuffer(
        hbm_ref=conv_state_ref,
        vmem_ref=conv_state_scratch_ref,
        metadata_ref=metadata_ref,
        cfgs=cfgs,
    )

    @pl.loop(0, cfgs.num_tiles)
    def loop_wrapper(p_id):
        inner_kernel(
            p_id=p_id,
            x_buffer=x_buffer,
            conv_state_buffer=conv_state_buffer,
            sem_ref=sem_ref,
            metadata_ref=metadata_ref,
            conv_rhs_ref=conv_rhs_ref,
            cfgs=cfgs,
        )


def preprocess_metadata(
    cfgs: ConvConfigs,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    has_initial_state: jax.Array,
    max_seqs: jax.Array,
) -> MetadataRef:
    """Preprocesses metadata for the convolution kernel."""
    num_seqs = state_indices.size

    # Mask out padded locations.
    max_token = query_start_loc[max_seqs]
    all_seqs = jnp.arange(num_seqs + 1)
    query_start_loc = jnp.where(all_seqs <= max_seqs, query_start_loc,
                                max_token)

    # Map batch index to sequence index.
    query_len = query_start_loc[1:] - query_start_loc[:-1]
    seqs = jnp.arange(num_seqs)
    b_idx_to_s_idx = jnp.repeat(seqs,
                                query_len,
                                total_repeat_length=cfgs.batch_size)
    b_idx_query_start_loc = query_start_loc[b_idx_to_s_idx]
    all_b_idx = jnp.arange(cfgs.batch_size)
    b_idx_query_len = 1 + all_b_idx - b_idx_query_start_loc

    b_idx_to_sz_from_new = jnp.minimum(b_idx_query_len, cfgs.kernel_size)
    b_idx_to_sz_from_old = cfgs.kernel_size - b_idx_to_sz_from_new
    b_idx_to_sz_from_old = jnp.minimum(b_idx_to_sz_from_old,
                                       cfgs.kernel_size - 1)

    b_idx_should_write = all_b_idx == (query_start_loc[b_idx_to_s_idx + 1] - 1)
    b_idx_should_write = b_idx_should_write.astype(jnp.int32)

    return MetadataRef(
        b_idx_to_s_idx=b_idx_to_s_idx,
        b_idx_to_sz_from_old=b_idx_to_sz_from_old,
        b_idx_should_write=b_idx_should_write,
        s_idx_to_state_idx=state_indices,
        s_idx_has_initial_state=has_initial_state,
    )


@jax.jit(donate_argnames=("x", "conv_state"),
         static_argnames=("kernel_size", ))
def ragged_causal_conv1d(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
    # Step 1: Validate inputs.
    num_seqs = state_indices.size
    num_tokens, dim = x.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert query_start_loc.shape == (num_seqs + 1, )
    assert state_indices.shape == (num_seqs, )
    assert distribution.shape == (3, )

    # Step 2: Input pre-processing.
    # TODO(kyuyeunk): Move this to SSM initialization logic.
    x_shape = x.shape
    x_dtype = x.dtype
    x_in = x.astype(jnp.float32)
    x_in = x_in.reshape(-1, 1, dim)

    conv_state_shape = conv_state.shape
    conv_state_dtype = conv_state.dtype
    conv_state_in = conv_state.astype(jnp.float32)
    conv_state_in = conv_state_in.reshape(-1, kernel_size - 1, 1, dim)

    vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    tile_size = min(64, num_tokens)
    cfgs = ConvConfigs(
        batch_size=num_tokens,
        kernel_size=kernel_size,
        tile_size=tile_size,
        dim_size=dim,
    )

    # Step 3: Wrap inputs for the kernel.
    metadata = preprocess_metadata(
        cfgs=cfgs,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        max_seqs=distribution[-1],
    )

    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(
        jnp.float32) if conv_bias is not None else None
    conv_rhs = ConvRhsRef(weight=conv_weight, bias=conv_bias)

    # Step 4: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    metadata_spec = MetadataRef(
        b_idx_to_s_idx=smem_spec,
        b_idx_to_sz_from_old=smem_spec,
        b_idx_should_write=smem_spec,
        s_idx_to_state_idx=smem_spec,
        s_idx_has_initial_state=smem_spec,
    )
    conv_rhs_spec = ConvRhsRef(
        weight=vmem_spec,
        bias=None if conv_bias is None else vmem_spec,
    )

    out, new_conv_state = pl.pallas_call(
        functools.partial(main_kernel, cfgs=cfgs),
        out_shape=(x_in, conv_state_in),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(metadata_spec, hbm_spec, hbm_spec, conv_rhs_spec),
            out_specs=(hbm_spec, hbm_spec),
            scratch_shapes=(
                pltpu.VMEM((2, cfgs.padded_tile_size, 1, cfgs.dim_size),
                           jnp.float32),
                pltpu.VMEM(
                    (2, cfgs.tile_size, cfgs.kernel_size - 1, 1,
                     cfgs.dim_size),
                    jnp.float32,
                ),
                pltpu.SemaphoreType.DMA((2, )),
            ),
        ),
        input_output_aliases={
            5: 0,
            6: 1
        },
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name="ragged_causal_conv1d_kernel",
    )(metadata, x_in, conv_state_in, conv_rhs)

    out = out.astype(x_dtype)
    out = out.reshape(x_shape)

    new_conv_state = new_conv_state.astype(conv_state_dtype)
    new_conv_state = new_conv_state.reshape(conv_state_shape)

    return out, new_conv_state
