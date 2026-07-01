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

from tpu_inference.kernels.causal_conv1d import strided_ldst


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvConfigs:
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvRhsRef:
    weight: Any
    bias: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    num_tiles: Any
    b_idx_to_s_idx: Any
    b_idx_to_sz_from_old: Any
    b_idx_should_write: Any
    s_idx_to_state_idx: Any
    s_idx_has_initial_state: Any
    b_idx_to_token_step: Any
    read_state_indices: Any
    write_state_indices: Any

    def __len__(self) -> int:
        return len(dataclasses.fields(self))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BufferWrapper(ABC):
    hbm_ref: Any
    vmem_ref: Any
    metadata_ref: MetadataRef
    cfgs: ConvConfigs

    def get_slot_vmem(self, slot):
        return self.vmem_ref.at[slot]

    def is_lower_oob(self, row) -> jax.Array:
        return row < 0

    def is_upper_oob(self, row) -> jax.Array:
        num_tiles = self.metadata_ref.num_tiles[...]
        last_row = num_tiles * self.cfgs.tile_size
        return row >= last_row

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
class XBuffer(BufferWrapper):

    def copy_in(self, b_start, slot, sem):
        is_no_op = self.is_upper_oob(b_start)
        dma_size = jnp.where(is_no_op, 0, self.cfgs.tile_size)
        b_start = jnp.where(is_no_op, 0, b_start)

        pltpu.make_async_copy(
            self.hbm_ref.at[pl.ds(b_start, dma_size)],
            self.get_slot_vmem(slot).at[pl.ds(0, dma_size)],
            sem,
        ).start()

    def wait_in(self, b_start, slot, sem):
        pltpu.make_async_copy(
            self.vmem_ref.at[0],
            self.vmem_ref.at[0],
            sem,
        ).wait()

    def copy_out(self, b_start, slot, sem):
        pltpu.make_async_copy(
            self.get_slot_vmem(slot),
            self.hbm_ref.at[pl.ds(b_start, self.cfgs.tile_size)],
            sem,
        ).start()

    def wait_out(self, b_start, slot, sem):
        is_no_op = self.is_lower_oob(b_start)
        dma_size = jnp.where(is_no_op, 0, self.cfgs.tile_size)
        pltpu.make_async_copy(
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            self.vmem_ref.at[0, pl.ds(0, dma_size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvStateBuffer(BufferWrapper):

    def copy_in(self, b_start, slot, sem):
        is_no_op = self.is_upper_oob(b_start)
        b_start = jnp.where(is_no_op, 0, b_start)

        for idx in range(self.cfgs.tile_size):
            b_idx = b_start + idx
            state_idx = self.metadata_ref.read_state_indices[b_idx]
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
            state_idx = self.metadata_ref.write_state_indices[b_idx]
            should_write = self.metadata_ref.b_idx_should_write[b_idx]

            pltpu.make_async_copy(
                self.get_slot_vmem(slot).at[pl.ds(idx, should_write)],
                self.hbm_ref.at[pl.ds(state_idx, should_write)],
                sem,
            ).start()

    def wait_out(self, b_start, slot, sem):
        is_no_op = self.is_lower_oob(b_start)
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


def inner_kernel(
    p_id: jax.Array,
    *,
    x_buffer: XBuffer,
    conv_state_buffer: ConvStateBuffer,
    sem_ref: jax.Array,
    metadata_ref: MetadataRef,
    conv_rhs_ref: ConvRhsRef,
    prev_x_scratch_ref: jax.Array,
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

    # Step 1: DMA prologue.

    # NOTE: SALU computation requirement of DMA can outweight VPU computation
    # requirement for Conv1D. By performing both DMA and Conv1D in a single
    # function, TPU can pipeline SALU and VPU operations.

    # Wait DMA read for current tile.
    x_buffer.wait_in(b_start, slot, recv_sem)
    conv_state_buffer.wait_in(b_start, slot, recv_sem)

    # Wait DMA write for previous tile.
    # NOTE: If current tile is the first tile, the size of DMA wait is set to zero
    # to ensure the kernel will not wait indefinitely for a DMA write that never
    # happened. This approach allows branchless execution of DMA and Conv1D.
    x_buffer.wait_out(prev_b_start, other_slot, send_sem)
    conv_state_buffer.wait_out(prev_b_start, other_slot, send_sem)

    # Start DMA read for next tile.
    # NOTE: Similar to wait_out, if current tile is the last tile, size of DMA
    # read is set to zero to allow branchless execution.
    x_buffer.copy_in(next_b_start, other_slot, recv_sem)
    conv_state_buffer.copy_in(next_b_start, other_slot, recv_sem)

    # Step 2: Computations.

    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding. By
    # leveraging compact layout, it ensures only one row is stored in a single
    # register and allows it to be reused across slides.  Instead of
    # pre-processing the inputs to use compact layout, performing strided load
    # allows performing relayout with zero-cost.
    x_compact = strided_ldst.load_large_to_compact(x_slot_ref, jnp.float32)

    # Load last prev_kernel_size rows of data.
    # NOTE: If the current tile is the first tile, VMEM will contain uninitialized
    # data.  In such cases, they will be overrided with either conv_state or zeros
    # during computation and does not cause any numeric issues.
    prev_x_scratch = prev_x_scratch_ref[...]
    x_compact = jnp.concat([prev_x_scratch, x_compact], axis=0)

    # NOTE: All conditionals below are static and evaluated during compile time.
    out_list = []
    for idx in range(cfgs.tile_size):
        b_idx = b_start + idx

        s_idx = metadata_ref.b_idx_to_s_idx[b_idx]
        sz_from_old = metadata_ref.b_idx_to_sz_from_old[b_idx]
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

        out = jnp.zeros((1, cfgs.dim_size), jnp.float32)

        for k in range(cfgs.kernel_size):
            # Computation for out[row] requires reading data
            # x[row - (kernel_size - 1) + k] where k iterates from 0 to
            # kernel_size - 1. Since x_compact is a concatenation of
            # x[b_start - kernel_size : b_start] and x[b_start : b_end].
            # x[row - (kernel_size - 1) + k] is equivalent to x_compact[idx + k],
            # where idx = row - b_start.
            in_idx = idx + k
            lhs = x_compact[in_idx]

            if k < cfgs.prev_kernel_size:
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

    out = jnp.stack(out_list, axis=0)
    # NOTE: Similar to strided load, strided store is performed to ensure no
    # post-processing is needed to the output.
    strided_ldst.store_compact_to_large(x_slot_ref, out)
    # NOTE: Write last prev_kernel_size rows of data to scratch memory to allow
    # next tile to read from it.
    prev_x_scratch_ref[...] = x_compact[cfgs.tile_size:]

    # Step 3: DMA epilogue.

    # Start DMA write for current tile.
    x_buffer.copy_out(b_start, slot, send_sem)
    conv_state_buffer.copy_out(b_start, slot, send_sem)


def main_kernel(
    # Inputs.
    metadata_ref: MetadataRef,
    x_ref: jax.Array,
    conv_state_ref: jax.Array,
    conv_rhs_ref: ConvRhsRef,
    # Outputs.
    x_out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    # Scratch.
    x_scratch_ref: jax.Array,
    conv_state_scratch_ref: jax.Array,
    prev_x_scratch_ref: jax.Array,
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

    recv_sem = sem_ref.at[0]
    send_sem = sem_ref.at[1]

    # Prologue: Start DMA read of the first tile.
    x_buffer.copy_in(0, 0, recv_sem)
    conv_state_buffer.copy_in(0, 0, recv_sem)

    num_tiles = metadata_ref.num_tiles[...]

    @pl.loop(0, num_tiles)
    def loop_wrapper(p_id):
        inner_kernel(
            p_id=p_id,
            x_buffer=x_buffer,
            conv_state_buffer=conv_state_buffer,
            sem_ref=sem_ref,
            metadata_ref=metadata_ref,
            conv_rhs_ref=conv_rhs_ref,
            prev_x_scratch_ref=prev_x_scratch_ref,
            cfgs=cfgs,
        )

    # Epilogue: Wait DMA write of the last tile.
    last_b_start = (num_tiles - 1) * cfgs.tile_size
    x_buffer.wait_out(last_b_start, 0, send_sem)
    conv_state_buffer.wait_out(last_b_start, 0, send_sem)


def preprocess_metadata(
    cfgs: ConvConfigs,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    has_initial_state: jax.Array,
    num_seqs: jax.Array,
) -> MetadataRef:
    """Preprocesses metadata required for DMA, and compute."""

    # NOTE: Following sequence of execution are the same for all layers. Compiler
    # may decide to perform CSE to remove redundant computations.

    max_seqs = state_indices.shape[0]

    # Mask out padded locations.
    num_tokens = query_start_loc[num_seqs]
    all_seqs = jnp.arange(max_seqs + 1)
    query_start_loc = jnp.where(all_seqs <= num_seqs, query_start_loc,
                                num_tokens)

    # Map batch index to sequence index.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    seqs = jnp.arange(max_seqs)
    b_idx_to_s_idx = jnp.repeat(seqs,
                                query_lens,
                                total_repeat_length=cfgs.batch_size)
    b_idx_query_start_loc = query_start_loc[b_idx_to_s_idx]
    all_b_idx = jnp.arange(cfgs.batch_size)
    b_idx_query_len = 1 + all_b_idx - b_idx_query_start_loc

    # Compute number of rows that needs to be fetched from conv_state (old) and
    # activation (new).
    b_idx_to_sz_from_new = jnp.minimum(b_idx_query_len, cfgs.kernel_size)
    b_idx_to_sz_from_old = cfgs.kernel_size - b_idx_to_sz_from_new
    b_idx_to_sz_from_old = jnp.minimum(b_idx_to_sz_from_old,
                                       cfgs.prev_kernel_size)

    # Determine if we should write the intermediate state history (Slot 1..num_spec+1).
    # We only write history when state_indices is 2D and the maximum query length
    # is <= num_spec + 1 (meaning it is a speculative verification step, not a prompt prefill).
    if state_indices.ndim == 2:
        num_spec = state_indices.shape[1] - 2
        max_query_len = jnp.max(query_lens)
        write_history = max_query_len <= num_spec + 1
    else:
        write_history = False

    # Determine which row needs to write its conv_state to HBM using tracer-safe jnp.where.
    if state_indices.ndim == 2:
        b_idx_should_write = jnp.where(
            write_history, all_b_idx < num_tokens, b_idx_query_start_loc +
            query_lens[b_idx_to_s_idx] - 1 == all_b_idx)
    else:
        b_idx_should_write = (b_idx_query_start_loc +
                              query_lens[b_idx_to_s_idx] - 1 == all_b_idx)
    b_idx_should_write = b_idx_should_write.astype(jnp.int32)

    # Compute 1D slot index arrays for read and write.
    if state_indices.ndim == 2:
        # All tokens load their initial state from Slot 0 (either prompt initial state or rollback initial state)
        read_state_indices = state_indices[b_idx_to_s_idx, 0]
        token_step_idx = all_b_idx - b_idx_query_start_loc
        write_state_indices = jnp.where(
            write_history, state_indices[b_idx_to_s_idx, token_step_idx + 1],
            state_indices[b_idx_to_s_idx, 0])
    else:
        read_state_indices = state_indices[b_idx_to_s_idx]
        write_state_indices = state_indices[b_idx_to_s_idx]

    # Compute the 0-based token step index within each request's speculative window.
    # This maps 1-to-1 with the token index in the request.
    b_idx_to_token_step = b_idx_query_len - 1

    return MetadataRef(
        num_tiles=pl.cdiv(num_tokens, cfgs.tile_size),
        b_idx_to_s_idx=b_idx_to_s_idx,
        b_idx_to_sz_from_old=b_idx_to_sz_from_old,
        b_idx_should_write=b_idx_should_write,
        s_idx_to_state_idx=state_indices,
        s_idx_has_initial_state=has_initial_state,
        b_idx_to_token_step=b_idx_to_token_step,
        read_state_indices=read_state_indices,
        write_state_indices=write_state_indices,
    )


@jax.jit(
    donate_argnames=("x", "conv_state"),
    static_argnames=("kernel_size", "tile_size"),
)
def ragged_causal_conv1d(
    x: jax.Array,  # (batch_size, dim_size)
    conv_state: jax.Array,  # (max_num_seqs, kernel_size - 1, dim_size)
    conv_weight: jax.Array,  # (dim_size, 1, kernel_size)
    conv_bias: jax.Array | None,  # (kernel_size,)
    query_start_loc: jax.Array,  # (max_num_seqs + 1,)
    state_indices: jax.Array,  # (max_num_seqs,)
    distribution: jax.Array,  # (3,)
    has_initial_state: jax.Array,  # (max_num_seqs,)
    *,
    kernel_size: int,
    tile_size: int = 64,  # TODO(kyuyeunk): Add more robust tiling logic.
) -> tuple[jax.Array, jax.Array]:
    """Perform Conv1D where input is a ragged sequence.

    Args:
        x: Input.
        conv_state: last (kernel_size - 1) rows of a given sequence.
        conv_weight: Conv1D weight parameter.
        conv_bias: Conv1D bias parameter.
        query_start_loc: Starting index on input x for a given sequence.
        state_indices: Pointer to index of conv_state for a given sequence.
        distribution: (i, j, k) that denotes seqs[:i] are decode, seqs[i:j] are
            prefill and seqs[j:k] are mixed.
        has_initial_state: Whether corresponding conv_state has a valid data.
        kernel_size: Number of previous rows including itself to use for compute.
        tile_size: Number of rows to process in a single tile.

    Returns:
        output: Output of conv1d.
        new_conv_state: Updated conv_state.
    """

    # Step 1: Validate inputs.
    num_seqs = state_indices.shape[0]
    batch_size, dim = x.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert query_start_loc.shape == (num_seqs + 1, )
    # state_indices can be 1D (normal decoding, shape (num_seqs,)) or 2D (speculative
    # decoding, shape (num_seqs, num_spec + 1)).
    assert state_indices.ndim in (1, 2)
    if state_indices.ndim == 1:
        assert state_indices.shape == (num_seqs, )
    else:
        # In speculative decoding, the first dimension represents the requests, and
        # the second dimension represents the history slots (which must be >= 2).
        assert state_indices.shape[0] == num_seqs
        assert state_indices.shape[1] >= 2
    assert distribution.shape == (3, )

    # Step 2: Input pre-processing.
    x_dtype = x.dtype
    sublane_tiling = pltpu.get_tpu_info().get_sublane_tiling(x_dtype)
    padded_batch_size = pl.cdiv(batch_size, sublane_tiling) * sublane_tiling
    tile_size = min(tile_size, padded_batch_size)
    # TODO(kyuyeunk): Eliminate the need for padding by using dynamic sized DMA.
    padded_batch_size = pl.cdiv(padded_batch_size, tile_size) * tile_size
    x = jnp.pad(x, ((0, padded_batch_size - batch_size), (0, 0)))

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): Perform this during model loading to eliminate runtime cost.
    conv_state_shape = conv_state.shape
    conv_state_dtype = conv_state.dtype
    # TODO(mhhuang): Remove the need for upcast.
    conv_state = conv_state.astype(jnp.float32)
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
    conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(
        jnp.float32) if conv_bias is not None else None

    cfgs = ConvConfigs(
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        dim_size=dim,
    )

    # Step 4: Metadata preprocessing.
    metadata = preprocess_metadata(
        cfgs=cfgs,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        num_seqs=distribution[-1],
    )

    # Step 5: Wrap inputs for the kernel.
    conv_rhs = ConvRhsRef(weight=conv_weight, bias=conv_bias)

    # Step 6: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata)
    conv_rhs_spec = jax.tree.map(lambda _: vmem_spec, conv_rhs)

    # Step 7: Perform computation.
    out, new_conv_state = pl.pallas_call(
        functools.partial(main_kernel, cfgs=cfgs),
        out_shape=(x, conv_state),
        in_specs=(metadata_spec, hbm_spec, hbm_spec, conv_rhs_spec),
        out_specs=(hbm_spec, hbm_spec),
        scratch_shapes=(
            pltpu.VMEM((2, cfgs.tile_size, cfgs.dim_size), x_dtype),
            pltpu.VMEM(
                (2, cfgs.tile_size, cfgs.prev_kernel_size, 1, cfgs.dim_size),
                jnp.float32,
            ),
            pltpu.VMEM((cfgs.prev_kernel_size, 1, cfgs.dim_size), jnp.float32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        input_output_aliases={
            len(metadata): 0,
            len(metadata) + 1: 1
        },
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="ragged_causal_conv1d_kernel",
    )(metadata, x, conv_state, conv_rhs)

    # Step 8: Output post-processing.
    out = out[:batch_size]
    new_conv_state = new_conv_state.astype(conv_state_dtype)
    new_conv_state = new_conv_state.reshape(conv_state_shape)

    return out, new_conv_state
