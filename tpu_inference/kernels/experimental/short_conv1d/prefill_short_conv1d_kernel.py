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
"""Ragged short causal 1D convolution with conv_state I/O on TPU."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tpu_inference.kernels.experimental.short_conv1d.util import (
    GDNChunkIndices, calculate_chunk_indices)


def get_default_block_sizes(H: int, D: int, W: int, dtype) -> int:
    """Choose a chunk size (``bt``) for ``prefill_short_conv1d``.

    Mirrors ``fused_recurrent_gdn_kernel.get_default_block_sizes``: budget
    VMEM as fixed bits (history slots + weight tile) plus per-bt bits
    (x and out tiles, both double-buffered by ``pltpu.emit_pipeline``),
    pick the largest ``bt`` that fits, then floor to the nearest power of 2.
    """
    ibits = dtypes.itemsize_bits(dtype)
    # Fixed VMEM (bits): two double-buffered history slots + weight tile.
    fixed_bits = 2 * (W - 1) * H * D * ibits + W * H * D * ibits
    # Per-bt VMEM: x and out tiles, both managed by emit_pipeline. Empirically
    # the pipeline allocates ~4x per tile (double-buffer with separate prefetch
    # and store buffers, plus alignment padding).
    per_bt_bits = 4 * (2 * H * D * ibits)
    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
    # Cap bt at 512: empirically the largest bt where MBU is highest
    bt = min(bt, 512)
    return 1 << (bt.bit_length() - 1)


def _prefill_short_conv1d_main(
    chunk_indices_smem,
    x_hbm,  # aliased to out_hbm; positions outside the active region
    weight_hbm,  # carry through unchanged from x_hbm.
    state_indices_smem,
    has_initial_state_smem,
    conv_state_in_hbm,
    out_hbm,
    new_conv_state_hbm,
    history_bufs,  # (2, W-1, H, D) double-buffered for state load
    load_sems,  # SemaphoreType.DMA(2) one per history_bufs slot
    store_sem,
    *,
    W: int,
    H: int,
    D: int,
    bt: int,
):
    bounded_bt = pl.BoundedSlice(bt)

    def token_map(block_id):
        t_start = chunk_indices_smem.block_id_to_t_offset[block_id]
        t_end = chunk_indices_smem.block_id_to_t_offset[block_id + 1]
        t_size = t_end - t_start
        return (pl.ds(t_start, t_size), 0, 0)

    x_spec = pl.BlockSpec((bounded_bt, H, D), token_map)
    o_spec = pl.BlockSpec((bounded_bt, H, D), token_map)
    weight_spec = pl.BlockSpec((W, H, D), lambda _: (0, 0, 0))

    num_blocks = chunk_indices_smem.num_blocks[0]

    # Prefetch the first sequence's state before entering the pipeline so it
    # overlaps with the first chunk's x DMA. Mirrors fused_recurrent_gdn.
    first_seq = chunk_indices_smem.block_id_to_seq_idx[0]
    first_state_idx = state_indices_smem[first_seq]
    first_buf = first_seq % 2
    pltpu.make_async_copy(
        conv_state_in_hbm.at[first_state_idx],
        history_bufs.at[first_buf],
        load_sems.at[first_buf],
    ).start()

    def kernel_body(
        x_ref,
        weight_ref,
        o_ref,
        history_bufs_s,
        load_sems_s,
        store_sem_s,
    ):
        block_id = pl.program_id(0)
        seq_idx = chunk_indices_smem.block_id_to_seq_idx[block_id]
        t_start = chunk_indices_smem.block_id_to_t_offset[block_id]
        t_end = chunk_indices_smem.block_id_to_t_offset[block_id + 1]
        block_len = t_end - t_start

        prev_seq_idx = chunk_indices_smem.block_id_to_seq_idx[jnp.maximum(
            block_id - 1, 0)]
        is_new_seq = (block_id == 0) | (seq_idx != prev_seq_idx)
        next_seq_idx = chunk_indices_smem.block_id_to_seq_idx[block_id + 1]
        is_seq_end = seq_idx != next_seq_idx

        buf_idx = seq_idx % 2
        safe_next_seq = jnp.maximum(next_seq_idx, 0)
        next_buf_idx = safe_next_seq % 2
        state_idx = state_indices_smem[seq_idx]
        next_state_idx = state_indices_smem[safe_next_seq]
        has_init = has_initial_state_smem[seq_idx]

        prefetch_cp = pltpu.make_async_copy(
            conv_state_in_hbm.at[next_state_idx],
            history_bufs_s.at[next_buf_idx],
            load_sems_s.at[next_buf_idx],
        )
        load_wait_cp = pltpu.make_async_copy(
            conv_state_in_hbm.at[state_idx],
            history_bufs_s.at[buf_idx],
            load_sems_s.at[buf_idx],
        )

        # Prefetch the next sequence's state on seq_end (if there is a next
        # sequence). The first sequence was prefetched outside the loop.
        @pl.when(is_seq_end & (next_seq_idx >= 0))
        def _prefetch_next():
            prefetch_cp.start()

        # Wait for the current sequence's load to land before reading history.
        @pl.when(is_new_seq)
        def _wait_load():
            load_wait_cp.wait()

        # Zero out history if the sequence has no initial state. Loaded data
        # from `conv_state_in_hbm[state_idx]` may be stale garbage in that case.
        @pl.when(is_new_seq & (has_init == 0))
        def _zero_state():
            history_bufs_s[buf_idx, :, :, :] = jnp.zeros(
                (W - 1, H, D), dtype=history_bufs_s.dtype)

        # Conv split into head (first W-1 rows, depend on history) and body
        # (rest, depend only on x_chunk). The body uses STATIC slices of
        # x_chunk: `x_chunk[k : k + bt - W + 1]` for each k, avoiding the
        # per-k jnp.concatenate that materialized a fresh (bt, H, D) bf16
        # tile every iteration. The head builds one (2(W-1), H, D) merged
        # tile once and reuses static slices of it. Result is written to
        # o_ref in two parts.
        history = history_bufs_s[buf_idx, :, :, :]
        x_chunk = x_ref[...]
        weight_block = weight_ref[...]
        # Upcast operands to fp32 before the multiply so the product is computed
        # in fp32; the bf16 multiplier's mantissa error is otherwise visible.
        body_len = bt - (W - 1)
        # Body: out_body[t] = sum_k weight[k] * x_chunk[t + k]  for t in [0, body_len)
        acc_body = jnp.zeros((body_len, H, D), jnp.float32)
        for k in range(W):
            cur_body = x_chunk[k:k + body_len, :, :]
            acc_body = acc_body + cur_body.astype(
                jnp.float32) * weight_block[k][None, :, :].astype(jnp.float32)
        # Head: build merged = [history; x_chunk[:W-1]] of length 2*(W-1)
        # and slice statically. head[i] depends on merged[i..i+W-1].
        merged_head = jnp.concatenate([history, x_chunk[:W - 1, :, :]], axis=0)
        acc_head = jnp.zeros((W - 1, H, D), jnp.float32)
        for k in range(W):
            cur_head = merged_head[k:k + W - 1, :, :]
            acc_head = acc_head + cur_head.astype(
                jnp.float32) * weight_block[k][None, :, :].astype(jnp.float32)
        # Write head and body to non-overlapping ranges of o_ref.
        o_ref[:W - 1, :, :] = acc_head.astype(o_ref.dtype)
        o_ref[W - 1:, :, :] = acc_body.astype(o_ref.dtype)

        # Update history slot in-place. The new history is the W-1 rows of
        # the "merged" stream [history; x_chunk] at indices [block_len,
        # block_len + W - 1). In the common case block_len >= W-1, all W-1
        # rows come from the tail of x_chunk: x_chunk[block_len - (W-1) :
        # block_len]. Issue this as a single (W-1)-row VMEM copy. The slow
        # path (block_len < W-1) only fires for sequences shorter than W-1
        # tokens (rare on prefill); it keeps the per-row dispatch from the
        # original implementation.
        @pl.when(block_len >= W - 1)
        def _history_update_fast():
            history_bufs_s[buf_idx, :, :, :] = x_ref[
                pl.ds(block_len - (W - 1), W - 1), :, :]

        @pl.when(block_len < W - 1)
        def _history_update_short_seq():
            for i in range(W - 1):
                src_idx = block_len + i

                @pl.when(src_idx < W - 1)
                def _from_history():
                    history_bufs_s[buf_idx,
                                   i, :, :] = history_bufs_s[buf_idx,
                                                             src_idx, :, :]

                @pl.when(src_idx >= W - 1)
                def _from_x():
                    history_bufs_s[buf_idx,
                                   i, :, :] = x_ref[src_idx - (W - 1), :, :]

        # On seq end: persist trailing state to HBM (sync).
        store_cp = pltpu.make_async_copy(
            history_bufs_s.at[buf_idx],
            new_conv_state_hbm.at[state_idx],
            store_sem_s,
        )

        @pl.when(is_seq_end)
        def _store_state():
            store_cp.start()
            store_cp.wait()

    pltpu.emit_pipeline(
        kernel_body,
        grid=(num_blocks, ),
        in_specs=[x_spec, weight_spec],
        out_specs=o_spec,
    )(
        x_hbm,
        weight_hbm,
        out_hbm,
        scratches=[history_bufs, load_sems, store_sem],
    )


@functools.partial(
    jax.jit,
    static_argnames=("block_n", "region_start_idx"),
    donate_argnames=("x", "conv_state"),
)
def prefill_short_conv1d(
    x: jax.Array,
    weight: jax.Array,
    conv_state: jax.Array,
    cu_seqlens: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    block_n: int | None = None,
    region_start_idx: int = 1,
):
    """Ragged per-channel short causal 1D convolution with state I/O.

    ``x`` is donated and aliased to the output: tokens in the active
    region are overwritten in place with the conv output; tokens outside
    the active region are returned unchanged.  The ``out`` return value
    IS the same buffer as the input ``x``.

    Args:
        x: ``[T, H, D]`` packed token activations (donated).
        weight: ``[W, H, D]`` causal kernel; ``weight[W-1]`` is the
            current-token coefficient.
        conv_state: ``[N_states, W-1, H, D]`` history slots, indexed by
            ``state_indices``.
        cu_seqlens: ``[num_seqs+1]`` int32 cumulative sequence lengths.
        state_indices: ``[max_reqs]`` int32, request -> conv_state slot.
        distribution: ``[3]`` int32 ``[decoding, recurrent, total]``.
        has_initial_state: ``[max_reqs]`` int32 / bool, whether to read
            ``conv_state[state_indices[s]]`` (else use zeros).
        block_n: chunk size on the token axis. ``None`` -> use
            ``get_default_block_sizes`` based on VMEM budget.
        region_start_idx: which slice of ``distribution`` to iterate; ``1``
            processes ``[distribution[1], distribution[2])``.

    Returns:
        ``(out, new_conv_state)`` with shapes matching the inputs.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [T, H, D]; got {x.shape}")
    T, H, D = x.shape
    W = weight.shape[0]
    if block_n is None:
        block_n = get_default_block_sizes(H, D, W, x.dtype)
    if weight.shape != (W, H, D):
        raise ValueError(
            f"weight must be [W, H, D]={W, H, D}; got {weight.shape}")
    if conv_state.shape[1:] != (W - 1, H, D):
        raise ValueError(
            f"conv_state must be [N, W-1, H, D]; got {conv_state.shape}")
    N_states = conv_state.shape[0]
    max_reqs = state_indices.shape[0]

    bt = block_n
    # Upper bound on chunks: ceil(T / bt) chunks for the densest packing,
    # plus up to max_reqs extra splits where a sequence boundary falls
    # mid-block.
    max_num_blocks = (T + bt - 1) // bt + max_reqs

    chunk_indices = calculate_chunk_indices(
        cu_seqlens,
        distribution,
        max_num_blocks,
        bt,
        region_start_idx=region_start_idx,
    )

    hbm_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    # Skip the kernel entirely when this region has no sequences to process.
    n_seqs = distribution[region_start_idx +
                          1] - distribution[region_start_idx]
    grid_dim = jnp.where(n_seqs > 0, 1, 0)

    out, new_conv_state = pl.pallas_call(
        functools.partial(_prefill_short_conv1d_main, W=W, H=H, D=D, bt=bt),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                GDNChunkIndices(
                    num_blocks=smem_spec,
                    block_id_to_seq_idx=smem_spec,
                    block_id_to_t_offset=smem_spec,
                ),
                hbm_spec,  # x (input, aliased to output 0)
                hbm_spec,  # weight
                smem_spec,  # state_indices
                smem_spec,  # has_initial_state
                hbm_spec,  # conv_state (input, aliased to output 1)
            ],
            out_specs=[hbm_spec, hbm_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                pltpu.VMEM((2, W - 1, H, D), x.dtype),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        # flat: 0..2=chunk_indices, 3=x, 4=weight, 5=state_indices,
        #       6=has_init, 7=conv_state.
        input_output_aliases={
            3: 0,
            7: 1
        },
        out_shape=[
            jax.ShapeDtypeStruct((T, H, D), x.dtype),
            jax.ShapeDtypeStruct((N_states, W - 1, H, D), conv_state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name="prefill_short_conv1d",
    )(
        chunk_indices,
        x,
        weight,
        state_indices,
        has_initial_state,
        conv_state,
    )

    return out, new_conv_state
