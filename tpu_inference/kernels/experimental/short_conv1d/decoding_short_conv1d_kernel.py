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
"""Decode-optimized short causal 1D convolution on TPU."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def get_default_block_sizes(N: int, H: int, D: int, W: int, dtype) -> int:
    """Choose ``bt`` (decode tokens per pallas iteration) for the decode kernel."""
    N = max(1, N)
    ibits = dtypes.itemsize_bits(dtype)
    fixed_bits = W * H * D * ibits  # weight tile (loaded once)
    # Per-bt: x and out tiles double-buffered, plus the
    # double-buffered state scratch (2 * bt * (W-1) * H * D * ibits).
    per_bt_bits = 2 * (2 * H * D * ibits) + 2 * (W - 1) * H * D * ibits
    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
    # Cap bt: keep at least ~4 chunks so emit_pipeline can double-buffer DMAs
    # against compute.
    bt = min(bt, N, max(128, N // 4), 1024)
    return 1 << (bt.bit_length() - 1)


def _decoding_short_conv1d_main(
    x_hbm,
    weight_hbm,
    state_indices_smem,
    has_initial_state_smem,
    distribution_smem,
    conv_state_in_hbm,
    out_hbm,
    new_conv_state_hbm,
    state_bufs,
    load_sems,
    store_sems,
    *,
    H: int,
    D: int,
    W: int,
    bt: int,
):
    decode_end = distribution_smem[0]
    nb_t = (decode_end + bt - 1) // bt

    bounded_bt = pl.BoundedSlice(bt)

    def token_map(i):
        t_start = i * bt
        t_size = jnp.minimum(bt, decode_end - t_start)
        return (pl.ds(t_start, t_size), 0, 0)

    x_spec = pl.BlockSpec((bounded_bt, H, D), token_map)
    o_spec = pl.BlockSpec((bounded_bt, H, D), token_map)
    weight_spec = pl.BlockSpec((W, H, D), lambda _: (0, 0, 0))

    # Prologue: prefetch the first tile's state slots.
    first_block_len = jnp.minimum(bt, decode_end)
    for i_t in range(bt):

        @pl.when(i_t < first_block_len)
        def _first_load():
            si = state_indices_smem[i_t]
            pltpu.make_async_copy(
                conv_state_in_hbm.at[si],
                state_bufs.at[0, i_t],
                load_sems.at[0],
            ).start()

    def _inner_kernel(
        x_ref,
        weight_ref,
        o_ref,
        state_bufs_s,
        load_sems_s,
        store_sems_s,
    ):
        block_id = pl.program_id(0)
        t_start = block_id * bt
        block_len = jnp.minimum(bt, decode_end - t_start)
        buf_idx = block_id % 2
        next_buf_idx = (block_id + 1) % 2

        # Step 1: prefetch the next tile's state slots (if any).
        next_t_start = t_start + bt
        next_block_len = jnp.maximum(
            jnp.minimum(bt, decode_end - next_t_start), 0)
        for i_t in range(bt):

            @pl.when(i_t < next_block_len)
            def _prefetch():
                next_si = state_indices_smem[next_t_start + i_t]
                pltpu.make_async_copy(
                    conv_state_in_hbm.at[next_si],
                    state_bufs_s.at[next_buf_idx, i_t],
                    load_sems_s.at[next_buf_idx],
                ).start()

        # Step 2: wait for the current tile's state loads (single drain).
        pltpu.make_async_copy(
            conv_state_in_hbm.at[pl.ds(0, block_len)],
            state_bufs_s.at[buf_idx, pl.ds(0, block_len)],
            load_sems_s.at[buf_idx],
        ).wait()

        # Step 3: compute. state_tile shape (bt, W-1, H, D).
        state_tile = state_bufs_s[buf_idx][...]
        x_chunk = x_ref[...]
        weight_block = weight_ref[...]

        # Mask out rows whose seq has no initial state. Read SMEM scalars
        # one-by-one so the lowering stays trivial; the compiler vectorizes
        # the resulting selects. The final decode tile can be partial, so keep
        # metadata reads in-bounds for inactive rows.
        masked_rows = []
        for i in range(bt):
            req_idx = t_start + i
            safe_req_idx = jnp.minimum(req_idx, jnp.maximum(decode_end - 1, 0))
            use_loaded_state = (i < block_len) & (
                has_initial_state_smem[safe_req_idx] != 0)
            masked_rows.append(
                jnp.where(
                    use_loaded_state,
                    state_tile[i],
                    jnp.zeros_like(state_tile[i]),
                ))
        masked_state = jnp.stack(masked_rows, axis=0)  # (bt, W-1, H, D)

        # out[t] = weight[W-1] * x[t] + sum_{k<W-1} weight[k] * masked_state[t, k]
        # Upcast operands to fp32 before the multiply so the product is computed
        # in fp32; the bf16 multiplier's mantissa error is otherwise visible.
        acc = x_chunk.astype(
            jnp.float32) * weight_block[W - 1][None, :, :].astype(jnp.float32)
        for k in range(W - 1):
            acc = acc + masked_state[:, k].astype(
                jnp.float32) * weight_block[k][None, :, :].astype(jnp.float32)
        o_ref[...] = acc.astype(o_ref.dtype)

        # New state: shift the (masked) state left by one and append x.
        # new_state[:, k] = masked_state[:, k+1] for k in [0, W-2)
        # new_state[:, W-2] = x_chunk
        new_state = jnp.concatenate(
            [masked_state[:, 1:, :, :], x_chunk[:, None, :, :]],
            axis=1,
        )
        state_bufs_s[buf_idx] = new_state.astype(state_bufs_s.dtype)

        # Step 4: wait for stores from 2 iterations ago (single drain).
        prev_t_start = jnp.maximum((block_id - 2) * bt, 0)
        prev_block_len = jnp.where(
            block_id >= 2,
            jnp.minimum(bt, decode_end - prev_t_start),
            0,
        )

        @pl.when(prev_block_len > 0)
        def _wait_prev_store():
            pltpu.make_async_copy(
                state_bufs_s.at[buf_idx, pl.ds(0, prev_block_len)],
                new_conv_state_hbm.at[pl.ds(0, prev_block_len)],
                store_sems_s.at[buf_idx],
            ).wait()

        # Step 5: start scatter stores back to conv_state.
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _start_store():
                si = state_indices_smem[t_start + i_t]
                pltpu.make_async_copy(
                    state_bufs_s.at[buf_idx, i_t],
                    new_conv_state_hbm.at[si],
                    store_sems_s.at[buf_idx],
                ).start()

    pltpu.emit_pipeline(
        _inner_kernel,
        grid=(nb_t, ),
        in_specs=[x_spec, weight_spec],
        out_specs=o_spec,
    )(
        x_hbm,
        weight_hbm,
        out_hbm,
        scratches=[state_bufs, load_sems, store_sems],
    )

    # Epilogue: drain the last two pending store batches.  Guarded so the
    # outer kernel is fully inert when ``decode_end == 0`` (nb_t == 0).
    @pl.when(nb_t > 0)
    def _epilogue_last():
        last_buf_idx = (nb_t - 1) % 2
        last_block_len = jnp.minimum(bt, decode_end - (nb_t - 1) * bt)
        pltpu.make_async_copy(
            state_bufs.at[last_buf_idx, pl.ds(0, last_block_len)],
            new_conv_state_hbm.at[pl.ds(0, last_block_len)],
            store_sems.at[last_buf_idx],
        ).wait()

    @pl.when(nb_t >= 2)
    def _drain_other():
        other_buf_idx = nb_t % 2
        other_block_len = jnp.minimum(bt, decode_end - (nb_t - 2) * bt)
        pltpu.make_async_copy(
            state_bufs.at[other_buf_idx,
                          pl.ds(0, other_block_len)],
            new_conv_state_hbm.at[pl.ds(0, other_block_len)],
            store_sems.at[other_buf_idx],
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=("block_n", ),
    donate_argnames=("x", "conv_state"),
)
def decoding_short_conv1d(
    x: jax.Array,
    weight: jax.Array,
    conv_state: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    block_n: int | None = None,
):
    """Decode-optimized short causal 1D convolution.

    ``x`` is donated and aliased to the output: tokens in
    ``[0, distribution[0])`` are overwritten in place with the conv
    output; the rest of ``x`` is returned unchanged.  The ``out`` return
    value IS the same buffer as the input ``x``.

    Args:
        x: ``[T, H, D]`` packed token activations (donated).
        weight: ``[W, H, D]`` causal kernel.
        conv_state: ``[N_states, W-1, H, D]`` history slots.
        state_indices: ``[max_reqs]`` int32, request -> conv_state slot.
        distribution: int32 array; ``distribution[0]`` is the decode end.
        has_initial_state: ``[max_reqs]`` int32.
        block_n: tokens processed per pallas iteration. ``None`` -> use
            ``get_default_block_sizes``.

    Returns:
        ``(out, new_conv_state)``.  ``out[:distribution[0]]`` is freshly
        written; the rest of ``out`` matches the input ``x`` (carried
        through via aliasing).  The ``state_indices[:distribution[0]]``
        slots of ``new_conv_state`` are written; the rest are unchanged.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [T, H, D]; got {x.shape}")
    T, H, D = x.shape
    W = weight.shape[0]
    if weight.shape != (W, H, D):
        raise ValueError(
            f"weight must be [W, H, D]={W, H, D}; got {weight.shape}")
    if conv_state.shape[1:] != (W - 1, H, D):
        raise ValueError(
            f"conv_state must be [N, W-1, H, D]; got {conv_state.shape}")
    N_states = conv_state.shape[0]
    max_reqs = state_indices.shape[0]
    if block_n is None:
        block_n = get_default_block_sizes(min(T, max_reqs), H, D, W, x.dtype)
    # Decode metadata is indexed by request id, while ``T`` includes prefill
    # tokens in mixed batches. Keep the tile within both static capacities.
    bt = max(1, min(block_n, T, max_reqs))

    hbm_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    # Skip the kernel entirely when there are no decode tokens.
    decode_end = distribution[0]
    grid_dim = jnp.where(decode_end > 0, 1, 0)

    out, new_conv_state = pl.pallas_call(
        functools.partial(_decoding_short_conv1d_main, H=H, D=D, W=W, bt=bt),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                hbm_spec,  # x
                hbm_spec,  # weight
                smem_spec,  # state_indices
                smem_spec,  # has_initial_state
                smem_spec,  # distribution
                hbm_spec,  # conv_state (input, aliased to output 1)
            ],
            out_specs=[hbm_spec, hbm_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                pltpu.VMEM((2, bt, W - 1, H, D), x.dtype),
                pltpu.SemaphoreType.DMA((2, )),
                pltpu.SemaphoreType.DMA((2, )),
            ],
        ),
        input_output_aliases={
            0: 0,
            5: 1
        },
        out_shape=[
            jax.ShapeDtypeStruct((T, H, D), x.dtype),
            jax.ShapeDtypeStruct((N_states, W - 1, H, D), conv_state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=f"decoding_short_conv1d-bt_{bt}",
    )(
        x,
        weight,
        state_indices,
        has_initial_state,
        distribution,
        conv_state,
    )

    return out, new_conv_state
