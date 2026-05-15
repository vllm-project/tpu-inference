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
"""Fused recurrent GDN forward kernel for TPU.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.gdn.fused_gdn_kernel_common import \
    validate_gdn_inputs


def get_default_block_sizes(
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
    state_dtype=jnp.float32,
) -> int:
    """Choose bt to maximize VMEM utilization.

    The recurrent kernel uses a fixed ``(2, H_v, K, V)`` state double
    buffer of ``state_dtype`` regardless of bt.  Only pipeline tiles scale with bt.
    """
    ibits = dtypes.itemsize_bits(dtype)
    sbits = dtypes.itemsize_bits(state_dtype)

    # Fixed (in bits): h_bufs (2, H_v, K, V) of state_dtype — always 2 buffers
    num_lanes = pltpu.get_tpu_info().num_lanes
    fixed_bits = 2 * H_v * K * V * sbits
    if use_gate_in_kernel:
        fixed_bits += 2 * H_v * num_lanes * 32  # a_log: (H_v, num_lanes) f32
    if has_dt_bias:
        fixed_bits += 2 * H_v * num_lanes * 32  # dt_bias: (H_v, num_lanes) f32

    # bt-proportional (in bits): pipeline tiles (×2 for emit_pipeline double buffering)
    #   q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
    #   g(bt,H_v,K) float32                     -> H_v·K·32
    #   v(bt,H_v,V) + o(bt,H_v,V)              -> 2·H_v·V·ibits
    #   b(bt,H_v,num_lanes)                     -> H_v·num_lanes·ibits
    per_bt_bits = 2 * (2 * H_qk * K * ibits + H_v * K * 32 +
                       2 * H_v * V * ibits + H_v * num_lanes * ibits)

    bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
    # Round down to nearest power of 2
    return 1 << (bt.bit_length() - 1)


# ── Metadata ──


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNChunkIndices:
    num_blocks: jax.Array  # [1] int32
    block_id_to_seq_idx: jax.Array  # [max_num_blocks + 1] int32 (sentinel at end)
    block_id_to_t_offset: jax.Array  # [max_num_blocks + 1] int32


@jax.named_scope("calculate_chunk_indices")
def calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks, bt: int):
    """Pre-compute per-block metadata as a standalone Pallas kernel.

    Iterates over sequences and splits each into BT-sized work
    items.  A sequence boundary that falls mid-block creates two work
    items for that block (one per sequence).

    Returns a GDNChunkIndices with num_blocks, block_id_to_seq_idx, block_id_to_t_offset.
    """

    def _kernel(
        cu_seqlens_ref,
        distribution_ref,
        meta_out,
        *,
        bt: int,
    ):
        seq_start = distribution_ref[0]
        seq_end = distribution_ref[1]
        n_seqs = seq_end - seq_start

        @jax.named_scope("inner_block_loop")
        def inner_block_loop(blk_rel, carry, *, seq_idx, eos):
            num_blocks, t_cursor = carry
            block_id = num_blocks + blk_rel

            t_start = t_cursor + blk_rel * bt
            t_end = jnp.minimum(t_start + bt, eos)

            meta_out.block_id_to_seq_idx[block_id] = seq_idx
            meta_out.block_id_to_t_offset[block_id] = t_start
            meta_out.block_id_to_t_offset[block_id + 1] = t_end

            return num_blocks, t_cursor

        @jax.named_scope("outer_seq_loop")
        def outer_seq_loop(seq_rel, carry):
            num_blocks, t_cursor = carry
            seq_idx = seq_start + seq_rel
            eos = cu_seqlens_ref[seq_idx + 1]

            seq_len_from_cursor = eos - t_cursor
            n_seq_blocks = pl.cdiv(seq_len_from_cursor, bt)

            loop_fn = functools.partial(
                inner_block_loop,
                seq_idx=seq_idx,
                eos=eos,
            )
            jax.lax.fori_loop(0, n_seq_blocks, loop_fn, (num_blocks, t_cursor))

            return num_blocks + n_seq_blocks, eos

        first_token = cu_seqlens_ref[seq_start]
        num_blocks, _ = jax.lax.fori_loop(
            0,
            n_seqs,
            outer_seq_loop,
            (jnp.int32(0), first_token),
        )
        # Sentinel for look-ahead: block_id+1 reads -1 past the last block
        meta_out.block_id_to_seq_idx[num_blocks] = jnp.int32(-1)
        meta_out.num_blocks[0] = num_blocks

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    meta = pl.pallas_call(
        functools.partial(_kernel, bt=bt),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            out_specs=GDNChunkIndices(
                num_blocks=smem_spec,
                block_id_to_seq_idx=smem_spec,
                block_id_to_t_offset=smem_spec,
            ),
            grid=(1, ),
        ),
        out_shape=GDNChunkIndices(
            num_blocks=jax.ShapeDtypeStruct((1, ), jnp.int32),
            block_id_to_seq_idx=jax.ShapeDtypeStruct((max_num_blocks + 1, ),
                                                     jnp.int32),
            block_id_to_t_offset=jax.ShapeDtypeStruct((max_num_blocks + 1, ),
                                                      jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(cu_seqlens, distribution)

    return meta


# ── Index maps ──


class _MetadataIndexMaps:
    """Index maps driven by pre-computed metadata arrays."""

    def __init__(self, meta: GDNChunkIndices):
        self.meta = meta

    def token_map(self, block_id):
        t_start = self.meta.block_id_to_t_offset[block_id]
        t_end = self.meta.block_id_to_t_offset[block_id + 1]
        t_size = t_end - t_start
        return (pl.ds(t_start, t_size), 0, 0)


# ── Outer kernel ──


def _recurrent_gdn_main(
    meta,  # GDNChunkIndices (SMEM)
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K]
    b_hbm,  # [T, H_v, num_lanes]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [H_v, num_lanes] or None
    dt_bias_hbm,  # [H_v, num_lanes] or None
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    has_initial_state_ref,  # [max_num_req] int32 (SMEM); 0 = zero out h0
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2, H_v, K, V] VMEM scratch (double buffer)
    h_load_sems,  # [2] DMA semaphores
    h_store_sems,  # [2] DMA semaphores
    *,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    bt: int,
):
    num_blocks = meta.num_blocks[0]
    repeat_factor = H_v // H_qk
    # Build index maps from metadata
    idx_maps = _MetadataIndexMaps(meta)
    bounded_bt = pl.BoundedSlice(bt)

    qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), idx_maps.token_map)
    g_spec = pl.BlockSpec((bounded_bt, H_v, K), idx_maps.token_map)
    v_spec = pl.BlockSpec((bounded_bt, H_v, V), idx_maps.token_map)
    o_spec = pl.BlockSpec((bounded_bt, H_v, V), idx_maps.token_map)
    if b_hbm is not None:
        b_last = b_hbm.shape[2]
        b_spec = pl.BlockSpec((bounded_bt, H_v, b_last), idx_maps.token_map)
    else:
        b_spec = None

    # ── Prologue: start h0 load for first sequence (don't wait) ──
    first_seq = meta.block_id_to_seq_idx[0]
    first_state_idx = state_indices_ref[first_seq]
    first_buf = first_seq % 2
    pltpu.make_async_copy(
        state_hbm.at[pl.ds(first_state_idx, 1), :, :, :],
        h_bufs.at[pl.ds(first_buf, 1), :, :, :],
        h_load_sems.at[first_buf],
    ).start()

    # ── Inner kernel ──
    def _inner_kernel_body(
            q_ref,  # [<=bt, H_qk, K]
            k_ref,  # [<=bt, H_qk, K]
            v_ref,  # [<=bt, H_v, V]
            g_ref,  # [<=bt, H_v, K]
            b_ref,  # [<=bt, H_v, num_lanes]
            a_log_ref,  # [H_v, num_lanes] or None
            dt_bias_ref,  # [H_v, num_lanes] or None
            o_ref,  # [<=bt, H_v, V]
            h_bufs_s,  # [2, H_v, K, V] VMEM scratch
            meta_s,  # GDNChunkIndices (SMEM)
            state_indices_s,  # [max_num_req] int32 (SMEM)
            has_initial_state_s,  # [max_num_req] int32 (SMEM)
            h_load_sems_s,  # [2] DMA semaphores
            h_store_sems_s,  # [2] DMA semaphores
    ):
        block_id = pl.program_id(0)
        seq_idx = meta_s.block_id_to_seq_idx[block_id]
        t_start = meta_s.block_id_to_t_offset[block_id]
        t_end = meta_s.block_id_to_t_offset[block_id + 1]
        block_len = t_end - t_start

        # Detect sequence start
        prev_seq_idx = meta_s.block_id_to_seq_idx[jnp.maximum(block_id - 1, 0)]
        is_new_seq = (block_id == 0) | (seq_idx != prev_seq_idx)

        # Look ahead: detect sequence end
        next_seq_idx = meta_s.block_id_to_seq_idx[block_id + 1]
        is_seq_end = seq_idx != next_seq_idx

        # Double-buffer index: alternate buffers per sequence
        buf_idx = seq_idx % 2
        safe_next_seq = jnp.maximum(next_seq_idx, 0)
        next_buf_idx = safe_next_seq % 2

        # Pool indices via state_indices
        state_idx = state_indices_s[seq_idx]
        safe_next_state_idx = state_indices_s[safe_next_seq]

        # ── Step 1: Prefetch next h0 & wait for current h0 load ──
        prefetch_cp = pltpu.make_async_copy(
            state_hbm.at[pl.ds(safe_next_state_idx, 1), :, :, :],
            h_bufs_s.at[pl.ds(next_buf_idx, 1), :, :, :],
            h_load_sems_s.at[next_buf_idx],
        )
        load_wait_cp = pltpu.make_async_copy(
            state_hbm.at[pl.ds(state_idx, 1), :, :, :],
            h_bufs_s.at[pl.ds(buf_idx, 1), :, :, :],
            h_load_sems_s.at[buf_idx],
        )

        @pl.when(is_seq_end & (next_seq_idx >= 0))
        def _prefetch():
            prefetch_cp.start()

        @pl.when(is_new_seq)
        def _wait_h0():
            load_wait_cp.wait()

        # If the request has no prior recurrent state (brand-new prefill
        # landing on a freshly-allocated mamba slot), the DMA-loaded h0
        # is stale data from a previous tenant. Overwrite with zeros so
        # the recurrent update starts from zero, mirroring the chunked
        # path's `init_states_for_seqs = jnp.where(has_initial_state,
        # ..., 0)` and GPU's `initial_state[~has_initial_state, ...] = 0`
        # in `gdn_linear_attn._forward_core`.
        has_init = has_initial_state_s[seq_idx]

        @pl.when(is_new_seq & (has_init == 0))
        def _zero_h0():
            h_bufs_s[buf_idx] = jnp.zeros((H_v, K, V), dtype=h_bufs_s.dtype)

        # ── Step 2: Compute ──
        h = h_bufs_s[buf_idx].astype(jnp.float32)

        if use_gate_in_kernel:
            a_val = jnp.exp(a_log_ref[:, 0].astype(jnp.float32))
            if dt_bias_ref is not None:
                dt_bias_tile = dt_bias_ref[...].astype(
                    jnp.float32)  # [H_v, num_lanes]
                if K > dt_bias_tile.shape[-1]:
                    dt_bias_val = jnp.concatenate(
                        [dt_bias_tile] * (K // dt_bias_tile.shape[-1]),
                        axis=-1)
                else:
                    dt_bias_val = dt_bias_tile

        def step(local_t, h):
            q_t = q_ref[local_t].astype(jnp.float32)
            k_t = k_ref[local_t].astype(jnp.float32)
            v_t = v_ref[local_t].astype(jnp.float32)
            g_t = g_ref[local_t].astype(jnp.float32)
            if b_ref is not None:
                b_tile = b_ref[local_t].astype(jnp.float32)  # [H_v, num_lanes]
                if V > b_tile.shape[-1]:
                    beta_t = jax.nn.sigmoid(
                        jnp.concatenate([b_tile] * (V // b_tile.shape[-1]),
                                        axis=-1))  # [H_v, V]
                else:
                    beta_t = jax.nn.sigmoid(
                        b_tile)  # [H_v, num_lanes] (== [H_v, V])

            if use_qk_l2norm:
                q_t = q_t / jnp.sqrt(
                    jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
                k_t = k_t / jnp.sqrt(
                    jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
            q_t = q_t * scale

            # GQA: repeat q/k from H_qk to H_v heads
            if repeat_factor > 1:
                q_t = jnp.repeat(q_t, repeat_factor, axis=0)
                k_t = jnp.repeat(k_t, repeat_factor, axis=0)

            if use_gate_in_kernel:
                if dt_bias_ref is not None:
                    g_t = g_t + dt_bias_val
                if lower_bound is not None:
                    gk = lower_bound / (1.0 + jnp.exp(-(a_val[:, None] * g_t)))
                else:
                    gk = -a_val[:, None] * jax.nn.softplus(g_t)
            else:
                gk = g_t

            # Same algebraic identity as fused_gdn_decode_kernel.py:
            # o = q @ h_pre + (q . k) * b_v
            # Lets MXU(o) and VPU(rank-1 update) run in parallel.
            h_pre = h * jnp.exp(gk[:, :, None])
            kh = jax.lax.dot_general(
                k_t.reshape(H_v, 1, K),
                h_pre,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).reshape(H_v, V)
            v_diff = v_t - kh
            b_v = beta_t * v_diff if b_ref is not None else v_diff

            o_step1 = jax.lax.dot_general(
                q_t.reshape(H_v, 1, K),
                h_pre,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).reshape(H_v, V)
            qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            o_t = o_step1 + qk_dot * b_v
            h = h_pre + k_t[:, :, None] * b_v[:, None, :]

            o_ref[local_t] = o_t.astype(o_ref.dtype)
            return h

        h = jax.lax.fori_loop(0, block_len, step, h, unroll=False)
        h_bufs_s[buf_idx] = h.astype(h_bufs_s.dtype)

        # ── Step 3: Wait prev store, start current store ──
        # Store updated state back at state_idx.
        store_cp = pltpu.make_async_copy(
            h_bufs_s.at[pl.ds(buf_idx, 1), :, :, :],
            state_hbm.at[pl.ds(state_idx, 1), :, :, :],
            h_store_sems_s.at[buf_idx],
        )
        # Wait for prev store from same buffer (S-2's store).
        # Skip for the first 2 sequences — no prior store on this sem.
        has_prev_same_buf = (seq_idx - first_seq) >= 2

        @pl.when(is_seq_end & has_prev_same_buf)
        def _wait_prev_store():
            store_cp.wait()

        # Start current store
        @pl.when(is_seq_end)
        def _start_store():
            store_cp.start()

    # Run pipeline — None specs/inputs are passed through as None refs
    if use_gate_in_kernel and a_log_hbm is not None:
        a_log_spec = pl.BlockSpec((H_v, a_log_hbm.shape[1]), lambda _: (0, 0))
    else:
        a_log_spec = None
    dt_bias_spec = (pl.BlockSpec((H_v, dt_bias_hbm.shape[1]), lambda _:
                                 (0, 0)) if dt_bias_hbm is not None else None)

    pltpu.emit_pipeline(
        _inner_kernel_body,
        grid=(num_blocks, ),
        in_specs=[
            qk_spec, qk_spec, v_spec, g_spec, b_spec, a_log_spec, dt_bias_spec
        ],
        out_specs=o_spec,
    )(
        q_hbm,
        k_hbm,
        v_hbm,
        g_hbm,
        b_hbm,
        a_log_hbm,
        dt_bias_hbm,
        o_hbm,
        scratches=[
            h_bufs, meta, state_indices_ref, has_initial_state_ref,
            h_load_sems, h_store_sems
        ],
    )

    # ── Epilogue: wait for outstanding stores ──
    last_seq = meta.block_id_to_seq_idx[jnp.maximum(num_blocks - 1, 0)]
    last_buf = last_seq % 2
    other_buf = 1 - last_buf
    # Always wait for last seq's store
    pltpu.make_async_copy(
        h_bufs.at[pl.ds(0, 1), :, :, :],
        state_hbm.at[pl.ds(0, 1), :, :, :],
        h_store_sems.at[last_buf],
    ).wait()
    # If >= 2 seqs, also wait for the other sem
    drain_other = pltpu.make_async_copy(
        h_bufs.at[pl.ds(0, 1), :, :, :],
        state_hbm.at[pl.ds(0, 1), :, :, :],
        h_store_sems.at[other_buf],
    )

    @pl.when(last_seq != first_seq)
    def _drain_other():
        drain_other.wait()


# ── Public API ──


def fused_recurrent_gdn(
        q,  # [T, H_qk, K]
        k,  # [T, H_qk, K]
        v,  # [T, H_v, V]
        cu_seqlens,  # [N+1] int32
        g,  # [T, H_v, K] float32
        initial_state,  # [num_states, H_v, K, V] float32
        state_indices,  # [max_num_req] int32
        b,  # [T, H_v, num_lanes] or None
        has_initial_state,  # [max_num_req] int32 (0/1)
        *,
        scale,  # float
        use_qk_l2norm,  # bool
        use_gate_in_kernel=False,  # bool
        A_log=None,  # [H_v, num_lanes] or None
        dt_bias=None,  # [H_v, num_lanes] or None
        lower_bound=None,  # float or None
        distribution,  # [2] int32
):
    """Run the pre-computed-metadata recurrent GDN pallas kernel.

    ``has_initial_state[i]`` indicates whether request ``i``'s recurrent
    slot already holds a valid prior state (continuation, prefix-cache
    hit) or whether the slot is freshly allocated and its contents must
    be treated as zeros for this call. Mirrors the chunked path's
    `init_states_for_seqs = jnp.where(has_initial_state, ..., 0)` and
    GPU's `initial_state[~has_initial_state, ...] = 0` in
    ``gdn_linear_attn._forward_core``. Pass an all-ones array if the
    caller wants the previous (no-op) behaviour.
    """
    T, H_qk, H_v, K, V, dtype, num_states, num_lanes, _ = (validate_gdn_inputs(
        q,
        k,
        v,
        g,
        initial_state,
        state_indices,
        b=b,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias))
    max_num_req = cu_seqlens.shape[0] - 1

    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = get_default_block_sizes(H_qk,
                                 H_v,
                                 K,
                                 V,
                                 dtype,
                                 use_gate_in_kernel,
                                 dt_bias is not None,
                                 vmem_bytes_limit,
                                 state_dtype=initial_state.dtype)

    # Worst case: cdiv(T, bt) base blocks + up to max_num_req-1 boundary splits
    max_num_blocks = (T + bt - 1) // bt + max_num_req - 1

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    o_shape = jax.ShapeDtypeStruct((T, H_v, V), dtype)
    state_shape = jax.ShapeDtypeStruct((num_states, H_v, K, V),
                                       initial_state.dtype)

    meta = calculate_chunk_indices(cu_seqlens, distribution, max_num_blocks,
                                   bt)

    n_seqs = distribution[1] - distribution[0]
    grid_dim = jnp.where(n_seqs > 0, 1, 0)

    n_b = (b is not None)
    n_gate = (A_log is not None) + (dt_bias is not None)

    scope_name = f"recurrent_gdn-bt_{bt}"

    o, state = pl.pallas_call(
        functools.partial(
            _recurrent_gdn_main,
            H_qk=H_qk,
            H_v=H_v,
            K=K,
            V=V,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            bt=bt,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                GDNChunkIndices(
                    num_blocks=smem_spec,
                    block_id_to_seq_idx=smem_spec,
                    block_id_to_t_offset=smem_spec,
                ),
                *([any_spec] * 4),  # q, k, v, g
                any_spec if b is not None else None,  # b
                smem_spec,  # state_indices
                any_spec if A_log is not None else None,
                any_spec if dt_bias is not None else None,
                any_spec,  # state_init (= initial_state)
                smem_spec,  # has_initial_state
            ],
            out_specs=[any_spec, any_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                # h_bufs match HBM dtype so the DMAs don't need conversion.
                # The compute path upcasts h_bufs to fp32 once per block
                # (h = h_bufs_s[buf_idx].astype(fp32) above) and carries it
                # as fp32 across the per-token fori_loop, so on-chip math is
                # fp32 regardless of HBM storage dtype.
                pltpu.VMEM((2, H_v, K, V),
                           initial_state.dtype),  # h_bufs (double buffer)
                pltpu.SemaphoreType.DMA((2, )),  # h_load_sems
                pltpu.SemaphoreType.DMA((2, )),  # h_store_sems
            ],
        ),
        # Aliases reference flat positional inputs. `meta` is a 3-leaf
        # pytree, so the absolute index of `v` is 5 (3 meta leaves + q, k)
        # and the absolute index of `initial_state` is 8 + n_b + n_gate.
        # `has_initial_state` is appended to the end and is not aliased.
        input_output_aliases={
            5: 0,
            8 + n_b + n_gate: 1
        },
        out_shape=[o_shape, state_shape],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=scope_name,
    )(
        meta,
        q,
        k,
        v,
        g,
        b,
        state_indices,
        A_log,
        dt_bias,
        initial_state,
        has_initial_state,
    )

    return o, state
