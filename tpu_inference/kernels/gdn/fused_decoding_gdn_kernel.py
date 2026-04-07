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
"""Fused recurrent GDN decoding kernel for TPU.

Processes ``bt`` decode tokens per pipeline step using ``emit_pipeline``
for q/k/v/g/beta tiling, with bulk manual DMA for state load/store via
``state_indices``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def get_default_block_sizes(
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
) -> int:
    """Choose bt to maximize VMEM utilization within vmem_bytes_limit.

    Accounts for state scratch ``(bt, H_v, K, V)`` float32, optional
    a_log / dt_bias, and bt-proportional tiles that ``emit_pipeline``
    double-buffers (q, k, v, g, beta, o).
    """
    ibits = dtypes.itemsize_bits(dtype)

    # Fixed (not bt-dependent), in bits
    fixed_bits = 0
    if use_gate_in_kernel:
        num_lanes = pltpu.get_tpu_info().num_lanes
        H_padded = ((H_v + num_lanes - 1) // num_lanes) * num_lanes
        fixed_bits += 2 * 1 * H_padded * 32  # a_log: (1, H_padded) f32
    if has_dt_bias:
        fixed_bits += 2 * H_v * K * 32  # dt_bias: (H_v, K) f32

    # bt-proportional (in bits):
    #   state scratch: (2*bt, H_v, K, V) float32 (double buffer)
    #   pipeline tiles (×2 for emit_pipeline double buffering):
    #     q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
    #     g(bt,H_v,K) float32                     -> H_v·K·32
    #     v(bt,H_v,V) + beta(bt,H_v,V) + o(bt,H_v,V) -> 3·H_v·V·ibits
    per_bt_bits = 2 * H_v * K * V * 32 + 2 * (
        2 * H_qk * K * ibits + H_v * K * 32 + 3 * H_v * V * ibits)

    bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
    # Round down to nearest power of 2
    return 1 << (bt.bit_length() - 1)


# ── Outer kernel ──────────────────────────────────────────────────────


def _decode_kernel_main(
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K] float32
    beta_hbm,  # [T, H_v, V]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [1, H_padded] or None
    dt_bias_hbm,  # [H_v, K] or None
    distribution_ref,  # [2] int32 (SMEM)
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2*bt, H_v, K, V] VMEM scratch (double buffer)
    h_load_sems,  # [2*bt] DMA semaphores
    h_store_sems,  # [2*bt] DMA semaphores
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
    decode_end = distribution_ref[0]
    nb_t = (decode_end + bt - 1) // bt
    repeat_factor = H_v // H_qk

    bounded_bt = pl.BoundedSlice(bt)

    def token_map(i):
        t_start = i * bt
        t_size = jnp.minimum(bt, decode_end - t_start)
        return (pl.ds(t_start, t_size), 0, 0)

    qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), token_map)
    g_spec = pl.BlockSpec((bounded_bt, H_v, K), token_map)
    v_spec = pl.BlockSpec((bounded_bt, H_v, V), token_map)

    if use_gate_in_kernel:
        num_lanes = pltpu.get_tpu_info().num_lanes
        H_padded = ((H_v + num_lanes - 1) // num_lanes) * num_lanes
        a_log_spec = pl.BlockSpec((1, H_padded), lambda _: (0, 0))
    else:
        a_log_spec = None
    dt_bias_spec = (pl.BlockSpec((H_v, K), lambda _:
                                 (0, 0)) if dt_bias_hbm is not None else None)

    # ── Prologue: start loading first bt-block's states ──
    for i_t in range(bt):

        @pl.when(i_t < decode_end)
        def _first_load():
            si = state_indices_ref[i_t]
            pltpu.make_async_copy(
                state_hbm.at[pl.ds(si, 1), :, :, :],
                h_bufs.at[pl.ds(i_t, 1), :, :, :],
                h_load_sems.at[i_t],
            ).start()

    # ── Inner kernel (runs per bt-block) ──
    def _inner_kernel(
            q_ref,  # [<=bt, H_qk, K]
            k_ref,  # [<=bt, H_qk, K]
            v_ref,  # [<=bt, H_v, V]
            g_ref,  # [<=bt, H_v, K]
            beta_ref,  # [<=bt, H_v, V]
            a_log_ref,  # [1, H_padded] or None
            dt_bias_ref,  # [H_v, K] or None
            o_ref,  # [<=bt, H_v, V]
            h_bufs_s,  # [2*bt, H_v, K, V] VMEM scratch (double buffer)
            state_indices_s,  # [max_num_req] int32 (SMEM)
            h_load_sems_s,  # [2*bt] DMA semaphores
            h_store_sems_s,  # [2*bt] DMA semaphores
    ):
        block_id = pl.program_id(0)
        t_start = block_id * bt
        block_len = jnp.minimum(bt, decode_end - t_start)

        buf_offset = (block_id % 2) * bt
        next_buf_offset = ((block_id + 1) % 2) * bt

        if use_gate_in_kernel:
            a_val = jnp.exp(a_log_ref[0][:H_v].astype(jnp.float32))
            if dt_bias_ref is not None:
                dt_bias_val = dt_bias_ref[...].astype(jnp.float32)

        # ── Step 1: Prefetch next bt-block's states ──
        next_t_start = t_start + bt
        next_block_len = jnp.maximum(
            jnp.minimum(bt, decode_end - next_t_start), 0)
        for i_t in range(bt):

            @pl.when(i_t < next_block_len)
            def _prefetch():
                next_si = state_indices_s[next_t_start + i_t]
                pltpu.make_async_copy(
                    state_hbm.at[pl.ds(next_si, 1), :, :, :],
                    h_bufs_s.at[pl.ds(next_buf_offset + i_t, 1), :, :, :],
                    h_load_sems_s.at[next_buf_offset + i_t],
                ).start()

        # ── Step 2: Wait for current bt-block's state loads ──
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _wait_load():
                pltpu.make_async_copy(
                    state_hbm.at[pl.ds(0, 1), :, :, :],
                    h_bufs_s.at[pl.ds(buf_offset + i_t, 1), :, :, :],
                    h_load_sems_s.at[buf_offset + i_t],
                ).wait()

        # ── Step 3: Compute ──
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _process_token():
                h0 = h_bufs_s[buf_offset + i_t].astype(jnp.float32)
                q_t = q_ref[i_t].astype(jnp.float32)
                k_t = k_ref[i_t].astype(jnp.float32)
                v_t = v_ref[i_t].astype(jnp.float32)
                g_t = g_ref[i_t]
                beta_t = beta_ref[i_t].astype(jnp.float32)

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
                    g_val = g_t
                    if dt_bias_ref is not None:
                        g_val = g_val + dt_bias_val
                    if lower_bound is not None:
                        gk = lower_bound / (1.0 +
                                            jnp.exp(-(a_val[:, None] * g_val)))
                    else:
                        gk = -a_val[:, None] * jnp.log(1.0 + jnp.exp(g_val))
                else:
                    gk = g_t

                h_new = h0 * jnp.exp(gk[:, :, None])
                b_v = beta_t * (v_t - jnp.sum(h_new * k_t[:, :, None], axis=1))
                h_new = h_new + k_t[:, :, None] * b_v[:, None, :]
                o_t = jnp.sum(h_new * q_t[:, :, None], axis=1)

                o_ref[i_t] = o_t.astype(o_ref.dtype)
                h_bufs_s[buf_offset + i_t] = h_new.astype(h_bufs_s.dtype)

        # ── Step 4: Wait for stores from 2 blocks ago (same buffer set) ──
        prev_t_start = jnp.maximum((block_id - 2) * bt, 0)
        prev_block_len = jnp.where(
            block_id >= 2,
            jnp.minimum(bt, decode_end - prev_t_start),
            0,
        )
        for i_t in range(bt):

            @pl.when(i_t < prev_block_len)
            def _wait_prev_store():
                pltpu.make_async_copy(
                    h_bufs_s.at[pl.ds(buf_offset + i_t, 1), :, :, :],
                    state_hbm.at[pl.ds(0, 1), :, :, :],
                    h_store_sems_s.at[buf_offset + i_t],
                ).wait()

        # ── Step 5: Start storing current bt-block's states ──
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _start_store():
                si = state_indices_s[t_start + i_t]
                pltpu.make_async_copy(
                    h_bufs_s.at[pl.ds(buf_offset + i_t, 1), :, :, :],
                    state_hbm.at[pl.ds(si, 1), :, :, :],
                    h_store_sems_s.at[buf_offset + i_t],
                ).start()

    pltpu.emit_pipeline(
        _inner_kernel,
        grid=(nb_t, ),
        in_specs=[
            qk_spec, qk_spec, v_spec, g_spec, v_spec, a_log_spec, dt_bias_spec
        ],
        out_specs=v_spec,
    )(
        q_hbm,
        k_hbm,
        v_hbm,
        g_hbm,
        beta_hbm,
        a_log_hbm,
        dt_bias_hbm,
        o_hbm,
        scratches=[h_bufs, state_indices_ref, h_load_sems, h_store_sems],
    )

    # ── Epilogue: drain outstanding stores ──
    last_buf_offset = ((nb_t - 1) % 2) * bt
    other_buf_offset = (nb_t % 2) * bt

    last_block_len = jnp.minimum(bt, decode_end - (nb_t - 1) * bt)
    for i_t in range(bt):

        @pl.when(i_t < last_block_len)
        def _drain_last():
            pltpu.make_async_copy(
                h_bufs.at[pl.ds(last_buf_offset + i_t, 1), :, :, :],
                state_hbm.at[pl.ds(0, 1), :, :, :],
                h_store_sems.at[last_buf_offset + i_t],
            ).wait()

    other_block_len = jnp.where(
        nb_t >= 2,
        jnp.minimum(bt, decode_end - (nb_t - 2) * bt),
        0,
    )
    for i_t in range(bt):

        @pl.when(i_t < other_block_len)
        def _drain_other():
            pltpu.make_async_copy(
                h_bufs.at[pl.ds(other_buf_offset + i_t, 1), :, :, :],
                state_hbm.at[pl.ds(0, 1), :, :, :],
                h_store_sems.at[other_buf_offset + i_t],
            ).wait()


# ── Public API ───────────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "lower_bound",
    ],
)
def fused_decoding_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    g: jax.Array,  # [T, H_v, K] float32
    initial_state: jax.Array,  # [num_states, H_v, K, V] float32
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    beta: jax.Array,  # [T, H_v, V]
    *,
    scale: float,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    a_log_input: jax.Array | None = None,  # [1, H_padded] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v, K] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Fused recurrent GDN single-step decode.

    Args:
        q: Queries ``[T, H_qk, K]``.
        k: Keys ``[T, H_qk, K]``.
        v: Values ``[T, H_v, V]``.
        g: Per-key gating ``[T, H_v, K]``, float32.
        initial_state: State cache ``[num_states, H_v, K, V]`` float32.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        beta: Betas ``[T, H_v, V]``.
        scale: Scale factor.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        a_log_input: Padded per-head log gate ``[1, H_padded]`` float32.
        dt_bias: Per-head-key bias ``[H_v, K]`` float32.
        lower_bound: If set, use sigmoid gate instead of softplus.

    Returns:
        ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]``.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]
    V = v.shape[-1]
    dtype = q.dtype
    num_states = initial_state.shape[0]

    if k.shape != (T, H_qk, K):
        raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of H_qk={H_qk}")
    if v.shape != (T, H_v, V):
        raise ValueError(f"v shape {v.shape} must be [{T}, {H_v}, {V}]")
    if g.shape != (T, H_v, K):
        raise ValueError(f"g shape {g.shape} must be [{T}, {H_v}, {K}]")
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 32 // dtypes.itemsize_bits(dtype)
    if K % num_lanes != 0 or V % num_lanes != 0:
        raise ValueError(f"K={K}, V={V} must be multiples of {num_lanes}")
    if H_qk % packing != 0:
        raise ValueError(
            f"H_qk={H_qk} must be a multiple of packing={packing} (32 // bitwidth)"
        )
    if H_v % packing != 0:
        raise ValueError(
            f"H_v={H_v} must be a multiple of packing={packing} (32 // bitwidth)"
        )
    if initial_state.shape[1:] != (H_v, K, V):
        raise ValueError(
            f"initial_state trailing dims {initial_state.shape[1:]} "
            f"must be ({H_v}, {K}, {V})")
    if state_indices.dtype != jnp.int32:
        raise ValueError(
            f"state_indices must be int32, got {state_indices.dtype}")
    if beta.shape != (T, H_v, V):
        raise ValueError(f"beta shape {beta.shape} must be [{T}, {H_v}, {V}]")
    if k.dtype != dtype or v.dtype != dtype or beta.dtype != dtype:
        raise ValueError(
            f"q/k/v/beta must share the same dtype, got q={dtype}, "
            f"k={k.dtype}, v={v.dtype}, beta={beta.dtype}")
    if g.dtype != jnp.float32:
        raise ValueError(f"g must be float32, got {g.dtype}")
    if initial_state.dtype != jnp.float32:
        raise ValueError(
            f"initial_state must be float32, got {initial_state.dtype}")
    if use_gate_in_kernel:
        if a_log_input is None:
            raise ValueError(
                "a_log_input is required when use_gate_in_kernel=True")
        if dt_bias is not None and dt_bias.shape != (H_v, K):
            raise ValueError(
                f"dt_bias shape {dt_bias.shape} must be [{H_v}, {K}]")
        if dt_bias is not None and dt_bias.dtype != jnp.float32:
            raise ValueError(f"dt_bias must be float32, got {dt_bias.dtype}")

    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = get_default_block_sizes(H_qk, H_v, K, V, dtype, use_gate_in_kernel,
                                 dt_bias is not None, vmem_bytes_limit)

    dt_bias_input = dt_bias

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    decode_end = distribution[0]
    grid_dim = jnp.where(decode_end > 0, 1, 0)

    n_gate = (a_log_input is not None) + (dt_bias is not None)

    scope_name = f"decoding_gdn-bt_{bt}"

    o, state = pl.pallas_call(
        functools.partial(
            _decode_kernel_main,
            H_qk=H_qk,
            H_v=H_v,
            K=K,
            V=V,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            bt=bt,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                *([any_spec] * 5),  # q, k, v, g, beta
                smem_spec,  # state_indices
                any_spec if a_log_input is not None else None,
                any_spec if dt_bias is not None else None,
                smem_spec,  # distribution
                any_spec,  # state_init
            ],
            out_specs=[any_spec, any_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                pltpu.VMEM((2 * bt, H_v, K, V),
                           jnp.float32),  # h_bufs (double buffer)
                pltpu.SemaphoreType.DMA((2 * bt, )),  # h_load_sems
                pltpu.SemaphoreType.DMA((2 * bt, )),  # h_store_sems
            ],
        ),
        input_output_aliases={
            2: 0,
            7 + n_gate: 1
        },
        out_shape=[
            jax.ShapeDtypeStruct((T, H_v, V), dtype),
            jax.ShapeDtypeStruct((num_states, H_v, K, V), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=scope_name,
    )(
        q,
        k,
        v,
        g,
        beta,
        state_indices,
        a_log_input,
        dt_bias_input,
        distribution,
        initial_state,
    )

    return o, state
