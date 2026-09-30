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

import jax
import jax.numpy as jnp

from tpu_inference.kernels.gdn.v3 import config


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt(
        jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
    return x / norm


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


# NOTE: Fork of recurrent_scan_v2.py but applied various optimizations.
def solve_triangular(t: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve `t @ x = rhs` for unit lower-triangular `t` via 2-block Neumann series."""
    out_dtype = rhs.dtype
    chunk = t.shape[-1]
    half = chunk // 2
    rem = chunk - half
    t_f32 = t.astype(jnp.float32)
    dn = (((2, ), (1, )), ((0, ), (0, )))

    iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
    eye = jnp.where(iota_r == iota_c, 1.0, 0.0)
    is_off_diag = (iota_r >= half) & (iota_c < half)
    a_diag = jnp.where((iota_r > iota_c) & ~is_off_diag, -t_f32, 0.0)
    d_inv = eye + a_diag
    a_pow = a_diag
    k = 1
    while k < rem:
        a_pow_bf16 = a_pow.astype(out_dtype)
        if k > 1:
            d_inv = d_inv + jax.lax.dot(d_inv.astype(out_dtype),
                                        a_pow_bf16,
                                        dimension_numbers=dn,
                                        preferred_element_type=jnp.float32)
        k *= 2
        if k < rem:
            a_pow = jax.lax.dot(a_pow_bf16,
                                a_pow_bf16,
                                dimension_numbers=dn,
                                preferred_element_type=jnp.float32)

    x_init = jax.lax.dot(d_inv,
                         rhs,
                         dimension_numbers=dn,
                         preferred_element_type=jnp.float32)
    x_0, x_1_init = x_init[:, :half, :], x_init[:, half:, :]
    x_0_pad = jnp.pad(x_0, ((0, 0), (0, rem), (0, 0)))
    t_10_x_0 = jax.lax.dot(t_f32[:, half:, :],
                           x_0_pad,
                           dimension_numbers=dn,
                           preferred_element_type=jnp.float32)
    t_10_pad = jnp.pad(t_10_x_0, ((0, 0), (half, 0), (0, 0)))
    corr = jax.lax.dot(d_inv[:, half:, :],
                       t_10_pad,
                       dimension_numbers=dn,
                       preferred_element_type=jnp.float32)
    return jnp.concat([x_0, x_1_init - corr], axis=1).astype(out_dtype)


def fused_transpose_broadcast(x: jax.Array,
                              src_dim: int,
                              dst_dim: int,
                              dst_size: int | None = None) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_shape[
        dst_dim] = dst_size if dst_size is not None else mask_shape[src_dim]
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def chunked_gdn_per_seq(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, 1, num_v_heads]
    beta: jax.Array,  # [1, 1, num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [num_heads, chunk, head_dim]."""

    # NOTE: Repeat along non lane/sublane dim is free.
    q_repeat = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
    k_repeat = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

    # Compute cumulative sum of decay.
    g_cum_sum_list = [gating_log[:, :1]]
    for row in range(1, cfg.chunk_size):
        g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row:row + 1])
    g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

    g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log,
                                              src_dim=2,
                                              dst_dim=0,
                                              dst_size=cfg.num_v_heads)
    beta_large = fused_transpose_broadcast(beta,
                                           src_dim=2,
                                           dst_dim=0,
                                           dst_size=cfg.num_v_heads)

    # [num_v_heads, 1, chunk]
    g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log,
                                                src_dim=1,
                                                dst_dim=2)
    # [num_v_heads, chunk, chunk]
    g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
    gating_map = jnp.exp(g_cum_sum_diff_log)
    # [num_v_heads, chunk, 1]
    gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
    # [num_v_heads, chunk, 1]
    gating_forward = jnp.exp(g_cum_sum_log)
    # [num_v_heads, 1, 1]
    gating_last = gating_forward[:, -1:]

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    identity_mask = iota_r == iota_c
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    # [num_v_heads, chunk, chunk]
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # [num_v_heads, chunk, kq_head_dim]
    k_beta_repeat = k_repeat * beta_large

    # Merge `k @ k.T` and `q @ k.T` into one 128-row MXU pass sharing `k_large`.
    kq_merged = jnp.concat([k_large, q_large], axis=1)
    kk_qk = jax.lax.dot(
        kq_merged,
        k_large,
        dimension_numbers=(((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    k_k_t, out_qk = jnp.split(kk_qk, 2, axis=1)
    beta_k_k_t = jnp.repeat(k_k_t, cfg.v_per_kq_head, axis=0) * beta_large
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t
    t = jnp.where(identity_mask, 1, gating_beta_k_k_t)

    # [num_v_heads, chunk, v_head_dim + kq_head_dim]
    v_beta_large = v_large * beta_large
    k_beta_gating = k_beta_repeat * gating_forward
    merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
    merged_uw = solve_triangular(t, merged_v_k)

    # [num_v_heads, chunk, v_head_dim]
    u, w = jnp.split(merged_uw, [cfg.v_head_dim], axis=-1)

    # Materialize recurrent state now, after `solve_triangular`.
    state_prev = state_prev()
    q_large_gating = q_repeat * gating_forward
    merged_w_q = jnp.concat([w, q_large_gating], axis=1)
    merged_ws_out_updated = jax.lax.dot(
        merged_w_q,
        state_prev,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    # NOTE: Splitting along non sublane/lane dim is free.
    ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
    ws = ws.astype(cfg.dtypes.compute)

    # [num_v_heads, chunk, v_head_dim]
    u_ws = u - ws

    # [num_v_heads, chunk, kq_head_dim]
    k_repeat_gating = k_repeat * gating_backward

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_new = jax.lax.dot(
        k_repeat_gating,
        u_ws,
        dimension_numbers=(((1, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    # [num_v_heads, kq_head_dim, v_head_dim]
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # [num_v_heads, chunk, chunk]
    out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
    out_qk = jnp.where(lower_mask, out_qk * gating_map, 0)

    # [num_v_heads, chunk, v_head_dim]
    out_new = jax.lax.dot(
        out_qk,
        u_ws,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )
    out = out_updated + out_new

    return out, state


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [seq, num_heads, chunk, head_dim].

    The returned state is [seq, 1, num_v_heads, kq_head_dim, v_head_dim]: the
    chunked path only produces the final state, so it is never taken when
    more than one checkpoint per sequence is required.
    """

    assert not cfg.use_recurrent

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(mask_dtype,
                                    (cfg.seq_tile_size, 1, cfg.chunk_size, 1),
                                    2)
    mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

    # [seqs, num_kq_heads, chunk, kq_head_dim]
    q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
    k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, v_head_dim]
    v_large = jnp.where(mask, v_large.astype(cfg.dtypes.compute), 0)

    b_large = b_large.astype(cfg.dtypes.compute)
    a_large = a_large.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

    # NOTE: Any element-wise computations should occur before repeat.
    q_large = l2_norm(q_large)
    q_scale = cfg.kq_head_dim**-0.5
    q_large *= q_scale
    k_large = l2_norm(k_large)

    # [seqs, 1, chunk, num_v_heads]
    beta = jax.nn.sigmoid(b_large)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)

    beta = jnp.where(mask, beta, 0)
    # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
    # be masked to 1 since it signifies that strength of state from previous row
    # will be 1 (i.e., no decay) if current row is invalid.
    gating_log = jnp.where(mask, gating_log, 0)

    out_list = []
    state_list = []
    for idx in range(cfg.seq_tile_size):
        out, state = chunked_gdn_per_seq(
            q_large[idx],
            k_large[idx],
            v_large[idx],
            gating_log[idx],
            beta[idx],
            lambda idx=idx: (state_prev(idx)
                             if callable(state_prev) else state_prev[idx]),
            cfg,
        )
        out_list.append(out.swapaxes(0, 1))
        state_list.append(state)
    out = jnp.stack(out_list, axis=0)
    # Single checkpoint per sequence, laid out like the recurrent path.
    state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
    return out, state


def recurrent_gdn_per_seq(
    q_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact_t: jax.Array,  # [num_kq_heads, chunk, kq_head_dim, 1]
    qk_dot: jax.Array,  # [num_kq_heads, chunk, 1, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim].

    The returned state stacks the post-token state of the last
    `cfgs.window_size` window positions ([window_size, num_v_heads,
    kq_head_dim, v_head_dim]); with `window_size == 1` that is just the final
    state.
    """

    out_list = []
    state_list = []
    for c_idx in range(cfgs.chunk_size):
        # [num_v_heads, 1, kq_head_dim]
        q_curr = jnp.repeat(q_compact[:, c_idx], cfgs.v_per_kq_head, axis=0)
        k_curr = jnp.repeat(k_compact[:, c_idx], cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, v_head_dim]
        v_curr = v_compact[:, c_idx]

        # [num_v_heads, kq_head_dim, 1]
        k_curr_t = jnp.repeat(k_compact_t[:, c_idx],
                              cfgs.v_per_kq_head,
                              axis=0)

        # [num_v_heads, 1, 1]
        qk_dot_curr = jnp.repeat(qk_dot[:, c_idx], cfgs.v_per_kq_head, axis=0)
        beta_curr = beta[:, c_idx]
        gating_curr = gating_log[:, c_idx]

        # [num_v_heads, kq_head_dim, v_head_dim]
        state_updated = state * gating_curr

        # Merge `k @ state` and `q @ state` into one 2-row matmul sharing `state_updated`.
        kq_state = jax.lax.dot(
            jnp.concatenate([k_curr, q_curr], axis=1),
            state_updated,
            dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        ).astype(cfgs.dtypes.compute)
        v_updated, q_state_updated = kq_state[:, 0:1, :], kq_state[:, 1:2, :]

        # [num_v_heads, 1, v_head_dim]
        v_diff = v_curr - v_updated
        v_new = beta_curr * v_diff

        # [num_v_heads, kq_head_dim, v_head_dim]
        state = state_updated + k_curr_t * v_new
        out = (q_state_updated + qk_dot_curr * v_new).astype(
            cfgs.dtypes.compute)

        out_list.append(out[:, 0, :])
        # NOTE: Rows past real_sizes are masked out by the caller, so the state
        # stops changing there and trailing checkpoints simply repeat the state
        # of the last real token. Checkpoints of the positions dropped here are
        # dead code the compiler eliminates.
        if c_idx >= cfgs.chunk_size - cfgs.window_size:
            state_list.append(state)

    return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim].

    The returned state has one checkpoint per window position: [seq,
    window_size, num_v_heads, kq_head_dim, v_head_dim]. Positions >=
    real_sizes repeat the last valid state; they are never written back to
    HBM.
    """

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2)
    mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

    # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
    q_compact = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0)
    k_compact = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, 1, v_head_dim]
    v_compact = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0)

    b_compact = b_compact.astype(cfg.dtypes.compute)
    a_compact = a_compact.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)

    qk_shape = q_compact.shape
    q_packed = l2_norm(q_compact.reshape(
        -1, cfg.kq_head_dim)) * (cfg.kq_head_dim**-0.5)
    k_packed = l2_norm(k_compact.reshape(-1, cfg.kq_head_dim))
    qk_dot = jnp.sum(q_packed * k_packed, axis=-1,
                     keepdims=True).reshape(cfg.seq_tile_size,
                                            cfg.num_kq_heads, cfg.chunk_size,
                                            1, 1)
    q_compact, k_compact = q_packed.reshape(qk_shape), k_packed.reshape(
        qk_shape)
    k_compact_t = fused_transpose_broadcast(k_compact,
                                            src_dim=4,
                                            dst_dim=3,
                                            dst_size=cfg.kq_head_dim)

    beta = jax.nn.sigmoid(b_compact)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)

    beta = jnp.where(mask, beta, 0)
    gating_log = jnp.exp(jnp.where(mask, gating_log, 0))

    beta = fused_transpose_broadcast(beta,
                                     src_dim=4,
                                     dst_dim=1,
                                     dst_size=cfg.num_v_heads)
    gating_log = fused_transpose_broadcast(gating_log,
                                           src_dim=4,
                                           dst_dim=1,
                                           dst_size=cfg.num_v_heads)

    out_list = []
    new_state_list = []

    for idx in range(cfg.seq_tile_size):
        out, state = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            k_compact_t[idx],
            qk_dot[idx],
            v_compact[idx],
            gating_log[idx],
            beta[idx],
            (state_prev(idx) if callable(state_prev) else state_prev[idx]),
            cfg,
        )
        out_list.append(out)
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state
