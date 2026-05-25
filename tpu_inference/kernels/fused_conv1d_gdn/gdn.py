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
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_conv1d_gdn import configs, ref_classes


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
def invert_triangular_matrix(t: jax.Array, block_size=16) -> jax.Array:
    """Compute invert trinagular when chunk_size > 1."""

    # NOTE: if chunk_size=1, compiler will perform DCE.
    out_dtype = t.dtype
    chunk = t.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = chunk // block_size

    def local_forward_sub(t_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_t = t_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, 1)
    iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, 2)
    identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = identity_mask[:, start:end, :]

        if i == 0:
            target_b = e_block
        else:
            interaction_t = t[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot_general(
                interaction_t,
                solved_x,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            )
            target_b = e_block - prev_sum

        # NOTE: Utilize fp32 to minimize cost of sublane rolling.
        local_t = t[:, start:end, start:end].astype(jnp.float32)
        x_block = local_forward_sub(local_t, target_b)
        x_blocks.append(x_block.astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


def fused_transpose_broadcast(x: jax.Array, src_dim: int,
                              dst_dim: int) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1
    mask_size = x.shape[src_dim]

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_shape[dst_dim] = mask_size
    mask_shape[src_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def chunked_gdn(
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    gdn_weights_ref: ref_classes.GDNWeightsRef,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array]:
    # (seqs, num_kq_heads, chunk, kq_head_dim)
    q_large = q_large.astype(cfgs.dtypes.compute)
    k_large = k_large.astype(cfgs.dtypes.compute)
    # (seqs, num_v_heads, chunk, v_head_dim)
    v_large = v_large.astype(cfgs.dtypes.compute)

    # (seqs, chunk, num_v_heads)
    b_large = b_large.astype(cfgs.dtypes.compute)
    a_large = a_large.astype(cfgs.dtypes.compute)

    a_log = gdn_weights_ref.a_log[...].reshape(1, 1, 1, -1)
    a_log = a_log.astype(cfgs.dtypes.compute)
    dt_bias = gdn_weights_ref.dt_bias[...].reshape(1, 1, 1, -1)
    dt_bias = dt_bias.astype(cfgs.dtypes.compute)

    # NOTE: Collapse sequence dim and num kv heads dim into a single dimension to
    # reduce number of non-elementwise instructions.  Leverage the fact that
    # results along both dimensions are completely independent from each. The
    # reamining code will simply treat as if there are seq_tile_size * num_v_heads
    # number of heads and will still yield numerically correct results.
    q_large = pltpu.einshape("snck->(sn)ck", q_large, True)
    k_large = pltpu.einshape("snck->(sn)ck", k_large, True)
    v_large = pltpu.einshape("sncv->(sn)cv", v_large, True)
    state_prev = pltpu.einshape("snkv->(sn)kv", state_prev, True)

    # NOTE: Any element-wise computations should occur before repeat.
    q_large = l2_norm(q_large)
    q_scale = cfgs.kq_head_dim**-0.5
    q_large *= q_scale
    k_large = l2_norm(k_large)

    # NOTE: repeat along non-sublane/lane dimension is free.
    # (num_v_heads, chunk, kq_head_dim)
    q_repeat = jnp.repeat(q_large, cfgs.v_per_kq_head, axis=0)
    k_repeat = jnp.repeat(k_large, cfgs.v_per_kq_head, axis=0)

    # (seqs, 1, chunk, num_v_heads)
    beta = jax.nn.sigmoid(b_large)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a_large + dt_bias)

    # Compute cumulative sum of decay.
    g_cum_sum_list = [gating_log[:, :, :1]]
    for row in range(1, cfgs.tok_tile_size):
        g_cum_sum_list.append(g_cum_sum_list[-1] +
                              gating_log[:, :, row:row + 1])
    # (seq, 1, chunk, num_v_heads)
    g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=2)

    # (seqs, num_v_heads, chunk, 1)
    g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log,
                                              src_dim=3,
                                              dst_dim=1)
    g_cum_sum_log = g_cum_sum_log[:, :cfgs.num_v_heads]
    beta = fused_transpose_broadcast(beta, src_dim=3, dst_dim=1)
    beta_large = beta[:, :cfgs.num_v_heads]

    # (seqs * num_v_heads, chunk, 1)
    # NOTE: Unlike qkv, num kv heads dim is transposed into batching dim starting
    # from this point. Therefore, collapsing can only happen here and not before.
    g_cum_sum_log = pltpu.einshape("snc1->(sn)c1", g_cum_sum_log, True)
    beta_large = pltpu.einshape("sn12->(sn)12", beta_large, True)

    # (seqs, 1, chunk)
    g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log,
                                                src_dim=1,
                                                dst_dim=2)
    # (num_v_heads, chunk, chunk)
    g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
    gating_map = jnp.exp(g_cum_sum_diff_log)
    # (num_v_heads, chunk, 1)
    gating_reverse = jnp.exp(-g_cum_sum_diff_log[..., -1:])
    # (num_v_heads, chunk, 1)
    gating = jnp.exp(g_cum_sum_log)
    # (num_v_heads, 1, 1)
    gating_last = gating[:, -1:]

    mask_dtype = get_mask_dtype(cfgs.dtypes.compute)
    iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    identity_mask = iota_r == iota_c
    strictly_lower_mask = iota_r > iota_c
    lower_mask = iota_r >= iota_c
    # (num_v_heads, chunk, chunk)
    gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # (num_v_heads, chunk, kq_head_dim)
    k_beta_repeat = k_repeat * beta_large

    # (num_v_heads, chunk, chunk)
    beta_k_k_t = jax.lax.dot_general(
        k_beta_repeat,
        k_repeat,
        (((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)
    gating_beta_k_k_t = gating_map_masked * beta_k_k_t
    t = jnp.where(identity_mask, 1, gating_beta_k_k_t)

    # (num_v_heads, chunk, chunk)
    t_inv = invert_triangular_matrix(t)

    # (num_v_heads, chunk, v_head_dim)
    v_beta_large = v_large * beta_large

    # (num_v_heads, chunk, v_head_dim)
    u = jax.lax.dot_general(
        t_inv,
        v_beta_large,
        (((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    # (num_v_heads, chunk, kq_head_dim)
    w = jax.lax.dot_general(
        t_inv,
        k_beta_repeat,
        (((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)
    w *= gating

    # (num_v_heads, chunk, kq_head_dim)
    q_large_gating = q_repeat * gating
    # NOTE: Concatenate lhs with same rhs to leverage weight
    # stationary architecture.
    # (num_v_heads, 2, chunk, kq_head_dim)
    merged_w_q = jnp.stack([w, q_large_gating], axis=1)
    # (num_v_heads, 2, chunk, v_head_dim)
    merged_ws_out_updated = jax.lax.dot_general(
        merged_w_q,
        state_prev,
        (((3, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    # splitting along non sublane/lane dim is free.
    ws = merged_ws_out_updated[:, 0].astype(cfgs.dtypes.compute)
    out_updated = merged_ws_out_updated[:, 1]

    # (num_v_heads, chunk, v_head_dim)
    u_ws = u - ws

    # (num_v_heads, chunk, kq_head_dim)
    k_repeat_gating = k_repeat * gating_reverse

    # (num_v_heads, kq_head_dim, v_head_dim)
    state_new = jax.lax.dot_general(
        k_repeat_gating,
        u_ws,
        (((1, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)

    # (num_v_heads, kq_head_dim, v_head_dim)
    state_updated = state_prev * gating_last
    state = state_updated + state_new

    # (num_kq_heads, chunk, chunk)
    out_qk = jax.lax.dot_general(
        q_large,
        k_large,
        (((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfgs.dtypes.compute)
    # NOTE: must perform repeat after matmul to reduce required compute.
    # (num_v_heads, chunk, chunk)
    out_qk = jnp.repeat(out_qk, cfgs.v_per_kq_head, axis=0)
    out_qk *= gating_map
    out_qk = jnp.where(lower_mask, out_qk, 0)

    # (num_v_heads, chunk, v_head_dim)
    out_new = jax.lax.dot_general(
        out_qk,
        u_ws,
        (((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    out = out_updated + out_new

    out = pltpu.einshape("(sn)cv->scnv", out, s=cfgs.seq_tile_size)
    state = pltpu.einshape("(sn)kv->snkv", state, True, s=cfgs.seq_tile_size)
    return out, state


def recurrent_gdn(
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    gdn_weights_ref: ref_classes.GDNWeightsRef,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array]:
    # (seqs, num_kq_heads, chunk, kq_head_dim)
    q_compact = q_compact.astype(cfgs.dtypes.compute)
    k_compact = k_compact.astype(cfgs.dtypes.compute)
    # (seqs, num_v_heads, chunk, v_head_dim)
    v_compact = v_compact.astype(cfgs.dtypes.compute)

    # (seqs, chunk, num_v_heads)
    b_compact = b_compact.astype(cfgs.dtypes.compute)
    a_compact = a_compact.astype(cfgs.dtypes.compute)

    a_log = gdn_weights_ref.a_log[...].reshape(1, 1, 1, 1, -1)
    a_log = a_log.astype(cfgs.dtypes.compute)
    dt_bias = gdn_weights_ref.dt_bias[...].reshape(1, 1, 1, 1, -1)
    dt_bias = dt_bias.astype(cfgs.dtypes.compute)

    # (seqs, num_kq_heads, toks, 1, kq_head_dim)
    q_compact = l2_norm(q_compact)
    q_scale = cfgs.kq_head_dim**-0.5
    q_compact *= q_scale
    k_compact = l2_norm(k_compact)
    k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

    # (seqs, num_v_heads, toks, 1, kq_head_dim)
    q_repeat = jnp.repeat(q_compact, cfgs.v_per_kq_head, axis=1)
    k_repeat = jnp.repeat(k_compact, cfgs.v_per_kq_head, axis=1)
    k_repeat_t = jnp.repeat(k_compact_t, cfgs.v_per_kq_head, axis=1)

    beta = jax.nn.sigmoid(b_compact)
    gating = -jnp.exp(a_log) * jax.nn.softplus(a_compact + dt_bias)
    gating = jnp.exp(gating)

    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, :cfgs.num_v_heads]
    gating = fused_transpose_broadcast(gating, src_dim=4, dst_dim=1)
    gating = gating[:, :cfgs.num_v_heads]

    out_list = []
    new_state_list = []

    for s_idx in range(cfgs.seq_tile_size):
        out_seq_list = []
        # (num_v_heads, kq_head_dim, v_head_dim)
        state = state_prev[s_idx]
        for t_idx in range(cfgs.tok_tile_size):
            # (num_v_heads, 1, kq_head_dim)
            q_curr = q_repeat[s_idx, :, t_idx]
            k_curr = k_repeat[s_idx, :, t_idx]
            # (num_v_heads, 1, v_head_dim)
            v_curr = v_compact[s_idx, :, t_idx]

            # (num_v_heads, kq_head_dim, 1)
            k_curr_t = k_repeat_t[s_idx, :, t_idx]

            # (num_v_heads, 1, 1)
            beta_curr = beta[s_idx, :, t_idx]
            gating_curr = gating[s_idx, :, t_idx]

            # (num_v_heads, kq_head_dim, v_head_dim)
            state_updated = state * gating_curr

            # (num_v_heads, 1, v_head_dim)
            v_updated = jax.lax.dot_general(
                k_curr,
                state_updated,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.dtypes.compute)

            # (num_v_heads, 1, v_head_dim)
            v_diff = v_curr - v_updated
            v_new = beta_curr * v_diff

            # (num_v_heads, kq_head_dim, v_head_dim)
            # NOTE: Multiplication with k_curr_t needs to be deferred as much as
            # possible as it expands the dimension size by kq_head_dim.
            state_new = k_curr_t * v_new
            # (num_v_heads, kq_head_dim, v_head_dim)
            state = state_updated + state_new

            # (num_v_heads, 1, v_head_dim)
            out = jax.lax.dot_general(
                q_curr,
                state,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.dtypes.compute)

            out_seq_list.append(out[:, 0, :])
        out_list.append(jnp.stack(out_seq_list, axis=0))
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state
