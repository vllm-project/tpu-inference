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

from tpu_inference.kernels.fused_conv1d_gdn import configs, ref_classes


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt(
        jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
    return x / norm


def fused_transpose_broadcast(x: jax.Array, src_dim: int,
                              dst_dim: int) -> jax.Array:
    assert x.shape[dst_dim] == 1
    mask_size = x.shape[src_dim]

    dtype = x.dtype
    match dtype.itemsize:
        case 4:
            mask_dtype = jnp.int32
        case 2:
            mask_dtype = jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")

    mask_shape = list(x.shape)
    mask_shape[dst_dim] = mask_size
    mask_shape[src_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


def split_and_compute_qkv(
        qkv: jax.Array,
        cfgs: configs.GDNConfigs) -> tuple[jax.Array, jax.Array, jax.Array]:
    k_offset = cfgs.kq_dim_size

    q_list = []
    k_list = []
    for row in range(cfgs.tile_size):
        q = qkv[row, :, :cfgs.kq_dim_size]
        k = qkv[row, :, k_offset:k_offset + cfgs.kq_dim_size]
        q_list.append(q)
        k_list.append(k)

    # compact layout with (batch, 1, kq_dim_size)
    q_compact = jnp.stack(q_list, axis=0)
    k_compact = jnp.stack(k_list, axis=0)

    q_list = []
    k_list = []

    q_scale = cfgs.kq_head_dim**-0.5
    for row in range(cfgs.tile_size):
        q_row_list = []
        k_row_list = []
        for i in range(cfgs.num_kq_heads):
            start = i * cfgs.kq_head_dim
            end = start + cfgs.kq_head_dim
            q = q_compact[row, :, start:end]
            k = k_compact[row, :, start:end]

            # NOTE: Apply elementwise or q/k computations before repeat to avoid
            # increase in required FLOPs.
            q = l2_norm(q)
            q *= q_scale
            k = l2_norm(k)

            # Repeat by v_per_kq_head.
            q_row_list += [q] * cfgs.v_per_kq_head
            k_row_list += [k] * cfgs.v_per_kq_head
        q_list.append(jnp.stack(q_row_list, axis=0))
        k_list.append(jnp.stack(k_row_list, axis=0))

    v_list = []
    v_offset = 2 * cfgs.kq_dim_size
    for row in range(cfgs.tile_size):
        v_row_list = []
        for i in range(cfgs.num_v_heads):
            start = v_offset + i * cfgs.v_head_dim
            end = start + cfgs.v_head_dim
            v = qkv[row, :, start:end]
            v_row_list.append(v)
        v_list.append(jnp.stack(v_row_list, axis=0))

    # (batch, num_v_heads, 1, kq_head_dim)
    q_split = jnp.stack(q_list, axis=0)
    k_split = jnp.stack(k_list, axis=0)
    # (batch, num_v_heads, 1, v_head_dim)
    v_split = jnp.stack(v_list, axis=0)

    return q_split, k_split, v_split


def recurrent_gdn(
    metadata_ref: ref_classes.MetadataRef,
    b_start: jax.Array,
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_states: jax.Array,
    prev_recurrent_state_ref: jax.Array,
    gdn_weights_ref: ref_classes.GDNWeightsRef,
    cfgs: configs.GDNConfigs,
):
    q, k, v = split_and_compute_qkv(qkv, cfgs)
    k_t = fused_transpose_broadcast(k, src_dim=3, dst_dim=2)

    a_log = gdn_weights_ref.a_log[...].reshape(1, 1, 1, -1)
    a_log = a_log.astype(cfgs.dtypes.compute)
    dt_bias = gdn_weights_ref.dt_bias[...].reshape(1, 1, 1, -1)
    dt_bias = dt_bias.astype(cfgs.dtypes.compute)

    beta = jax.nn.sigmoid(b)
    gating = -jnp.exp(a_log) * jax.nn.softplus(a + dt_bias)
    gating = jnp.exp(gating)

    beta = fused_transpose_broadcast(beta, src_dim=3, dst_dim=1)
    beta = beta[:, :cfgs.num_v_heads]
    gating = fused_transpose_broadcast(gating, src_dim=3, dst_dim=1)
    gating = gating[:, :cfgs.num_v_heads]

    out_list = []
    new_recurrent_state_list = []

    for head_start in range(0, cfgs.num_v_heads, cfgs.head_tile_size):
        head_end = min(head_start + cfgs.head_tile_size, cfgs.num_v_heads)
        head_slice = slice(head_start, head_end)
        prev_state = prev_recurrent_state_ref[head_slice]
        zero_state = jnp.zeros_like(prev_state)

        out_head_list = []
        new_recurrent_state_head_list = []

        for row in range(cfgs.tile_size):
            b_idx = b_start + row
            sz_from_old = metadata_ref.b_idx_to_sz_from_old[b_idx]
            s_idx = metadata_ref.b_idx_to_s_idx[b_idx]

            # (num_v_heads, kq_head_dim, v_head_dim)
            state_curr = recurrent_states[row, head_slice]
            has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]
            state_curr = jnp.where(has_initial_state, state_curr, zero_state)
            state_curr = jnp.where(sz_from_old != cfgs.prev_kernel_size,
                                   prev_state,
                                   state_curr).astype(cfgs.dtypes.compute)

            # (num_v_heads, 1, kq_head_dim)
            q_curr = q[row, head_slice]
            k_curr = k[row, head_slice]
            # (num_v_heads, 1, v_head_dim)
            v_curr = v[row, head_slice]

            # (num_v_heads, kq_head_dim, 1)
            k_curr_t = k_t[row, head_slice]

            # (num_v_heads, 1, 1)
            beta_curr = beta[row, head_slice]
            gating_curr = gating[row, head_slice]

            # (num_v_heads, kq_head_dim, v_head_dim)
            state_updated = state_curr * gating_curr

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

            out = jax.lax.dot_general(
                q_curr,
                state,
                (((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.dtypes.compute)

            prev_state = state

            out_head_list.append(out[:, 0, :])
            new_recurrent_state_head_list.append(state)

        out_list.append(jnp.stack(out_head_list, axis=0))
        new_recurrent_state_list.append(
            jnp.stack(new_recurrent_state_head_list, axis=0))

        prev_recurrent_state_ref[head_slice] = prev_state.astype(
            cfgs.dtypes.recurrent_state)

    out = jnp.concat(out_list, axis=1)
    new_recurrent_state = jnp.concat(new_recurrent_state_list, axis=1)

    return out, new_recurrent_state
