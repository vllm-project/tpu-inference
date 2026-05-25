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


def load_as_qkv_large(
        qkv_vmem_ref: jax.Ref,
        cfgs: configs.GDNConfigs) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Use strided LDST to split qkv along last dim and perform transpose."""

    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = qkv_vmem_ref.shape[-1] // num_lanes
    kq_lanes_per_head = cfgs.kq_head_dim // num_lanes
    k_offset = cfgs.num_kq_heads * kq_lanes_per_head

    q_large_list = []
    k_large_list = []
    v_large_list = []

    qkv_slot_flat_ref = qkv_vmem_ref.reshape(-1, num_lanes)
    for kq_head in range(cfgs.num_kq_heads):
        q_head_list = []
        k_head_list = []
        for lane in range(kq_lanes_per_head):
            q_lane = kq_head * kq_lanes_per_head + lane
            k_lane = k_offset + q_lane

            q_head_list.append(qkv_slot_flat_ref[q_lane::lanes_per_col])
            k_head_list.append(qkv_slot_flat_ref[k_lane::lanes_per_col])
        q_large_list.append(jnp.concat(q_head_list, axis=-1))
        k_large_list.append(jnp.concat(k_head_list, axis=-1))
    v_offset = kq_lanes_per_head * cfgs.num_kq_heads * 2
    v_lanes_per_head = cfgs.v_head_dim // num_lanes
    for v_head in range(cfgs.num_v_heads):
        v_head_list = []
        for lane in range(v_lanes_per_head):
            v_lane = v_offset + v_head * v_lanes_per_head + lane
            v_head_list.append(qkv_slot_flat_ref[v_lane::lanes_per_col])
        v_large_list.append(jnp.concat(v_head_list, axis=-1))

    q_large = jnp.stack(q_large_list, axis=0)
    k_large = jnp.stack(k_large_list, axis=0)
    v_large = jnp.stack(v_large_list, axis=0)

    return q_large, k_large, v_large


def load_as_qkv_compact(
    qkv_vmem_ref: jax.Ref,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array]:

    # (seqs, chunk, 1, num_kq_heads * kq_head_dim * 2 + num_v_heads * v_head_dim)
    k_offset = cfgs.num_kq_heads * cfgs.kq_head_dim
    v_offset = cfgs.num_kq_heads * 2 * cfgs.kq_head_dim

    q_compact_list = []
    k_compact_list = []
    v_compact_list = []

    for kq_head in range(cfgs.num_kq_heads):
        q_start = kq_head * cfgs.kq_head_dim
        q_end = q_start + cfgs.kq_head_dim
        k_start = k_offset + q_start
        k_end = k_start + cfgs.kq_head_dim
        q_compact_list.append(qkv_vmem_ref[..., q_start:q_end])
        k_compact_list.append(qkv_vmem_ref[..., k_start:k_end])
    for v_head in range(cfgs.num_v_heads):
        v_start = v_offset + v_head * cfgs.v_head_dim
        v_end = v_start + cfgs.v_head_dim
        v_compact_list.append(qkv_vmem_ref[..., v_start:v_end])

    q_compact = jnp.stack(q_compact_list, axis=1)
    k_compact = jnp.stack(k_compact_list, axis=1)
    v_compact = jnp.stack(v_compact_list, axis=1)

    return q_compact, k_compact, v_compact


def load_compact_to_large(vmem_ref) -> jax.Array:
    # NOTE: Only support 32-bits for now.
    assert vmem_ref.dtype.itemsize == 4
    assert vmem_ref.shape[-2] == 1
    col_size = vmem_ref.shape[-1]
    new_shape = vmem_ref.shape[:-2] + (col_size, )
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes

    vreg_list = []
    vmem_ref = vmem_ref.reshape(-1, col_size)
    for col_start in range(0, col_size, num_lanes):
        col_end = min(col_start + num_lanes, col_size)
        vreg = vmem_ref[..., col_start:col_end]
        vreg_list.append(vreg)
    return jnp.concat(vreg_list, axis=-1).reshape(new_shape)


def load_and_mask_states(
    metadata_ref: ref_classes.MetadataRef,
    st_idx: jax.Array,
    conv_state_slot_ref: jax.Array,
    recurrent_slot_ref: jax.Array,
    prev_qkv_scratch_ref: jax.Array | None,
    prev_recurrent_state_scratch_ref: jax.Array | None,
    cfgs: configs.GDNConfigs,
) -> tuple[list[jax.Array | int], jax.Array, jax.Array]:
    real_size_list = []
    prev_conv_state_list = []
    prev_recurrent_state_list = []

    for idx in range(cfgs.seq_tile_size):
        curr_st_idx = st_idx * cfgs.seq_tile_size + idx
        s_idx = metadata_ref.st_idx_to_s_idx[curr_st_idx]
        real_size = metadata_ref.st_idx_to_b_size[curr_st_idx]

        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]
        conv_state = conv_state_slot_ref[idx].astype(jnp.float32)
        conv_state = jnp.where(has_initial_state, conv_state, 0)

        recurrent_state = recurrent_slot_ref[idx].astype(cfgs.dtypes.compute)
        recurrent_state = jnp.where(has_initial_state, recurrent_state, 0)

        prev_conv_state = conv_state
        prev_recurrent_state = recurrent_state
        if prev_qkv_scratch_ref is not None:
            is_first_tile = metadata_ref.st_idx_is_first_tile[curr_st_idx]
            prev_qkv_scratch = prev_qkv_scratch_ref[idx]
            prev_conv_state = jnp.where(is_first_tile, conv_state,
                                        prev_qkv_scratch)

        if prev_recurrent_state_scratch_ref is not None:
            is_first_tile = metadata_ref.st_idx_is_first_tile[curr_st_idx]
            prev_recurrent_state_scratch = prev_recurrent_state_scratch_ref[
                idx]
            prev_recurrent_state = jnp.where(is_first_tile, recurrent_state,
                                             prev_recurrent_state_scratch)

        real_size_list.append(real_size)
        prev_conv_state_list.append(prev_conv_state)
        prev_recurrent_state_list.append(prev_recurrent_state)

    prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
    prev_recurrent_state = jnp.stack(prev_recurrent_state_list, axis=0)

    return real_size_list, prev_conv_state, prev_recurrent_state


def load_activation_as_compact(
    real_size: list[jax.Array | int],
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    for s_idx in range(cfgs.seq_tile_size):
        qkv_s = qkv_vreg[s_idx]
        iota = jax.lax.broadcasted_iota(jnp.int32, qkv_s.shape, 0)
        qkv_vmem_ref[s_idx] = jnp.where(iota < real_size[s_idx], qkv_s, 0)

    q_compact, k_compact, v_compact = load_as_qkv_compact(qkv_vmem_ref, cfgs)

    b_compact_list = []
    a_compact_list = []
    for s_idx in range(cfgs.seq_tile_size):
        b_s = b_vmem_ref[s_idx]
        a_s = a_vmem_ref[s_idx]
        # NOTE: When using compact layout, certain operations offer no benefit from
        # using 16-bits compute.
        iota = jax.lax.broadcasted_iota(jnp.int32, b_s.shape, 0)
        b_s = jnp.where(iota < real_size[s_idx], b_s, 0)
        a_s = jnp.where(iota < real_size[s_idx], a_s, 0)
        b_compact_list.append(b_s)
        a_compact_list.append(a_s)
    b_compact = jnp.stack(b_compact_list, axis=0)
    a_compact = jnp.stack(a_compact_list, axis=0)
    b_compact = jnp.expand_dims(b_compact, axis=1)
    a_compact = jnp.expand_dims(a_compact, axis=1)

    return q_compact, k_compact, v_compact, b_compact, a_compact


def load_activation_as_large(
    real_size: list[jax.Array | int],
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: configs.GDNConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    qkv_vmem_ref[...] = qkv_vreg

    q_large_list = []
    k_large_list = []
    v_large_list = []
    b_large_list = []
    a_large_list = []
    for s_idx in range(cfgs.seq_tile_size):
        q_large, k_large, v_large = load_as_qkv_large(qkv_vmem_ref.at[s_idx],
                                                      cfgs)
        b_large = load_compact_to_large(b_vmem_ref.at[s_idx])
        a_large = load_compact_to_large(a_vmem_ref.at[s_idx])
        b_large = jnp.expand_dims(b_large, axis=0)
        a_large = jnp.expand_dims(a_large, axis=0)

        q_large = q_large.astype(cfgs.dtypes.compute)
        k_large = k_large.astype(cfgs.dtypes.compute)
        v_large = v_large.astype(cfgs.dtypes.compute)

        iota = jax.lax.broadcasted_iota(jnp.int16, (1, cfgs.tok_tile_size, 1),
                                        0)
        mask = iota < real_size[s_idx]

        q_large_list.append(
            jnp.where(mask, q_large.astype(cfgs.dtypes.compute), 0))
        k_large_list.append(
            jnp.where(mask, k_large.astype(cfgs.dtypes.compute), 0))
        v_large_list.append(
            jnp.where(mask, v_large.astype(cfgs.dtypes.compute), 0))
        b_large_list.append(
            jnp.where(mask, b_large.astype(cfgs.dtypes.compute), 0))
        a_large_list.append(
            jnp.where(mask, a_large.astype(cfgs.dtypes.compute), 0))

    q_large = jnp.stack(q_large_list, axis=0)
    k_large = jnp.stack(k_large_list, axis=0)
    v_large = jnp.stack(v_large_list, axis=0)
    b_large = jnp.stack(b_large_list, axis=0)
    a_large = jnp.stack(a_large_list, axis=0)

    return q_large, k_large, v_large, b_large, a_large
