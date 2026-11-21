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
"""Fast head-major single-pass Pallas GDN decode kernel for TPU.

Supports optional in-register fused RMSNormGated (norm_before_gate=True,
activation="silu"/"swish"), vectorized 8-sublane conv1d/SiLU/L2-norm/norm,
for 1-token decode sequences [0, dist[0]).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def can_use_fast_decode(
    *,
    num_spec_tokens: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    dim: int,
    kernel_size: int,
    batch_size: int,
    num_seqs: int,
) -> bool:
    """Return True if fast_decode_conv1d_gdn supports the given GDN geometry."""
    return (num_spec_tokens == 0 and batch_size <= num_seqs and n_kq > 0
            and n_v > 0 and n_v % n_kq == 0 and n_v <= 128 and d_k == 128
            and d_v == 128 and dim % 128 == 0 and 2 <= kernel_size <= 8)


def fast_decode_conv1d_gdn(
    qkv: jax.Array,  # [padded_batch_size, 1, dim] or [batch_size, dim]
    b: jax.Array,  # [padded_batch_size, 1, aligned_nv] or [batch_size, n_v]
    a: jax.Array,  # [padded_batch_size, 1, aligned_nv] or [batch_size, n_v]
    conv_state: jax.
    Array,  # [num_blocks, prev_k, 1, dim] or [num_blocks, prev_k, dim]
    recurrent_state: jax.Array,  # [num_blocks, n_v, d_k, d_v]
    conv_weight: jax.
    Array,  # [kernel_size, 1, dim] (swapped) or [dim, 1, kernel_size]
    conv_bias: jax.Array | None,  # [dim]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    state_indices: jax.Array,  # [num_seqs]
    read_state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    tile_s: int = 8,
    zero_initialize_out: bool = False,
    act_out_dtype=None,
    conv_weight_swapped: bool = False,
    z: jax.Array
    | None = None,  # [batch_size, n_v * d_v] or [batch_size, n_v, d_v]
    norm_weight: jax.Array | None = None,  # [d_v]
    norm_eps: float = 1e-6,
    norm_bf16_roundtrip: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused conv1d + SiLU + recurrent GDN (+ optional RMSNormGated) for 1-token decode.

    Uses double-buffered whole-sequence async DMAs for conv_state and
    recurrent_state, 8-sublane vectorized conv1d/SiLU/L2-norm/RMSNormGated,
    matching GDN v3 MXU dot numerics bit-for-bit.
    """
    cs_is_4d = conv_state.ndim == 4
    qkv_is_3d = qkv.ndim == 3
    if qkv_is_3d:
        batch_size = qkv.shape[0]
        dim = qkv.shape[2]
        qkv_2d = qkv[:, 0, :]
        b_2d = b[:, 0, :]
        a_2d = a[:, 0, :]
    else:
        batch_size, dim = qkv.shape
        qkv_2d = qkv
        b_2d = b
        a_2d = a

    fuse_norm = z is not None and norm_weight is not None
    val_dim = n_v * d_v
    num_seqs = state_indices.shape[0]
    v_per_kq = n_v // n_kq
    prev_k = kernel_size - 1
    q_scale = float(d_k**-0.5)
    pad_nv = 128
    out_dtype = qkv.dtype if act_out_dtype is None else act_out_dtype

    if conv_weight_swapped:
        cw_raw = conv_weight[:, 0, :].astype(jnp.float32)
    else:
        cw_raw = jnp.transpose(conv_weight[:, 0, :],
                               (1, 0)).astype(jnp.float32)
    cw_t = jnp.pad(cw_raw, ((0, 8 - kernel_size), (0, 0)))

    has_bias = conv_bias is not None
    if has_bias:
        cb_p = jnp.broadcast_to(
            conv_bias.astype(jnp.float32).reshape(1, dim), (8, dim))
    else:
        cb_p = jnp.zeros((8, dim), dtype=jnp.float32)

    a_log_p = jnp.broadcast_to(
        jnp.pad(a_log.astype(jnp.float32),
                (0, pad_nv - n_v)).reshape(1, pad_nv),
        (8, pad_nv),
    )
    dt_bias_p = jnp.broadcast_to(
        jnp.pad(dt_bias.astype(jnp.float32),
                (0, pad_nv - n_v)).reshape(1, pad_nv),
        (8, pad_nv),
    )

    if fuse_norm:
        nw_p = jnp.broadcast_to(
            norm_weight.astype(jnp.float32).reshape(1, d_v), (8, d_v))
    else:
        nw_p = jnp.zeros((8, d_v), dtype=jnp.float32)

    n_tiles_max = (batch_size + tile_s - 1) // tile_s
    padded_T = n_tiles_max * tile_s
    pad_t = padded_T - batch_size
    qkv_p = jnp.pad(qkv_2d, ((0, pad_t), (0, 0))) if pad_t > 0 else qkv_2d
    b_pad_c = pad_nv - b_2d.shape[1]
    b_p = (jnp.pad(b_2d, ((0, pad_t), (0, b_pad_c))) if
           (pad_t > 0 or b_pad_c > 0) else b_2d)
    a_p = (jnp.pad(a_2d, ((0, pad_t), (0, b_pad_c))) if
           (pad_t > 0 or b_pad_c > 0) else a_2d)

    if fuse_norm:
        z_2d = z.reshape(batch_size, val_dim)
        z_p = jnp.pad(z_2d, ((0, pad_t), (0, 0))) if pad_t > 0 else z_2d
    else:
        z_p = None

    if padded_T > num_seqs:
        pad_n = padded_T - num_seqs
        state_idx_p = jnp.pad(state_indices, (0, pad_n))
        read_idx_p = jnp.pad(read_state_indices, (0, pad_n))
        seq_lens_p = jnp.pad(seq_lens, (0, pad_n))
    else:
        state_idx_p = state_indices
        read_idx_p = read_state_indices
        seq_lens_p = seq_lens

    cs_buf_shape = ((2, tile_s, prev_k, 1, dim) if cs_is_4d else
                    (2, tile_s, prev_k, dim))
    use_2d_out = not qkv_is_3d
    out_shape_tuple = (padded_T, val_dim) if use_2d_out else (padded_T, n_v,
                                                              d_v)
    out_buf_shape = (2, tile_s, val_dim) if use_2d_out else (2, tile_s, n_v,
                                                             d_v)

    def kernel(
        state_idx_ref,
        read_idx_ref,
        dist_ref,
        seq_lens_ref,
        cw_ref,
        cb_ref,
        alog_ref,
        dtb_ref,
        nw_ref,
        qkv_hbm,
        b_hbm,
        a_hbm,
        z_hbm,
        cs_hbm,
        rs_hbm,
        in_out_hbm,
        out_hbm,
        cs_out_hbm,
        rs_out_hbm,
        *scratches,
    ):
        del in_out_hbm
        it = iter(scratches)
        qkv_buf = next(it)
        b_buf = next(it)
        a_buf = next(it)
        z_buf = next(it) if fuse_norm else None
        out_buf = next(it)
        cs_in_buf = next(it)
        rs_in_buf = next(it)
        cs_out_buf = next(it)
        rs_out_buf = next(it)
        read_sems = next(it)
        write_sems = next(it)

        num_decode = dist_ref[0]
        num_active_tiles = (num_decode + tile_s - 1) // tile_s

        cb_tile = cb_ref[...] if has_bias else None
        a_neg_exp = -jnp.exp(alog_ref[0:1, :])
        dt_bias_row = dtb_ref[0:1, :]
        nw_tile = nw_ref[...] if fuse_norm else None
        lane_iota = jax.lax.broadcasted_iota(jnp.int32, (tile_s, 128), 1)

        def start_read(t_idx):
            slot = t_idx & 1
            base = t_idx * tile_s
            pltpu.make_async_copy(
                qkv_hbm.at[pl.ds(base, tile_s), :],
                qkv_buf.at[slot],
                read_sems.at[slot, 0],
            ).start()
            pltpu.make_async_copy(
                b_hbm.at[pl.ds(base, tile_s), :],
                b_buf.at[slot],
                read_sems.at[slot, 0],
            ).start()
            pltpu.make_async_copy(
                a_hbm.at[pl.ds(base, tile_s), :],
                a_buf.at[slot],
                read_sems.at[slot, 0],
            ).start()
            if fuse_norm:
                pltpu.make_async_copy(
                    z_hbm.at[pl.ds(base, tile_s), :],
                    z_buf.at[slot],
                    read_sems.at[slot, 0],
                ).start()
            for j in range(tile_s):
                seq_i = base + j

                @pl.when(seq_i < num_decode)
                def _do_r():
                    r_slot = read_idx_ref[seq_i]
                    pltpu.make_async_copy(
                        cs_hbm.at[r_slot],
                        cs_in_buf.at[slot, j],
                        read_sems.at[slot, 0],
                    ).start()
                    pltpu.make_async_copy(
                        rs_hbm.at[r_slot],
                        rs_in_buf.at[slot, j],
                        read_sems.at[slot, 1],
                    ).start()

        def wait_read(t_idx):
            slot = t_idx & 1
            base = t_idx * tile_s
            pltpu.make_async_copy(
                qkv_hbm.at[pl.ds(base, tile_s), :],
                qkv_buf.at[slot],
                read_sems.at[slot, 0],
            ).wait()
            pltpu.make_async_copy(
                b_hbm.at[pl.ds(base, tile_s), :],
                b_buf.at[slot],
                read_sems.at[slot, 0],
            ).wait()
            pltpu.make_async_copy(
                a_hbm.at[pl.ds(base, tile_s), :],
                a_buf.at[slot],
                read_sems.at[slot, 0],
            ).wait()
            if fuse_norm:
                pltpu.make_async_copy(
                    z_hbm.at[pl.ds(base, tile_s), :],
                    z_buf.at[slot],
                    read_sems.at[slot, 0],
                ).wait()
            for j in range(tile_s):
                seq_i = base + j

                @pl.when(seq_i < num_decode)
                def _do_wr():
                    r_slot = read_idx_ref[seq_i]
                    pltpu.make_async_copy(
                        cs_hbm.at[r_slot],
                        cs_in_buf.at[slot, j],
                        read_sems.at[slot, 0],
                    ).wait()
                    pltpu.make_async_copy(
                        rs_hbm.at[r_slot],
                        rs_in_buf.at[slot, j],
                        read_sems.at[slot, 1],
                    ).wait()

        def start_write(t_idx):
            slot = t_idx & 1
            base = t_idx * tile_s
            if use_2d_out:
                pltpu.make_async_copy(
                    out_buf.at[slot],
                    out_hbm.at[pl.ds(base, tile_s), :],
                    write_sems.at[slot, 0],
                ).start()
            else:
                pltpu.make_async_copy(
                    out_buf.at[slot],
                    out_hbm.at[pl.ds(base, tile_s), :, :],
                    write_sems.at[slot, 0],
                ).start()
            for j in range(tile_s):
                seq_i = base + j

                @pl.when(seq_i < num_decode)
                def _do_w():
                    w_slot = state_idx_ref[seq_i]
                    pltpu.make_async_copy(
                        cs_out_buf.at[slot, j],
                        cs_out_hbm.at[w_slot],
                        write_sems.at[slot, 0],
                    ).start()
                    pltpu.make_async_copy(
                        rs_out_buf.at[slot, j],
                        rs_out_hbm.at[w_slot],
                        write_sems.at[slot, 1],
                    ).start()

        def wait_write(t_idx):
            slot = t_idx & 1
            base = t_idx * tile_s
            if use_2d_out:
                pltpu.make_async_copy(
                    out_buf.at[slot],
                    out_hbm.at[pl.ds(base, tile_s), :],
                    write_sems.at[slot, 0],
                ).wait()
            else:
                pltpu.make_async_copy(
                    out_buf.at[slot],
                    out_hbm.at[pl.ds(base, tile_s), :, :],
                    write_sems.at[slot, 0],
                ).wait()
            for j in range(tile_s):
                seq_i = base + j

                @pl.when(seq_i < num_decode)
                def _do_ww():
                    w_slot = state_idx_ref[seq_i]
                    pltpu.make_async_copy(
                        cs_out_buf.at[slot, j],
                        cs_out_hbm.at[w_slot],
                        write_sems.at[slot, 0],
                    ).wait()
                    pltpu.make_async_copy(
                        rs_out_buf.at[slot, j],
                        rs_out_hbm.at[w_slot],
                        write_sems.at[slot, 1],
                    ).wait()

        @pl.when(num_active_tiles > 0)
        def _run():
            start_read(0)

            def body(step_i, _):
                slot = step_i & 1
                base = step_i * tile_s

                @pl.when(step_i >= 2)
                def _ww_prev2():
                    wait_write(step_i - 2)

                @pl.when(step_i + 1 < num_active_tiles)
                def _pref_next():
                    start_read(step_i + 1)

                wait_read(step_i)

                qkv_tile = qkv_buf[slot].astype(jnp.float32)
                b_tile = b_buf[slot].astype(jnp.float32)
                a_tile = a_buf[slot].astype(jnp.float32)

                beta_tile = jax.nn.sigmoid(b_tile)
                g_tile = jnp.exp(a_neg_exp *
                                 jax.nn.softplus(a_tile + dt_bias_row))

                # Vectorized 8-sublane conv1d + SiLU across the tile.
                cs_tap_rows = [[] for _ in range(prev_k)]
                for j in range(tile_s):
                    seq_i = base + j
                    has_init = (seq_i < num_decode) & (seq_lens_ref[seq_i] > 1)
                    if cs_is_4d:
                        cs_j = cs_in_buf[slot, j, :, 0, :].astype(jnp.float32)
                    else:
                        cs_j = cs_in_buf[slot, j].astype(jnp.float32)
                    cs_j = jnp.where(has_init, cs_j, 0.0)
                    for k in range(prev_k):
                        cs_tap_rows[k].append(cs_j[k:k + 1, :])
                    x_j = qkv_tile[j:j + 1, :]
                    if prev_k > 1:
                        new_cs = jnp.concatenate([cs_j[1:, :], x_j], axis=0)
                    else:
                        new_cs = x_j
                    if cs_is_4d:
                        cs_out_buf[slot, j] = new_cs.reshape(
                            prev_k, 1, dim).astype(cs_out_buf.dtype)
                    else:
                        cs_out_buf[slot, j] = new_cs.astype(cs_out_buf.dtype)

                conv_tile = jnp.concatenate(cs_tap_rows[0],
                                            axis=0) * jnp.broadcast_to(
                                                cw_ref[0:1, :], (tile_s, dim))
                for k in range(1, prev_k):
                    conv_tile = conv_tile + jnp.concatenate(
                        cs_tap_rows[k], axis=0) * jnp.broadcast_to(
                            cw_ref[k:k + 1, :], (tile_s, dim))
                conv_tile = conv_tile + qkv_tile * jnp.broadcast_to(
                    cw_ref[prev_k:prev_k + 1, :], (tile_s, dim))
                if has_bias:
                    conv_tile = conv_tile + cb_tile
                conv_tile = jax.nn.silu(conv_tile)

                # Vectorized 8-sublane L2 norm for Q and K heads (matches v3
                # compute_gdn.recurrent_gdn_per_seq bit-for-bit via / sqrt).
                q_norm_tile = []
                k_norm_tile = []
                for kh in range(n_kq):
                    q_h = conv_tile[:, kh * d_k:(kh + 1) * d_k]
                    k_h = conv_tile[:, (n_kq + kh) * d_k:(n_kq + kh + 1) * d_k]
                    q_norm_tile.append((q_h / jnp.sqrt(
                        jnp.sum(q_h * q_h, axis=-1, keepdims=True) + 1e-6)) *
                                       q_scale)
                    k_norm_tile.append(k_h / jnp.sqrt(
                        jnp.sum(k_h * k_h, axis=-1, keepdims=True) + 1e-6))

                # Vectorized 8-sublane beta/g head extraction (n_v ops instead of tile_s * n_v).
                beta_per_h = []
                g_per_h = []
                for vh in range(n_v):
                    m = lane_iota == vh
                    beta_per_h.append(
                        jnp.sum(jnp.where(m, beta_tile, 0.0),
                                axis=-1,
                                keepdims=True))
                    g_per_h.append(
                        jnp.sum(jnp.where(m, g_tile, 0.0),
                                axis=-1,
                                keepdims=True))

                v_base = 2 * n_kq * d_k
                out_rows = []
                for j in range(tile_s):
                    seq_i = base + j
                    has_init = (seq_i < num_decode) & (seq_lens_ref[seq_i] > 1)
                    h_outs = []
                    for kh in range(n_kq):
                        q_h = q_norm_tile[kh][j:j + 1, :]
                        k_h = k_norm_tile[kh][j:j + 1, :]
                        k_col = jnp.broadcast_to(k_h, (128, 128)).T
                        for sub_v in range(v_per_kq):
                            vh = kh * v_per_kq + sub_v
                            v_h = conv_tile[j:j + 1, v_base + vh * d_v:v_base +
                                            (vh + 1) * d_v]
                            beta_h = beta_per_h[vh][j:j + 1, :]
                            g_h = g_per_h[vh][j:j + 1, :]

                            S = rs_in_buf[slot, j, vh].astype(jnp.float32)
                            S = jnp.where(has_init, S * g_h, 0.0)
                            v_pred = jax.lax.dot(
                                k_h, S, preferred_element_type=jnp.float32)
                            v_new = beta_h * (v_h - v_pred)
                            S = S + k_col * v_new
                            o_h = jax.lax.dot(
                                q_h, S, preferred_element_type=jnp.float32)
                            rs_out_buf[slot, j,
                                       vh] = S.astype(rs_out_buf.dtype)
                            h_outs.append(o_h)
                    out_rows.append(jnp.concatenate(h_outs,
                                                    axis=1))  # (1, val_dim)

                out_tile = jnp.concatenate(out_rows,
                                           axis=0)  # (tile_s, val_dim)
                if fuse_norm:
                    z_tile = z_buf[slot].astype(jnp.float32)
                    normed_h = []
                    for vh in range(n_v):
                        oh_t = out_tile[:, vh * d_v:(vh + 1) * d_v]
                        if norm_bf16_roundtrip:
                            oh_t = oh_t.astype(out_dtype).astype(jnp.float32)
                        var_t = jnp.mean(oh_t * oh_t, axis=-1, keepdims=True)
                        zh_t = z_tile[:, vh * d_v:(vh + 1) * d_v]
                        normed_h.append(
                            (oh_t * jax.lax.rsqrt(var_t + norm_eps) * nw_tile)
                            * jax.nn.silu(zh_t))
                    out_tile = jnp.concatenate(normed_h, axis=1)

                if use_2d_out:
                    out_buf[slot] = out_tile.astype(out_buf.dtype)
                else:
                    out_buf[slot] = out_tile.reshape(tile_s, n_v,
                                                     d_v).astype(out_buf.dtype)
                start_write(step_i)
                return None

            jax.lax.fori_loop(0, num_active_tiles, body, None)

            wait_write(num_active_tiles - 1)

            @pl.when(num_active_tiles >= 2)
            def _ww_last2():
                wait_write(num_active_tiles - 2)

    out_struct = jax.ShapeDtypeStruct(out_shape_tuple, out_dtype)
    in_out_act = jnp.zeros_like(out_struct) if zero_initialize_out else None
    in_out_spec = (pl.BlockSpec(
        memory_space=pltpu.HBM) if in_out_act is not None else None)
    z_spec = pl.BlockSpec(memory_space=pltpu.HBM) if fuse_norm else None
    # Input indices after PyTree flattening (None arguments are omitted):
    # 0..3: SMEM (state_idx, read_idx, dist, seq_lens)
    # 4..8: VMEM (cw, cb, alog, dtb, nw)
    # 9..11: HBM (qkv, b, a)
    # 12 (if fuse_norm): HBM (z_p)
    # cs_idx, cs_idx+1, cs_idx+2: HBM (cs, rs, in_out_act)
    cs_idx = 13 if fuse_norm else 12
    aliases = {cs_idx: 1, cs_idx + 1: 2}
    if in_out_act is not None:
        aliases[cs_idx + 2] = 0

    out_shape = (
        out_struct,
        jax.ShapeDtypeStruct(conv_state.shape, conv_state.dtype),
        jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
    )
    scratch_shapes = [
        pltpu.VMEM((2, tile_s, dim), qkv_p.dtype),
        pltpu.VMEM((2, tile_s, pad_nv), b_p.dtype),
        pltpu.VMEM((2, tile_s, pad_nv), a_p.dtype),
    ]
    if fuse_norm:
        scratch_shapes.append(pltpu.VMEM((2, tile_s, val_dim), z_p.dtype))
    scratch_shapes.extend([
        pltpu.VMEM(out_buf_shape, out_dtype),
        pltpu.VMEM(cs_buf_shape, conv_state.dtype),
        pltpu.VMEM((2, tile_s, n_v, d_k, d_v), recurrent_state.dtype),
        pltpu.VMEM(cs_buf_shape, conv_state.dtype),
        pltpu.VMEM((2, tile_s, n_v, d_k, d_v), recurrent_state.dtype),
        pltpu.SemaphoreType.DMA((2, 2)),
        pltpu.SemaphoreType.DMA((2, 2)),
    ])
    out, ncs, nrs = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            z_spec,
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            in_out_spec,
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        name=f"fast_gdn_decode_ts{tile_s}_fn{int(fuse_norm)}",
    )(
        state_idx_p,
        read_idx_p,
        distribution,
        seq_lens_p,
        cw_t,
        cb_p,
        a_log_p,
        dt_bias_p,
        nw_p,
        qkv_p,
        b_p,
        a_p,
        z_p,
        conv_state,
        recurrent_state,
        in_out_act,
    )
    return out[:batch_size], ncs, nrs
