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
"""
Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl

"""
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.gdn.v3 import wrapper
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


def _row_perm(pcp: int) -> np.ndarray:
    """Head-tail chunk order of the concatenated per-rank shards.

    Under prefill context parallelism (PCP) a request's ``2*pcp`` equal token
    chunks are load-balanced head-tail: rank ``r`` owns global chunk ``r`` (head)
    and chunk ``2*pcp-1-r`` (tail), stored in that order. All-gathering the
    per-rank shards over the pcp axis and flattening therefore yields the global
    chunks in the order ``[0, 2P-1, 1, 2P-2, ...]`` -- this permutation.
    """
    return np.asarray([c for r in range(pcp) for c in (r, 2 * pcp - 1 - r)],
                      dtype=np.int64)


def _inv_row(pcp: int) -> np.ndarray:
    """Inverse of :func:`_row_perm`: global chunk index -> its gathered slot."""
    inv = np.empty(2 * pcp, np.int64)
    inv[_row_perm(pcp)] = np.arange(2 * pcp)
    return inv


def _take_group_last(x: jnp.ndarray, n_heads: int, head_dim: int,
                     group: jax.Array, num_groups: int) -> jnp.ndarray:
    """Keep contiguous head ``group`` of ``num_groups`` along the last axis.

    ``x`` is ``[..., n_heads * head_dim]``; returns
    ``[..., (n_heads // num_groups) * head_dim]``. ``group`` may be traced.
    """
    per = n_heads // num_groups
    lead = x.shape[:-1]
    x = x.reshape(*lead, n_heads, head_dim)
    x = lax.dynamic_slice_in_dim(x, group * per, per, axis=x.ndim - 2)
    return x.reshape(*lead, per * head_dim)


def _take_group_first(x: jnp.ndarray, n_heads: int, head_dim: int,
                      group: jax.Array, num_groups: int) -> jnp.ndarray:
    """Keep contiguous head ``group`` of ``num_groups`` along the first axis.

    ``x`` is ``[n_heads * head_dim, ...]``; returns
    ``[(n_heads // num_groups) * head_dim, ...]``.
    """
    per = n_heads // num_groups
    trail = x.shape[1:]
    x = x.reshape(n_heads, head_dim, *trail)
    x = lax.dynamic_slice_in_dim(x, group * per, per, axis=0)
    return x.reshape(per * head_dim, *trail)


def _split_qkv(x: jnp.ndarray, n_kq: int, n_v: int, d_k: int, d_v: int,
               axis: int):
    """Split a fused ``[q | k | v]`` tensor into its three blocks along ``axis``."""
    kq = n_kq * d_k
    return (lax.slice_in_dim(x, 0, kq, axis=axis),
            lax.slice_in_dim(x, kq, 2 * kq, axis=axis),
            lax.slice_in_dim(x, 2 * kq, 2 * kq + n_v * d_v, axis=axis))


def _gdn_local_pcp(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    *,
    pcp_axis: str,
    pcp_size: int,
    n_kq_local: int,
    n_v_local: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Per-shard GDN body for PCP: gather tokens, distribute heads over pcp.

    Runs *inside* the ``shard_map``. On entry the local shard holds this rank's
    head-tail token slice (``[2*C, dim_tp]``) with the full TP head set; the
    per-request metadata (``query_start_loc`` / ``seq_lens`` / ``distribution`` /
    ``state_indices``) and the state caches are replicated over the pcp axis.

    Strategy (no wasted compute -- every linear-attention head is processed on
    exactly one device):

    1. All-gather the token tensors over pcp and un-permute the head-tail layout,
       reconstructing the *full* padded sequence a single-device run would see.
    2. Slice this rank's contiguous head group, so heads are distributed over
       ``tp * pcp`` devices: each runs the full sequence for ``1/pcp`` of its TP
       heads.
    3. Run the ordinary fused conv1d+GDN kernel on the full sequence.
    4. All-gather the per-group output / conv-state / recurrent-state back over
       the head axis (every rank ends with the full TP-head result) and re-shard
       the output tokens back to this rank's head-tail slice.
    """
    r = lax.axis_index(pcp_axis)
    local_tokens, dim_tp = j_mixed_qkv.shape
    full_s = local_tokens * pcp_size
    chunk = local_tokens // 2  # C: each rank owns 2 chunks (head + tail)
    inv = _inv_row(pcp_size)

    def gather_tokens(x: jnp.ndarray) -> jnp.ndarray:
        # [2*C, w] -> all-gather -> [full_s, w] in rank order -> natural order.
        g = lax.all_gather(x, pcp_axis, axis=0, tiled=True)
        w = x.shape[1:]
        return g.reshape(2 * pcp_size, chunk, *w)[inv].reshape(full_s, *w)

    qkv_full = gather_tokens(j_mixed_qkv)
    b_full = gather_tokens(j_b)
    a_full = gather_tokens(j_a)

    # --- slice this rank's contiguous head group (distribute heads over pcp) ---
    q_blk, k_blk, v_blk = _split_qkv(qkv_full,
                                     n_kq_local,
                                     n_v_local,
                                     d_k,
                                     d_v,
                                     axis=1)
    qkv_r = jnp.concatenate([
        _take_group_last(q_blk, n_kq_local, d_k, r, pcp_size),
        _take_group_last(k_blk, n_kq_local, d_k, r, pcp_size),
        _take_group_last(v_blk, n_v_local, d_v, r, pcp_size),
    ],
                            axis=1)
    b_r = _take_group_last(b_full, n_v_local, 1, r, pcp_size)
    a_r = _take_group_last(a_full, n_v_local, 1, r, pcp_size)

    cw_q, cw_k, cw_v = _split_qkv(j_conv_weight,
                                  n_kq_local,
                                  n_v_local,
                                  d_k,
                                  d_v,
                                  axis=0)
    conv_weight_r = jnp.concatenate([
        _take_group_first(cw_q, n_kq_local, d_k, r, pcp_size),
        _take_group_first(cw_k, n_kq_local, d_k, r, pcp_size),
        _take_group_first(cw_v, n_v_local, d_v, r, pcp_size),
    ],
                                    axis=0)
    conv_bias_r = None
    if j_conv_bias is not None:
        cb_q, cb_k, cb_v = _split_qkv(j_conv_bias,
                                      n_kq_local,
                                      n_v_local,
                                      d_k,
                                      d_v,
                                      axis=0)
        conv_bias_r = jnp.concatenate([
            _take_group_first(cb_q, n_kq_local, d_k, r, pcp_size),
            _take_group_first(cb_k, n_kq_local, d_k, r, pcp_size),
            _take_group_first(cb_v, n_v_local, d_v, r, pcp_size),
        ],
                                      axis=0)

    gv = n_v_local // pcp_size
    a_log_r = lax.dynamic_slice_in_dim(j_A_log, r * gv, gv, axis=0)
    dt_bias_r = lax.dynamic_slice_in_dim(j_dt_bias, r * gv, gv, axis=0)
    recurrent_r = lax.dynamic_slice_in_dim(recurrent_state, r * gv, gv, axis=1)

    cs_q, cs_k, cs_v = _split_qkv(conv_state,
                                  n_kq_local,
                                  n_v_local,
                                  d_k,
                                  d_v,
                                  axis=2)
    conv_state_r = jnp.concatenate([
        _take_group_last(cs_q, n_kq_local, d_k, r, pcp_size),
        _take_group_last(cs_k, n_kq_local, d_k, r, pcp_size),
        _take_group_last(cs_v, n_v_local, d_v, r, pcp_size),
    ],
                                   axis=2)

    (new_conv_r, new_rec_r), out_r = wrapper.fused_conv1d_gdn(
        qkv_r,
        b_r,
        a_r,
        conv_state_r,
        recurrent_r,
        conv_weight_r,
        conv_bias_r,
        a_log_r,
        dt_bias_r,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        n_kq=n_kq_local // pcp_size,
        n_v=gv,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    # --- regather the head-distributed results over the pcp head axis ---------
    out_full = lax.all_gather(out_r, pcp_axis, axis=1,
                              tiled=True)  # [full_s, n_v_local*d_v]
    new_rec = lax.all_gather(new_rec_r, pcp_axis, axis=1,
                             tiled=True)  # [nb, n_v_local, d_k, d_v]

    ncs_q, ncs_k, ncs_v = _split_qkv(new_conv_r,
                                     n_kq_local // pcp_size,
                                     gv,
                                     d_k,
                                     d_v,
                                     axis=2)
    new_conv = jnp.concatenate([
        lax.all_gather(ncs_q, pcp_axis, axis=2, tiled=True),
        lax.all_gather(ncs_k, pcp_axis, axis=2, tiled=True),
        lax.all_gather(ncs_v, pcp_axis, axis=2, tiled=True),
    ],
                               axis=2)

    # --- re-shard the output tokens back to this rank's head-tail slice -------
    w = out_full.shape[-1]
    out_rank = out_full.reshape(2 * pcp_size, chunk, w)[_row_perm(pcp_size)]
    out_local = lax.dynamic_index_in_dim(out_rank.reshape(
        pcp_size, 2 * chunk, w),
                                         r,
                                         axis=0,
                                         keepdims=False)
    return (new_conv, new_rec), out_local


def run_jax_gdn_attention(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the Jax GDN attention mechanism.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        conv_state: Convolutional state tensor of shape `(num_blocks, kernel_size
          - 1, dim)`. `num_blocks` is always equal or larger than `max_seqs +
          1`. The first block is a null_block and only used for padded / invalid
          tokens.
        recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
          d_v)`.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used inside the local function
          to derive ``has_initial_state``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(num_blocks, kernel_size - 1, dim)`
          - new_recurrent_state: `(num_blocks, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    # NOTE ON PCP SHARDING: ``ATTN_DATA`` includes the ``pcp`` axis, so the token
    # tensors (qkv/b/a) and the output are split head-tail over pcp -- that is the
    # PCP token layout. Everything that is *per request* rather than *per token* --
    # the two state caches (indexed by request slot on their leading dim) and the
    # ragged metadata -- must be REPLICATED over pcp, i.e. sharded on ``BATCH``
    # (``ATTN_DATA`` minus ``pcp``). Sharding the state caches on ``ATTN_DATA``
    # (the pre-PCP behavior) silently splits each request's state along the block
    # axis over pcp and corrupts it. ``BATCH == ATTN_DATA`` whenever pcp is absent,
    # so this is a no-op for the non-PCP / decode-only paths.
    in_specs = (
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_b
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_a
        P(ShardingAxisName.BATCH, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state
        P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD, None,
          None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(ShardingAxisName.BATCH),  # query_start_loc
        P(ShardingAxisName.BATCH),  # state_indices
        P(ShardingAxisName.BATCH),  # distribution
        P(ShardingAxisName.BATCH),  # seq_lens
    )

    out_specs = (
        (
            P(ShardingAxisName.BATCH, None,
              ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(ShardingAxisName.BATCH, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    pcp_size = get_mesh_shape_product(mesh, ShardingAxisName.PREFILL_CONTEXT)

    if pcp_size > 1:
        n_kq_local = n_kq // tp_size
        n_v_local = n_v // tp_size
        # Heads are distributed over ``tp * pcp`` devices, so each rank must own a
        # whole number of both key/query and value heads after the pcp split. This
        # holds for the hybrid models we target (e.g. Qwen3-Next n_kq=16, n_v=32).
        if n_kq_local % pcp_size != 0 or n_v_local % pcp_size != 0:
            raise ValueError(
                "GDN+PCP requires n_kq and n_v to be divisible by tp_size * "
                f"pcp_size; got n_kq={n_kq}, n_v={n_v}, tp_size={tp_size}, "
                f"pcp_size={pcp_size} (per-tp n_kq={n_kq_local}, n_v={n_v_local})."
            )
        p_run_jax_gdn_attention_local = functools.partial(
            _gdn_local_pcp,
            pcp_axis=ShardingAxisName.PREFILL_CONTEXT,
            pcp_size=pcp_size,
            n_kq_local=n_kq_local,
            n_v_local=n_v_local,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )
    else:
        p_run_jax_gdn_attention_local = functools.partial(
            wrapper.fused_conv1d_gdn,
            n_kq=n_kq // tp_size,
            n_v=n_v // tp_size,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )

    return (new_conv_state, new_recurrent_state), output
