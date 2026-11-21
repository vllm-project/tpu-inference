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
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.gdn.v3 import fast_decode
from tpu_inference.kernels.gdn.v3.rms_norm_gated import pallas_rms_norm_gated
from tpu_inference.layers.common.attention_backend import (
    Conv1dGDNKernel, GDNLinearAttentionBackend)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product


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
    read_state_indices: Optional[jnp.ndarray] = None,
    conv1d_gdn_kernel: Optional[Conv1dGDNKernel] = None,
    read_offsets: Optional[jnp.ndarray] = None,
    num_spec_tokens: int = 0,
    j_z: Optional[jnp.ndarray] = None,
    j_norm_weight: Optional[jnp.ndarray] = None,
    norm_eps: float = 1e-6,
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
        read_state_indices: Optional tensor of shape `(max_reqs,)` mapping
          request index to the state index its initial state is read from.
          Defaults to `state_indices` (read and write the same slot). Mamba
          prefix caching passes a different slot here, since a request
          resumes from the cached state block of the last block boundary and
          checkpoints into the block covering its current position.
        conv1d_gdn_kernel: The linear attention backend's fused conv1d + GDN
          kernel. Defaults to the GDN v3 kernel.

    Returns:
        A tuple containing:
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(num_blocks, kernel_size - 1, dim)`
          - new_recurrent_state: `(num_blocks, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    if read_state_indices is None:
        read_state_indices = state_indices
    if num_spec_tokens > 0:
        assert read_offsets is not None, (
            "read_offsets is required when num_spec_tokens > 0")

    fuse_norm = j_z is not None and j_norm_weight is not None
    in_specs = (
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_b
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_a
        (
            P(ShardingAxisName.ATTN_DATA, None,
              ShardingAxisName.ATTN_HEAD),  # conv_state
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
              None),  # recurrent_state
        ),
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(ShardingAxisName.ATTN_DATA),  # query_start_loc
        P(ShardingAxisName.ATTN_DATA),  # state_indices
        P(ShardingAxisName.ATTN_DATA),  # distribution
        P(ShardingAxisName.ATTN_DATA),  # seq_lens
        P(ShardingAxisName.ATTN_DATA),  # read_state_indices
    )
    if read_offsets is not None:
        in_specs = in_specs + (
            P(ShardingAxisName.ATTN_DATA),  # read_offsets
        )
    if fuse_norm:
        in_specs = in_specs + (
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_z
            P(None),  # j_norm_weight
        )

    out_specs = (
        (
            P(ShardingAxisName.ATTN_DATA, None,
              ShardingAxisName.ATTN_HEAD),  # conv_state
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
              None),  # recurrent_state
        ),
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    n_kq_local = n_kq // tp_size
    n_v_local = n_v // tp_size
    base_kernel = (conv1d_gdn_kernel
                   or GDNLinearAttentionBackend().conv1d_gdn_kernel())
    has_read_offsets = read_offsets is not None

    def _run_local(*args):
        (
            qkv_l,
            b_l,
            a_l,
            states_l,
            cw_l,
            cb_l,
            alog_l,
            dtb_l,
            qsl_l,
            si_l,
            dist_l,
            sl_l,
            rsi_l,
        ) = args[:13]
        idx = 13
        ro_l = args[idx] if has_read_offsets else None
        if has_read_offsets:
            idx += 1
        z_l = args[idx] if fuse_norm else None
        nw_l = args[idx + 1] if fuse_norm else None

        batch_size, dim = qkv_l.shape
        num_seqs = si_l.size
        cs_l, rs_l = states_l

        def _call_base(st):
            kw = dict(
                n_kq=n_kq_local,
                n_v=n_v_local,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
            )
            if has_read_offsets:
                kw["num_spec_tokens"] = num_spec_tokens
                out_st, out_a = base_kernel(
                    qkv_l,
                    b_l,
                    a_l,
                    st,
                    cw_l,
                    cb_l,
                    alog_l,
                    dtb_l,
                    qsl_l,
                    si_l,
                    dist_l,
                    sl_l,
                    rsi_l,
                    ro_l,
                    **kw,
                )
            else:
                out_st, out_a = base_kernel(
                    qkv_l,
                    b_l,
                    a_l,
                    st,
                    cw_l,
                    cb_l,
                    alog_l,
                    dtb_l,
                    qsl_l,
                    si_l,
                    dist_l,
                    sl_l,
                    rsi_l,
                    **kw,
                )
            if fuse_norm:
                out_a = pallas_rms_norm_gated(
                    out_a,
                    z_l,
                    nw_l,
                    eps=norm_eps,
                    head_dim=d_v,
                )
            return out_st, out_a

        # Route pure-decode buckets (T == num_seqs >= 128) through the
        # head-major Pallas fast_decode kernel with in-register RMSNormGated
        # when all active sequences are 1-token decodes (dist[-1] == dist[0]).
        # Smaller buckets (T < 128) stay on base_kernel for exact parity.
        if (batch_size >= 128 and batch_size == num_seqs
                and fast_decode.can_use_fast_decode(
                    num_spec_tokens=num_spec_tokens,
                    n_kq=n_kq_local,
                    n_v=n_v_local,
                    d_k=d_k,
                    d_v=d_v,
                    dim=dim,
                    kernel_size=kernel_size,
                    batch_size=batch_size,
                    num_seqs=num_seqs,
                )):

            def _pure_decode(st):
                cs_in, rs_in = st
                out_a, out_cs, out_rs = fast_decode.fast_decode_conv1d_gdn(
                    qkv_l,
                    b_l,
                    a_l,
                    cs_in,
                    rs_in,
                    cw_l,
                    cb_l,
                    alog_l,
                    dtb_l,
                    si_l,
                    rsi_l.astype(si_l.dtype),
                    dist_l,
                    sl_l,
                    n_kq=n_kq_local,
                    n_v=n_v_local,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    tile_s=8,
                    zero_initialize_out=True,
                    act_out_dtype=qkv_l.dtype,
                    conv_weight_swapped=False,
                    z=z_l,
                    norm_weight=nw_l,
                    norm_eps=norm_eps,
                    norm_bf16_roundtrip=True,
                )
                return (out_cs, out_rs), out_a

            return jax.lax.cond(
                dist_l[-1] == dist_l[0],
                _pure_decode,
                _call_base,
                (cs_l, rs_l),
            )

        return _call_base((cs_l, rs_l))

    mapped_fn = jax.shard_map(
        _run_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    mapped_args = (
        j_mixed_qkv,
        j_b,
        j_a,
        (conv_state, recurrent_state),
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
        read_state_indices,
    )
    if read_offsets is not None:
        mapped_args = mapped_args + (read_offsets, )
    if fuse_norm:
        mapped_args = mapped_args + (j_z, j_norm_weight)

    (new_conv_state, new_recurrent_state), output = mapped_fn(*mapped_args)

    return (new_conv_state, new_recurrent_state), output
