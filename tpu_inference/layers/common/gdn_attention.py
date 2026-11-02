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
import dataclasses
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.attention_backend import (
    Conv1dGDNKernel, GDNLinearAttentionBackend)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import truncate_sharded_tensor
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

    in_specs = (
        P(ShardingAxisName.ATTN_DATA,
          ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_b
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # j_a
        P(ShardingAxisName.ATTN_DATA, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
          None),  # recurrent_state
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
    if num_spec_tokens > 0 or read_offsets is not None:
        in_specs = in_specs + (
            P(ShardingAxisName.ATTN_DATA)
            if read_offsets is not None else None,  # read_offsets
        )

    out_specs = (
        (
            P(ShardingAxisName.ATTN_DATA, None,
              ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)

    partial_kwargs = dict(
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )
    if num_spec_tokens > 0 or read_offsets is not None:
        partial_kwargs["num_spec_tokens"] = num_spec_tokens

    p_run_jax_gdn_attention_local = functools.partial(
        conv1d_gdn_kernel or GDNLinearAttentionBackend().conv1d_gdn_kernel(),
        **partial_kwargs,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    mapped_args = (
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
        read_state_indices,
    )
    if num_spec_tokens > 0 or read_offsets is not None:
        mapped_args = mapped_args + (read_offsets, )

    (new_conv_state, new_recurrent_state), output = mapped_fn(*mapped_args)

    return (new_conv_state, new_recurrent_state), output


def compute_align_spec_state_indices(
    block_tables: jnp.ndarray,
    query_lens: jnp.ndarray,
    seq_lens: jnp.ndarray,
    is_window: jnp.ndarray,
    scratch_bases: jnp.ndarray,
    read_offsets: jnp.ndarray,
    mamba_block_size: int,
    num_local_rows: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """GDN state slots for mamba prefix caching ("align" mode) + spec decode.

    The scheduler gives each request one mamba block per `mamba_block_size`
    tokens, while a speculative verify window needs `num_spec_tokens + 1`
    consecutive slots for its per-position checkpoints. Each request
    therefore also owns a runner-side scratch group (base `scratch_bases`):

    * Verify windows (`is_window`) write their checkpoints to the scratch
      group. Prefill chunks write their final state to the block covering
      their last token, exactly as in align mode without spec decode.
    * A request resumes from scratch checkpoint `scratch_base + offset` when
      its previous step was a verify window (`read_offsets >= 0`), and from
      the block covering its last computed token otherwise (`-1`: after a
      prefill chunk or a prefix-cache hit).
    * When a window covers a block boundary `b` (`n < b <= n + q_len` with
      `n` computed tokens), its checkpoint for `b` (window position
      `b - n - 1`) is copied into the block holding tokens up to `b`, which
      the scheduler caches once `b` is verified. The copy happens whether or
      not the drafts past `b` are later accepted: the block is only cached
      after a later step verified `b`, and the last window that covers `b`
      is always the one whose accepted tokens reach it.

    All arrays are per request, flattened over DP ranks, with rank-local slot
    ids.

    Args:
        block_tables: `(num_reqs, max_blocks)` mamba block table.
        query_lens: `(num_reqs,)` scheduled tokens.
        seq_lens: `(num_reqs,)` computed + scheduled tokens.
        is_window: `(num_reqs,)` bool, True for verify-window (decode
            segment) sequences.
        scratch_bases: `(num_reqs,)` scratch group base slot.
        read_offsets: `(num_reqs,)` scratch checkpoint to resume from, or -1
            to resume from the block table.
        mamba_block_size: Tokens per mamba block.
        num_local_rows: Rows in each rank's shard of the state arrays; ids
            are clamped below it since the kernel DMAs them unchecked.

    Returns:
        `(state_indices, read_state_indices, copy_src, copy_dst, copy_mask)`
        where the kernel writes from `state_indices` (scratch group base for
        windows) and reads from `read_state_indices` (offset already applied),
        and `copy_mask` selects the boundary checkpoints to copy from
        `copy_src` to `copy_dst` after the kernel.
    """
    num_reqs = seq_lens.shape[0]
    batch_idx = jnp.arange(num_reqs)
    max_row = max(num_local_rows - 1, 0)
    num_computed = seq_lens - query_lens

    def block_at(col):
        col = jnp.clip(col, 0, block_tables.shape[1] - 1)
        return jnp.clip(block_tables[batch_idx, col], 0, max_row)

    read_block = block_at(jnp.maximum(num_computed - 1, 0) // mamba_block_size)
    write_block = block_at(jnp.maximum(seq_lens - 1, 0) // mamba_block_size)

    from_scratch = read_offsets >= 0
    state_indices = jnp.where(is_window, scratch_bases, write_block)
    read_state_indices = jnp.where(
        from_scratch, jnp.clip(scratch_bases + read_offsets, 0, max_row),
        read_block)

    boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
    copy_dst = block_at(num_computed // mamba_block_size)
    copy_src = jnp.clip(scratch_bases + (boundary - num_computed - 1), 0,
                        max_row)
    copy_mask = (is_window & (query_lens > 0) & (boundary <= seq_lens)
                 & (scratch_bases > 0) & (copy_dst > 0))
    return (state_indices.astype(jnp.int32),
            read_state_indices.astype(jnp.int32), copy_src.astype(jnp.int32),
            copy_dst.astype(jnp.int32), copy_mask)


def copy_gdn_state_slots(
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    src: jnp.ndarray,
    dst: jnp.ndarray,
    mask: jnp.ndarray,
    mesh: jax.sharding.Mesh,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Copies GDN state rows `src[i] -> dst[i]` wherever `mask[i]`.

    `src`, `dst` and `mask` are sharded like the requests (over
    `ATTN_DATA`) and hold rank-local row ids. The selected pairs are
    compacted per rank and copied in a loop whose trip count is the number
    of selected pairs, so the cost scales with the (rare) copies rather than
    with the batch size. Sources and destinations must not overlap.
    """
    state_specs = (
        P(ShardingAxisName.ATTN_DATA, None, ShardingAxisName.ATTN_HEAD),
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None, None),
    )

    def _local(conv, rec, src, dst, mask):

        def _copy_selected(state):
            conv, rec = state
            n = mask.shape[0]
            (selected, ) = jnp.nonzero(mask, size=n, fill_value=0)
            count = jnp.sum(mask.astype(jnp.int32))
            src_sel = src[selected]
            dst_sel = dst[selected]

            def body(i, carry):
                conv, rec = carry
                s, d = src_sel[i], dst_sel[i]
                conv = jax.lax.dynamic_update_slice_in_dim(
                    conv, jax.lax.dynamic_slice_in_dim(conv, s, 1, axis=0), d,
                    0)
                rec = jax.lax.dynamic_update_slice_in_dim(
                    rec, jax.lax.dynamic_slice_in_dim(rec, s, 1, axis=0), d, 0)
                return conv, rec

            return jax.lax.fori_loop(0, count, body, (conv, rec))

        return jax.lax.cond(jnp.any(mask), _copy_selected, lambda s: s,
                            (conv, rec))

    mapped_fn = jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=state_specs + (P(ShardingAxisName.ATTN_DATA), ) * 3,
        out_specs=state_specs,
        check_vma=False,
    )
    return mapped_fn(conv_state, recurrent_state, src, dst, mask)


@dataclasses.dataclass(frozen=True)
class GDNStateSlots:
    """Per-request GDN state slots for one step (see `resolve_gdn_state_slots`)."""
    state_indices: jnp.ndarray
    read_state_indices: jnp.ndarray
    read_offsets: Optional[jnp.ndarray]
    num_spec_tokens: int
    boundary_copies: Optional[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]

    def kernel_kwargs(self) -> dict:
        kwargs = dict(read_state_indices=self.read_state_indices)
        if self.read_offsets is not None:
            kwargs.update(read_offsets=self.read_offsets,
                          num_spec_tokens=self.num_spec_tokens)
        return kwargs

    def apply_boundary_copies(
        self,
        conv_state: jnp.ndarray,
        recurrent_state: jnp.ndarray,
        mesh: jax.sharding.Mesh,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.boundary_copies is None:
            return conv_state, recurrent_state
        return copy_gdn_state_slots(conv_state,
                                    recurrent_state,
                                    *self.boundary_copies,
                                    mesh=mesh)


def resolve_gdn_state_slots(
    attn_metadata,
    *,
    seq_lens: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    padded_num_reqs_per_dp: int,
    dp_size: int,
    num_local_rows: int,
    mamba_block_size: int,
) -> GDNStateSlots:
    """Picks the GDN state slots each request reads and writes this step."""
    raw_spec = getattr(attn_metadata, "num_spec_tokens", 0)
    num_spec_tokens = int(raw_spec) if isinstance(raw_spec, int) else 0
    gdn_read_offsets = getattr(attn_metadata, "gdn_read_offsets", None)
    mamba_state_indices = getattr(attn_metadata, "mamba_state_indices", None)
    max_row = max(num_local_rows - 1, 0)

    def truncate(x):
        return truncate_sharded_tensor(x.astype(jnp.int32),
                                       padded_num_reqs_per_dp, dp_size)

    if not mamba_block_size:
        state_indices = truncate(mamba_state_indices)
        read_offsets = None
        if num_spec_tokens > 0 and gdn_read_offsets is not None:
            read_offsets = jnp.maximum(truncate(gdn_read_offsets), 0)
        return GDNStateSlots(state_indices, state_indices, read_offsets,
                             num_spec_tokens if read_offsets is not None else 0,
                             None)

    num_reqs_total = attn_metadata.seq_lens.shape[0]
    block_tables = attn_metadata.block_tables.reshape(num_reqs_total, -1)
    max_num_reqs_per_dp = num_reqs_total // dp_size
    block_tables = block_tables.reshape(
        dp_size, max_num_reqs_per_dp,
        -1)[:, :padded_num_reqs_per_dp, :].reshape(-1, block_tables.shape[-1])
    qsl = query_start_loc.reshape(dp_size, padded_num_reqs_per_dp + 1)
    query_lens = (qsl[:, 1:] - qsl[:, :-1]).reshape(-1)

    if (num_spec_tokens > 0 and mamba_state_indices is not None
            and gdn_read_offsets is not None):
        num_windows = attn_metadata.request_distribution.reshape(
            dp_size, -1)[:, :1]
        is_window = (jnp.arange(padded_num_reqs_per_dp)[None, :]
                     < num_windows).reshape(-1)
        (state_indices, read_state_indices, copy_src, copy_dst,
         copy_mask) = compute_align_spec_state_indices(
             block_tables,
             query_lens,
             seq_lens,
             is_window,
             truncate(mamba_state_indices),
             truncate(gdn_read_offsets),
             mamba_block_size,
             num_local_rows,
         )
        return GDNStateSlots(state_indices, read_state_indices,
                             jnp.zeros_like(state_indices), num_spec_tokens,
                             (copy_src, copy_dst, copy_mask))

    num_computed = seq_lens - query_lens
    batch_idx = jnp.arange(seq_lens.shape[0])
    read_col = jnp.maximum(num_computed - 1, 0) // mamba_block_size
    write_col = jnp.maximum(seq_lens - 1, 0) // mamba_block_size
    read_state_indices = jnp.clip(block_tables[batch_idx, read_col], 0,
                                  max_row).astype(jnp.int32)
    state_indices = jnp.clip(block_tables[batch_idx, write_col], 0,
                             max_row).astype(jnp.int32)
    return GDNStateSlots(state_indices, read_state_indices, None, 0, None)
