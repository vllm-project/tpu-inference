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
"""FlyWheel TPU attention backend
(--attention-backend flywheel)."""

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import flywheel_tpu
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flywheel_tpu import flash_attn_func, flash_attn_with_kvcache
from flywheel_tpu.flash_attn_interface import PAGE_SIZE_MULTIPLE
from flywheel_tpu.pallas.block_sizes import FWD_BLOCKS, NUM_LANES
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheGroupSpec,
                                        UniformTypeKVCacheSpecs)

from tpu_inference import envs
from tpu_inference.layers.common.attention_backend import (AttentionBackend,
                                                           HostStepInputs)
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import (DeviceBuffer, align_to, device_array,
                                 get_mesh_shape_product)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Note: jit here so that every layer calls one shared traced kernel,
# as flash_attn_with_kvcache and the KV writer already do. Inlined, every layer
# would lower its own copy, which a warm compile cache does not skip.
flash_attn_varlen_func = jax.jit(
    flywheel_tpu.flash_attn_varlen_func,
    static_argnames=("max_seqlen_q", "max_seqlen_k", "softmax_scale", "causal",
                     "interpret", "token_major"),
)


def align_head_dim(head_dim: int) -> int:
    """The head_dim the kernels run at: head_dim padded to whole 128 lanes."""
    if not 0 < head_dim <= 256:
        raise NotImplementedError(
            "The flywheel attention backend needs 0 < head_dim <= 256; "
            f"got {head_dim}.")
    return align_to(head_dim, 128)


# Note: at a padded head_dim with 2 KV heads per chip the decode
# kernel's rows come out near zero, so it runs only where no padding is needed.
_DECODE_HEAD_DIMS = (128, 256)


def kv_write_slice_bound(num_tokens: int, max_num_seqs: int,
                         page_size: int) -> int:
    """Upper bound on the page pieces one step can write, which sizes the
    host slice table.

    A request writing q_len tokens touches at most q_len // page_size + 2
    pages, so the bound depends only on the token bucket and the request
    slots, never on the batch.
    """
    return num_tokens // page_size + 2 * max_num_seqs


def build_kv_write_slices(
        cu_q_lens: np.ndarray,  # i32[num_reqs + 1]
        kv_lens: np.ndarray,  # i32[num_reqs], includes this step's tokens
        block_table: np.ndarray,  # i32[num_reqs, pages_per_seq]
        *,
        page_size: int,
        out: np.ndarray,  # i32[3, bound], overwritten
) -> int:
    """Fill out with the token slices that copy this step's new K/V rows into
    the paged cache and return their count.

    Column j < count is (cache_row_start, new_row_start, length) in rows of
    the flat (num_pages * page_size, ...) cache view, each piece inside one
    page; the remaining columns are zero.
    """
    # Note: built on the host, since deriving it on the device scans
    # every (request slot, page) pair in every layer.
    num_reqs = kv_lens.shape[0]
    q_lens = cu_q_lens[1:num_reqs + 1] - cu_q_lens[:num_reqs]
    kv_start = kv_lens - q_lens
    first_page = kv_start // page_size
    last_page = (kv_lens - 1) // page_size
    counts = np.where(q_lens > 0, last_page - first_page + 1, 0)
    total = int(counts.sum())
    req_idx = np.repeat(np.arange(num_reqs), counts)
    piece_start = np.repeat(np.cumsum(counts) - counts, counts)
    page_slot = first_page[req_idx] + (np.arange(total) - piece_start)
    page_start_token = page_slot * page_size
    write_start = np.maximum(kv_start[req_idx], page_start_token)
    write_end = np.minimum(kv_lens[req_idx], page_start_token + page_size)
    out[:] = 0
    out[0, :total] = (block_table[req_idx, page_slot] * page_size +
                      (write_start - page_start_token))
    out[1, :total] = (cu_q_lens[:num_reqs][req_idx] +
                      (write_start - kv_start[req_idx]))
    out[2, :total] = write_end - write_start
    return total


_PREFILL_PHASE = "prefill"
_DECODE_PHASE = "decode"
_EXTEND_PHASE = "extend"
_ATTENTION_PHASES = (_PREFILL_PHASE, _DECODE_PHASE, _EXTEND_PHASE)
_DENSE_KERNEL = "dense"

# Note: flash_attn_func pads the sequence axis to a multiple of
# NUM_LANES, then runs one block up to the largest of FWD_BLOCKS (the forward
# kernels' block sizes), else whole blocks of the largest FWD_BLOCKS size
# that divides the axis. The dense cost model describes an unpadded bucket of
# whole min(bucket, largest) blocks only, so any other bucket stays ragged.
_DENSE_PAD_MULTIPLE = NUM_LANES
_DENSE_MAX_BLOCK = max(FWD_BLOCKS)


def select_attention_phase(num_decode: int, cu_q_lens: np.ndarray,
                           kv_lens: np.ndarray) -> str:
    """A rank's static kernel choice, from this step's own host lengths.

    PREFILL is a step of fresh prefill requests alone (kv_len == q_len and no
    decode row), which the prefill kernels serve from this step's K/V. A
    mixed step, or a request whose keys are partly in the pages, is EXTEND.
    """
    q_lens = np.diff(cu_q_lens)
    if np.any(q_lens[:num_decode] != 1):
        # Note: the decode kernel takes the leading num_decode rows
        # as one token each, so a runner that stops ordering decode requests
        # first has to fail here instead of attending the wrong rows.
        raise ValueError(
            "A batch must put its decode requests first, one query each; got "
            f"q_lens {q_lens[:num_decode].tolist()} for the leading "
            f"{num_decode}.")
    elif num_decode == kv_lens.size:
        return _DECODE_PHASE
    elif num_decode > 0 or np.any(kv_lens != q_lens):
        # Note: a mixed step is EXTEND even when its prefill requests
        # are fresh. The ragged kernel would need the decode rows merged into
        # its output, an extra [T, N, D] copy per layer that blocks the MLP
        # weight prefetch.
        return _EXTEND_PHASE
    else:
        return _PREFILL_PHASE


def use_dense_prefill(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
    ragged_block_span: int,
) -> bool:
    """Whether a prefill step runs the dense kernel instead of the ragged one
    at ragged_block_span: every DP rank schedules one request, the dense
    kernel can tile the token bucket, and the cost model favours dense.

    The runner packs requests end to end and pads the tail, so one request is
    already a (1, bucket) dense row: the causal mask keeps every real query
    off the padded keys, and the KV writer copies real rows only.
    """
    bucket = padded_num_tokens_per_dp_rank
    if not (all(len(lengths) == 1 for lengths in scheduled_tokens_per_dp_rank)
            and is_dense_tileable(bucket)):
        return False
    # Note: both kernels pay for the bucket's pad tail. The dense cost
    # has no per-step overhead term because its schedule is built on the host.
    block = min(bucket, _DENSE_MAX_BLOCK)
    num_blocks = bucket // block
    dense_cost = num_blocks * (num_blocks + 1) // 2 * block * block
    return dense_cost <= ragged_step_cost(scheduled_tokens_per_dp_rank, bucket,
                                          ragged_block_span)


# Note: the varlen kernel caps its q/kv block at max_seqlen_q/k, picking
# from FWD_BLOCKS, so a span is one of those block sizes. The smallest, 128,
# is left out, saving a backbone per bucket: on v6e 256 matched or beat it on
# every sampled batch, and on v7x 128 won only small buckets packed with short
# requests.
_VARLEN_BLOCK_SPANS = tuple(
    sorted(block for block in FWD_BLOCKS if block > min(FWD_BLOCKS)))

# Note: fixed cost of one ragged kernel step, written as the side of a
# square tile of equal cost. It was fitted on v6e; on v7x it also picks the
# fastest span on nearly every sampled batch. It belongs to the kernel's
# schedule rather than to the head count or head_dim, so the host model needs
# neither.
_VARLEN_STEP_OVERHEAD_TOKENS = 704


def varlen_block_spans(padded_num_tokens_per_dp_rank: int) -> tuple[int, ...]:
    """Every block span varlen_block_span can return at this token bucket,
    which is the set of ragged prefill backbones the warmup compiles."""
    spans = tuple(s for s in _VARLEN_BLOCK_SPANS
                  if s <= padded_num_tokens_per_dp_rank)
    return spans or (padded_num_tokens_per_dp_rank, )


def varlen_block_span(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
) -> int:
    """The varlen_block_spans entry with the lowest ragged_step_cost for this
    step (one executable serves every DP rank). min keeps the first of equal
    costs, and the spans go widest first, so a tie goes to the wider span."""
    return min(
        reversed(varlen_block_spans(padded_num_tokens_per_dp_rank)),
        key=lambda span: ragged_step_cost(scheduled_tokens_per_dp_rank,
                                          padded_num_tokens_per_dp_rank, span))


def ragged_step_cost(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
    span: int,
) -> int:
    """Modelled cost of one ragged prefill step at this block span. It is a
    relative cost: only its ratio across spans, and to the dense cost in
    use_dense_prefill, matters.

    The ragged kernel cuts each sequence into blocks of span tokens and runs
    one step per (query block, key block) tile on or below the causal
    diagonal, so a sequence of t = ceil(length / span) blocks takes
    t * (t + 1) / 2 steps. The bucket's unfilled tail runs as one more
    sequence. A step costs its span x span tile plus a fixed overhead, written
    as a tile of _VARLEN_STEP_OVERHEAD_TOKENS x _VARLEN_STEP_OVERHEAD_TOKENS:

        cost = max over DP ranks of sum(t * (t + 1) / 2)
               * (span**2 + _VARLEN_STEP_OVERHEAD_TOKENS**2)

    The DP ranks run at the same time, so the step costs its slowest rank.
    A wide span computes more of the tiles past the end of short sequences
    and along the diagonal; a narrow one takes more steps, each paying the
    overhead. For example, one 3000-token request in a 4096-token bucket (a
    1096-token tail) costs (21 + 6) * (512**2 + 704**2) = 20.5M at span 512,
    and (6 + 3) * (1024**2 + 704**2) = 13.9M at span 1024.
    """
    max_num_steps = 0
    for lengths in scheduled_tokens_per_dp_rank:
        tail = padded_num_tokens_per_dp_rank - sum(lengths)
        num_blocks = [
            -(-length // span) for length in [*lengths, tail] if length > 0
        ]
        max_num_steps = max(max_num_steps,
                            sum(t * (t + 1) // 2 for t in num_blocks))
    return max_num_steps * (span * span + _VARLEN_STEP_OVERHEAD_TOKENS**2)


def is_dense_tileable(padded_num_tokens_per_dp_rank: int) -> bool:
    """Whether the dense kernel can tile a (1, bucket) row at this token
    bucket."""
    bucket = padded_num_tokens_per_dp_rank
    return (bucket % _DENSE_PAD_MULTIPLE == 0
            and (bucket <= _DENSE_MAX_BLOCK or bucket % _DENSE_MAX_BLOCK == 0))


def fold_heads_outer(x: jax.Array) -> jax.Array:
    """[T, H, D] -> (H, 1, T, D), flash_attn_func's heads-outer layout of one
    sequence."""
    # Note: XLA already lays the projections out heads-outermost, so
    # this transpose compiles to a bitcast; a batch-outer fold would copy.
    return x.transpose(1, 0, 2)[:, None]


def paged_cache_decode(
    q: jax.Array,  # [T, N, D]
    k: jax.Array | None,  # [T, H, D], or None when the cache already holds
    v: jax.Array | None,  # this step's tokens
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs], includes this step's token
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    num_active: jax.Array,  # i32 scalar: leading rows that are requests
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """One query token per row against the paged cache, for the leading
    min(max_num_seqs, T) rows of the token bucket; returns those rows, NOT
    padded back to T, and the cache the kernel aliased through.

    Passing k/v lets the kernel append them to the pages itself; None means
    the DMA writer has already copied this step's tokens there.
    """
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    batch = min(max_num_seqs, q.shape[0])
    block_table = page_indices[:batch * pages_per_seq].reshape(
        batch, pages_per_seq)
    has_new_kv = k is not None
    # Note: kv_lens count this step's token, so an in-kernel append
    # takes the length before it (padding rows go to -1, which the kernel
    # never reads); without the append that token is in its page already.
    cache_seqlens = kv_lens[:batch] - 1 if has_new_kv else kv_lens[:batch]
    # Note: num_active stops the kernel at the real requests instead
    # of attending the padding rows.
    output, cache = flash_attn_with_kvcache(
        q[:batch, None],
        kv_cache,
        None,
        k[:batch, None] if has_new_kv else None,
        v[:batch, None] if has_new_kv else None,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        num_active=num_active,
        softmax_scale=sm_scale,
    )
    return output[:, 0], cache


def _paged_extend_rows(
    q: jax.Array,  # [T, N, D]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_decode: jax.Array,
    num_active: jax.Array,
    *,
    sm_scale: float,
) -> jax.Array:
    """The paged varlen kernel over the requests after the leading
    num_decode decode rows, reading their cached prefix through the block
    table. Rows it does not own are unspecified; the caller merges them."""
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    block_table = page_indices.reshape(max_num_seqs, -1)
    pages_per_seq = block_table.shape[1]
    # Note: the window slides past the decode requests, since a
    # request clamped to zero tokens in place still costs a step. Each array
    # is padded by its own length to keep dynamic_slice's start in range; the
    # kernel reads only the first num_active - num_decode slots plus
    # cu_q_lens[num_reqs], never the padding.
    cu_q_window = lax.dynamic_slice(jnp.pad(cu_q_lens, (0, max_num_seqs)),
                                    (num_decode, ), (max_num_seqs + 1, ))
    kv_lens_window = lax.dynamic_slice(jnp.pad(kv_lens, (0, max_num_seqs)),
                                       (num_decode, ), (max_num_seqs, ))
    block_table_window = lax.dynamic_slice(
        jnp.pad(block_table, ((0, max_num_seqs), (0, 0))), (num_decode, 0),
        (max_num_seqs, pages_per_seq))
    # Note: max_seqlen_q/k are static block-size bounds, not lengths
    # (the token bucket and the table's capacity), so one executable serves
    # every kv_len of a bucket. The kernel clamps page indices into each
    # sequence's own pages, so its table needs no capacity padding.
    return flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        kv_cache,
        None,
        cu_q_window,
        None,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=pages_per_seq * page_size,
        softmax_scale=sm_scale,
        causal=True,
        token_major=False,
        block_table=block_table_window,
        seqused_k=kv_lens_window,
        num_active=num_active - num_decode,
    ).transpose(1, 0, 2)


def _dense_prefill_rows(q: jax.Array, k: jax.Array, v: jax.Array, *,
                        sm_scale: float) -> jax.Array:
    """The dense kernel on the step's one (1, T) row of this step's K/V."""
    num_tokens, num_heads, head_dim = q.shape
    output = flash_attn_func(
        fold_heads_outer(q),
        fold_heads_outer(k),
        fold_heads_outer(v),
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
        heads_outer_fold=True,
    )
    # Note: a trailing transpose, which XLA folds into o_proj instead of
    # copying.
    return output.reshape(num_heads, num_tokens, head_dim).transpose(1, 0, 2)


def _ragged_prefill_rows(q: jax.Array, k: jax.Array, v: jax.Array,
                         cu_q_lens: jax.Array, *, span: int,
                         sm_scale: float) -> jax.Array:
    """The ragged kernel at block span `span` on this step's K/V."""
    return flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        k.transpose(1, 0, 2),
        v.transpose(1, 0, 2),
        cu_q_lens,
        cu_q_lens,
        max_seqlen_q=span,
        max_seqlen_k=span,
        causal=True,
        softmax_scale=sm_scale,
        token_major=False,
    ).transpose(1, 0, 2)


def step_kernels(padded_num_tokens_per_dp_rank: int) -> tuple[str | int, ...]:
    """The kernels one backbone holds at this token bucket for a step's rows
    after its decode rows, in lax.switch order: none (a decode step), the
    paged extend kernel, dense prefill when the bucket is tileable, and
    ragged prefill at each block span (an int)."""
    bucket = padded_num_tokens_per_dp_rank
    kernels: list[str | int] = [_DECODE_PHASE, _EXTEND_PHASE]
    if is_dense_tileable(bucket):
        kernels.append(_DENSE_KERNEL)
    kernels.extend(varlen_block_spans(bucket))
    return tuple(kernels)


def flywheel_tpu_attention(
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,  # [num_pages, page_size, 2 * H, D]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    kv_write_slices: jax.Array,  # i32[3 * bound]
    kv_write_num_slices: jax.Array,  # i32[1]
    request_distribution: jax.Array,  # i32[3]
    kernel_index: jax.Array,  # i32[1]: into step_kernels(T)
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """One shard's attention for any step, in one executable per bucket.

    The DMA writer copies the K/V of the rows after the leading decode rows
    into the pages, and the decode kernel appends and attends the decode
    rows. Then lax.switch runs step_kernels(T)[kernel_index] on the other
    rows. Every switch branch only reads the cache: a branch that wrote it
    would make XLA copy the whole cache per layer.

    request_distribution is this rank's (decode_end, prefill_end, mixed_end),
    RPA's cumulative request counts of its three request groups. The runner
    puts no request in the middle, prefill-only group, so it is
    [num_decode, num_decode, num_reqs], and only entries 0 and 2 are read.
    """
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape != k.shape:
        raise ValueError(
            "Q/K/V must share head_dim and K/V shapes must match.")
    padded_head_dim = align_head_dim(head_dim)
    if padded_head_dim == head_dim:
        padded_q, padded_k, padded_v = q, k, v
    else:
        # Note: the cache is allocated at the padded head_dim once,
        # so only this step's activations need padding.
        padding = ((0, 0), (0, 0), (0, padded_head_dim - head_dim))
        padded_q, padded_k, padded_v = (jnp.pad(activation, padding)
                                        for activation in (q, k, v))
    num_decode = request_distribution[0]
    num_active = request_distribution[2]
    # Note: per-rank metadata is sharded as flat arrays, so the
    # writer's (3, bound) table arrives flattened.
    cache = write_kv_cache_pages(padded_k, padded_v, kv_cache,
                                 kv_write_slices.reshape(3, -1),
                                 kv_write_num_slices)
    # Note: the cache threads writer -> decode kernel so the one donated
    # buffer aliases the chain, outside the switch.
    decode_output, cache = paged_cache_decode(padded_q,
                                              padded_k,
                                              padded_v,
                                              cache,
                                              kv_lens,
                                              page_indices,
                                              num_decode,
                                              sm_scale=sm_scale)

    def _branch(kernel: str | int):

        def run(q, k, v, cache, kv_lens, page_indices, cu_q_lens, num_decode,
                num_active):
            if kernel == _DECODE_PHASE:
                return jnp.zeros_like(q)
            elif kernel == _EXTEND_PHASE:
                return _paged_extend_rows(q,
                                          cache,
                                          kv_lens,
                                          page_indices,
                                          cu_q_lens,
                                          num_decode,
                                          num_active,
                                          sm_scale=sm_scale)
            elif kernel == _DENSE_KERNEL:
                return _dense_prefill_rows(q, k, v, sm_scale=sm_scale)
            else:
                return _ragged_prefill_rows(q,
                                            k,
                                            v,
                                            cu_q_lens,
                                            span=kernel,
                                            sm_scale=sm_scale)

        return run

    output = lax.switch(
        kernel_index[0],
        [_branch(kernel) for kernel in step_kernels(padded_q.shape[0])],
        padded_q, padded_k, padded_v, cache, kv_lens, page_indices, cu_q_lens,
        num_decode, num_active)
    # Note: the leading rows take the decode kernel's output below
    # num_decode and zeros past the scheduled tokens: a branch need not own
    # the rows below its first request (row num_decode), and a decode-only
    # rank leaves it no request. num_decode never exceeds num_leading_rows.
    num_leading_rows = min(align_to(decode_output.shape[0], 8),
                           padded_q.shape[0])
    # Note: shape padding only; num_decode cannot reach these rows.
    padded_decode_output = jnp.pad(
        decode_output,
        ((0, num_leading_rows - decode_output.shape[0]), (0, 0), (0, 0)))
    row_index = jnp.arange(num_leading_rows)[:, None, None]
    num_scheduled_tokens = cu_q_lens[num_active]
    leading_output = jnp.where(
        row_index < num_decode, padded_decode_output,
        jnp.where(row_index < num_scheduled_tokens, output[:num_leading_rows],
                  0))
    output = output.at[:num_leading_rows].set(leading_output)
    return output[..., :head_dim], cache


def sharded_flywheel_tpu_attention(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_write_slices: jax.Array,  # i32[dp * 3 * bound]
    kv_write_num_slices: jax.Array,  # i32[dp]
    request_distribution: jax.Array,  # i32[dp * 3]
    kernel_index: jax.Array,  # i32[dp]
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Shards along KV heads, mirroring sharded_ragged_paged_attention."""
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    num_kv_heads = k.shape[1]
    if num_kv_heads >= tp_size:
        sharded_k, sharded_v = k, v
    elif tp_size % num_kv_heads:
        raise ValueError(
            f"For GQA/MQA, tp_size {tp_size} must be divisible by "
            f"num_kv_heads {num_kv_heads}")
    else:
        kv_head_repeats = tp_size // num_kv_heads
        sharded_k = jnp.repeat(k, kv_head_repeats, axis=1)
        sharded_v = jnp.repeat(v, kv_head_repeats, axis=1)

    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    # Note: the cache's head axis splits into each shard's
    # contiguous run of interleaved (K, V) row pairs.
    kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD, None)
    in_specs = ((qkv_spec, qkv_spec, qkv_spec, kv_cache_spec) +
                (P(ShardingAxisName.ATTN_DATA), ) * 7)
    out_specs = (qkv_spec, kv_cache_spec)

    def _attention(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
        return flywheel_tpu_attention(*args, sm_scale=sm_scale)

    return jax.shard_map(
        _attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(q, sharded_k, sharded_v, kv_cache, kv_lens, page_indices, cu_q_lens,
      kv_write_slices, kv_write_num_slices, request_distribution, kernel_index)


def check_supported(
    q: jax.Array,
    *,
    sinks: jax.Array | None,
    attention_chunk_size: int | None,
    attn_logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    update_kv_cache: bool,
    use_causal_mask: bool,
) -> None:
    """Raises NotImplementedError for a feature of one attention layer that
    the backend does not support. FlywheelAttentionBackend.check_config has
    already rejected, at startup, what the engine configuration shows
    (parallelism, speculative decoding, dtypes)."""
    # Note: every served model decodes, and check_config can only check
    # the model's one get_head_size(); a layer of another head_dim would
    # decode wrong rows rather than fail.
    if q.shape[-1] not in _DECODE_HEAD_DIMS:
        raise NotImplementedError(
            f"The flywheel attention backend decodes only head_dim in "
            f"{_DECODE_HEAD_DIMS}; got {q.shape[-1]}.")
    unsupported = {
        "sinks":
        sinks is not None,
        "sliding window (attention_chunk_size)":
        attention_chunk_size is not None,
        "logits soft cap":
        attn_logits_soft_cap is not None,
        "q_scale/k_scale/v_scale (quantized KV)":
        any(s is not None for s in (q_scale, k_scale, v_scale)),
        "update_kv_cache=False (KV-share layers)":
        not update_kv_cache,
        "use_causal_mask=False (non-causal attention)":
        not use_causal_mask,
    }
    for name, is_unsupported in unsupported.items():
        if is_unsupported:
            raise NotImplementedError(
                f"The flywheel attention backend does not support {name}.")


def attention_group_page_sizes(
        kv_cache_groups: Sequence[KVCacheGroupSpec]) -> dict[int, int]:
    """KV cache group id -> tokens per page, for the attention groups only
    (a mamba group has no pages to write)."""
    page_sizes: dict[int, int] = {}
    for gid, kv_cache_group in enumerate(kv_cache_groups):
        spec = kv_cache_group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            layer_spec = next(iter(spec.kv_cache_specs.values()))
        else:
            layer_spec = spec
        if isinstance(layer_spec, AttentionSpec):
            page_sizes[gid] = spec.block_size
    return page_sizes


_KERNEL_INDEX_KEY = "flywheel_kernel_index"


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["kv_write_slices", "kv_write_num_slices", "kernel_index"],
    meta_fields=[],
)
@dataclass
class FlywheelMetadata:
    """AttentionMetadata.backend_metadata of an attention KV cache group in
    one step under --attention-backend flywheel. Every field is data, so one
    backbone per token bucket serves every step."""
    # The KV writer's slice table, flat (dp_size * 3 * bound,) int32, and its
    # (dp_size,) int32 slice counts, split per DP rank like block_tables.
    kv_write_slices: jax.Array
    kv_write_num_slices: jax.Array
    # (dp_size,) int32: the step's kernel, an index into step_kernels(bucket).
    kernel_index: jax.Array


class FlywheelAttentionBackend(AttentionBackend):
    """--attention-backend flywheel."""

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Round the block size the platform chose up to what the kernels can
        index, then reject every configuration this backend cannot serve,
        listing each unsupported feature in one error."""
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        problems: list[str] = []

        if not cache_config.user_specified_block_size:
            block_size = align_to(cache_config.block_size, PAGE_SIZE_MULTIPLE)
            if block_size != cache_config.block_size:
                logger.info(
                    "The flywheel attention backend indexes whole %d-token "
                    "pages; increasing the block size from %d to %d.",
                    PAGE_SIZE_MULTIPLE, cache_config.block_size, block_size)
                cache_config.block_size = block_size
        elif cache_config.block_size % PAGE_SIZE_MULTIPLE:
            problems.append(
                f"block size {cache_config.block_size} is not a multiple of "
                f"{PAGE_SIZE_MULTIPLE}; pass a --block-size that is, or leave "
                "it unset")

        if envs.USE_BATCHED_RPA_KERNEL:
            problems.append("USE_BATCHED_RPA_KERNEL selects an RPA kernel; "
                            "unset it")
        if vllm_config.speculative_config is not None:
            problems.append("speculative decoding is not supported")
        if vllm_config.additional_config.get("enable_continue_decode", False):
            # Note: the fused on-device decode loop rebuilds
            # AttentionMetadata without the KV write slice table or the static
            # attention phase this backend reads.
            problems.append("continue_decode is not supported; drop "
                            "additional_config.enable_continue_decode")
        if vllm_config.kv_transfer_config is not None:
            problems.append(
                "KV transfer (disaggregated serving) is not supported")
        if parallel_config.decode_context_parallel_size > 1:
            problems.append("decode context parallelism is not supported")
        if parallel_config.prefill_context_parallel_size > 1:
            problems.append("prefill context parallelism is not supported")
        if model_config.use_mla:
            problems.append("MLA models are not supported")
        # Note: every served model decodes, so a head_dim the decode
        # kernel rejects fails here rather than at the first decode step.
        if model_config.get_head_size() not in _DECODE_HEAD_DIMS:
            problems.append(
                f"head_dim {model_config.get_head_size()} is not one of the "
                f"decode kernel's {_DECODE_HEAD_DIMS}")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            problems.append(
                f"kv cache dtype {cache_config.cache_dtype!r} is not bfloat16")
        if model_config.dtype != torch.bfloat16:
            problems.append(
                f"model dtype {model_config.dtype} is not bfloat16")

        if problems:
            raise ValueError(
                "The flywheel attention backend cannot serve this "
                "configuration: " + "; ".join(problems) + ".")

    def get_kv_cache_shape(self, total_num_pages: int, page_size: int,
                           num_kv_heads: int, head_dim: int,
                           kv_dtype: jnp.dtype) -> tuple[int, ...]:
        # Note: K and V rows interleave per KV head, [k0, v0, k1, v1,
        # ...] (RPA v3's order). The kernels read a head's (K, V) pair as one
        # u32 word per lane and stage it as a 2-row window, and the layout has
        # no singleton axis at one KV head per chip, which XLA laid out with
        # two whole-cache copies per written layer. bf16 only, so kv_dtype
        # does not size the cache.
        return (total_num_pages, page_size, 2 * num_kv_heads,
                align_head_dim(head_dim))

    def attention(
        self,
        kv_cache: jax.Array,
        q: jax.Array,  # [T, N, D]
        k: jax.Array,  # [T, H, D]
        v: jax.Array,  # [T, H, D]
        md: AttentionMetadata,
        shared_md: AttentionMetadata | SharedAttentionMetadata,
        mesh: Mesh,
        *,
        head_dim_original: int,
        sm_scale: float,
        attention_chunk_size: int | None,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        sinks: jax.Array | None,
        update_kv_cache: bool,
        use_causal_mask: bool,
        attn_logits_soft_cap: float | None,
        decode_query_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        # Note: the kernels pad head_dim themselves (align_head_dim), so q
        # arrives at the model's head_dim and head_dim_original is unused;
        # decode_query_size is 1 for every vLLM model.
        check_supported(
            q,
            sinks=sinks,
            attention_chunk_size=attention_chunk_size,
            attn_logits_soft_cap=attn_logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            update_kv_cache=update_kv_cache,
            use_causal_mask=use_causal_mask,
        )
        flywheel_md = md.backend_metadata
        if not isinstance(flywheel_md, FlywheelMetadata):
            raise ValueError(
                "The flywheel attention backend needs its step metadata "
                "(AttentionMetadata.backend_metadata, built by "
                "FlywheelAttentionBackend.prepare_step); got "
                f"{type(flywheel_md).__name__}.")
        output, kv_cache = sharded_flywheel_tpu_attention(
            mesh,
            q,
            k,
            v,
            kv_cache,
            shared_md.seq_lens,
            md.block_tables,
            shared_md.query_start_loc,
            flywheel_md.kv_write_slices,
            flywheel_md.kv_write_num_slices,
            shared_md.request_distribution,
            flywheel_md.kernel_index,
            sm_scale=sm_scale,
        )
        return kv_cache, output

    def prepare_step(self, step: HostStepInputs,
                     device_buffer: DeviceBuffer) -> None:
        """Decide the step's kernel, and append its index and each attention
        group's KV write slice table to device_buffer."""

        def _rank_slices(dp_rank: int) -> tuple[int, slice, slice]:
            num_reqs = int(step.request_distribution[3 * dp_rank + 2])
            req_offset = dp_rank * step.max_num_reqs_per_dp_rank
            query_offset = dp_rank * (step.max_num_reqs_per_dp_rank + 1)
            return (num_reqs, slice(req_offset, req_offset + num_reqs),
                    slice(query_offset, query_offset + num_reqs + 1))

        attention_phases: set[str] = set()
        for dp_rank in range(step.dp_size):
            num_reqs, req_range, query_range = _rank_slices(dp_rank)
            if num_reqs > 0:
                attention_phases.add(
                    select_attention_phase(
                        int(step.request_distribution[3 * dp_rank]),
                        step.query_start_loc[query_range],
                        step.seq_lens[req_range]))
        # Note: DP ranks that disagree settle on extend, the only phase
        # that serves cached and fresh requests alike.
        if len(attention_phases) == 1:
            attention_phase = attention_phases.pop()
        else:
            attention_phase = _EXTEND_PHASE

        # Note: the prefill kernel and the ragged block span are decided
        # here on the host from this step's request lengths. Only a prefill
        # step has a choice.
        bucket = step.padded_num_scheduled_tokens_per_dp_rank
        kernel: str | int = attention_phase
        if attention_phase == _PREFILL_PHASE:
            scheduled_tokens_by_rank = [
                step.scheduled_tokens_per_dp_rank[dp_rank]
                for dp_rank in range(step.dp_size)
            ]
            ragged_block_span = varlen_block_span(scheduled_tokens_by_rank,
                                                  bucket)
            kernel = (_DENSE_KERNEL if use_dense_prefill(
                scheduled_tokens_by_rank, bucket, ragged_block_span) else
                      ragged_block_span)

        # Note: the slice table's bound follows the token bucket and
        # the request slots alone, keeping the compiled shapes fixed across
        # steps.
        page_sizes = attention_group_page_sizes(step.kv_cache_groups)
        for gid, page_size in page_sizes.items():
            max_num_slices = kv_write_slice_bound(
                bucket, step.max_num_reqs_per_dp_rank, page_size)
            slices_view = device_buffer.get_view(
                (step.dp_size * 3, max_num_slices),
                key=f"kv_write_slices_gid_{gid}")
            num_slices_view = device_buffer.get_view(
                (step.dp_size, ), key=f"kv_write_num_slices_gid_{gid}")
            for dp_rank in range(step.dp_size):
                _, req_range, query_range = _rank_slices(dp_rank)
                # Note: the decode kernel appends the leading decode rows
                # itself, so the writer takes only the rows after them.
                num_decode = int(step.request_distribution[3 * dp_rank])
                num_slices_view[dp_rank] = build_kv_write_slices(
                    step.query_start_loc[query_range][num_decode:],
                    step.seq_lens[req_range][num_decode:],
                    step.block_tables[gid][req_range][num_decode:],
                    page_size=page_size,
                    out=slices_view[dp_rank * 3:(dp_rank + 1) * 3])
        kernel_index_view = device_buffer.get_view((step.dp_size, ),
                                                   key=_KERNEL_INDEX_KEY)
        kernel_index_view[:] = step_kernels(bucket).index(kernel)

    def backend_metadata(self, metadata: Mapping[str, jax.Array],
                         gid: int) -> FlywheelMetadata | None:
        # Note: prepare_step and warmup_arrays write a slice table for
        # each attention group only; a mamba group has no pages to write.
        if f"kv_write_slices_gid_{gid}" not in metadata:
            return None
        return FlywheelMetadata(
            kv_write_slices=metadata[f"kv_write_slices_gid_{gid}"],
            kv_write_num_slices=metadata[f"kv_write_num_slices_gid_{gid}"],
            kernel_index=metadata[_KERNEL_INDEX_KEY])

    def warmup_arrays(self, kv_cache_groups: Sequence[KVCacheGroupSpec], *,
                      num_tokens: int, max_num_reqs: int, dp_size: int,
                      mesh: Mesh,
                      sharding: NamedSharding) -> Mapping[str, jax.Array]:
        # Note: the primer stands for a step with no active request, and the
        # extend kernel only clears the whole output bucket, first tile
        # included, when query_start_loc[0] is zero.
        arrays: dict[str, jax.Array] = {
            "seq_lens":
            device_array(mesh,
                         np.zeros(max_num_reqs, dtype=np.int32),
                         sharding=sharding),
            "query_start_loc":
            device_array(mesh,
                         np.zeros(max_num_reqs + dp_size, dtype=np.int32),
                         sharding=sharding),
        }
        # Note: at runtime every attention group carries a KV write slice
        # table, so the primer must carry one of the same shape or the cached
        # HLO is not reused.
        for gid, page_size in attention_group_page_sizes(
                kv_cache_groups).items():
            max_num_slices = kv_write_slice_bound(num_tokens // dp_size,
                                                  max_num_reqs // dp_size,
                                                  page_size)
            arrays[f"kv_write_slices_gid_{gid}"] = device_array(
                mesh,
                np.zeros(dp_size * 3 * max_num_slices, dtype=np.int32),
                sharding=sharding)
            arrays[f"kv_write_num_slices_gid_{gid}"] = device_array(
                mesh, np.zeros(dp_size, dtype=np.int32), sharding=sharding)
        arrays[_KERNEL_INDEX_KEY] = device_array(mesh,
                                                 np.zeros(dp_size,
                                                          dtype=np.int32),
                                                 sharding=sharding)
        return arrays
