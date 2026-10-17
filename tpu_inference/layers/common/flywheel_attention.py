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
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import flywheel_tpu
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flywheel_tpu import flash_attn_func, flash_attn_with_kvcache
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheGroupSpec,
                                        UniformTypeKVCacheSpecs)

from tpu_inference import envs
from tpu_inference.layers.common.attention_backend import (AttentionBackend,
                                                           AttentionFields,
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
DECODE_HEAD_DIMS = (128, 256)


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


# Note: these kernels tile the KV token axis in 128-token blocks, so a
# page that is not a whole number of blocks cannot be indexed.
PAGE_SIZE_MULTIPLE = 128

PREFILL_PHASE = "prefill"
DECODE_PHASE = "decode"
EXTEND_PHASE = "extend"
ATTENTION_PHASES = (PREFILL_PHASE, DECODE_PHASE, EXTEND_PHASE)

DENSE_MIN_SEQ_LEN = 128

# Note: the dense kernel's largest physical block. A longer axis that
# is not a whole number of these falls to a smaller divisor, a schedule the
# cost model does not describe, so such a bucket stays ragged.
DENSE_MAX_BLOCK = 2048

# Note: flash_attn_with_kvcache reads the cache in the largest KV
# block up to its VMEM target that divides the row capacity, so each row is
# padded until every power-of-two block up to this many tokens divides it;
# otherwise a prime page count leaves only one-page blocks.
DECODE_CAPACITY_ALIGN_TOKENS = 4096


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
        return DECODE_PHASE
    elif num_decode > 0 or np.any(kv_lens != q_lens):
        # Note: a mixed step is EXTEND even when its prefill requests
        # are fresh. The ragged kernel would need the decode rows merged into
        # its output, an extra [T, N, D] copy per layer that blocks the MLP
        # weight prefetch.
        return EXTEND_PHASE
    else:
        return PREFILL_PHASE


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
    block = min(bucket, DENSE_MAX_BLOCK)
    num_blocks = bucket // block
    dense_cost = num_blocks * (num_blocks + 1) // 2 * block * block
    return dense_cost <= ragged_step_cost(scheduled_tokens_per_dp_rank, bucket,
                                          ragged_block_span)


# Note: the ragged kernel's own block sizes without 128, which 256
# matches or beats. A span reaches the kernel as max_seqlen_q/k, which it
# turns into the q/kv block cap.
VARLEN_BLOCK_SPANS = (256, 512, 1024, 2048)

# Note: fixed cost of one ragged kernel step, as the side of a tile of
# equal compute, fitted on v6e. It belongs to the schedule rather than to the
# head count or head_dim, so the host model needs neither.
VARLEN_STEP_OVERHEAD_TOKENS = 704


def varlen_block_spans(padded_num_tokens_per_dp_rank: int) -> tuple[int, ...]:
    """Every block span varlen_block_span can return at this token bucket,
    which is the set of ragged prefill backbones the warmup compiles."""
    spans = tuple(s for s in VARLEN_BLOCK_SPANS
                  if s <= padded_num_tokens_per_dp_rank)
    return spans or (padded_num_tokens_per_dp_rank, )


def varlen_block_span(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
) -> int:
    """The varlen_block_spans entry minimising the modelled ragged kernel cost
    of the step (one executable serves every DP rank); ties go wider."""
    return min(varlen_block_spans(padded_num_tokens_per_dp_rank),
               key=lambda span:
               (ragged_step_cost(scheduled_tokens_per_dp_rank,
                                 padded_num_tokens_per_dp_rank, span), -span))


def ragged_step_cost(
    scheduled_tokens_per_dp_rank: Sequence[Sequence[int]],
    padded_num_tokens_per_dp_rank: int,
    span: int,
) -> int:
    """Modelled cost of one ragged prefill step at this block span."""
    # Note: a sequence of t blocks visits the t (t + 1) / 2
    # (q block, kv block) pairs of its causal block-diagonal, so many short
    # sequences push the span down and one long one pushes it up. The kernel
    # runs the bucket's unfilled tail as one more sequence, so the model counts
    # it too.
    # Note: the DP ranks run concurrently, so the step costs the
    # slowest rank.
    max_num_steps = 0
    for lengths in scheduled_tokens_per_dp_rank:
        tail = padded_num_tokens_per_dp_rank - sum(lengths)
        num_blocks = [
            -(-length // span) for length in [*lengths, tail] if length > 0
        ]
        max_num_steps = max(max_num_steps,
                            sum(t * (t + 1) // 2 for t in num_blocks))
    return max_num_steps * (span * span + VARLEN_STEP_OVERHEAD_TOKENS**2)


def is_dense_tileable(padded_num_tokens_per_dp_rank: int) -> bool:
    """Whether the dense kernel can tile a (1, bucket) row at this token
    bucket."""
    bucket = padded_num_tokens_per_dp_rank
    return (bucket >= DENSE_MIN_SEQ_LEN and bucket % PAGE_SIZE_MULTIPLE == 0
            and (bucket <= DENSE_MAX_BLOCK or bucket % DENSE_MAX_BLOCK == 0))


def fold_heads_outer(x: jax.Array) -> jax.Array:
    """[T, H, D] -> (H, 1, T, D), flash_attn_func's heads-outer layout of one
    sequence."""
    # Note: XLA already lays the projections out heads-outermost, so
    # this transpose compiles to a bitcast; a batch-outer fold would copy.
    return x.transpose(1, 0, 2)[:, None]


def flywheel_tpu_prefill(
    q: jax.Array,  # [T, N, D]
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    *,
    sm_scale: float,
    dense_prefill: bool,
    varlen_block_span: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Prefill with the dense kernel on the step's one (1, T) row, otherwise
    with the ragged one at the host-chosen block span.

    Every request is a fresh prefill one: select_attention_phase sends a step
    with a decode row or a cached prefix to extend, so both kernels attend
    this step's own K/V and the cache is only written.
    """
    # Note: the fused writer interleaves K and V in VMEM; stacking
    # them in XLA first costs two more HBM passes.
    cache = write_kv_cache_pages(k, v, kv_cache, kv_write_slices,
                                 kv_write_num_slices)
    num_tokens, num_heads, head_dim = q.shape
    if dense_prefill:
        output = flash_attn_func(
            fold_heads_outer(q),
            fold_heads_outer(k),
            fold_heads_outer(v),
            causal=True,
            softmax_scale=sm_scale,
            token_major=False,
            heads_outer_fold=True,
        )
        # Note: a trailing transpose, which XLA folds into o_proj
        # instead of copying.
        return output.reshape(num_heads, num_tokens,
                              head_dim).transpose(1, 0, 2), cache
    elif varlen_block_span is not None and 0 < varlen_block_span <= num_tokens:
        output = flash_attn_varlen_func(
            q.transpose(1, 0, 2),
            k.transpose(1, 0, 2),
            v.transpose(1, 0, 2),
            cu_q_lens,
            cu_q_lens,
            max_seqlen_q=varlen_block_span,
            max_seqlen_k=varlen_block_span,
            causal=True,
            softmax_scale=sm_scale,
            token_major=False,
        )
        return output.transpose(1, 0, 2), cache
    else:
        raise ValueError(
            "ragged prefill needs the host-chosen block span "
            "(FlywheelMetadata.varlen_block_span, see the varlen_block_span "
            f"helper) in [1, {num_tokens}]; got {varlen_block_span}.")


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
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    batch = min(max_num_seqs, q.shape[0])
    # Note: the padding entries are only read by the zero-length tail
    # copies of a row's last block (the kernel clips each page copy to what is
    # left of the row), so page 0 is a safe filler.
    align_pages = math.lcm(DECODE_CAPACITY_ALIGN_TOKENS,
                           page_size) // page_size
    padded_pages_per_seq = align_to(pages_per_seq, align_pages)
    block_table = jnp.pad(
        page_indices[:batch * pages_per_seq].reshape(batch, pages_per_seq),
        ((0, 0), (0, padded_pages_per_seq - pages_per_seq)))
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


def flywheel_tpu_extend(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_write_slices: jax.Array,
    kv_write_num_slices: jax.Array,
    num_decode: jax.Array,  # i32 scalar: leading one-token requests
    num_active: jax.Array,  # i32 scalar: leading rows that are requests
    *,
    sm_scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Append only the scheduled tokens, then attend to their entire cached
    prefix; the phase for a prefill request whose keys are partly in the pages.

    A mixed step splits over the same written cache: its leading num_decode
    one-token rows go to the decode kernel, the prefill requests to the paged
    varlen kernel, over a request window slid past them.
    """
    cache = write_kv_cache_pages(k, v, kv_cache, kv_write_slices,
                                 kv_write_num_slices)
    # Note: the decode rows go to the batched decode kernel because
    # the varlen kernel walks requests serially at a per-request cost that
    # ignores the query length. The cache threads writer -> decode kernel so
    # the one donated buffer aliases the chain; the varlen kernel only reads
    # it.
    decode_output, cache = paged_cache_decode(q,
                                              None,
                                              None,
                                              cache,
                                              kv_lens,
                                              page_indices,
                                              num_decode,
                                              sm_scale=sm_scale)
    page_size = cache.shape[1]
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
    # sequence's own pages, so its table needs none of the decode kernel's
    # capacity padding.
    output = flash_attn_varlen_func(
        q.transpose(1, 0, 2),
        cache,
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
    # Note: the varlen kernel does not own the rows below its first
    # request (row num_decode): its first q block may start anywhere below it,
    # and a decode-only DP rank leaves it no request. num_decode never exceeds
    # num_leading_rows, so rebuilding the leading rows covers all of them.
    num_leading_rows = min(align_to(decode_output.shape[0], 8), q.shape[0])
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
    return output.at[:num_leading_rows].set(leading_output), cache


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
    request_distribution: jax.
    Array,  # i32[3]: num_decode, num_decode, num_reqs
    *,
    phase: str,  # one of ATTENTION_PHASES, static
    sm_scale: float,
    dense_prefill: bool = False,
    varlen_block_span: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """One shard's attention for a fresh prefill, cached extend, or decode
    step, dispatched on the host-decided phase and prefill kernel choice.

    request_distribution is this rank's: its last entry bounds the decode
    kernel's active rows and its first splits a mixed batch between the
    decode kernel and the paged varlen kernel.
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
    # Note: per-rank metadata is sharded as flat arrays, so the
    # writer's (3, bound) table arrives flattened.
    kv_write_slices = kv_write_slices.reshape(3, -1)

    if phase == DECODE_PHASE:
        # Note: the decode kernel appends each row's K/V itself, so a
        # decode step needs neither the DMA writer nor its slice table.
        decode_output, kv_cache = paged_cache_decode(padded_q,
                                                     padded_k,
                                                     padded_v,
                                                     kv_cache,
                                                     kv_lens,
                                                     page_indices,
                                                     request_distribution[2],
                                                     sm_scale=sm_scale)
        output = jnp.pad(decode_output,
                         ((0, padded_q.shape[0] - decode_output.shape[0]),
                          (0, 0), (0, 0)))
    elif phase == EXTEND_PHASE:
        output, kv_cache = flywheel_tpu_extend(padded_q,
                                               padded_k,
                                               padded_v,
                                               kv_cache,
                                               kv_lens,
                                               page_indices,
                                               cu_q_lens,
                                               kv_write_slices,
                                               kv_write_num_slices,
                                               request_distribution[0],
                                               request_distribution[2],
                                               sm_scale=sm_scale)
    elif phase == PREFILL_PHASE:
        output, kv_cache = flywheel_tpu_prefill(
            padded_q,
            padded_k,
            padded_v,
            kv_cache,
            cu_q_lens,
            kv_write_slices,
            kv_write_num_slices,
            sm_scale=sm_scale,
            dense_prefill=dense_prefill,
            varlen_block_span=varlen_block_span)
    else:
        raise ValueError(f"phase must be one of {ATTENTION_PHASES}; got "
                         f"{phase!r}.")
    return output[..., :head_dim], kv_cache


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
    *,
    phase: str,
    sm_scale: float,
    dense_prefill: bool,
    varlen_block_span: int | None,
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
                (P(ShardingAxisName.ATTN_DATA), ) * 6)
    out_specs = (qkv_spec, kv_cache_spec)

    def _attention(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
        return flywheel_tpu_attention(*args,
                                      phase=phase,
                                      sm_scale=sm_scale,
                                      dense_prefill=dense_prefill,
                                      varlen_block_span=varlen_block_span)

    return jax.shard_map(
        _attention,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(q, sharded_k, sharded_v, kv_cache, kv_lens, page_indices, cu_q_lens,
      kv_write_slices, kv_write_num_slices, request_distribution)


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
    if q.shape[-1] not in DECODE_HEAD_DIMS:
        raise NotImplementedError(
            f"The flywheel attention backend decodes only head_dim in "
            f"{DECODE_HEAD_DIMS}; got {q.shape[-1]}.")
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


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["kv_write_slices", "kv_write_num_slices"],
    meta_fields=["attention_phase", "dense_prefill", "varlen_block_span"],
)
@dataclass
class FlywheelMetadata:
    """AttentionMetadata.backend_metadata of an attention KV cache group in
    one step under --attention-backend flywheel.

    The static fields pick the kernels on the host, so each value compiles
    its own backbone. RPA reads the step's phases from request_distribution
    inside one kernel; picking between these kernels on the device instead
    made XLA copy the whole KV cache once per layer.
    """
    # STATIC: one of ATTENTION_PHASES.
    attention_phase: str
    # The KV writer's slice table, flat (dp_size * 3 * bound,) int32, and its
    # (dp_size,) int32 slice counts, split per DP rank like block_tables.
    kv_write_slices: jax.Array
    kv_write_num_slices: jax.Array
    # STATIC, prefill only: the dense kernel on the step's one (1, bucket)
    # row, or else the ragged kernel at this block span.
    dense_prefill: bool = False
    varlen_block_span: int | None = None


@dataclass(frozen=True)
class FlywheelStepFields:
    """The AttentionFields of one step: each attention KV cache group in
    attention_gids gets a FlywheelMetadata with its KV write slice table from
    the unpacked device buffer, any other group (mamba) None."""
    attention_phase: str
    attention_gids: frozenset[int]
    dense_prefill: bool = False
    varlen_block_span: int | None = None

    def __call__(self, metadata: Mapping[str, jax.Array],
                 gid: int) -> FlywheelMetadata | None:
        if gid not in self.attention_gids:
            return None
        return FlywheelMetadata(
            attention_phase=self.attention_phase,
            kv_write_slices=metadata[f"kv_write_slices_gid_{gid}"],
            kv_write_num_slices=metadata[f"kv_write_num_slices_gid_{gid}"],
            dense_prefill=self.dense_prefill,
            varlen_block_span=self.varlen_block_span)


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
        if model_config.get_head_size() not in DECODE_HEAD_DIMS:
            problems.append(
                f"head_dim {model_config.get_head_size()} is not one of the "
                f"decode kernel's {DECODE_HEAD_DIMS}")
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
            phase=flywheel_md.attention_phase,
            sm_scale=sm_scale,
            dense_prefill=flywheel_md.dense_prefill,
            varlen_block_span=flywheel_md.varlen_block_span,
        )
        return kv_cache, output

    def prepare_step(self, step: HostStepInputs,
                     device_buffer: DeviceBuffer) -> AttentionFields:
        """Decide the step's static phase and prefill kernel, and append each
        attention group's KV write slice table to device_buffer."""

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
        # Note: the step's phase is static metadata, one backbone
        # executable per phase, so DP ranks that disagree have to settle on
        # extend, the only phase that serves cached and fresh requests alike.
        if len(attention_phases) == 1:
            attention_phase = attention_phases.pop()
        else:
            attention_phase = EXTEND_PHASE

        # Note: the prefill kernel and the ragged block span are static
        # metadata, so both are decided here on the host from this step's
        # request lengths. Only a prefill step has a choice.
        bucket = step.padded_num_scheduled_tokens_per_dp_rank
        dense_prefill = False
        block_span = None
        if attention_phase == PREFILL_PHASE:
            scheduled_tokens_by_rank = [
                step.scheduled_tokens_per_dp_rank[dp_rank]
                for dp_rank in range(step.dp_size)
            ]
            ragged_block_span = varlen_block_span(scheduled_tokens_by_rank,
                                                  bucket)
            dense_prefill = use_dense_prefill(scheduled_tokens_by_rank, bucket,
                                              ragged_block_span)
            if not dense_prefill:
                block_span = ragged_block_span

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
                num_slices_view[dp_rank] = build_kv_write_slices(
                    step.query_start_loc[query_range],
                    step.seq_lens[req_range],
                    step.block_tables[gid][req_range],
                    page_size=page_size,
                    out=slices_view[dp_rank * 3:(dp_rank + 1) * 3])

        return FlywheelStepFields(
            attention_phase=attention_phase,
            attention_gids=frozenset(page_sizes),
            dense_prefill=dense_prefill,
            varlen_block_span=block_span,
        )

    def warmup_variants(self, kv_cache_groups: Sequence[KVCacheGroupSpec], *,
                        num_tokens: int, num_tokens_paddings: Sequence[int],
                        max_num_reqs: int,
                        dp_size: int) -> list[AttentionFields]:
        # Note: RPA serves every phase from one kernel and one
        # executable per bucket. Here the phase and the prefill kernel are
        # static metadata, so each choice a step can make at this bucket is
        # its own backbone; one the warmup skipped would compile while
        # serving, even with a warm persistent compile cache.
        bucket = num_tokens // dp_size
        attention_gids = frozenset(attention_group_page_sizes(kv_cache_groups))
        variants: list[AttentionFields] = [
            FlywheelStepFields(attention_phase=PREFILL_PHASE,
                               attention_gids=attention_gids,
                               varlen_block_span=span)
            for span in varlen_block_spans(bucket)
        ]
        if is_dense_tileable(bucket):
            variants.append(
                FlywheelStepFields(attention_phase=PREFILL_PHASE,
                                   attention_gids=attention_gids,
                                   dense_prefill=True))
        variants.append(
            FlywheelStepFields(attention_phase=EXTEND_PHASE,
                               attention_gids=attention_gids))
        # Note: a decode step carries one token per request, so the
        # bucket a full request batch rounds up to is the largest one decode
        # can reach.
        max_decode_bucket = next(
            (p for p in num_tokens_paddings if p >= max_num_reqs),
            num_tokens_paddings[-1])
        if num_tokens <= max_decode_bucket:
            variants.append(
                FlywheelStepFields(attention_phase=DECODE_PHASE,
                                   attention_gids=attention_gids))
        return variants

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
        return arrays
