# Functions copied from https://source.corp.google.com/h/vertex-model-garden/hex-llm/+/main:/hex_llm/models/jax/layers.py
import functools
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# TODO(pooyam): consolidate `kernel_hexllm.py` with `kernel.py` in there.
from tpu_commons.kernels.ragged_paged_attention.kernel_hexllm import \
    ragged_paged_attention
from tpu_commons.models.jax.layers.attention import MAX_ALLOWED_PAGE_INDICES_N
from tpu_commons.utils_jax import get_megacore


@functools.partial(
    jax.jit,
    donate_argnames=["cache", "operand"],
)
def chunked_prefill_update_cache(cache, indices, operand, num_decode_seqs):
    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    assert B == 1
    operand = jnp.squeeze(operand, 0)
    # operand now: KTH

    # Handle Decode tokens kv cache update.
    decode_indices = jax.lax.slice(indices, (0, ), (T, ))

    # Number of valid indice update could be much smaller
    # than T. We improve performance by skipping the
    # update for those padded indices as much as possible.
    # TODO(b/396129273): tune the value of DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE
    # base on benchmarking
    DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE = 16
    assert T % DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE == 0
    decode_indices = decode_indices.reshape((
        T // DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
    ))
    decode_update_operand = operand.reshape(
        K,
        T // DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        H,
    )
    cache = cache.reshape(K, L * S, H)
    cache = jax.lax.fori_loop(
        0,
        jnp.ceil(num_decode_seqs[0] /
                 DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE).astype(jnp.int32),
        body_fun=lambda i, c: c.at[jnp.arange(K)[..., None], decode_indices[
            i], :].set(decode_update_operand[:, i, :, :]),
        init_val=cache,
    )
    cache = cache.reshape(K, L, S, H)

    # Handle Prefill tokens kv cache update.
    # ruff: noqa: E741
    I = T // S
    prefill_indices = jax.lax.slice(indices, (T, ), (T + I, ))
    # cache: (K, L, S, H)
    # prefill_operand: (K, T, H) -> (K, I, S, H)
    # prefill_indices: (I,)
    prefill_operand = operand.reshape(K, I, S, H)
    cache = cache.at[:, prefill_indices, :, :].set(prefill_operand)
    return cache


def sharded_chunked_prefill_update_cache(mesh: Mesh, ) -> Callable[..., Any]:
    """Shards along KV heads."""
    in_specs = (
        P("model", None, None),  # cache [K, L, S, H]
        P(),  # indice
        P(None, "model", None, None),  # operand [B, K, T, H]
        P(),  # num_decode_seqs
    )
    out_specs = P("model", None, None, None)

    return jax.jit(
        shard_map.shard_map(
            chunked_prefill_update_cache,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


def sharded_chunked_prefill_attention(mesh: Mesh, ) -> Callable[..., Any]:
    """Shards along KV heads."""
    # q: BNTH
    in_specs = (
        P(None, "model", None, None),  # q
        P("model", None, None, None),  # k_pages
        P("model", None, None, None),  # v_pages
        P(),  # decode_lengths
        P(),  # decode_page_indices
        P(),  # num_decode_seqs
        P(),  # prefill_lengths
        P(),  # prefill_page_indices
        P(),  # prefill_query_start_offsets
        P(),  # num_prefill_seqs
    )
    # output: BNTH
    out_specs = P(None, "model", None, None)

    return jax.jit(
        shard_map.shard_map(
            chunked_prefill_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


# Attention that can handle a token batch with a mix of decode and prefill tokens.
#
# Layout of q: [1, num_heads, num_tokens_in_batch, head_dim]
# For the 3rd dimension, decode tokens come first, followed by one or more
# prefill segments. There may be padding between the last decode token
# and the start of first prefill token, as well as between consecutive prefill
# segments, so that each prefill sequence segments are page-aligned.
@functools.partial(jax.jit, donate_argnames=["q"])
def chunked_prefill_attention(
        q: jax.Array,  # [1, num_heads, num_tokens_in_batch, head_dim]
        k_pages: jax.
    Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
        v_pages: jax.
    Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
        decode_lengths: jax.Array,  # [max_num_decode_seqs]
        decode_page_indices: jax.
    Array,  # [max_num_decode_seqs, pages_per_sequence]
        num_decode_seqs: jax.Array,  # [1]
        prefill_lengths: jax.Array,  # [max_num_prefill_seqs]
        prefill_page_indices: jax.
    Array,  # [max_num_prefill_seqs, pages_per_sequence]
        prefill_query_start_offsets: jax.Array,  # [max_num_prefill_seqs + 1]
        num_prefill_seqs: jax.Array,  # [1]
):
    batch_size, _, num_tokens_in_batch, _ = q.shape
    assert batch_size == 1
    q = jnp.squeeze(jnp.swapaxes(q, 1, 2), 0)
    # Now q: [num_tokens_in_batch, num_heads, head_dim]

    max_num_decode_seqs = decode_lengths.shape[0]
    assert max_num_decode_seqs <= num_tokens_in_batch
    decode_outputs = jnp.empty_like(q)
    decode_outputs = jax.lax.cond(
        num_decode_seqs[0] > 0,
        lambda: decode_outputs.at[0:max_num_decode_seqs].set(
            _attention_decode(
                q[0:max_num_decode_seqs],
                k_pages,
                v_pages,
                decode_lengths,
                decode_page_indices,
                num_decode_seqs,
            )),
        lambda: decode_outputs,
    )

    # TODO(b/396129273): Tune num_kv_pages_per_compute_block, num_queries_per_compute_block
    # in a generic way.
    prefill_outputs = ragged_paged_attention(
        q=q,
        k_pages=k_pages,
        v_pages=v_pages,
        kv_lens=prefill_lengths,
        page_indices=prefill_page_indices,
        cu_q_lens=prefill_query_start_offsets,
        num_seqs=num_prefill_seqs,
        num_kv_pages_per_block=16,
        num_queries_per_block=128,
    )

    ret = jnp.where(
        jnp.expand_dims(
            jnp.arange(num_tokens_in_batch) < num_decode_seqs[0], (1, 2)),
        decode_outputs,
        prefill_outputs,
    )

    return jnp.expand_dims(jnp.swapaxes(ret, 0, 1), 0)


@jax.jit
def _attention_decode(
    q,  # [max_num_decode_tokens, num_heads, head_dim]
    k_pages,
    v_pages,
    lengths,
    page_indices,
    num_decode_seqs,
):
    num_tokens_in_batch, blocks_per_seq = page_indices.shape
    paged_attention_fn = functools.partial(
        paged_attention,
        pages_per_compute_block=16,
        megacore_mode="kv_head" if get_megacore() else None,
    )

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_fn(q, k_pages, v_pages, lengths, page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(num_tokens_in_batch, mini_batch_size)
    num_mini_batches = num_tokens_in_batch // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_mini_batches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_mini_batches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_mini_batches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_mini_batches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_mini_batches):
        outputs = jax.lax.cond(
            i * mini_batch_size < num_decode_seqs[0],
            lambda q, k_pages, v_pages, seq_lens, block_indices: outputs.at[i].
            set(
                paged_attention_fn(
                    q,
                    k_pages,
                    v_pages,
                    seq_lens,
                    block_indices,
                )),
            lambda q, k_pages, v_pages, seq_lens, block_indices: outputs,
            q[i],
            k_pages,
            v_pages,
            seq_lens[i],
            block_indices[i],
        )
    return outputs.reshape((num_tokens_in_batch, *outputs.shape[2:]))
