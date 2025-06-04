# ruff: noqa: E731, E741, F722
import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.models.jax.param_init import sharding_init
from tpu_commons.utils_jax import get_megacore

MAX_ALLOWED_PAGE_INDICES_N = (
    128 * 1024
)  # Based on experiments on v5e, 256x1024 results in smem oom but 128x1024 not. TODO: Adjust this based on TPU version.


def shard_array(x: jax.Array, sharding_names: Tuple[str, ...],
                mesh: jax.sharding.Mesh) -> jax.Array:
    # Single device sharding requires this special handling
    # to avoid the recursive jit error.
    if math.prod(mesh.axis_sizes) == 1:
        return jax.device_put(x, jax.devices()[0])
    return jax.device_put(x, NamedSharding(mesh, P(*sharding_names)))


class Einsum(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh),
            self.shape,
            self.dtype,
        )
        return jnp.einsum(eqn, x, w)


class EinsumBias(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh

    # We need this because not every EinsumBias usage is for qkv_proj.
    bias_shape: Optional[Tuple[int, ...]] = None
    bias_named_axes: Optional[Tuple[str, ...]] = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        bias_shape = self.bias_shape or (
            self.shape[0],
            self.shape[2],
        )  # (num_heads or num_kv_heads, head_dim)
        bias_named_axes = self.bias_named_axes or ("model", None)

        b = self.param(
            "bias",
            sharding_init(bias_named_axes, self.mesh),
            bias_shape,
            self.dtype,
        )

        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh),
            self.shape,
            self.dtype,
        )

        return jnp.einsum(eqn, x, w), b


class Embedder(nn.Module):
    vocab_size: int
    hidden_size: int
    dtype: jnp.dtype
    mesh: Mesh

    def setup(self) -> None:
        self.input_embedding_table = self.param(
            "weight",
            sharding_init(
                ("model", None),
                self.mesh,
            ),
            (self.vocab_size, self.hidden_size),
            self.dtype,
        )

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.input_embedding_table[(x, )]
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return jnp.dot(x, self.input_embedding_table.T)


class RMSNorm(nn.Module):
    rms_norm_eps: float
    dtype: jnp.dtype
    mesh: Mesh

    @nn.compact
    def __call__(self, x) -> jax.Array:
        scale = self.param(
            "weight",
            sharding_init((None, ), self.mesh),
            (x.shape[-1], ),
            self.dtype,
        )
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(
            x * jnp.reciprocal(jnp.sqrt(var + self.rms_norm_eps)))
        normed_inputs = normed_inputs * scale
        return normed_inputs


def sharded_flash_attention(mesh: Mesh,
                            cache_attention_scores: bool = False,
                            causal: bool = True) -> Callable[..., Any]:
    in_specs = (
        P("data", "model", None, None),  # q
        P("data", "model", None, None),  # k
        P("data", "model", None, None),  # vx
    )
    flash_attention_fn = flash_attention
    out_specs = P("data", "model", None, None)
    return jax.jit(
        shard_map.shard_map(
            functools.partial(flash_attention_fn, causal=causal),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: Optional[float] = None,
    cache_attention_scores: bool = False,
) -> Callable[..., Any]:
    """Shards GQA PagedAttention along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P("model", None, None, None),  # k
        P("model", None, None, None),  # v
        P(),  # lengths
        P(),  # page_indices
    )
    paged_attention_fn = paged_attention
    out_specs = P(None, "model", None)

    def _paged_attention_fn(q, k, v, lengths, page_indices):
        if page_indices.size > MAX_ALLOWED_PAGE_INDICES_N:
            raise ValueError(
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` to run with minibatches."
            )
        return paged_attention_fn(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        shard_map.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


@functools.partial(jax.jit, static_argnums=[0])
def paged_attention_with_guarded_smem(
    paged_attention_kernel: Callable,
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
):
    # Addresses b/336316706. Summary:
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent. Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths,
                                      page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(q[i], k_pages, v_pages, seq_lens[i],
                                   block_indices[i]))

    outputs = outputs.reshape((batch_size, *outputs.shape[2:]))

    return outputs


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "seq_lens",
        "block_indices",
        "kv_cache_write_indices",
        "decode_lengths",
        "decode_page_indices",
        "num_decode_seqs",
        "prefill_lengths",
        "prefill_page_indices",
        "prefill_query_start_offsets",
        "num_prefill_seqs",
    ],
    meta_fields=["chunked_prefill_enabled"],
)
@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]

    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]


def update_cache(
    is_prefill,
    cache,
    indices,
    operand,
    prefill_seq_len=None,
    sliding_window=None,
) -> jax.Array:

    # (8, 55640, 32, 128) (1, 8, 256, 128) -> K (8, 8, 32, 128)
    # I = B * T // S
    # k cache, operand

    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op inserted by
    #    the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from actual prompt, not from the padded tokens.
        if sliding_window and T > sliding_window:
            assert B == 1
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            operand = jax.lax.dynamic_slice_in_dim(
                operand, start_index, sliding_window,
                axis=2)  # TODO: @pooyam Perf check this.
            T = sliding_window

        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)
        indices = indices.reshape(I)
        cache = cache.at[:, indices, :, :].set(operand)
    else:
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        cache = cache.reshape(K, L * S, H)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache
