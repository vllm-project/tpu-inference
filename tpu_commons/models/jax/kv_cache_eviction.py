import functools
from typing import Callable, Optional, Tuple, Type, TypeAlias, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_commons.models.jax import layers
from tpu_commons.models.jax.block import BlockTable
from tpu_commons.models.jax.config import CacheConfig

KVCache: TypeAlias = Tuple[jax.Array, jax.Array]

# (input, input_positions) -> output
PrepareQKFn: TypeAlias = Callable[[jax.Array, jax.Array], jax.Array]


class KVCacheUpdater:
    eviction_algorithm: Optional[str] = None

    sliding_window: Optional[int]
    sink_size: Optional[int]
    cache_attention_scores: bool
    prefill_repeat_kv: int
    is_mqa: bool

    def __init__(
        self,
        sliding_window: Optional[int],
        sink_size: Optional[int],
        mesh: Mesh,
        cache_attention_scores: bool = False,
        prefill_repeat_kv: int = 1,
        is_mqa: bool = False,
    ):
        self.sliding_window = sliding_window
        self.sink_size = sink_size
        self.cache_attention_scores = cache_attention_scores
        self.prefill_repeat_kv = prefill_repeat_kv
        self.is_mqa = is_mqa

        self.flash_attention = layers.sharded_flash_attention(
            mesh, cache_attention_scores=cache_attention_scores)
        self.splash_attention = layers.sharded_splash_attention(
            mesh, None, None, is_mqa)
        # TODO: Deprecate usage of `paged_attention` (Instead use `guarded_paged_attention`) in H2O or attention score caching.
        self.paged_attention = layers.sharded_paged_attention(
            mesh, cache_attention_scores=cache_attention_scores)
        self.guarded_paged_attention = functools.partial(
            layers.paged_attention_with_guarded_smem,
            self.paged_attention,
        )
        self.sharded_chunked_prefill_attention = (
            layers.sharded_chunked_prefill_attention(mesh))
        self.sharded_chunked_prefill_update_cache = (
            layers.sharded_chunked_prefill_update_cache(mesh))

    def _update_cache(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        k: jax.Array,
        v: jax.Array,
        prefill_seq_len: jax.Array,
        kv_cache_write_indices: jax.Array,
        chunked_prefill_enabled: bool = False,
        num_decode_seqs=None,
    ) -> KVCache:

        if kv_cache is None:
            return k, v

        k_cache, v_cache = kv_cache

        if chunked_prefill_enabled:
            k_cache = self.sharded_chunked_prefill_update_cache(
                k_cache, kv_cache_write_indices, k, num_decode_seqs)
            v_cache = self.sharded_chunked_prefill_update_cache(
                v_cache, kv_cache_write_indices, v, num_decode_seqs)
        else:
            k_cache = layers.update_cache(
                is_prefill,
                k_cache,
                kv_cache_write_indices,
                k,
                prefill_seq_len,
                self.sliding_window,
                self.sink_size,
            )
            v_cache = layers.update_cache(
                is_prefill,
                v_cache,
                kv_cache_write_indices,
                v,
                prefill_seq_len,
                self.sliding_window,
                self.sink_size,
            )
        return k_cache, v_cache

    def _prepare_prefill_kv(
        self,
        k: jax.Array,
        v: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        block_indices: jax.Array,
        kv_cache_write_indices: jax.Array,
        prepare_qk_fn:
        PrepareQKFn,  # This should be set by the caller. Because different layers can have different rope mechanism.
        input_positions: Optional[jax.Array] = None,
    ) -> KVCache:
        """
        Prepares keys and values during prefill. If input_positions is provided,
        applies RoPE to the cached keys.
        """
        if block_indices.shape[1] > kv_cache_write_indices.shape[1]:
            # This means we just prefilled the non-cached succeeding tokens.
            # We need to fetch all kv tensors for the whole sequence before
            # passing in flash attention.
            k = layers.fetch_cache(k_cache, block_indices)
            v = layers.fetch_cache(v_cache, block_indices)
        if input_positions is not None:
            k = prepare_qk_fn(k, input_positions)
        # When it's not MQA, we need to repeat KV heads to use MHA
        if self.prefill_repeat_kv > 1 and not self.is_mqa:
            k = jnp.repeat(k, self.prefill_repeat_kv, axis=1)
            v = jnp.repeat(v, self.prefill_repeat_kv, axis=1)
        return k, v

    def _prefill_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
    ) -> jax.Array:
        # Although current flash attention kernel implementation
        # supports varient lengths for q and k/v, it aligns q to k/v's
        # left ending (instead of right ending). The workaround is we
        # use splash attention in case of cache hit.
        left_paddings = k.shape[2] - q.shape[2]
        assert left_paddings >= 0
        # TODO (b/410665507): Enable splash_attention by default.
        outputs = (self.splash_attention(q, k, v) if left_paddings
                   or self.is_mqa else self.flash_attention(q, k, v))
        return outputs

    def _attention_with_sliding_window(
        self,
        is_prefill: bool,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        prepare_qk_fn: PrepareQKFn,
        kv_cache_position_indices: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        attn_scores = None

        block_indices = attention_metadata.block_indices
        kv_cache_write_indices = attention_metadata.kv_cache_write_indices
        input_positions = attention_metadata.input_positions
        seq_lens = attention_metadata.seq_lens

        if is_prefill:
            k, v = self._prepare_prefill_kv(
                k,
                v,
                k_cache,
                v_cache,
                block_indices,
                kv_cache_write_indices,
                prepare_qk_fn,
                input_positions,
            )
            outputs = self._prefill_attention(q, k, v)
            if self.cache_attention_scores:
                outputs, attn_scores = outputs
        else:
            # (B, N, H)
            q = jnp.squeeze(q, 2)

            k_cache_selected = k_cache.at[:, block_indices, :, :].get(
            )  # (K, B, num_blocks_sliding_window, S, H)
            K, B, num_blocks_sliding_window, S, H = k_cache_selected.shape

            v_cache_selected = v_cache.at[:, block_indices, :, :].get()
            v_cache_selected = v_cache_selected.reshape(K, -1, S, H)

            k_cache_selected = jnp.transpose(k_cache_selected, (1, 0, 2, 3, 4))
            k_cache_selected = k_cache_selected.reshape(
                B, K, -1, H)  # (B, K, num_blocks_sliding_window * S, H)
            k_cache_selected = prepare_qk_fn(
                k_cache_selected,
                kv_cache_position_indices,
            )
            k_cache_selected = k_cache_selected.reshape(
                B, K, num_blocks_sliding_window, S, H)
            k_cache_selected = jnp.transpose(k_cache_selected, (1, 0, 2, 3, 4))
            k_cache_selected = k_cache_selected.reshape(
                K, B * num_blocks_sliding_window, S,
                H)  # (K, B * num_blocks_sliding_window, S, H)

            block_indices_selected = jnp.arange(
                block_indices.shape[0] * block_indices.shape[1]).reshape(
                    block_indices.shape)

            outputs = self.paged_attention(
                q,
                k_cache_selected,
                v_cache_selected,
                seq_lens,
                block_indices_selected,
            )
            if self.cache_attention_scores:
                outputs, attn_scores = outputs
            # (B, N, 1, H)
            outputs = jnp.expand_dims(outputs, 2)
        return outputs, attn_scores

    def update_cache_with_attn(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        prepare_qk_fn: PrepareQKFn,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
    ) -> Tuple[KVCache, jax.Array, Optional[jax.Array]]:
        """
        Updates the KV cache and calculates the attention. It also applies RoPE
        to KV since different caching algorithms might apply RoPE at different
        steps.

        Returns: KVCache, output, optional attention scores.
        """
        del kv_cache_position_indices
        del evict_write_indices
        del replacement_write_indices

        input_positions = attention_metadata.input_positions
        seq_lens = attention_metadata.seq_lens
        kv_cache_write_indices = attention_metadata.kv_cache_write_indices
        block_indices = attention_metadata.block_indices
        chunked_prefill_enabled = attention_metadata.chunked_prefill_enabled

        k = prepare_qk_fn(k, input_positions)
        k_cache, v_cache = self._update_cache(
            is_prefill,
            kv_cache,
            k,
            v,
            None if seq_lens is None else seq_lens[0],
            kv_cache_write_indices,
            chunked_prefill_enabled=chunked_prefill_enabled,
            num_decode_seqs=attention_metadata.num_decode_seqs,
        )

        if chunked_prefill_enabled:
            outputs = self.sharded_chunked_prefill_attention(
                q,
                k_cache,
                v_cache,
                attention_metadata.decode_lengths,
                attention_metadata.decode_page_indices,
                attention_metadata.num_decode_seqs,
                attention_metadata.prefill_lengths,
                attention_metadata.prefill_page_indices,
                attention_metadata.prefill_query_start_offsets,
                attention_metadata.num_prefill_seqs,
            )
        elif is_prefill:
            k, v = self._prepare_prefill_kv(
                k,
                v,
                k_cache,
                v_cache,
                block_indices,
                kv_cache_write_indices,
                prepare_qk_fn,
            )
            outputs = self._prefill_attention(q, k, v)
        else:
            # (B, N, H)
            q = jnp.squeeze(q, 2)
            outputs = self.guarded_paged_attention(q, k_cache, v_cache,
                                                   seq_lens, block_indices)
            # (B, N, 1, H)
            outputs = jnp.expand_dims(outputs, 2)

        return (k_cache, v_cache), outputs, None

    def update_kv_caches(
        self,
        is_prefill: bool,
        x: jax.Array,
        kv_caches: list[KVCache],
        seq_lens: jax.Array,
        accumulated_attn_scores: jax.Array,
        eviction_score_mask: jax.Array,
        kv_cache_write_indices: jax.Array,
    ) -> Tuple[list[KVCache], Optional[jax.Array]]:
        """
        Optionally updates the full KV cache.

        Returns: The updated KV caches and cached attention scores if applicable.
        """
        del (
            is_prefill,
            x,
            seq_lens,
            accumulated_attn_scores,
            eviction_score_mask,
            kv_cache_write_indices,
        )
        return kv_caches, None


class StreamingLlmKVCacheUpdater(KVCacheUpdater):
    eviction_algorithm: Optional[str] = "streamingllm"

    def update_cache_with_attn(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        prepare_qk_fn: PrepareQKFn,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
    ) -> Tuple[KVCache, jax.Array, Optional[jax.Array]]:
        del evict_write_indices
        del replacement_write_indices

        k_cache, v_cache = self._update_cache(
            is_prefill,
            kv_cache,
            k,
            v,
            attention_metadata.seq_lens[0],
            attention_metadata.kv_cache_write_indices,
        )

        outputs, _ = self._attention_with_sliding_window(
            is_prefill,
            q,
            k,
            v,
            k_cache,
            v_cache,
            attention_metadata,
            prepare_qk_fn,
            kv_cache_position_indices,
        )

        return (k_cache, v_cache), outputs, None


class H2oKVCacheUpdater(KVCacheUpdater):
    eviction_algorithm: Optional[str] = "h2o"

    def update_cache_with_attn(
        self,
        is_prefill: bool,
        kv_cache: Optional[KVCache],
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        attention_metadata: layers.AttentionMetadata,
        prepare_qk_fn: PrepareQKFn,
        kv_cache_position_indices: Optional[jax.Array] = None,
        evict_write_indices: Optional[jax.Array] = None,
        replacement_write_indices: Optional[jax.Array] = None,
    ) -> Tuple[KVCache, jax.Array, Optional[jax.Array]]:

        # (K, L, S, H)
        if kv_cache is None:
            k_cache, v_cache = k, v
        else:
            k_cache, v_cache = kv_cache
            if not is_prefill:
                # Evict the least important KV from the cache.
                # Replace the evicted KV with the replacement KV.
                k_replacement = get_replacement_token(
                    k_cache, self.eviction_algorithm,
                    replacement_write_indices)
                v_replacement = get_replacement_token(
                    v_cache, self.eviction_algorithm,
                    replacement_write_indices)
                k_cache = layers.update_cache(is_prefill, k_cache,
                                              evict_write_indices,
                                              k_replacement)
                v_cache = layers.update_cache(is_prefill, v_cache,
                                              evict_write_indices,
                                              v_replacement)
            # For KV cache eviction, we initially store the entire prompt in the KV cache.
            # KV cache will be compressed later, at the end of prefill phase, after all layers have been processed.
            k_cache = layers.update_cache(
                is_prefill, k_cache, attention_metadata.kv_cache_write_indices,
                k)
            v_cache = layers.update_cache(
                is_prefill, v_cache, attention_metadata.kv_cache_write_indices,
                v)

        outputs, attn_scores = self._attention_with_sliding_window(
            is_prefill,
            q,
            k,
            v,
            k_cache,
            v_cache,
            attention_metadata,
            prepare_qk_fn,
            kv_cache_position_indices,
        )
        attn_scores = attn_scores.sum(
            axis=-2)  # sum over all heads. Shape = (batch_size, seq_len)
        return (k_cache, v_cache), outputs, attn_scores

    def update_kv_caches(
        self,
        is_prefill: bool,
        x: jax.Array,
        kv_caches: list[KVCache],
        seq_lens: jax.Array,
        accumulated_attn_scores: jax.Array,
        eviction_score_mask: jax.Array,
        kv_cache_write_indices: jax.Array,
    ) -> Tuple[list[KVCache], Optional[jax.Array]]:
        # Evict the least important tokens from the KV cache and the attention score cache.
        if not is_prefill:
            return
        padded_prompt_len = x.shape[1]
        kv_caches, accumulated_attn_scores = compress_caches_prefill(
            kv_caches,
            accumulated_attn_scores,
            self.eviction_algorithm,
            seq_lens[0],
            padded_prompt_len,
            eviction_score_mask,
            self.sliding_window,
            self.sink_size,
            kv_cache_write_indices,
        )
        return kv_caches, accumulated_attn_scores


def get_kv_cache_updater_class(
    eviction_algorithm: Optional[str] = None, ) -> Type[KVCacheUpdater]:
    if not eviction_algorithm:
        return KVCacheUpdater
    elif eviction_algorithm == "streamingllm":
        return StreamingLlmKVCacheUpdater
    elif eviction_algorithm == "h2o":
        return H2oKVCacheUpdater
    else:
        raise ValueError(f"Unknown eviction algorithm: {eviction_algorithm}")


@functools.partial(jax.jit,
                   static_argnames=["eviction_algorithm", "is_prefill"],
                   donate_argnums=0)
def update_attn_score_cache(
    attn_score_cache: jax.Array,
    eviction_algorithm: str,
    single_step_attn_scores: jax.Array,
    running_indices: jax.Array,
    is_prefill: bool,
) -> jax.Array:
    """
    Updates attn_score_cache with the new attention scores produced from the most recent forward pass.
    """
    if eviction_algorithm == "h2o":
        # single_step_attn_scores: (batch_size, sink_size + sliding_window)
        # attn_score_cache: (max_decode_seqs + 1, sink_size + sliding_window)
        if is_prefill:
            attn_score_cache = attn_score_cache.at[running_indices, :].set(
                single_step_attn_scores.astype(attn_score_cache.dtype))
        else:
            attn_score_cache = attn_score_cache.at[running_indices, :].add(
                single_step_attn_scores.astype(attn_score_cache.dtype))
    return attn_score_cache


@functools.partial(jax.jit,
                   static_argnames=["eviction_algorithm", "cache_config"])
def get_evict_idx(
    attn_score_cache: jax.Array,
    eviction_algorithm: str,
    running_idx: int,
    cache_config: CacheConfig,
) -> Union[jax.Array, int]:
    """
    IMPORTANT: This is the place to implement your customized EVICTION policy (decoding phase).

    Get the logical index of the least important token.
    """
    evict_idx = None
    if eviction_algorithm == "h2o":
        evict_idx = jnp.argmin(attn_score_cache[running_idx,
                                                cache_config.sink_size])
    return evict_idx


@functools.partial(jax.jit,
                   static_argnames=["eviction_algorithm", "cache_config"])
def get_replace_idx(
    attn_score_cache: jax.Array,
    eviction_algorithm: str,
    running_idx: int,
    cache_config: CacheConfig,
    position_in_cache: Optional[int] = None,
) -> int:
    """
    IMPORTANT: This is the place to implement your customized REPLACEMENT policy (decoding phase).

    Get the logical index of the replacement token, which is used later to
    replace the evicted token (in both the attention score cache and KV cache).
    """
    replace_idx = None
    if eviction_algorithm == "h2o":
        # Same as position_in_cache. The oldest token in the sliding window.
        replace_idx = position_in_cache
    return replace_idx


@functools.partial(jax.jit,
                   static_argnames=["eviction_algorithm", "cache_config"],
                   donate_argnums=0)
def evict_and_replace_attn_score_cache(
    attn_score_cache: jax.Array,
    eviction_algorithm: str,
    running_idx: int,
    evict_idx: int,
    replace_idx: int,
    cache_config: CacheConfig,
) -> jax.Array:
    """
    Evict from attn_score_cache the score of the least important token.
    Then replace it with the score of the replacement token.

    Returns the updated attn_score_cache.
    """
    if eviction_algorithm == "h2o":
        # Replace the evicted score, with the oldest score in the sliding window
        attn_score_cache = attn_score_cache.at[running_idx, evict_idx].set(
            attn_score_cache[running_idx, replace_idx])
        # Replace the oldest score in the sliding window, with the newest score (zero)
        attn_score_cache = attn_score_cache.at[running_idx, replace_idx].set(0)
    return attn_score_cache


@functools.partial(jax.jit, static_argnames=["eviction_algorithm"])
def get_replacement_token(
    cache: jax.Array,
    eviction_algorithm: str,
    replacement_write_indices: jax.Array,
) -> jax.Array:
    """
    Retrieve the replacement KV, using physical indices of the replacement tokens.
    """
    oldest_in_sliding_window = None
    if eviction_algorithm == "h2o":
        # In H2O, the evicted token is the token in HH cache with the lowest accumulated attention score.
        # The replacement token is the oldest token in Window cache.
        # Replacement token is moved to HH cache to replace the evicted token.
        num_heads, num_blocks_sliding_window, block_size, head_dim = cache.shape
        cache = cache.reshape(
            num_heads, -1, head_dim
        )  # (num_heads, num_blocks_sliding_window * block_size, head_dim)

        oldest_in_sliding_window = cache.at[:, replacement_write_indices, :].get(
        )  # (num_heads, batch_size, head_dim)
        oldest_in_sliding_window = jnp.transpose(
            oldest_in_sliding_window,
            (1, 0, 2))  # (batch_size, num_heads, head_dim)
        oldest_in_sliding_window = oldest_in_sliding_window[:, :,
                                                            None, :]  # (batch_size, num_heads, 1, head_dim)

        cache = cache.reshape(num_heads, num_blocks_sliding_window, block_size,
                              head_dim)
    return oldest_in_sliding_window


def convert_logical_idx_to_physical_idx(logical_idx: int,
                                        block_table: BlockTable,
                                        block_size: int) -> int:
    block_id = logical_idx // block_size
    block_offset = logical_idx % block_size
    return block_table[block_id] * block_size + block_offset


@functools.partial(
    jax.jit,
    static_argnames=[
        "eviction_algorithm",
        "padded_prompt_len",
        "sliding_window",
        "sink_size",
    ],
    donate_argnums=(0, 1),
)
def compress_caches_prefill(
    kv_caches: jax.Array,
    accumulated_attn_scores: jax.Array,
    eviction_algorithm: str,
    prefill_seq_len: int,
    padded_prompt_len: int,
    eviction_score_mask: jax.Array,
    sliding_window: int,
    sink_size: int,
    kv_cache_write_indices: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    IMPORTANT: This is the place to implement your customized EVICTION policy (prefill phase).

    (PREFILL ONLY) Evict the least important tokens from the KV cache and the attention score cache.

    Returns:
    - Compressed KV cache.
    - Compressed attention score cache.
    """
    if eviction_algorithm == "h2o":
        aggregate_window_size = sliding_window + sink_size
        assert accumulated_attn_scores.shape == eviction_score_mask.shape

        if padded_prompt_len > aggregate_window_size:
            # Prompt length exceeds cache limit. Compress the caches.
            num_layers = len(kv_caches)
            num_heads, num_blocks_kv_cache, block_size, head_dim = kv_caches[
                0][0].shape

            # Convert block indices to actual KV write indices.
            kv_cache_write_indices = kv_cache_write_indices.squeeze()
            unfold_write_indices = jnp.tile(
                jnp.arange(block_size),
                len(kv_cache_write_indices)) + jnp.repeat(
                    kv_cache_write_indices * block_size,
                    block_size)  # ( len(kv_cache_write_indices) * block_size,)

            # Find the top K tokens with the highest accumulated attention scores.
            # We don't include tokens in the Window cache for this ranking process (masked out).
            masked_scores = accumulated_attn_scores + eviction_score_mask.astype(
                accumulated_attn_scores.dtype)
            _, top_k_indices = jax.lax.top_k(masked_scores.squeeze(),
                                             k=sink_size)
            top_k_indices = jnp.sort(top_k_indices)  # (sink_size,)

            # Compress the attention score cache.
            start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
            hh_scores = accumulated_attn_scores[:,
                                                top_k_indices]  # (B, sink_size)
            window_scores = jax.lax.dynamic_slice_in_dim(
                accumulated_attn_scores, start_index, sliding_window,
                axis=-1)  # (B, sliding_window)
            accumulated_attn_scores = jnp.concatenate(
                [hh_scores, window_scores],
                axis=-1)  # (B, aggregate_window_size)

            # Get KV write indices for HH cache and Window cache.
            top_k_write_indices = unfold_write_indices.at[top_k_indices].get(
            )  # (sink_size,)
            window_write_indices = jax.lax.dynamic_slice_in_dim(
                unfold_write_indices, start_index, sliding_window,
                axis=0)  # (sliding_window,)

            # Compress the KV cache.
            for i in range(num_layers):
                k_cache, v_cache = kv_caches[i]
                k_cache = compress_kv_cache_prefill_h2o(
                    k_cache,
                    top_k_write_indices,
                    window_write_indices,
                    unfold_write_indices[:aggregate_window_size],
                )
                v_cache = compress_kv_cache_prefill_h2o(
                    v_cache,
                    top_k_write_indices,
                    window_write_indices,
                    unfold_write_indices[:aggregate_window_size],
                )
                kv_caches[i] = (k_cache, v_cache)

        else:
            # Prompt length is within the cache limit. No compression needed.
            accumulated_attn_scores *= eviction_score_mask.astype(
                accumulated_attn_scores.dtype
            )  # zeros out the scores of the padding tokens
            if padded_prompt_len < aggregate_window_size:
                # pad the scores with zeros.
                accumulated_attn_scores = jnp.pad(
                    accumulated_attn_scores,
                    pad_width=((0, 0),
                               (0, aggregate_window_size - padded_prompt_len)),
                    mode="constant",
                    constant_values=0,
                )

    return kv_caches, accumulated_attn_scores


@functools.partial(jax.jit, donate_argnums=0)
def compress_kv_cache_prefill_h2o(
    cache: jax.Array,
    top_k_write_indices: jax.Array,
    window_write_indices: jax.Array,
    aggregate_write_indices: jax.Array,
) -> jax.Array:
    num_heads, num_blocks_kv_cache, block_size, head_dim = cache.shape
    cache = cache.reshape(num_heads, -1, head_dim)

    hh_cache = cache[:,
                     top_k_write_indices, :]  # (num_heads, sink_size, head_dim)
    window_cache = cache[:,
                         window_write_indices, :]  # (num_heads, slding_window, head_dim)

    compressed_cache = jnp.concatenate(
        [hh_cache, window_cache],
        axis=1)  # (num_heads, sink_size + slding_window, head_dim)
    cache = cache.at[:, aggregate_write_indices, :].set(compressed_cache)
    return cache.reshape(num_heads, num_blocks_kv_cache, block_size, head_dim)


def update_eviction_score_mask(
    eviction_score_mask: np.ndarray,
    eviction_algorithm: str,
    mask_value: float,
    **kwargs,
):
    if eviction_algorithm == "h2o":
        prompt_len = kwargs.get("prompt_len", None)
        sliding_window = kwargs.get("sliding_window", None)
        seq_idx = kwargs.get("seq_idx", None)
        assert (
            prompt_len is not None and sliding_window is not None
            and seq_idx is not None
        ), "prompt_len, sliding_window, and seq_idx must be provided for H2O eviction score mask."
        eviction_score_mask[seq_idx, prompt_len - sliding_window:] = mask_value
        return eviction_score_mask
