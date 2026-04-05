# Copyright 2025 Google LLC
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
"""DFlash proposer for speculative decoding on JAX/TPU."""

import functools
from dataclasses import replace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.utils import device_array

logger = init_logger(__name__)


class DFlashProposer:
    """Proposer for speculative decoding using DFlash block diffusion."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        runner: Any,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.mesh = runner.mesh
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens)

        hf_config = self.draft_model_config.hf_config
        self.block_size = getattr(hf_config, "block_size",
                                  self.num_speculative_tokens + 1)
        dflash_config = getattr(hf_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get("mask_token_id", 0)
        self.hidden_size = hf_config.hidden_size
        self.num_layers = hf_config.num_hidden_layers

        self.rng_key = jax.random.key(self.vllm_config.model_config.seed)
        self.max_num_tokens = runner.max_num_tokens
        self.max_model_len = runner.max_model_len

        # Host-side context projection buffer
        self._ctx_buf: Optional[np.ndarray] = None
        self._ctx_len: int = 0

        # On-device KV caches (allocated in load_model)
        self._draft_kv_caches: Optional[list[jax.Array]] = None
        self._cache_len: int = 0
        self._max_kv_len: int = 0

        # Track previous seq_len for GPU-compatible crop semantics.
        # GPU calls past_key_values_draft.crop(start) AFTER each forward
        # pass, where start = beginning of the CURRENT iteration's block.
        # This equals the seq_len from the PREVIOUS call to prepare_inputs.
        # We must match this: cache_len = prev_seq_len, not current seq_len.
        self._prev_seq_len: int = 0

    def load_model(self, target_model: Any) -> None:
        """Load the DFlash draft model and share embeddings from target."""
        (
            self.model_fn,
            self.compute_logits_fn,
            self.combine_hidden_states_fn,
            _,
            self.state,
            _,
            _,
        ) = get_model(self.vllm_config,
                      self.rng_key,
                      self.mesh,
                      is_draft_model=True)

        # Share the target model's embedding with the draft model.
        draft_embed = getattr(self.state.model, "embed_tokens", None)
        target_embed = getattr(target_model.model, "embed_tokens", None)
        if target_embed is None:
            target_embed = getattr(target_model.model, "embed", None)
        if target_embed is not None:
            if draft_embed is None or not jnp.any(draft_embed.embedding):
                logger.info(
                    "Sharing target model embedding with DFlash draft model.")
                self.state.model.embed_tokens = target_embed
            elif jnp.array_equal(draft_embed.embedding,
                                 target_embed.embedding):
                logger.info("Draft embedding identical to target; sharing.")
                self.state.model.embed_tokens = target_embed

        # Allocate on-device KV caches
        hf_config = self.draft_model_config.hf_config
        from tpu_inference import utils
        from tpu_inference.layers.common.sharding import ShardingAxisName
        from tpu_inference.utils import get_mesh_shape_product

        sharding_size = get_mesh_shape_product(self.mesh,
                                               ShardingAxisName.MLP_TENSOR)
        num_heads = utils.get_padded_num_heads(hf_config.num_attention_heads,
                                               sharding_size)
        head_dim_orig = getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads)
        head_dim = utils.get_padded_head_dim(head_dim_orig)

        self._max_kv_len = self._next_padded_size(self.max_model_len)
        cache_shape = (1, num_heads, self._max_kv_len, head_dim)
        self._draft_kv_caches = []
        for _ in range(self.num_layers):
            k_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
            v_cache = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
            self._draft_kv_caches.append(k_cache)
            self._draft_kv_caches.append(v_cache)
        self._cache_len = 0

        logger.info(
            "Allocated DFlash on-device KV caches: %d layers, shape %s",
            self.num_layers,
            cache_shape,
        )

    def _project_aux_hidden(
            self, aux_hidden_states: tuple[jax.Array, ...]) -> jax.Array:
        """Project and normalise auxiliary hidden states."""
        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        return self.combine_hidden_states_fn(self.state, raw)

    def _update_context_buffer(
        self,
        projected: jax.Array,
        seq_len: int,
    ) -> Optional[np.ndarray]:
        """Append newly-accepted projected hidden states to the buffer.

        Returns the NEW context tokens as a numpy array, or None on rejection.
        """
        proj_np = np.asarray(projected)

        if self._ctx_buf is None:
            self._ctx_buf = np.zeros(
                (self.max_model_len, proj_np.shape[-1]),
                dtype=proj_np.dtype,
            )

        num_new = seq_len - self._ctx_len
        if num_new <= 0:
            self._ctx_len = seq_len
            self._cache_len = min(self._cache_len, seq_len)
            return None

        end = min(self._ctx_len + num_new, self.max_model_len)
        n_copy = end - self._ctx_len
        self._ctx_buf[self._ctx_len:end] = proj_np[:n_copy]
        new_ctx = proj_np[:n_copy].copy()
        self._ctx_len = end
        return new_ctx

    @staticmethod
    def _next_padded_size(n: int) -> int:
        """Round n up to the next power-of-two (min 16)."""
        if n <= 16:
            return 16
        p = 16
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _pad_context(ctx: np.ndarray) -> np.ndarray:
        """Pad context array to the next power-of-2 size (min 16).

        Args:
            ctx: (T, D) numpy array of context features.

        Returns:
            (T_padded, D) numpy array with zero-padding appended.
        """
        T = ctx.shape[0]
        T_padded = DFlashProposer._next_padded_size(T)
        if T_padded == T:
            return ctx
        pad = np.zeros((T_padded - T, ctx.shape[1]), dtype=ctx.dtype)
        return np.concatenate([ctx, pad], axis=0)

    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _build_noise_block(
        self,
        seq_len_arr: jax.Array,
        next_token_ids: jax.Array,
        mask_token_id: int,
        block_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Build noise block and positions (JIT-compiled)."""
        seq_len = seq_len_arr[0]
        first_token = next_token_ids[0]
        noise_input_ids = jnp.full((block_size, ),
                                   mask_token_id,
                                   dtype=jnp.int32)
        noise_input_ids = noise_input_ids.at[0].set(first_token)
        noise_positions = jnp.arange(block_size, dtype=jnp.int32) + seq_len
        return noise_input_ids, noise_positions

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        next_token_ids: jax.Array,
        num_rejected_tokens: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare DFlash inputs with on-device KV cache."""
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0

        # 1. Current sequence length
        seq_len_jax = attn_metadata.seq_lens[0]
        seq_len = int(jax.device_get(seq_len_jax))

        # 2. Crop cache to match GPU DynamicCache.crop(start) semantics.
        #
        # GPU reference (zhongyan_dev/dflash/model/dflash.py line 246):
        #   past_key_values_draft.crop(start)
        # where `start` = beginning of the CURRENT block = position of
        # the first accepted token from the previous iteration.
        #
        # After crop, GPU cache_seq_len = start, which equals the seq_len
        # from the PREVIOUS prepare_inputs call (not the current one).
        # Context + noise are then written starting from this position.
        #
        # Bug was: self._cache_len = seq_len (CURRENT accepted position),
        # which left stale noise K/V entries from the previous iteration
        # in positions [prev_seq_len, seq_len) and shifted all subsequent
        # RoPE positions, accumulating errors every iteration.
        if self._prev_seq_len > 0:
            self._cache_len = self._prev_seq_len

        if seq_len < self._ctx_len:
            self._ctx_len = seq_len
        self._prev_seq_len = seq_len

        # 3. Project new auxiliary hidden states
        projected = self._project_aux_hidden(aux_hidden_states)

        # 4. Update context buffer and get NEW tokens
        new_ctx_np = self._update_context_buffer(projected, seq_len)

        if new_ctx_np is None or len(new_ctx_np) == 0:
            # Full rejection — all padded entries are zeros, and noise
            # writes at cache_len + 0, completely overwriting them.
            actual_new_ctx_count = 0
            new_ctx_np = np.zeros((16, self.hidden_size), dtype=np.float32)
        else:
            actual_new_ctx_count = len(new_ctx_np)
            new_ctx_np = self._pad_context(new_ctx_np)

        # 5. Upload padded context to device.
        # Padding to power-of-2 sizes (16/32/64/128) means JIT only
        # traces ~4 unique shapes, eliminating per-token retracing.
        new_ctx_jax = device_array(
            self.mesh,
            jnp.array(new_ctx_np, dtype=jnp.bfloat16),
        )

        # 6. Build noise block
        seq_len_arr = device_array(self.mesh,
                                   np.array([seq_len], dtype=np.int32))
        noise_input_ids, noise_positions = self._build_noise_block(
            seq_len_arr,
            next_token_ids,
            self.mask_token_id,
            self.block_size,
        )

        # 7. Pack target_hidden_states as 3-tuple (always same pytree shape)
        cache_len_arr = device_array(
            self.mesh, np.array([self._cache_len], dtype=np.int32))
        actual_ctx_count_arr = device_array(
            self.mesh, np.array([actual_new_ctx_count], dtype=np.int32))
        target_hidden = (new_ctx_jax, cache_len_arr, actual_ctx_count_arr)

        # 8. Build draft attention metadata
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id].
            get_cpu_tensor().reshape(-1))
        num_reqs = attn_metadata.seq_lens.shape[0]
        draft_attn_metadata = replace(
            attn_metadata,
            input_positions=noise_positions,
            query_start_loc=jnp.array([0, self.block_size], dtype=jnp.int32),
            block_tables=device_array(self.mesh, block_tables),
        )

        dummy_last_indices = jnp.zeros(num_reqs, dtype=jnp.int32)
        return (
            target_hidden,
            noise_input_ids,
            dummy_last_indices,
            draft_attn_metadata,
        )

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _sample_block_draft_tokens(
        self,
        state: nnx.State,
        hidden_states: jax.Array,
    ) -> jax.Array:
        """Greedy-sample draft tokens from the block output."""
        draft_hidden = hidden_states[1:1 + self.num_speculative_tokens]
        logits = self.compute_logits_fn(state, draft_hidden, None)
        draft_ids = jnp.argmax(logits, axis=-1)
        return lax.with_sharding_constraint(
            draft_ids, NamedSharding(self.mesh, PartitionSpec()))

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Generate all draft tokens in one forward pass."""
        # Use our own on-device KV caches
        draft_kv_caches, hidden_states, _ = self.model_fn(
            self.state,
            self._draft_kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
        )

        # Update cached references
        self._draft_kv_caches = draft_kv_caches

        # Update cache_len: model wrote actual_ctx_count + T_noise entries.
        # This will be corrected at the start of the next prepare_inputs
        # to match the actual accepted seq_len.
        _, cache_len_arr, actual_ctx_count_arr = target_hidden_states
        old_cache_len = int(jax.device_get(cache_len_arr)[0])
        actual_ctx_count = int(jax.device_get(actual_ctx_count_arr)[0])
        T_noise = self.block_size
        self._cache_len = old_cache_len + actual_ctx_count + T_noise

        draft_token_ids = self._sample_block_draft_tokens(
            self.state, hidden_states)

        if draft_token_ids.ndim == 1:
            draft_token_ids = draft_token_ids[jnp.newaxis, :]

        # Pass the FRAMEWORK kv_caches through unchanged
        return kv_caches, draft_token_ids
