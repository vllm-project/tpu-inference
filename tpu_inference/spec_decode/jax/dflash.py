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

        # On-device KV caches (allocated in load_model)
        self._draft_kv_caches: Optional[list[jax.Array]] = None
        self._max_kv_len: int = 0

    def load_model(self, target_model: Any) -> None:
        """Load the DFlash draft model and share embeddings from target."""
        draft_mi = get_model(self.vllm_config,
                             self.rng_key,
                             self.mesh,
                             is_draft_model=True)
        self.model_fn = draft_mi.model_fn
        self.compute_logits_fn = draft_mi.compute_logits_fn
        self.combine_hidden_states_fn = draft_mi.combine_hidden_states_fn
        self.state = draft_mi.state

        # Share the target model's embedding with the draft model.
        # Only applicable to flax_nnx state (has .model); vllm-impl state is
        # a params dict, in which case the draft keeps its own embedding.
        if hasattr(self.state, "model") and hasattr(target_model, "model"):
            draft_embed = getattr(self.state.model, "embed_tokens", None)
            target_embed = getattr(target_model.model, "embed_tokens", None)
            if target_embed is None:
                target_embed = getattr(target_model.model, "embed", None)
            if target_embed is not None:
                if draft_embed is None or not jnp.any(draft_embed.embedding):
                    logger.info(
                        "Sharing target model embedding with DFlash draft model."
                    )
                    self.state.model.embed_tokens = target_embed
                elif jnp.array_equal(draft_embed.embedding,
                                     target_embed.embedding):
                    logger.info(
                        "Draft embedding identical to target; sharing.")
                    self.state.model.embed_tokens = target_embed

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _project_aux_hidden(
            self, state: nnx.State,
            aux_hidden_states: tuple[jax.Array, ...]) -> jax.Array:
        """Project and normalise auxiliary hidden states."""
        raw = jnp.concatenate(aux_hidden_states, axis=-1)
        return self.combine_hidden_states_fn(state, raw)

    @staticmethod
    def _next_padded_size(n: int) -> int:
        """Round n up to the next power-of-two (min 16)."""
        if n <= 16:
            return 16
        p = 16
        while p < n:
            p *= 2
        return p

    @functools.partial(jax.jit, static_argnums=(0, 3, 4))
    def _build_noise_block(
        self,
        seq_lens: jax.Array,
        next_token_ids: jax.Array,
        mask_token_id: int,
        block_size: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Build noise block and positions for a batch (JIT-compiled)."""
        num_reqs = next_token_ids.shape[0]
        noise_input_ids = jnp.full((num_reqs, block_size),
                                   mask_token_id,
                                   dtype=jnp.int32)
        noise_input_ids = noise_input_ids.at[:, 0].set(next_token_ids)
        offsets = jnp.arange(block_size, dtype=jnp.int32)[jnp.newaxis, :]
        noise_positions = seq_lens[:, jnp.newaxis] + offsets
        return noise_input_ids.flatten(), noise_positions.flatten()

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

        # Project new auxiliary hidden states (on-device, JIT'd)
        # Target hidden states are no longer sliced manually because they contain
        # only the newly accepted tokens computed in this forward pass.
        projected = self._project_aux_hidden(self.state, aux_hidden_states)
        target_hidden = projected.astype(jnp.bfloat16)

        # Build noise block (vectorized for batching)
        noise_input_ids, noise_positions = self._build_noise_block(
            attn_metadata.seq_lens,
            next_token_ids,
            self.mask_token_id,
            self.block_size,
        )

        # Build draft attention metadata
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id].
            get_cpu_tensor().reshape(-1))
        num_reqs = attn_metadata.seq_lens.shape[0]
        draft_attn_metadata = replace(
            attn_metadata,
            input_positions=noise_positions,
            query_start_loc=jnp.arange(num_reqs + 1, dtype=jnp.int32) *
            self.block_size,
            block_tables=device_array(self.mesh, block_tables),
        )

        dummy_last_indices = jnp.zeros(num_reqs, dtype=jnp.int32)
        return (
            (target_hidden, attn_metadata.query_start_loc,
             attn_metadata.input_positions),
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
        num_reqs = hidden_states.shape[0] // self.block_size
        hidden_states_batched = hidden_states.reshape(
            (num_reqs, self.block_size, -1))
        draft_hidden = hidden_states_batched[:, 1:1 +
                                             self.num_speculative_tokens, :]
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
        """Generate all draft tokens in one forward pass using paged KV cache."""
        kv_caches, hidden_states, _ = self.model_fn(
            self.state,
            kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
        )

        draft_token_ids = self._sample_block_draft_tokens(
            self.state, hidden_states)

        if draft_token_ids.ndim == 1:
            draft_token_ids = draft_token_ids[jnp.newaxis, :]

        return kv_caches, draft_token_ids
