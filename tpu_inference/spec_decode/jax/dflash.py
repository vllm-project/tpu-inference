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
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import get_model
from tpu_inference.utils import device_array

logger = init_logger(__name__)

DRAFT_EMBED_PATHS = [
    "model.embed_tokens.weight", "model.embed.embedding",
    "model.embed_tokens.embedding", "lm_head.embedding", "lm_head.weight"
]

TARGET_EMBED_PATHS = [
    "model.embed_tokens.weight", "model.embed_tokens.embedding",
    "model.embed.embedding", "embed_tokens.embedding", "embed.embedding"
]

DRAFT_LM_HEAD_PATHS = [
    "lm_head.weight", "lm_head.kernel", "model.embed_tokens.weight",
    "model.embed.embedding", "model.embed_tokens.embedding",
    "lm_head.embedding"
]

TARGET_LM_HEAD_PATHS = [
    "model.lm_head", "lm_head", "lm_head.weight", "lm_head.kernel",
    "model.embed_tokens.weight", "model.embed.embedding",
    "model.embed_tokens.embedding", "embed.embedding", "embed_tokens.embedding"
]


def _find_param(state: Any, paths: list[str]) -> Optional[Any]:
    from tpu_inference.models.jax.utils.weight_utils import get_param
    for path in paths:
        try:
            return get_param(state, path)
        except ValueError:
            continue
    return None


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
        if self.mesh.shape.get(ShardingAxisName.ATTN_DATA, 1) > 1:
            raise NotImplementedError(
                "DFlash currently does not support Data Parallelism (DP) attention "
                "(ATTN_DATA > 1).")
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
        from tpu_inference import envs
        from tpu_inference.models.common.model_loader import \
            resolve_model_architecture

        draft_mi = get_model(self.vllm_config,
                             self.rng_key,
                             self.mesh,
                             is_draft_model=True)
        self.model_fn = draft_mi.model_fn
        self.compute_logits_fn = draft_mi.compute_logits_fn
        self.combine_hidden_states_fn = draft_mi.combine_hidden_states_fn
        self.state = draft_mi.state

        draft_model_impl = envs.DRAFT_MODEL_IMPL_TYPE
        target_model_impl = envs.MODEL_IMPL_TYPE
        if draft_model_impl == 'auto':
            draft_model_impl = resolve_model_architecture(
                self.vllm_config, True)
        if target_model_impl == 'auto':
            target_model_impl = resolve_model_architecture(
                self.vllm_config, False)
        if draft_model_impl != target_model_impl:
            raise ValueError(
                "Draft model implementation must match target model.")

        if draft_model_impl == "flax_nnx":

            def get_target_value(target, paths):
                # 1. Check PyTorch/vLLM backend
                if (hasattr(self.runner, "model")
                        and hasattr(self.runner.model, "model")
                        and hasattr(self.runner.model.model, "vllm_model")):
                    vllm_model = self.runner.model.model.vllm_model
                    for name, param in vllm_model.named_parameters():
                        full_name = f"vllm_model.{name}"
                        for path in paths:
                            if path == full_name or full_name.endswith(
                                    path) or name == path or name.endswith(
                                        path):
                                from torchax.interop import jax_view
                                return jax_view(param.data)

                # 2. Check flat dict state lookup
                if isinstance(target,
                              dict) and not hasattr(target, "flat_state"):
                    for path in paths:
                        if path in target:
                            val = target[path]
                            return val.value if hasattr(val, "value") else val

                # 3. Check JAX nnx.State
                param = _find_param(target, paths)
                return param.value if param is not None else None

            # Resolve draft and target embeddings
            draft_embed_param = _find_param(self.state, DRAFT_EMBED_PATHS)
            target_embed_value = get_target_value(target_model,
                                                  TARGET_EMBED_PATHS)

            if draft_embed_param is not None and target_embed_value is not None:
                try:
                    if not jnp.any(draft_embed_param.value):
                        logger.info(
                            "Setting draft model embedding to target embedding."
                        )
                        draft_embed_param.value = target_embed_value
                    elif jnp.array_equal(draft_embed_param.value,
                                         target_embed_value):
                        logger.info("Draft and target models share embedding.")
                    else:
                        logger.warning(
                            "Draft and target models DO NOT share embedding.")
                except Exception:
                    logger.info(
                        "Directly setting draft model embedding to target embedding."
                    )
                    draft_embed_param.value = target_embed_value
            else:
                logger.warning(
                    "Failed to locate draft or target embedding parameter.")

            # Resolve draft and target lm_head
            draft_lm_head_param = _find_param(self.state, DRAFT_LM_HEAD_PATHS)
            target_lm_head_value = get_target_value(target_model,
                                                    TARGET_LM_HEAD_PATHS)

            if draft_lm_head_param is not None and target_lm_head_value is not None:
                try:
                    if draft_lm_head_param.value.shape == target_lm_head_value.shape:
                        logger.info(
                            "Sharing target model lm_head via .value assignment."
                        )
                        draft_lm_head_param.value = target_lm_head_value
                    elif draft_lm_head_param.value.shape == target_lm_head_value.T.shape:
                        logger.info(
                            "Sharing target model lm_head via transposed .value assignment."
                        )
                        draft_lm_head_param.value = target_lm_head_value.T
                    else:
                        logger.warning(
                            f"Shape mismatch: draft {draft_lm_head_param.value.shape} vs target {target_lm_head_value.shape}."
                        )
                except Exception:
                    logger.info(
                        "Directly setting draft model lm_head to target lm_head."
                    )
                    draft_lm_head_param.value = target_lm_head_value
            else:
                logger.warning(
                    "Failed to locate draft or target lm_head parameter.")

        if isinstance(self.state, nnx.State):
            self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))
        else:
            self.state_leaves = self.state
        draft_mi = replace(draft_mi, state_leaves=self.state_leaves)
        self.draft_mi = draft_mi

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _project_aux_hidden(self, state_leaves,
                            aux_hidden_states: list[jax.Array]) -> jax.Array:
        """Project auxiliary hidden states (JIT-compiled within prepare_inputs)."""
        concat_hidden = jnp.concatenate(aux_hidden_states, axis=-1)
        return self.combine_hidden_states_fn(state_leaves, concat_hidden)

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

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _compute_token_indices_and_filter(
        self,
        is_in_prefill: jax.Array,
        next_prompt_token_id: jax.Array,
        last_sampled_token_id: jax.Array,
        query_start_loc: jax.Array,
        seq_lens: jax.Array,
        num_reqs: jax.Array,
        num_rejected_tokens: jax.Array,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        input_positions: jax.Array,
    ):
        """Computes query metadata after filtering out rejected draft tokens.

        During speculative verification, some draft tokens might get rejected by the target model.
        This function:
        1. Compares sequence metadata and rolls back seq_lens based on rejected tokens.
        2. Derives 1D flat indices (`token_indices`) corresponding ONLY to the accepted tokens.
        3. Slices the target input IDs, position IDs, and target auxiliary hidden states
           using these token indices to filter out states/tokens belonging to rejected draft positions.
        """

        def _compute_token_indices(is_in_prefill, next_prompt_token_id,
                                   last_sampled_token_id, query_start_loc,
                                   seq_lens, num_reqs, num_rejected_tokens,
                                   input_ids):
            # 1. Determine next token to start generation (prompt token if in prefill, last accepted token if decoding)
            next_token_ids = jnp.where(is_in_prefill, next_prompt_token_id,
                                       last_sampled_token_id)

            # 2. Compute original query length and mask padded requests
            query_len_per_req = (query_start_loc[1:] - query_start_loc[:-1])
            query_len_per_req = jnp.where(
                jnp.arange(query_len_per_req.shape[0]) < num_reqs,
                query_len_per_req, 0)
            num_rejected_tokens = jnp.where(
                jnp.arange(num_rejected_tokens.shape[0]) < num_reqs,
                num_rejected_tokens, 0)

            # 3. Calculate number of accepted tokens per request and derive new query boundaries
            # Example: query_len_per_req = [5, 5], num_rejected_tokens = [1, 2] -> num_tokens_per_req = [4, 3]
            num_tokens_per_req = (query_len_per_req - num_rejected_tokens)
            new_query_start_loc = jnp.cumsum(num_tokens_per_req)
            new_query_start_loc = jnp.pad(
                new_query_start_loc, (1, 0),
                constant_values=0)  # Example: -> [0, 4, 7]

            # 4. Map the new (filtered) token positions back to their original index locations in input_ids.
            # This is a JAX-friendly vectorized mapping trick to avoid recompilations from dynamic shapes.
            total_num_tokens = input_ids.shape[0]

            # Repeat the new starting locations for each token in the accepted sequence.
            # Example: repeat [0, 4] by [4, 3] -> [0, 0, 0, 0, 4, 4, 4, 0, 0, 0] (padded to total_num_tokens)
            expanded_new_query_start_loc = jnp.repeat(
                new_query_start_loc[:-1],
                num_tokens_per_req,
                total_repeat_length=total_num_tokens)

            # Calculate offset/index within each new filtered segment.
            # Example: [0..9] - [0,0,0,0,4,4,4,...] -> [0, 1, 2, 3, 0, 1, 2, ...]
            token_offsets = jnp.arange(total_num_tokens, dtype=jnp.int32)
            token_offsets -= expanded_new_query_start_loc

            # Repeat the old query start locations.
            # Example: repeat [0, 5] by [4, 3] -> [0, 0, 0, 0, 5, 5, 5, 0, 0, 0]
            old_query_start_loc_expanded = jnp.repeat(
                query_start_loc[:-1],
                num_tokens_per_req,
                total_repeat_length=total_num_tokens)

            # Add relative offset to original start index to get absolute token index in the original sequence.
            # Example: [0,1,2,3,0,1,2,...] + [0,0,0,0,5,5,5,...] -> [0, 1, 2, 3, 5, 6, 7, ...]
            token_indices = token_offsets + old_query_start_loc_expanded

            # 5. Rollback sequence lengths based on the number of rejected tokens
            new_seq_lens = seq_lens - num_rejected_tokens

            return (next_token_ids, token_indices, new_query_start_loc,
                    new_seq_lens)

        data_spec = PartitionSpec(ShardingAxisName.ATTN_DATA)
        (next_token_ids, token_indices, new_query_start_loc,
         new_seq_lens) = jax.shard_map(
             _compute_token_indices,
             mesh=self.mesh,
             in_specs=(data_spec, ) * 8,
             out_specs=(data_spec, ) * 4,
         )(is_in_prefill, next_prompt_token_id, last_sampled_token_id,
           query_start_loc, seq_lens, num_reqs, num_rejected_tokens, input_ids)

        def _sharded_select_target_tokens_and_hidden_states(
                input_ids, token_indices, input_positions, aux_hidden_states):
            # Extract only the token IDs of accepted tokens
            target_token_ids = input_ids[token_indices]

            # Slice position IDs for accepted tokens (supporting both 2D and 1D position layouts)
            if input_positions.ndim == 2:
                input_positions = input_positions[:, token_indices]
            else:
                input_positions = input_positions[token_indices]

            # Filter target auxiliary hidden states from target layers, keeping only accepted token steps
            aux_states_processed = [
                h[token_indices] for h in aux_hidden_states
            ]
            return target_token_ids, input_positions, aux_states_processed

        positions_spec = (PartitionSpec(None, ShardingAxisName.ATTN_DATA)
                          if input_positions.ndim == 2 else data_spec)

        target_token_ids, new_input_positions, aux_states_processed = jax.shard_map(
            _sharded_select_target_tokens_and_hidden_states,
            mesh=self.mesh,
            in_specs=(data_spec, data_spec, positions_spec, data_spec),
            out_specs=(data_spec, positions_spec, data_spec),
        )(input_ids, token_indices, input_positions, aux_hidden_states)

        return (next_token_ids, new_query_start_loc, new_seq_lens,
                target_token_ids, new_input_positions, aux_states_processed)

    def prepare_inputs(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """Prepare DFlash inputs with on-device KV cache."""
        assert aux_hidden_states is not None and len(aux_hidden_states) > 0

        # The last KV cache group is for the draft model.
        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = (
            self.runner.input_batch.block_table[draft_kv_cache_group_id].
            get_cpu_tensor().reshape(-1))
        block_tables = device_array(self.mesh,
                                    block_tables,
                                    sharding=PartitionSpec(
                                        ShardingAxisName.ATTN_DATA))

        return self._prepare_inputs(
            self.state_leaves,
            block_tables,
            attn_metadata,
            input_ids,
            aux_hidden_states,
            last_sampled_token_id,
            next_prompt_token_id,
            is_in_prefill,
            num_rejected_tokens,
            num_reqs_dp,
        )

    @functools.partial(
        jax.jit,
        static_argnums=(0, ),
    )
    def _prepare_inputs(
        self,
        state_leaves: Any,
        block_tables: jax.Array,
        attn_metadata: AttentionMetadata,
        input_ids: jax.Array,
        aux_hidden_states: tuple[jax.Array, ...],
        last_sampled_token_id: jax.Array,
        next_prompt_token_id: jax.Array,
        is_in_prefill: jax.Array,
        num_rejected_tokens: jax.Array,
        num_reqs_dp: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, AttentionMetadata]:
        """JIT-compiled core logic for preparing inputs for the draft model.

        Performs:
        1. Filters out rejected token metadata via `_compute_token_indices_and_filter`.
        2. Projects the accepted target hidden states into draft model input space.
        3. Prepares a masked 'noise block' of token IDs and position IDs for generating 
           speculative proposals.
        4. Updates AttentionMetadata for the draft model forward pass.
        """
        (next_token_ids, new_query_start_loc, new_seq_lens, target_token_ids,
         new_input_positions,
         aux_states_processed) = self._compute_token_indices_and_filter(
             is_in_prefill, next_prompt_token_id, last_sampled_token_id,
             attn_metadata.query_start_loc, attn_metadata.seq_lens,
             num_reqs_dp, num_rejected_tokens, input_ids, aux_hidden_states,
             attn_metadata.input_positions)

        # Project new auxiliary hidden states
        projected = self._project_aux_hidden(state_leaves,
                                             aux_states_processed)
        target_hidden = projected.astype(jnp.bfloat16)

        # Build noise block (vectorized for batching)
        noise_input_ids, noise_positions = self._build_noise_block(
            new_seq_lens,
            next_token_ids,
            self.mask_token_id,
            self.block_size,
        )

        num_reqs = attn_metadata.seq_lens.shape[0]
        draft_attn_metadata = replace(
            attn_metadata,
            input_positions=noise_positions,
            seq_lens=new_seq_lens,
            query_start_loc=jnp.arange(num_reqs + 1, dtype=jnp.int32) *
            self.block_size,
            block_tables=block_tables,
        )

        dummy_last_indices = jnp.zeros(num_reqs, dtype=jnp.int32)
        return (
            (target_hidden, new_query_start_loc, new_input_positions),
            noise_input_ids,
            dummy_last_indices,
            draft_attn_metadata,
        )

    @functools.partial(
        jax.jit,
        static_argnums=(0, ),
        donate_argnames=("kv_caches", ),
        out_shardings=(
            None,  # kv_caches - keep original sharding
            None,  # draft_token_ids
        ),
        compiler_options={
            "xla_tpu_all_gather_collective_matmul_mode":
            "post_spmd_conservative",
            "xla_tpu_reduce_scatter_collective_matmul_mode":
            "post_spmd_conservative"
        },
    )
    def _propose(
        self,
        state_leaves: Any,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        target_hidden_states: jax.Array,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """JIT-compiled forward pass of the draft model.

        1. Runs the draft model forward pass using the target model's projected hidden states.
        2. Reshapes the output hidden states to locate the positions corresponding to the
           new speculative tokens.
        3. Computes vocabulary logits and greedily samples the token IDs.
        4. Constrains the sharding of the drafted token IDs back to ATTN_DATA.
        """
        kv_caches, hidden_states, *_ = self.model_fn(
            state_leaves,
            kv_caches,
            input_ids,
            target_hidden_states,
            attn_metadata,
        )

        num_reqs = hidden_states.shape[0] // self.block_size
        hidden_states_batched = hidden_states.reshape(
            (num_reqs, self.block_size, -1))
        draft_hidden = hidden_states_batched[:, 1:1 +
                                             self.num_speculative_tokens, :]

        # Flatten draft_hidden to 2D for compute_logits_fn which expects 2D
        # shape to satisfy PartitionSpec('data', 'model') for the output logits
        draft_hidden_flat = draft_hidden.reshape(-1, draft_hidden.shape[-1])
        logits_flat = self.compute_logits_fn(state_leaves, draft_hidden_flat,
                                             None)

        logits = logits_flat.reshape(num_reqs, self.num_speculative_tokens, -1)

        draft_ids = jnp.argmax(logits, axis=-1)
        draft_token_ids = lax.with_sharding_constraint(
            draft_ids,
            NamedSharding(self.mesh,
                          PartitionSpec(ShardingAxisName.ATTN_DATA, None)))

        if draft_token_ids.ndim == 1:
            draft_token_ids = draft_token_ids[jnp.newaxis, :]

        return kv_caches, draft_token_ids

    def propose(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attn_metadata: AttentionMetadata,
        last_token_indices: jax.Array,
        target_hidden_states,
    ) -> tuple[list[jax.Array], jnp.ndarray]:
        """Generate all draft tokens in one forward pass using paged KV cache."""
        return self._propose(
            self.state_leaves,
            kv_caches,
            input_ids,
            attn_metadata,
            target_hidden_states,
        )
