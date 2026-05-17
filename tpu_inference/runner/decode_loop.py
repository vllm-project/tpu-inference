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

import functools
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from tpu_inference.layers.common.attention_metadata import AttentionMetadata


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "current_tokens",
        "active_mask",
        "attn_metadata",
        "step_counter",
    ],
    meta_fields=[],
)
@dataclass
class TpuSamplingState:
    # (batch_size,)
    current_tokens: jax.Array
    # (batch_size,)
    active_mask: jax.Array
    # AttentionMetadata PyTree
    attn_metadata: AttentionMetadata
    # scalar
    step_counter: jax.Array


@functools.partial(
    jax.jit,
    static_argnames=["eos_token_id", "padding_token_id", "dp_size", "pad_len"])
def _update_loop_state(
    next_tokens: jax.Array,
    active_mask: jax.Array,
    input_positions: jax.Array,
    seq_lens: jax.Array,
    eos_token_id: tuple[int, ...],
    padding_token_id: int,
    dp_size: int,
    pad_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    eos_arr = jnp.atleast_1d(jnp.array(eos_token_id, dtype=jnp.int32))
    is_eos = jnp.any(next_tokens[:, None] == eos_arr[None, :], axis=-1)
    new_active_mask = jnp.logical_and(active_mask, jnp.logical_not(is_eos))
    next_input_ids = jnp.where(new_active_mask, next_tokens, padding_token_id)
    increment = new_active_mask.astype(jnp.int32)
    new_positions = input_positions + increment

    # Fix DP padding bug by exposing DP dimension as 2D before padding
    increment_2d = increment.reshape(dp_size, -1)
    seq_lens_2d = seq_lens.reshape(dp_size, -1)
    if pad_len > 0:
        padded_increment_2d = jnp.pad(increment_2d, ((0, 0), (0, pad_len)))
    else:
        padded_increment_2d = increment_2d[:, :seq_lens_2d.shape[1]]
    new_seq_lens = (seq_lens_2d + padded_increment_2d).ravel()

    # Compute the token to record in generated_tokens
    step_record_tokens = jnp.where(active_mask, next_tokens, padding_token_id)

    any_hit_eos = jnp.any(jnp.logical_and(active_mask, is_eos))

    return new_active_mask, next_input_ids, new_positions, new_seq_lens, step_record_tokens, any_hit_eos


@functools.partial(jax.jit, static_argnums=(1, 2))
def _split_rngs(rng, static_size, dynamic_size):
    all_rngs = jax.random.split(rng, static_size + 1)
    return all_rngs[:dynamic_size], all_rngs[dynamic_size]


def continue_decode(
    state: dict,
    model_fn: Callable,
    compute_logits_fn: Callable,
    sample_fn: Callable,
    init_state: TpuSamplingState,
    kv_caches: Any,
    max_decode_steps: int,
    static_max_decode_steps: int,
    eos_token_id: tuple[int, ...],
    padding_token_id: int,
    rng: jax.Array,
    terminate_on_any_eos: bool = False,
    inputs_embeds: jax.Array | None = None,
    layer_name_to_kvcache_index: tuple[tuple[str, int], ...] = (),
    lora_metadata: dict | None = None,
    intermediate_tensors: dict[str, jax.Array] | None = None,
    is_first_rank: bool = True,
    is_last_rank: bool = True,
    dp_size: int = 1,
) -> tuple[list[jax.Array], Any, TpuSamplingState, jax.Array, list[jax.Array]
           | None]:
    """Helper function to run the decode loop on TPU.

    Args:
      state: Model state dict.
      model_fn: Function to run the model forward pass.
      compute_logits_fn: Function to compute logits from hidden states.
      sample_fn: Function to sample next tokens.
      init_state: Initial TpuSamplingState.
      kv_caches: KV caches.
      max_decode_steps: Maximum number of steps to run the decode loop.
      static_max_decode_steps: Static maximum number of steps to split RNG.
      eos_token_id: EOS token ID.
      padding_token_id: Padding token ID.
      rng: RNG key.
      terminate_on_any_eos: Whether to terminate the loop early if any request
        hits EOS.
      inputs_embeds: Optional input embeddings.
      layer_name_to_kvcache_index: Mapping from layer name to KV cache index.
      lora_metadata: Optional LoRA metadata.
      intermediate_tensors: Optional intermediate tensors.
      is_first_rank: Whether this is the first PP rank.
      is_last_rank: Whether this is the last PP rank.
      dp_size: Data parallel size.

    Returns:
      Tuple of (generated_tokens, final_kv_caches, final_state, final_rng, all_expert_indices).
    """

    batch_size = init_state.current_tokens.shape[0]
    seq_lens_size = init_state.attn_metadata.seq_lens.shape[0]
    pad_len = (seq_lens_size - batch_size) // dp_size

    current_tokens = init_state.current_tokens
    active_mask = init_state.active_mask
    attn_metadata = init_state.attn_metadata
    step_rngs, current_rng = _split_rngs(rng, static_max_decode_steps,
                                         max_decode_steps)

    token_list = []
    expert_indices_list = []

    for step_idx in range(max_decode_steps):
        step_rng = step_rngs[step_idx]

        # 1. Forward pass
        kv_caches, hidden_states, _, expert_indices_step = model_fn(
            state,
            kv_caches,
            current_tokens,
            attn_metadata,
            inputs_embeds,
            attn_metadata.input_positions,
            layer_name_to_kvcache_index,
            lora_metadata,
            intermediate_tensors,
            is_first_rank,
            is_last_rank,
        )

        # Record expert indices if returned
        if expert_indices_step is not None:
            expert_indices_list.append(expert_indices_step)

        # 2. Compute logits and sample
        logits = compute_logits_fn(state, hidden_states, None)
        logits = logits.astype(jnp.float32)
        next_tokens, _ = sample_fn(step_rng, logits)

        # 3. Update loop state via fused JIT helper
        new_active_mask, next_input_ids, new_positions, new_seq_lens, step_record_tokens, any_hit_eos = _update_loop_state(
            next_tokens,
            active_mask,
            attn_metadata.input_positions,
            attn_metadata.seq_lens,
            eos_token_id,
            padding_token_id,
            dp_size,
            pad_len,
        )

        new_attn_metadata = AttentionMetadata(
            input_positions=new_positions,
            block_tables=attn_metadata.block_tables,
            seq_lens=new_seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            request_distribution=attn_metadata.request_distribution,
            mamba_state_indices=attn_metadata.mamba_state_indices,
        )

        # 4. Record generated tokens
        token_list.append(step_record_tokens)

        # Update loop variables
        current_tokens = next_input_ids
        active_mask = new_active_mask
        attn_metadata = new_attn_metadata

        if terminate_on_any_eos and bool(any_hit_eos):
            actual_steps = step_idx + 1
            break
    else:
        actual_steps = max_decode_steps

    generated_tokens = token_list
    all_expert_indices = expert_indices_list if expert_indices_list else None

    final_state = TpuSamplingState(
        current_tokens=current_tokens,
        active_mask=active_mask,
        attn_metadata=attn_metadata,
        step_counter=jnp.array(actual_steps, dtype=jnp.int32),
    )

    return generated_tokens, kv_caches, final_state, current_rng, all_expert_indices
