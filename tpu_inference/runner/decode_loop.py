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


def continue_decode(
    state: Any,
    model_fn: Callable,
    compute_logits_fn: Callable,
    sample_fn: Callable,
    init_state: TpuSamplingState,
    kv_caches: Any,
    max_decode_steps: int,
    eos_token_id: int,
    padding_token_id: int,
    rng: jax.Array,
    terminate_on_any_eos: bool = False,
    inputs_embeds: Any = None,
    layer_name_to_kvcache_index: Any = (),
    lora_metadata: Any = None,
    intermediate_tensors: Any = None,
    is_first_rank: bool = True,
    is_last_rank: bool = True,
    dp_size: int = 1,
) -> tuple[jax.Array, Any, TpuSamplingState, jax.Array]:
    """Continues decoding on TPU using a Python loop (no host sync during loop).

    Args:
        state: Model weights and state.
        model_fn: Function to run the model forward pass.
        compute_logits_fn: Function to compute logits from hidden states.
        sample_fn: Function to sample next tokens.
        init_state: Initial TpuSamplingState.
        kv_caches: Initial KV caches.
        max_decode_steps: Maximum steps to decode (Python int).
        eos_token_id: EOS token ID.
        padding_token_id: Padding token ID.
        rng: Initial PRNG key for sampling.
        terminate_on_any_eos: If True, stops as soon as any request hits EOS (handled on CPU).
                             If False, continues until all requests hit EOS.
        dp_size: Data parallel size (needed for correct metadata padding).

    Returns:
    Tuple of (generated_tokens, final_kv_caches, final_state, final_rng, all_expert_indices).
    """

    batch_size = init_state.current_tokens.shape[0]
    generated_tokens = jnp.full((max_decode_steps, batch_size),
                                padding_token_id,
                                dtype=jnp.int32)

    current_tokens = init_state.current_tokens
    active_mask = init_state.active_mask
    attn_metadata = init_state.attn_metadata
    current_rng = rng

    all_expert_indices = None

    for step_idx in range(max_decode_steps):
        # Split RNG for current step
        current_rng, step_rng = jax.random.split(current_rng)

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

        # Initialize and record expert indices if returned
        if expert_indices_step is not None:
            if step_idx == 0:
                num_layers, _, top_k = expert_indices_step.shape
                all_expert_indices = jnp.zeros(
                    (max_decode_steps, num_layers, batch_size, top_k),
                    dtype=expert_indices_step.dtype,
                )
            all_expert_indices = all_expert_indices.at[step_idx].set(
                expert_indices_step)

        # 2. Compute logits and sample
        logits = compute_logits_fn(state, hidden_states, None)
        logits = logits.astype(jnp.float32)
        next_tokens, _ = sample_fn(step_rng, logits)

        # 3. Check for EOS and update mask
        is_eos = next_tokens == eos_token_id
        new_active_mask = jnp.logical_and(active_mask, jnp.logical_not(is_eos))

        # 4. Update tokens for next step (pad finished ones)
        next_input_ids = jnp.where(new_active_mask, next_tokens,
                                   padding_token_id)

        # 5. Update AttentionMetadata for next step
        increment = new_active_mask.astype(jnp.int32)
        new_positions = attn_metadata.input_positions + increment

        # Fix DP padding bug by exposing DP dimension as 2D before padding
        increment_2d = increment.reshape(dp_size, -1)
        seq_lens_2d = attn_metadata.seq_lens.reshape(dp_size, -1)

        pad_len = seq_lens_2d.shape[1] - increment_2d.shape[1]
        padded_increment_2d = jnp.pad(increment_2d, ((0, 0), (0, pad_len)))

        new_seq_lens = (seq_lens_2d + padded_increment_2d).ravel()

        new_attn_metadata = AttentionMetadata(
            input_positions=new_positions,
            block_tables=attn_metadata.block_tables,
            seq_lens=new_seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            request_distribution=attn_metadata.request_distribution,
            mamba_state_indices=attn_metadata.mamba_state_indices,
        )

        # 6. Record generated tokens
        generated_tokens = generated_tokens.at[step_idx].set(
            jnp.where(active_mask, next_tokens, padding_token_id))

        # Update loop variables
        current_tokens = next_input_ids
        active_mask = new_active_mask
        attn_metadata = new_attn_metadata

    final_state = TpuSamplingState(
        current_tokens=current_tokens,
        active_mask=active_mask,
        attn_metadata=attn_metadata,
        step_counter=jnp.array(max_decode_steps, dtype=jnp.int32),
    )

    return generated_tokens, kv_caches, final_state, current_rng, all_expert_indices
