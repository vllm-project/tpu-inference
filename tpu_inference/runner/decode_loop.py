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
from typing import Any, Callable, NamedTuple

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


class DecodeLoopCarry(NamedTuple):
    step_idx: jax.Array
    current_tokens: jax.Array
    active_mask: jax.Array
    attn_metadata: AttentionMetadata
    current_rng: jax.Array
    kv_caches: Any
    generated_tokens: jax.Array
    all_expert_indices: Any


@functools.partial(
    jax.jit,
    donate_argnames=("kv_caches", ),
    static_argnames=(
        "model_fn",
        "compute_logits_fn",
        "sample_fn",
        "mesh",
        "max_decode_steps",
        "eos_token_id",
        "padding_token_id",
        "terminate_on_any_eos",
        "layer_name_to_kvcache_index",
        "is_first_rank",
        "is_last_rank",
        "dp_size",
        "expert_indices_static_shape",
        "expert_indices_dtype",
    ),
)
def continue_decode(
    model_fn: Callable,
    compute_logits_fn: Callable,
    sample_fn: Callable,
    mesh: Any,
    init_state: TpuSamplingState,
    kv_caches: Any,
    sampling_metadata: Any,
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
    expert_indices_static_shape: Any = None,
    expert_indices_dtype: Any = None,
) -> tuple[jax.Array, Any, TpuSamplingState, jax.Array, Any]:
    """Continues decoding on TPU using an optimized lax.while_loop for native hardware early termination.

    Args:
        model_fn: Function to run the model forward pass.
        compute_logits_fn: Function to compute logits from hidden states.
        sample_fn: Function to sample next tokens.
        mesh: Sharding mesh.
        init_state: Initial TpuSamplingState.
        kv_caches: Initial KV caches.
        sampling_metadata: Sampling parameters PyTree.
        max_decode_steps: Maximum steps to decode (Python int).
        eos_token_id: EOS token ID.
        padding_token_id: Padding token ID.
        rng: Initial PRNG key for sampling.
        terminate_on_any_eos: If True, stops as soon as any request hits EOS.
                             If False, continues until all requests hit EOS.
        dp_size: Data parallel size (needed for correct metadata padding).
        expert_indices_static_shape: Static tuple of (num_layers, top_k) if MoE tracking enabled.
        expert_indices_dtype: Dtype for expert indices accumulation.

    Returns:
    Tuple of (generated_tokens, final_kv_caches, final_state, final_rng, all_expert_indices).
    """

    batch_size = init_state.current_tokens.shape[0]
    generated_tokens = jnp.full((max_decode_steps, batch_size),
                                padding_token_id,
                                dtype=jnp.int32)

    if expert_indices_static_shape is not None:
        num_layers, top_k = expert_indices_static_shape
        expert_dtype = (expert_indices_dtype
                        if expert_indices_dtype is not None else jnp.int32)
        all_expert_indices = jnp.zeros(
            (max_decode_steps, num_layers, batch_size, top_k),
            dtype=expert_dtype,
        )
    else:
        all_expert_indices = None

    # Padded request slots initially have active_mask == False.
    # Capture this static valid batch mask to exclude padding slots from triggering false early exits on hardware.
    valid_batch_mask = init_state.active_mask

    init_carry = DecodeLoopCarry(
        step_idx=jnp.array(0, dtype=jnp.int32),
        current_tokens=init_state.current_tokens,
        active_mask=init_state.active_mask,
        attn_metadata=init_state.attn_metadata,
        current_rng=rng,
        kv_caches=kv_caches,
        generated_tokens=generated_tokens,
        all_expert_indices=all_expert_indices,
    )

    def cond_fun(carry: DecodeLoopCarry) -> jax.Array:
        within_bounds = carry.step_idx < max_decode_steps
        if terminate_on_any_eos:
            # Mask out trailing padding slots from forcing premature hardware loop termination
            streams_active = jnp.all(
                jnp.logical_or(carry.active_mask,
                               jnp.logical_not(valid_batch_mask)))
        else:
            streams_active = jnp.any(carry.active_mask)
        return jnp.logical_and(within_bounds, streams_active)

    def body_fun(carry: DecodeLoopCarry) -> DecodeLoopCarry:
        step_idx = carry.step_idx
        current_rng, step_rng = jax.random.split(carry.current_rng)

        # 1. Forward pass
        kv_caches, hidden_states, _, expert_indices_step = model_fn(
            carry.kv_caches,
            carry.current_tokens,
            carry.attn_metadata,
            inputs_embeds,
            carry.attn_metadata.input_positions,
            layer_name_to_kvcache_index,
            lora_metadata,
            intermediate_tensors,
            is_first_rank,
            is_last_rank,
        )

        all_expert_indices = carry.all_expert_indices
        if expert_indices_step is not None and all_expert_indices is not None:
            all_expert_indices = all_expert_indices.at[step_idx].set(
                expert_indices_step)

        # 2. Compute logits and sample
        logits = compute_logits_fn(hidden_states, None)

        logits = logits.astype(jnp.float32)
        next_tokens, _ = sample_fn(step_rng, mesh, logits, sampling_metadata)

        # 3. Check for EOS and update mask
        is_eos = next_tokens == eos_token_id
        new_active_mask = jnp.logical_and(carry.active_mask,
                                          jnp.logical_not(is_eos))

        # 4. Update tokens for next step (pad finished ones)
        next_input_ids = jnp.where(new_active_mask, next_tokens,
                                   padding_token_id)

        # 5. Update AttentionMetadata for next step
        increment = new_active_mask.astype(jnp.int32)
        new_positions = carry.attn_metadata.input_positions + increment

        # Fix DP padding bug by exposing DP dimension as 2D before padding
        increment_2d = increment.reshape(dp_size, -1)
        seq_lens_2d = carry.attn_metadata.seq_lens.reshape(dp_size, -1)

        pad_len = seq_lens_2d.shape[1] - increment_2d.shape[1]
        padded_increment_2d = jnp.pad(increment_2d, ((0, 0), (0, pad_len)))

        new_seq_lens = (seq_lens_2d + padded_increment_2d).ravel()

        new_attn_metadata = AttentionMetadata(
            input_positions=new_positions,
            block_tables=carry.attn_metadata.block_tables,
            seq_lens=new_seq_lens,
            query_start_loc=carry.attn_metadata.query_start_loc,
            request_distribution=carry.attn_metadata.request_distribution,
            mamba_state_indices=carry.attn_metadata.mamba_state_indices,
        )

        # 6. Record generated tokens
        generated_tokens = carry.generated_tokens.at[step_idx].set(
            jnp.where(carry.active_mask, next_tokens, padding_token_id))

        return DecodeLoopCarry(
            step_idx=step_idx + 1,
            current_tokens=next_input_ids,
            active_mask=new_active_mask,
            attn_metadata=new_attn_metadata,
            current_rng=current_rng,
            kv_caches=kv_caches,
            generated_tokens=generated_tokens,
            all_expert_indices=all_expert_indices,
        )

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)

    final_state = TpuSamplingState(
        current_tokens=final_carry.current_tokens,
        active_mask=final_carry.active_mask,
        attn_metadata=final_carry.attn_metadata,
        step_counter=final_carry.step_idx,
    )

    return (
        final_carry.generated_tokens,
        final_carry.kv_caches,
        final_state,
        final_carry.current_rng,
        final_carry.all_expert_indices,
    )
