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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.common.binary_search import topk_mask, topp_mask
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import VllmSchedulerOutput

    from tpu_inference.runner.input_batch import CachedRequestState

_SAMPLING_EPS = 1e-5


@dataclass
class PromptLogprobsReqSnap:
    """Per-request state snapshotted at step N for use in get_output()."""
    req_id: str
    req_state: "CachedRequestState"  # Stable request state reference; CPU buffer is pre-allocated.
    req_offset: int  # Absolute row index into the full-batch logprobs tensor.
    start_idx: int  # Number of computed tokens.
    num_logits: int  # Number of rows to copy from the TPU tensor to the CPU accumulator.
    is_last_chunk: bool  # True if this is the final chunk of the prompt logprobs.
    num_k: int  # Number of top logprobs to retain for this request.


@dataclass
class PromptLogprobsAsyncData:
    """Holds async-copied prompt logprob tensors + per-request snapshots for get_output()."""
    tensors: LogprobsTensors  # Result of _jax_logprobs_copy_to_host_async (pending transfer).
    req_snaps: List[PromptLogprobsReqSnap]


def _jax_logprobs_copy_to_host_async(
        logprobs_tensors: LogprobsTensors) -> LogprobsTensors:
    """Initiate non-blocking TPU-to-host copies for all logprobs arrays."""
    return LogprobsTensors(
        logprob_token_ids=jax.copy_to_host_async(
            logprobs_tensors.logprob_token_ids),
        logprobs=jax.copy_to_host_async(logprobs_tensors.logprobs),
        selected_token_ranks=jax.copy_to_host_async(
            logprobs_tensors.selected_token_ranks),
    )


def _apply_sampling_transforms(
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    """Apply temperature scaling, top-k, and top-p filtering to logits.

    This extracts the common logit processing logic used by both the sampling
    path and the processed-logprobs path so that the transformations are
    applied identically.

    Args:
        logits: (B, vocab_size) raw logits in float32.
        tpu_sampling_metadata: Sampling parameters (temperature, top_k, top_p).

    Returns:
        Processed logits with temperature, top-k, and top-p applied.
    """
    # Temperature scaling
    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits = logits / temperatures

    # Only apply top-k masking if k > 0 for each token
    top_k = tpu_sampling_metadata.top_k
    should_apply_topk = jnp.expand_dims(top_k > 0, axis=-1)
    topk_masked = topk_mask(logits, top_k, replace_val=-1e12)
    logits = jnp.where(should_apply_topk, topk_masked, logits)

    # Only apply top-p masking if p < 1.0 for each token
    top_p = tpu_sampling_metadata.top_p
    should_apply_topp = jnp.expand_dims(top_p < 1.0, axis=-1)
    topp_masked = topp_mask(logits, top_p, replace_val=-1e12)
    logits = jnp.where(should_apply_topp, topp_masked, logits)

    return logits


@jax.jit(static_argnames=["mesh"])
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    if tpu_sampling_metadata._cache_collision_dummy is not None:
        # Force a dependency on the dummy tensor's shape to ensure unique HLO.
        logits = logits + 0 * jnp.sum(
            tpu_sampling_metadata._cache_collision_dummy)

    if tpu_sampling_metadata.do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        # TODO(gxd3): revisit if the 2nd dimension of the logits can be sharded
        # instead of being replicated.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))

    greedy_tokens = jnp.argmax(logits, axis=-1)
    logits = logits.astype(jnp.float32)
    if not tpu_sampling_metadata.do_sampling:
        ret_tokens = greedy_tokens
        ret_logits = logits
    else:
        processed_logits = _apply_sampling_transforms(logits,
                                                      tpu_sampling_metadata)
        # (batch_size,)
        next_tokens = jax.random.categorical(rng, processed_logits)
        # Note: avoid using the sample result when temperature < _SAMPLING_EPS
        # If temperature < 0, logits /= temperatures will flip the result, causing error.
        is_greedy = tpu_sampling_metadata.temperature < _SAMPLING_EPS
        ret_tokens = jnp.where(is_greedy, greedy_tokens, next_tokens)
        ret_logits = jnp.where(jnp.expand_dims(is_greedy, axis=-1), logits,
                               processed_logits)
    # Replicate the result so that in multi-controller jax setup
    # (i.e. Ray based multi-host setup), we won't hit error like
    # RuntimeError: Fetching value for `jax.Array` that spans non-addressable
    # (non process local) devices is not possible.
    next_tokens = jax.lax.with_sharding_constraint(ret_tokens,
                                                   NamedSharding(mesh, P()))
    return next_tokens, ret_logits


def diffusion_commit(
    logits: jax.Array,
    mask: jax.Array,
    threshold: float,
    temperature: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Per-position, threshold-based commit step for block-diffusion decode.

    Given per-position vocab logits and a boolean mask marking the positions
    that are still "masked" (i.e. not yet committed), greedily commit the
    positions whose top-1 probability exceeds ``threshold``. To guarantee
    forward progress, the single highest-confidence still-masked position in
    each row is always committed even if it is below ``threshold`` (unless the
    row has no masked positions left).

    Args:
        logits: (..., L, V) float logits, where ``L`` is the number of
            positions per row and ``V`` is the vocab size. Any number of
            leading batch/row dims is supported (e.g. (B, L, V) or (L, V)).
        mask: (..., L) boolean array; ``True`` marks positions that are still
            masked and therefore eligible to be committed this step.
        threshold: scalar confidence threshold in [0, 1]. A masked position
            commits when its top-1 softmax probability is strictly greater than
            this value.
        temperature: optional softmax temperature. When ``> 0`` the logits are
            divided by it before the softmax (sharpening for ``< 1``, flattening
            for ``> 1``); when ``0`` (default) the raw logits are used.

    Returns:
        A tuple ``(committed_token_ids, new_mask)`` where:
            - ``committed_token_ids``: (..., L) int32 array holding the argmax
              (top-1) token id for every position. Values at positions that
              commit this step (``mask & ~new_mask``) are the tokens to write;
              values at still-masked positions should be ignored by the caller.
            - ``new_mask``: (..., L) boolean array equal to ``mask`` with the
              newly-committed positions cleared (a strictly shrinking mask).
    """
    if temperature > 0.0:
        logits = logits / temperature
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)

    # Top-1 probability and token id per position -> (..., L).
    top_prob = jnp.max(probs, axis=-1)
    top_tok = jnp.argmax(probs, axis=-1).astype(jnp.int32)

    mask_bool = mask.astype(bool)

    # Threshold commit: only masked positions confident enough are committed.
    commit = (top_prob > threshold) & mask_bool

    # Progress guarantee: force the single highest-confidence still-masked
    # position in each row to commit. Non-masked positions are excluded from
    # the argmax so an already-committed position can never be re-selected, and
    # rows with no masked positions left force nothing.
    neg_inf = jnp.array(-jnp.inf, dtype=top_prob.dtype)
    masked_conf = jnp.where(mask_bool, top_prob, neg_inf)
    forced_idx = jnp.argmax(masked_conf, axis=-1)
    row_has_mask = jnp.any(mask_bool, axis=-1)
    forced_onehot = jax.nn.one_hot(forced_idx, mask_bool.shape[-1], dtype=bool)
    forced_onehot = forced_onehot & jnp.expand_dims(row_has_mask, axis=-1)
    commit = commit | forced_onehot

    new_mask = mask_bool & (~commit)
    return top_tok, new_mask


def compute_logprobs(logits: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(logits, axis=-1)


@jax.jit(static_argnames=("max_logprobs", ))
def compute_and_gather_logprobs(
    logits: jax.Array,
    next_tokens: jax.Array,
    max_logprobs: int,
) -> LogprobsTensors:
    """Compute logprobs from logits and gather the requested top-k."""
    logprobs = compute_logprobs(logits)
    return gather_logprobs(logprobs, next_tokens, max_logprobs)


@jax.jit(static_argnames=("max_logprobs", ))
def compute_and_gather_prompt_logprobs(
    logits: jax.Array,
    input_ids: jax.Array,
    max_logprobs: int,
) -> LogprobsTensors:
    """Compute logprobs from full logits and gather the requested top-k for prompt tokens."""
    prompt_target_ids = jnp.roll(input_ids, -1, axis=0)
    return compute_and_gather_logprobs(logits, prompt_target_ids, max_logprobs)


def compute_prompt_logprobs(
    full_logits: Optional[jax.Array],
    input_ids: Optional[jax.Array],
    num_prompt_logprobs: Dict[str, int],
    requests: Dict[str, "CachedRequestState"],
    scheduler_output: "VllmSchedulerOutput",
    req_ids_dp: Optional[Dict[int, List[str]]],
    dp_size: int,
    max_logprobs: int,
) -> Optional[PromptLogprobsAsyncData]:
    """Dispatches prompt logprob computation on TPU and snapshots per-request state.
    Returns PromptLogprobsAsyncData containing the async-copied tensors and
    the snapshotted state needed to safely slice them in get_output().
    """
    if (not num_prompt_logprobs or full_logits is None or input_ids is None):
        return None

    # Gather compact [total_padded_tokens, max_logprobs+1] tensors on TPU and
    # start async transfer to host (overlaps with next step's execute_model).
    # We use the statically precompiled max_logprobs instead of the dynamic user max_k
    # to avoid triggering JAX recompilation. The correct num_k is preserved in req_snaps.
    prompt_lp_tensors = compute_and_gather_prompt_logprobs(
        full_logits, input_ids, max_logprobs)
    prompt_lp_tensors = _jax_logprobs_copy_to_host_async(prompt_lp_tensors)

    # Snapshot all mutable per-request state before update_states(N+1) runs.
    padded_tokens_per_dp = full_logits.shape[0] // dp_size
    req_snaps: List[PromptLogprobsReqSnap] = []
    if req_ids_dp:
        for dp_rank, req_id_list in req_ids_dp.items():
            dp_token_offset = dp_rank * padded_tokens_per_dp
            local_token_offset = 0
            for req_id in req_id_list:
                num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
                if req_id in num_prompt_logprobs:
                    num_k = num_prompt_logprobs[req_id]
                    req_state = requests[req_id]
                    start_idx = req_state.num_computed_tokens
                    num_remaining = req_state.num_prompt_tokens - (start_idx +
                                                                   1)
                    if num_scheduled <= num_remaining:
                        num_logits = num_scheduled
                        is_last_chunk = False
                    else:
                        num_logits = num_remaining
                        is_last_chunk = True
                    req_snaps.append(
                        PromptLogprobsReqSnap(
                            req_id=req_id,
                            req_state=req_state,
                            req_offset=dp_token_offset + local_token_offset,
                            start_idx=start_idx,
                            num_logits=num_logits,
                            is_last_chunk=is_last_chunk,
                            num_k=num_k,
                        ))
                local_token_offset += num_scheduled

    return PromptLogprobsAsyncData(tensors=prompt_lp_tensors,
                                   req_snaps=req_snaps)


def gather_logprobs(
    logprobs: jax.Array,
    token_ids: jax.Array,
    num_logprobs: int,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        token_ids: prompt tokens (if prompt logprobs)
                    or sampled tokens (if sampled
                    logprobs); 1D token ID tensor
                    with (num tokens) elements
        num_logprobs: minimum number of logprobs to
                    retain per token


    Returns:
        Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
        Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
        Sampled token rank tensor, (num tokens)
    """
    # Find the topK values.
    topk_logprobs, topk_indices = jax.lax.top_k(logprobs, k=num_logprobs)

    # Get with the logprob of the prompt or sampled token.
    token_ids = jnp.expand_dims(token_ids, axis=-1)
    token_logprobs = jnp.take_along_axis(logprobs, token_ids, axis=-1)

    # Compute the ranks of the actual token.
    token_ranks = jnp.sum(logprobs >= token_logprobs, axis=-1)

    # Concatenate together with the topk.
    indices = jnp.concatenate((token_ids, topk_indices), axis=1)
    logprobs = jnp.concatenate((token_logprobs, topk_logprobs), axis=1)

    # Use int32 to reduce the tensor size.
    indices = jnp.int32(indices)

    return LogprobsTensors(indices, logprobs, token_ranks)
