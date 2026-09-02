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

from tpu_inference import envs
from tpu_inference.layers.common.binary_search import topk_mask, topp_mask
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import VllmSchedulerOutput

    from tpu_inference.runner.input_batch import CachedRequestState

_SAMPLING_EPS = 1e-5

# Candidates each vocab shard contributes to vocab-sharded sampling. Rows with
# 0 < top_k <= this many candidates are sampled exactly (see
# _can_sample_vocab_sharded).
VOCAB_SHARDED_SAMPLING_NUM_CANDIDATES = 64
_MASKED_LOGIT = -1e12


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


def _mesh_axes(mesh: Mesh, axis_name) -> tuple[str, ...]:
    """The names in `axis_name` (a name or a tuple of names) present in mesh."""
    names = axis_name if isinstance(axis_name,
                                    (tuple, list)) else (axis_name, )
    return tuple(name for name in names if name in mesh.axis_names)


def _vocab_sharded_specs(mesh: Mesh):
    """(batch_axes, vocab_axis) for vocab-sharded sampling, or None.

    Returns None when the mesh has no model axis to shard the vocab over, in
    which case there is nothing to gain from the sharded path.
    """
    vocab_axes = _mesh_axes(mesh, ShardingAxisName.MODEL)
    if not vocab_axes:
        return None
    batch_axes = _mesh_axes(mesh, ShardingAxisName.ATTN_DATA)
    return (batch_axes or None), vocab_axes[0]


def _can_sample_vocab_sharded(
        tpu_sampling_metadata: TPUSupportedSamplingMetadata) -> jax.Array:
    """Whether every row is sampled exactly by the vocab-sharded path.

    A row is exact when it is greedy (the argmax is always among the
    candidates) or when 0 < top_k <= VOCAB_SHARDED_SAMPLING_NUM_CANDIDATES and
    top_p > 0, since the union of the per-shard top candidates then contains
    the whole top-k set and the top-p cutoff is computed over that set. Padded
    request slots carry DEFAULT_SAMPLING_PARAMS (temperature -1, top_k 0), so
    they count as greedy and never force the fallback.
    """
    is_greedy = tpu_sampling_metadata.temperature < _SAMPLING_EPS
    top_k = tpu_sampling_metadata.top_k
    exact = ((top_k > 0) & (top_k <= VOCAB_SHARDED_SAMPLING_NUM_CANDIDATES) &
             (tpu_sampling_metadata.top_p > 0.0))
    return jnp.all(is_greedy | exact)


def _merged_candidates(logits: jax.Array, vocab_axis: str,
                       num_candidates: int) -> tuple[jax.Array, jax.Array]:
    """Top candidates of every vocab shard, gathered along the vocab axis.

    Args:
        logits: this shard's slice, (batch_shard, vocab_shard).
        vocab_axis: mesh axis the vocab is sharded over.
        num_candidates: candidates taken from each shard.

    Returns:
        (values, token_ids), each (batch_shard, num_shards * num_candidates),
        with token_ids in the global vocab.
    """
    vocab_shard = logits.shape[-1]
    values, local_ids = jax.lax.top_k(logits, num_candidates)
    token_ids = local_ids + jax.lax.axis_index(vocab_axis) * vocab_shard
    values = jax.lax.all_gather(values, vocab_axis)
    token_ids = jax.lax.all_gather(token_ids, vocab_axis)
    num_shards = values.shape[0]
    values = jnp.transpose(values,
                           (1, 0, 2)).reshape(-1, num_shards * num_candidates)
    token_ids = jnp.transpose(token_ids,
                              (1, 0, 2)).reshape(-1,
                                                 num_shards * num_candidates)
    return values, token_ids


def _sample_vocab_sharded_block(
    rng: jax.Array,
    logits: jax.Array,
    temperature: jax.Array,
    top_k: jax.Array,
    top_p: jax.Array,
    batch_axes: tuple[str, ...] | None,
    vocab_axis: str,
) -> jax.Array:
    """Per-shard body of vocab-sharded sampling; returns (batch_shard,) tokens.

    Mirrors sample()'s replicated path on the merged candidates: temperature,
    top-k (ties at the k-th value kept, like topk_mask), top-p (smallest
    prefix of the sorted probabilities with mass >= p, like topp_mask), then a
    categorical draw; greedy rows take the argmax (lowest id among ties, like
    jnp.argmax). Candidates that are not in a row's top-k set are masked, so
    the softmax normalizes over exactly the same set as the replicated path.
    """
    num_candidates = VOCAB_SHARDED_SAMPLING_NUM_CANDIDATES
    values, token_ids = _merged_candidates(logits, vocab_axis, num_candidates)

    row_max = jnp.max(values, axis=-1, keepdims=True)
    greedy_tokens = jnp.min(jnp.where(values >= row_max, token_ids,
                                      jnp.iinfo(jnp.int32).max),
                            axis=-1)

    scaled = values / temperature[:, None]
    sorted_desc = -jnp.sort(-scaled, axis=-1)
    kth = jnp.take_along_axis(sorted_desc,
                              jnp.clip(top_k, 1, scaled.shape[-1])[:, None] -
                              1,
                              axis=-1)
    masked = jnp.where((top_k > 0)[:, None] & (scaled < kth), _MASKED_LOGIT,
                       scaled)

    probs = jax.nn.softmax(masked, axis=-1)
    sorted_probs = -jnp.sort(-probs, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)
    cutoff_index = jnp.argmax(cumulative >= top_p[:, None], axis=-1)
    cutoff = jnp.take_along_axis(sorted_probs, cutoff_index[:, None], axis=-1)
    masked = jnp.where((top_p < 1.0)[:, None] & (probs < cutoff),
                       _MASKED_LOGIT, masked)

    # Every batch shard receives the same key; fold in the shard's index so
    # rows on different shards draw independent noise.
    shard_index = jnp.int32(0)
    for axis in batch_axes or ():
        shard_index = (shard_index * jax.lax.axis_size(axis) +
                       jax.lax.axis_index(axis))
    choice = jax.random.categorical(jax.random.fold_in(rng, shard_index),
                                    masked)
    sampled = jnp.take_along_axis(token_ids, choice[:, None], axis=-1)[:, 0]
    tokens = jnp.where(temperature < _SAMPLING_EPS, greedy_tokens, sampled)
    # The value is already identical on every vocab shard (it is built from
    # the gathered candidates); the reduction makes that replication explicit
    # for the shard_map output spec.
    return jax.lax.pmax(tokens, vocab_axis)


def _sample_vocab_sharded(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
    batch_axes: tuple[str, ...] | None,
    vocab_axis: str,
) -> jax.Array:
    """Samples from logits sharded over the vocab; returns replicated tokens."""
    batch_spec = P(batch_axes)
    logits_spec = P(batch_axes, vocab_axis)
    logits = jax.lax.with_sharding_constraint(logits,
                                              NamedSharding(mesh, logits_spec))

    def block(rng, logits, temperature, top_k, top_p):
        return _sample_vocab_sharded_block(rng, logits, temperature, top_k,
                                           top_p, batch_axes, vocab_axis)

    tokens = jax.shard_map(
        block,
        mesh=mesh,
        in_specs=(P(), logits_spec, batch_spec, batch_spec, batch_spec),
        out_specs=batch_spec,
    )(rng, logits, tpu_sampling_metadata.temperature.astype(jnp.float32),
      tpu_sampling_metadata.top_k, tpu_sampling_metadata.top_p)
    return jax.lax.with_sharding_constraint(tokens, NamedSharding(mesh, P()))


@jax.jit(static_argnames=["mesh"])
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    """Samples the next token of every row of (B, vocab_size) logits.

    Returns (tokens, logits): the logits are the processed (temperature /
    top-k / top-p) ones for sampled rows and the raw ones for greedy rows.

    With USE_VOCAB_SHARDED_SAMPLING the logits are not all-gathered over the
    vocab: each vocab shard contributes its top candidates and sampling runs on
    those (see _sample_vocab_sharded_block). The returned logits are then the
    raw, still vocab-sharded logits, which is why that mode rejects
    logprobs_mode="processed_*" (see TpuPlatform.check_and_update_config).
    """
    # (B, vocab_size)
    if tpu_sampling_metadata._cache_collision_dummy is not None:
        # Force a dependency on the dummy tensor's shape to ensure unique HLO.
        logits = logits + 0 * jnp.sum(
            tpu_sampling_metadata._cache_collision_dummy)

    vocab_sharded_specs = (_vocab_sharded_specs(mesh)
                           if tpu_sampling_metadata.do_sampling
                           and envs.USE_VOCAB_SHARDED_SAMPLING else None)
    if vocab_sharded_specs is not None:
        # Both branches of the cond return the logits with the vocab still
        # sharded, so choosing the fallback at runtime never adds the
        # all-gather to the sharded path.
        batch_axes, vocab_axis = vocab_sharded_specs
        logits = logits.astype(jnp.float32)
        logits_sharding = NamedSharding(mesh, P(batch_axes, vocab_axis))

        def sharded(operands):
            rng, logits = operands
            tokens = _sample_vocab_sharded(rng, mesh, logits,
                                           tpu_sampling_metadata, batch_axes,
                                           vocab_axis)
            return tokens, logits

        def replicated(operands):
            rng, logits = operands
            tokens, ret_logits = _sample_replicated(rng, mesh, logits,
                                                    tpu_sampling_metadata)
            return tokens, jax.lax.with_sharding_constraint(
                ret_logits, logits_sharding)

        return jax.lax.cond(_can_sample_vocab_sharded(tpu_sampling_metadata),
                            sharded, replicated, (rng, logits))

    return _sample_replicated(rng, mesh, logits, tpu_sampling_metadata)


def _sample_replicated(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> tuple[jax.Array, jax.Array]:
    """sample() over logits replicated along the vocab."""
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
