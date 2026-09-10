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
from jax import lax
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


def _distributed_sampling_max_top_k() -> int:
    """Static top-k capacity used to shape the compiled candidate sampler."""
    max_top_k = envs.DISTRIBUTED_SAMPLING_MAX_TOP_K
    if max_top_k < 1:
        raise ValueError("DISTRIBUTED_SAMPLING_MAX_TOP_K must be >= 1, got "
                         f"{max_top_k}")
    return max_top_k


def _distributed_sampling_candidates_per_shard() -> int:
    # Retaining twice the supported top-k reduces tie-overflow fallbacks while
    # preserving the existing 128 candidates for the default max_top_k of 64.
    return 2 * _distributed_sampling_max_top_k()


def _distributed_sampling_fits(mesh: Mesh, vocab_size: int) -> bool:
    """Whether each vocab shard can provide the static candidate capacity."""
    tensor_axes = ShardingAxisName.MLP_TENSOR
    tensor_axes = tensor_axes if isinstance(tensor_axes, (tuple, list)) else (
        tensor_axes, )
    tensor_axes = tuple(axis for axis in tensor_axes
                        if axis is not None and axis in mesh.axis_names)
    if not tensor_axes:
        return False
    num_vocab_shards = 1
    for axis in tensor_axes:
        num_vocab_shards *= mesh.shape[axis]
    local_vocab_size = vocab_size // num_vocab_shards
    return local_vocab_size >= _distributed_sampling_candidates_per_shard()


def distributed_sampling_allowed(logprobs: bool, logprobs_mode) -> bool:
    """Whether sampling can return raw logits for the requested logprob mode."""
    return not (logprobs and str(logprobs_mode).startswith("processed"))


def _can_sample_distributed(
        tpu_sampling_metadata: TPUSupportedSamplingMetadata) -> jax.Array:
    """Whether every row is supported by distributed candidate sampling."""
    is_greedy = tpu_sampling_metadata.temperature < _SAMPLING_EPS
    supported = ((tpu_sampling_metadata.top_k > 0) &
                 (tpu_sampling_metadata.top_k <=
                  _distributed_sampling_max_top_k()) &
                 (tpu_sampling_metadata.top_p > 0.0))
    return jnp.all(is_greedy | supported)


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


def _merge_topk_candidates(
    candidate_values: jax.Array,
    candidate_ids: jax.Array,
    top_k: jax.Array,
    top_p: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Applies exact threshold top-k and top-p to gathered candidates.

    The production top-k retains every value tied with the requested rank. A
    larger per-shard candidate set lets this path retain those ties too. The
    result is incomplete only when a shard's last retained value reaches the
    global threshold, because that shard may have omitted more qualifying
    values.
    """
    candidates_per_shard = _distributed_sampling_candidates_per_shard()
    if candidate_values.shape[-1] % candidates_per_shard != 0:
        raise ValueError("Candidate dimension must contain complete shards")
    global_topk, _ = lax.top_k(candidate_values,
                               _distributed_sampling_max_top_k())
    threshold = jnp.take_along_axis(global_topk, top_k[:, None] - 1,
                                    axis=-1)[:, 0]
    shard_candidates = candidate_values.reshape(
        candidate_values.shape[0], -1, candidates_per_shard)
    shard_tails = shard_candidates[:, :, -1]
    incomplete = jnp.any(shard_tails >= threshold[:, None], axis=-1)
    topk_values = jnp.where(candidate_values >= threshold[:, None],
                            candidate_values, -1e12)
    filtered_values = topp_mask(topk_values, top_p, replace_val=-1e12)
    return filtered_values, candidate_ids, incomplete


def _distributed_topk_sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    temperature: jax.Array,
    top_k: jax.Array,
    top_p: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Samples from the exact requested global top-k using sharded candidates.

    Every TP shard contributes a static number of local candidates. The
    requested global top-k value is used as a dynamic threshold so all boundary
    ties are retained. Callers fall back if a shard may have omitted additional
    values at that threshold.

    Returns sampled global token IDs and a replicated scalar indicating that
    the gathered candidates may not contain the complete top-k tie group.
    """
    data_spec = P(ShardingAxisName.MLP_DATA)
    logits_spec = P(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR)
    replicated = P()

    def local_sample(local_rng, local_logits, local_temperature, local_top_k,
                     local_top_p):
        candidates_per_shard = _distributed_sampling_candidates_per_shard()
        local_vocab_size = local_logits.shape[-1]
        if local_vocab_size < candidates_per_shard:
            raise ValueError(
                "Distributed top-k sampling requires at least "
                f"{candidates_per_shard} logits per vocabulary shard")

        data_axis = ShardingAxisName.MLP_DATA
        if data_axis in mesh.axis_names and mesh.shape[data_axis] > 1:
            local_rng = jax.random.fold_in(local_rng,
                                           lax.axis_index(data_axis))
        # Preserve the candidate sampler's existing key derivation.
        sample_rng = jax.random.split(local_rng, 1)[0]
        shard_index = lax.axis_index(ShardingAxisName.MLP_TENSOR)

        # Greedy rows do not consume the categorical result. A safe positive
        # temperature avoids reversing their candidate ordering.
        safe_temperature = jnp.where(
            local_temperature < _SAMPLING_EPS,
            jnp.ones_like(local_temperature),
            local_temperature,
        )
        scaled_logits = local_logits / safe_temperature[:, None]
        local_values, local_ids = lax.top_k(scaled_logits,
                                            candidates_per_shard)
        local_ids = (local_ids + shard_index * local_vocab_size).astype(
            jnp.int32)

        candidate_values = lax.all_gather(
            local_values,
            ShardingAxisName.MLP_TENSOR,
            axis=-1,
            tiled=True,
        )
        candidate_ids = lax.all_gather(
            local_ids,
            ShardingAxisName.MLP_TENSOR,
            axis=-1,
            tiled=True,
        )
        safe_top_k = jnp.clip(local_top_k, 1,
                              _distributed_sampling_max_top_k())
        filtered_values, candidate_ids, incomplete = _merge_topk_candidates(
            candidate_values, candidate_ids, safe_top_k, local_top_p)
        incomplete = jnp.logical_and(incomplete,
                                     local_temperature >= _SAMPLING_EPS)
        sampled_positions = jax.random.categorical(sample_rng, filtered_values)
        sampled_ids = jnp.take_along_axis(candidate_ids,
                                          sampled_positions[:, None],
                                          axis=-1)[:, 0]
        return sampled_ids, jnp.any(incomplete)

    return jax.shard_map(
        local_sample,
        mesh=mesh,
        in_specs=(replicated, logits_spec, data_spec, data_spec, data_spec),
        out_specs=(data_spec, replicated),
        check_vma=False,
    )(rng, logits, temperature, top_k, top_p)


@jax.jit(static_argnames=["mesh", "allow_distributed_sampling"])
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
    allow_distributed_sampling: bool = True,
) -> jax.Array:
    # (B, vocab_size)
    if tpu_sampling_metadata._cache_collision_dummy is not None:
        # Force a dependency on the dummy tensor's shape to ensure unique HLO.
        logits = logits + 0 * jnp.sum(
            tpu_sampling_metadata._cache_collision_dummy)

    greedy_tokens = jnp.argmax(logits, axis=-1)
    logits = logits.astype(jnp.float32)
    if not tpu_sampling_metadata.do_sampling:
        ret_tokens = greedy_tokens
        ret_logits = logits
    else:
        is_greedy = tpu_sampling_metadata.temperature < _SAMPLING_EPS

        def sample_full_vocab(_):
            full_logits = jax.lax.with_sharding_constraint(
                logits,
                NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
            processed_logits = _apply_sampling_transforms(
                full_logits, tpu_sampling_metadata)
            sampled_tokens = jax.random.categorical(rng, processed_logits)
            tokens = jnp.where(is_greedy, greedy_tokens, sampled_tokens)
            output_logits = jnp.where(is_greedy[:, None], full_logits,
                                      processed_logits)
            return tokens, output_logits

        use_distributed_candidates = (
            allow_distributed_sampling
            and _distributed_sampling_fits(mesh, logits.shape[-1]))
        if use_distributed_candidates:
            # Candidate shapes use a trace-time maximum; each request's top-k
            # remains dynamic. Greedy and padded rows do not consume a sample.
            supported = _can_sample_distributed(tpu_sampling_metadata)

            def sample_candidates(_):
                sampled_tokens, incomplete_candidates = (
                    _distributed_topk_sample(
                        rng,
                        mesh,
                        logits,
                        tpu_sampling_metadata.temperature,
                        tpu_sampling_metadata.top_k,
                        tpu_sampling_metadata.top_p,
                    ))

                def use_candidate_result(_):
                    tokens = jnp.where(is_greedy, greedy_tokens,
                                       sampled_tokens)
                    # Processed-logit modes disable this path. Returning the
                    # raw input supports raw logprobs without materializing
                    # full-vocabulary filtered logits.
                    return tokens, logits

                return lax.cond(incomplete_candidates,
                                sample_full_vocab,
                                use_candidate_result,
                                operand=None)

            ret_tokens, ret_logits = lax.cond(
                supported,
                sample_candidates,
                sample_full_vocab,
                operand=None,
            )
        else:
            ret_tokens, ret_logits = sample_full_vocab(None)
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
