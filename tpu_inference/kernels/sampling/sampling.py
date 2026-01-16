"""
vLLM top-k top-p sampling, using two pallas functions
"""

import functools
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.kernels.sampling.top_p_and_sample import top_p_and_sample, _top_p_and_sample
from tpu_inference.kernels.sampling.divide_and_filter_topk import top_bounded_k
from tpu_inference.kernels.sampling.bitonic_topk import bitonic_topk

@functools.partial(
  jax.jit, static_argnames=("max_k", "num_bins", "bins_topm_schedule", "sampling_eps", "replace_val")
)
def topk_topp_and_sample(
  rng_key,
  logits,
  tpu_sampling_metadata,
  max_k: int,
  num_bins: int | None = None,
  bins_topm_schedule: int | None = None,
  sampling_eps: float = 1e-5,
  replace_val: float = -1e12,
):
  """Combined top-k, top-p filtering, and sampling for vLLM inference.

  Args:
    rng_key: RNG key for sampling.
    logits: Input logits of shape [batch_size, vocab_size].
    tpu_sampling_metadata: Metadata containing top_k, top_p, and temperature.
    max_k: Maximum k value for top-k computation.
    num_bins: Optional number of bins for divide-and-filter algorithm.
    bins_topm_schedule: Optional custom schedule for binned top-m computation.
    sampling_eps: Use greedy token if temperature < eps
    replace_val: Replace padding entries in probabilities with constant

  Returns:
    Sampled token indices.
  """
  vocab_size = logits.shape[1]
  topk_logits, topk_idxs = top_bounded_k(
    logits,
    k=tpu_sampling_metadata.top_k,
    replace_val=replace_val,
    max_k=max_k,
    num_bins=num_bins,
    bins_topm_schedule=bins_topm_schedule,
    guarantee_convergence=True,
  )
  if rng_key.shape == ():
    rng_key = jax.random.key_data(rng_key)
  return top_p_and_sample(
    topk_logits,
    topk_idxs,
    rng_key,
    top_p=tpu_sampling_metadata.top_p,
    temperature=tpu_sampling_metadata.temperature,
    vocab_size=vocab_size,
    replace_val=replace_val,
    sampling_eps=sampling_eps,
  )
  
  
@functools.partial(
  jax.jit, static_argnames=("mesh", "max_k", "num_bins", "bins_topm_schedule", "sampling_eps", "replace_val")
)
def topk_topp_and_sample_shmap(
  rng_key,
  mesh,
  logits,
  tpu_sampling_metadata,
  max_k: int,
  num_bins: int | None = None,
  bins_topm_schedule: int | None = None,
  sampling_eps: float = 1e-5,
  replace_val: float = -1e12,
):
  """Combined top-k, top-p filtering, and sampling for vLLM inference.

  Args:
    rng_key: RNG key for sampling.
    logits: Input logits of shape [batch_size, vocab_size].
    tpu_sampling_metadata: Metadata containing top_k, top_p, and temperature.
    max_k: Maximum k value for top-k computation.
    num_bins: Optional number of bins for divide-and-filter algorithm.
    bins_topm_schedule: Optional custom schedule for binned top-m computation.
    sampling_eps: Use greedy token if temperature < eps
    replace_val: Replace padding entries in probabilities with constant

  Returns:
    Sampled token indices.
  """
  batch_axis, vocab_axis = (ShardingAxisName.MLP_DATA, ShardingAxisName.VOCAB)
  def shmap_topk(logits, k):
    topk_logits, topk_idxs = top_bounded_k(
      logits,
      k=tpu_sampling_metadata.top_k,
      replace_val=replace_val,
      max_k=max_k,
      num_bins=num_bins,
      bins_topm_schedule=bins_topm_schedule,
      guarantee_convergence=True,
    )
    # convert idxs to global frame
    i = jax.lax.axis_index(vocab_axis)
    topk_idxs += i * logits.shape[1]
    # all-gather and top-k
    operands = [
      jax.lax.collapse(
        jax.lax.all_gather(x, vocab_axis, axis=1),
      1)
      for x in (topk_logits, topk_idxs)
    ]
    topk_logits, topk_idxs = bitonic_topk(operands, k=max_k)
    topk_logits = jnp.where(
      jax.lax.broadcasted_iota(jnp.int32, topk_logits.shape, 1) < k[:, None],
      topk_logits,
      replace_val,
    )
    return topk_logits, topk_idxs

  topk_logits, topk_idxs = jax.shard_map(
    shmap_topk,
    mesh=mesh,
    in_specs=(P(batch_axis, vocab_axis), P(batch_axis)),
    out_specs=(P(batch_axis), P(batch_axis)),
    check_vma=False,
  )(logits, tpu_sampling_metadata.top_k)

  if rng_key.shape == ():
    rng_key = jax.random.key_data(rng_key)

  def shmap_topp_and_sample(topk_logits, topk_idxs, rng_key, top_p, temperature):
    # Pass global sharded axis offset to maintain jax.random.categorical sampled values
    dim0_offset = jax.lax.axis_index(batch_axis) * topk_logits.shape[0]
    vocab_size=logits.shape[1] * jax.lax.axis_size(vocab_axis)
    return _top_p_and_sample(
      topk_logits,
      topk_idxs,
      rng_key,
      top_p,
      temperature,
      vocab_size=vocab_size,
      replace_val=replace_val,
      sampling_eps=sampling_eps,
      dim0_offset=dim0_offset,
    )

  return jax.shard_map(
    shmap_topp_and_sample,
    mesh=mesh,
    in_specs=(
      P(batch_axis),  # topk_logits
      P(batch_axis),  # topk_idx
      P(),            # rng_key (replicated)
      P(batch_axis),  # top_p
      P(batch_axis),  # temperature
    ),
    out_specs=P(batch_axis),  # output tokens
    check_vma=False,
  )(topk_logits, topk_idxs, rng_key, tpu_sampling_metadata.top_p, tpu_sampling_metadata.temperature)
