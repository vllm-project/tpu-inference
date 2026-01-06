"""
vLLM top-k top-p sampling, using two pallas functions
"""

import functools
import jax
from tpu_inference.kernels.sampling.top_p_and_sample import top_p_and_sample
from tpu_inference.kernels.sampling.divide_and_filter_topk import top_bounded_k


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
