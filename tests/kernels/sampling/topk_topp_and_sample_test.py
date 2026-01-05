import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.sampling import topk_topp_and_sample as pallas_sample
from tpu_inference.layers.jax.sample.sampling import _sample as tpu_inference_sample
from tpu_inference.layers.jax.sample.sampling_metadata import TPUSupportedSamplingMetadata
from tpu_inference.kernels.sampling.utils import is_cpu_platform
from tests.kernels.sampling.test_utils import uniquely_define_topk


@pytest.mark.parametrize(
  "shape",
  [
    (16, 16384),
    (13, 11792),
    (256, 2048),
    (256, 8192),
    (279, 3570),
    (279, 7593),
  ],
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("case", ["random", "worst_case"])
@pytest.mark.parametrize("max_k", [5, 17, 64, 128, 137])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_sample(shape, dtype, case, max_k, seed):
  """Test topk_topp_and_sample implementation against layers reference.

  Tests both random and worst-case logits distributions.
  Validates that kernel implementation matches layers sampling behavior.
  """
  num_tokens, vocab_size = shape

  # Split main seed into all needed keys
  key = jax.random.PRNGKey(seed)
  key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(
    key, 6
  )

  # Create sampling metadata with varying top_k, top_p, and temperature
  # We use varying k and temperatures of 10**normal(0,1) so that sometimes random gumbel noise dominates,
  # sometimes logits values dominates. Similarly, varying p threshold in top-p
  tpu_sampling_metadata = TPUSupportedSamplingMetadata(
    top_k=jax.random.randint(topk_key, (num_tokens,), 1, max_k+1, dtype=jnp.int32),
    top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
    temperature=10
    ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
    do_sampling=True,
    logprobs=False,
    use_pallas_kernel=True,
  )

  # Generate logits based on case
  logits = jax.random.normal(logits_key, shape).astype(dtype)
  if case == "worst_case":
    logits = logits.at[:, 13::256].add(100)

  logits = jax.vmap(uniquely_define_topk)(logits, tpu_sampling_metadata.top_k)

  # Run both implementations
  pallas_result = pallas_sample(
    sample_key, logits, tpu_sampling_metadata, max_k=max_k
  )

  tpu_inference_result = tpu_inference_sample(sample_key, logits, tpu_sampling_metadata)

  # Compare results - expect exact match
  # barring f32 summation order errors affecting top-p which are rare
  np.testing.assert_array_equal(
    pallas_result,
    tpu_inference_result,
    err_msg=f"Kernel sampling should exactly match layers sampling for "
    f"shape={shape}, dtype={dtype}, case={case}, seed={seed}",
  )
