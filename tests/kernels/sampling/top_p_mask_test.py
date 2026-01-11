import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.sampling.top_p_and_sample import top_p_mask as pallas_top_p_mask
from tpu_inference.layers.common.binary_search import (
  topp_mask as tpu_inference_top_p_mask,
)


@pytest.mark.parametrize(
  "shape",
  [
    (8, 128),
    (16, 256),
    (13, 167),
    (21, 128),
    (256, 128),
    (137, 17),
    (137, 193),
  ],
)
@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("p_threshold", [0.001, 0.1, 0.5, 0.999, 1.0, None])
def test_top_p_mask(shape, seed, p_threshold):
  """Test pallas_top_p_mask for exact match against tpu_inference_top_p_mask.

  Strategy:
  1. Generate random p values per batch element (or use fixed p_threshold)
  2. Sort input and get argsort indices (axis=1)
  3. Apply pallas_top_p_mask to sorted input
  4. Reverse argsort to return to original order
  5. Apply tpu_inference_top_p_mask to unsorted input (per-batch element)
  6. Compare results (should match exactly in f32)
  """
  key = jax.random.key(seed)
  key, logits_key, p_key = jax.random.split(key, 3)

  # Generate random logits (f32)
  logits = jax.random.normal(logits_key, shape, dtype=jnp.float32)

  # Generate p values: None means random uniform, otherwise use fixed threshold
  if p_threshold is None:
    p_array = jax.random.uniform(p_key, shape[:1], dtype=jnp.float32)
  else:
    p_array = jnp.full(shape[:1], p_threshold, dtype=jnp.float32)

  replace_val = -1e12

  # Sort logits in descending order and get indices (axis=1)
  sort_indices = jnp.argsort(logits, axis=1, descending=True)
  sorted_logits = jnp.take_along_axis(logits, sort_indices, axis=1)

  # Transpose for pallas_top_p_mask (expects axis=0)
  sorted_logits_transposed = sorted_logits.T

  # Apply pallas_top_p_mask (axis=0 on transposed logits)
  result_pallas_top_p_mask_sorted = pallas_top_p_mask(
    topk_logits=sorted_logits_transposed,
    p=p_array,
    replace_val=replace_val,
    axis=0,
    no_pallas_code=True,
  )

  # Transpose back to (batch, vocab)
  result_pallas_top_p_mask_sorted = result_pallas_top_p_mask_sorted.T

  # Reverse the argsort to get back to original order
  # Create inverse permutation
  inverse_sort_indices = jnp.argsort(sort_indices, axis=1)
  result_pallas_original_order = jnp.take_along_axis(
    result_pallas_top_p_mask_sorted, inverse_sort_indices, axis=1
  )

  # Apply tpu_inference_top_p_mask (uses unsorted logits)
  result_tpu_inference = tpu_inference_top_p_mask(
    logits, p_array, replace_val
  )

  # Compare results in original order
  # (should match exactly barring summing many values in f32 rounding errors
  # which should be very rare)
  np.testing.assert_array_equal(
    result_pallas_original_order,
    result_tpu_inference,
    err_msg=f"pallas_top_p_mask should match tpu_inference_top_p_mask for shape={shape}, seed={seed}, p={p_threshold}",
  )


if __name__ == "__main__":
  print("Running top_p_mask tests...")
  shapes = [(8, 128), (16, 256), (13, 167), (32, 128)]
  seeds = [42, 123, 456]
  p_thresholds = [0.001, 0.1, 0.5, 0.999, 1.0, None]

  for shape in shapes:
    for seed in seeds:
      for p_threshold in p_thresholds:
        print(
          f"Testing shape={shape}, seed={seed}, p_threshold={p_threshold}..."
        )
        test_top_p_mask(shape, seed, p_threshold)
        print("  ✓ Passed")

  print("\nAll top_p_mask tests passed!")