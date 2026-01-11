import pytest
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from tpu_inference.kernels.sampling.divide_and_filter_topk import topk as pallas_topk
from tpu_inference.kernels.sampling.utils import is_cpu_platform, NUM_LANES
from tests.kernels.sampling.test_utils import verify_topk_output


@pytest.mark.parametrize(
  "shape",
  [
    (8, 128),
    (16, 256),
    (13, 167),
    (256, 256),
    (173, 195),
    (16, 16384),
    (13, 11571),
  ],
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="Divide and filter top-k tests require TPU/GPU - CPU uses interpret mode which is slow",
)
def test_divide_and_filter_topk(shape, dtype):
  """Test divide and filter top-k implementation with exact match validation."""
  k = min(137, shape[1] // 2)

  # Generate test data
  key = jax.random.key(0)
  if jnp.isdtype(dtype, "real floating"):
    logits = jax.random.normal(key, shape, dtype=dtype)
  else:
    logits = jax.random.randint(key, shape, 0, 1000, dtype=dtype)

  # Run divide and filter top-k implementation
  outputs = pallas_topk(
    logits,
    k=k,
    interpret=is_cpu_platform(),
    num_bins=128 if shape[1] <= 128 else 256,
  )

  # Validate results using verify_topk_output (axis=1 is default)
  validation = verify_topk_output(logits, outputs, axis=1)

  assert validation.all(), (
    f"Divide and filter top-k validation failed for shape {shape}, dtype {dtype}: "
    f"{int(validation.sum())}/{shape[0]} rows passed"
  )


# tests the merging unconverged bins logic
@pytest.mark.parametrize("topk_distribution", ["random", "worst_case"])
@pytest.mark.parametrize("shape", [(14, 1957), (16, 16384), (13, 11571)])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.parametrize("k", [17, 128, 157])
@pytest.mark.parametrize("num_bins", [128, 384, 1024])
@pytest.mark.parametrize("schedule", [(1,), (2,), (1, 4), (4, 5), None])
@pytest.mark.skipif(
  is_cpu_platform(),
  reason="Divide and filter top-k tests require TPU/GPU - CPU uses interpret mode which is slow",
)
def test_divide_and_filter_topk_worst_case_values(
  topk_distribution, shape, dtype, k, num_bins, schedule
):
  """Test divide and filter top-k implementation with exact match validation."""
  if schedule is not None and any(m * num_bins < k for m in schedule):
    # will try to top-k less than k values
    pytest.skip("Unsupported setup")
  max_m = max(schedule) if schedule is not None else 1
  if pl.cdiv(k, max_m) > min(num_bins, NUM_LANES):
    pytest.skip("Unsupported setup")

  # Generate test data
  key = jax.random.key(0)
  if jnp.isdtype(dtype, "real floating"):
    logits = jax.random.normal(key, shape, dtype=dtype)
  else:
    logits = jax.random.randint(key, shape, 0, 1000, dtype=dtype)

  # organize that a single bin contains the largest values
  if topk_distribution == "worst_case":
    logits = logits.at[:, num_bins - 17 :: num_bins].add(1000)

  # Run divide and filter top-k implementation
  outputs = pallas_topk(
    logits,
    k=k,
    interpret=is_cpu_platform(),
    num_bins=num_bins,
    bins_topm_schedule=schedule,
  )

  # Validate results using verify_topk_output (axis=1 is default)
  validation = verify_topk_output(logits, outputs, axis=1)

  assert validation.all(), (
    f"Divide and filter top-k validation failed for shape {shape}, dtype {dtype}: "
    f"{int(validation.sum())}/{shape[0]} rows passed"
  )
