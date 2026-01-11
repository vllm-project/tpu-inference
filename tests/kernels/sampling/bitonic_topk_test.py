import functools
import pytest
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tpu_inference.kernels.sampling.bitonic_topk import (
  bitonic_topk_arrays as pallas_bitonic_topk_arrays,
  max_arrays as pallas_max_arrays,
)
from tpu_inference.kernels.sampling.utils import is_cpu_platform
from tests.kernels.sampling.test_utils import verify_topk_output


@pytest.mark.parametrize(
  "shape",
  [
    (8, 64),
    (17, 37),
    (8, 128),
    (16, 256),
    (13, 167),
    (256, 256),
    (173, 195),
    (16, 16384),
    (13, 11571),
  ],
)
@pytest.mark.parametrize("k", [1, 5, 23, 128, 137])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
@pytest.mark.parametrize("axis", [1])
def test_bitonic_topk(shape, dtype, axis, k):
  """Test bitonic_topk using bitonic_topk_arrays wrapped in pallas."""
  interpret = is_cpu_platform()
  if interpret and (shape[1] > 256):
    pytest.skip("Test too large for CPU, as compilation is very slow")
  key = jax.random.PRNGKey(axis)

  # Generate test data based on dtype
  if dtype == jnp.float32:
    arr = jax.random.normal(key, shape).astype(dtype)
  else:
    arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

  k = min(k, shape[axis])

  # Create indices array for argsort
  indices = jax.lax.broadcasted_iota(jnp.int32, shape, axis)
  out_shape = list(shape)
  out_shape[axis] = k

  def topk_refs(values_ref, indices_ref, out_values_ref, out_indices_ref):
    """Top-k refs kernel using bitonic_topk_arrays."""
    result_values, result_indices = pallas_bitonic_topk_arrays(
      [values_ref[...], indices_ref[...]], k=k, axis=axis
    )
    out_values_ref[...] = result_values
    out_indices_ref[...] = result_indices

  @functools.partial(jax.jit, static_argnames=("interpret",))
  def topk_pallas(values, indices, interpret=False):
    call_kwargs = {
      "out_shape": [
        jax.ShapeDtypeStruct(out_shape, values.dtype),
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
      ],
      "interpret": interpret,
    }
    if not interpret:
      call_kwargs["compiler_params"] = pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27))
    return pl.pallas_call(topk_refs, **call_kwargs)(values, indices)

  result_values, result_indices = topk_pallas(arr, indices, interpret=interpret)

  valid = verify_topk_output(arr, (result_values, result_indices), axis=axis)
  assert valid.all(), (
    f"Top-k validation failed for shape {shape}, dtype {dtype}, axis {axis}"
  )


@pytest.mark.parametrize(
  "shape",
  [(8, 128), (16, 256), (128, 8), (256, 16), (256, 256), (173, 195), (8, 1024)],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("axis", [0, 1])
def test_top1_pallas(shape, dtype, axis):
  """Test top1 wrapped in pallas kernel for both axes."""
  interpret = is_cpu_platform()
  key = jax.random.PRNGKey(1 + axis)

  # Generate test data based on dtype
  if dtype == jnp.float32:
    arr = jax.random.normal(key, shape).astype(dtype)
  else:
    arr = jax.random.randint(key, shape, 0, 1000).astype(dtype)

  indices = jax.lax.broadcasted_iota(jnp.int32, shape, axis)
  out_shape_1d = (shape[1 - axis],)

  def top1_refs(values_ref, indices_ref, out_values_ref, out_indices_ref):
    """Top1 refs kernel."""
    result_values, result_indices = pallas_max_arrays(
      [values_ref[...], indices_ref[...]], axis=axis
    )
    out_values_ref[...] = result_values
    out_indices_ref[...] = result_indices

  @functools.partial(jax.jit, static_argnames=("interpret",))
  def top1_pallas(values, indices, interpret=False):
    call_kwargs = {
      "out_shape": [
        jax.ShapeDtypeStruct(out_shape_1d, values.dtype),
        jax.ShapeDtypeStruct(out_shape_1d, jnp.int32),
      ],
      "interpret": interpret,
    }
    if not interpret:
      call_kwargs["compiler_params"] = pltpu.CompilerParams(vmem_limit_bytes=int(0.9 * 2**27))
    return pl.pallas_call(top1_refs, **call_kwargs)(values, indices)

  outputs = top1_pallas(arr, indices, interpret=interpret)

  # Reshape 1D outputs to 2D for verify_topk_output
  outputs = tuple(jnp.expand_dims(x, axis=axis) for x in outputs)
  valid = verify_topk_output(arr, outputs, axis=axis)
  assert valid.all(), (
    f"Top1 validation failed for shape {shape}, dtype {dtype}, axis={axis}"
  )
