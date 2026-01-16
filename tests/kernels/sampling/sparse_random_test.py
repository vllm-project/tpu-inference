import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.sampling.sparse_random import (
  sparse_random_uniform as pallas_sparse_random_uniform,
  sparse_random_categorical as pallas_sparse_random_categorical,
)


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize(
  "minval,maxval", [(0.0, 1.0), (-1.0, 1.0), (5.0, 10.0)]
)
def test_sparse_random_uniform(seed, minval, maxval):
  """Test sparse_random_uniform by comparing against indexed dense array."""
  key = jax.random.key(seed)
  key, subkey1, subkey2 = jax.random.split(key, 3)

  # Generate dense random array
  dense_shape = (16, 256)
  dense_uniform = jax.random.uniform(
    key, shape=dense_shape, dtype=jnp.float32, minval=minval, maxval=maxval
  )

  # Generate random sparse indices
  sparse_shape = (8, 128)
  indices_0 = jax.random.randint(subkey1, sparse_shape, 0, dense_shape[0])
  indices_1 = jax.random.randint(subkey2, sparse_shape, 0, dense_shape[1])

  # Generate sparse random values
  sparse_uniform = pallas_sparse_random_uniform(
    key,
    [indices_0, indices_1],
    dim1_size=dense_shape[1],
    dtype=jnp.float32,
    minval=minval,
    maxval=maxval,
  )

  # Index into dense array at the sparse positions
  expected = dense_uniform[indices_0, indices_1]

  # Should match exactly (or within FP32 epsilon for scaled ranges)
  if minval == 0.0 and maxval == 1.0:
    # For [0, 1) range, should be exact
    np.testing.assert_array_equal(
      sparse_uniform,
      expected,
      err_msg="sparse_random_uniform should match indexed dense array exactly",
    )
  else:
    # For scaled ranges, allow 1 ULP difference due to FP arithmetic
    np.testing.assert_allclose(
      sparse_uniform,
      expected,
      rtol=0,
      atol=1e-6,
      err_msg="sparse_random_uniform should match indexed dense array",
    )


@pytest.mark.parametrize("seed", [789, 321, 654])
@pytest.mark.parametrize("axis", [0, 1])
def test_sparse_random_categorical(seed, axis):
  """Test sparse_random_categorical by comparing against masked dense array."""
  key = jax.random.key(seed)
  key, logits_key, indices_key = jax.random.split(key, 3)

  batch_dim, dense_dim, sparse_dim = 16, 256, 128

  # Always work with (batch, dense) shape, transpose at end if needed
  sparse_logits = jax.random.normal(logits_key, (batch_dim, sparse_dim))

  dense_iota = jax.lax.broadcasted_iota(jnp.int32, (batch_dim, dense_dim), 1)
  dense_choices = jax.vmap(
    lambda k, iota: jax.random.choice(
      k, iota, shape=(sparse_dim,), replace=False
    )
  )(jax.random.split(indices_key, batch_dim), dense_iota)

  indices_0 = jax.lax.broadcasted_iota(jnp.int32, (batch_dim, sparse_dim), 0)
  indices_1 = dense_choices

  dense_masked = (
    jnp.full((batch_dim, dense_dim), -1e12)
    .at[indices_0, indices_1]
    .set(sparse_logits)
  )

  # Transpose for axis=0 and swap indices
  if axis == 0:
    sparse_logits, indices_0, indices_1, dense_masked = (
      x.T for x in (sparse_logits, indices_1, indices_0, dense_masked)
    )

  dense_result = jax.random.categorical(key, dense_masked, axis=axis)
  sparse_result = pallas_sparse_random_categorical(
    key,
    sparse_logits,
    [indices_0, indices_1],
    dim1_size=dense_masked.shape[1],
    axis=axis,
  )[axis]

  np.testing.assert_array_equal(
    sparse_result,
    dense_result,
    err_msg=f"sparse_random_categorical should match dense categorical for axis={axis}",
  )


if __name__ == "__main__":
  print("Running sparse_random_uniform tests...")
  for seed in [42, 123, 456]:
    for minval, maxval in [(0.0, 1.0), (-1.0, 1.0), (5.0, 10.0)]:
      test_sparse_random_uniform(seed, minval, maxval)
  print("sparse_random_uniform tests passed!")

  print("\nRunning sparse_random_categorical tests...")
  for seed in [789, 321, 654]:
    for axis in [0, 1]:
      print(f"  Testing seed={seed}, axis={axis}...")
      test_sparse_random_categorical(seed, axis)
  print("sparse_random_categorical tests passed!")

  print("\nAll tests passed!")
