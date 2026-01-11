import pytest
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from tpu_inference.kernels.sampling.cumsum import cumsum as pallas_cumsum
from tpu_inference.kernels.sampling.utils import is_cpu_platform


@pytest.mark.parametrize(
  "shape", [(8, 128), (16, 256), (128, 8), (256, 16), (13, 167)]
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum(shape, axis, dtype, reverse):
  """Test cumsum for various shapes, axes, dtypes, and reverse parameter."""
  key = jax.random.key(42)

  if dtype == jnp.float32:
    x = jax.random.normal(key, shape, dtype=dtype)
  else:
    x = jax.random.randint(key, shape, 0, 100, dtype=dtype)

  # Reference using lax.cumsum
  expected = lax.cumsum(x, axis=axis, reverse=reverse)

  interpret = is_cpu_platform()
  actual = pallas_cumsum(x, axis=axis, reverse=reverse, interpret=interpret)

  # Use close match for float32, exact for int32
  if dtype == jnp.float32:
    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=1e-5)
  else:
    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  test_cumsum()
