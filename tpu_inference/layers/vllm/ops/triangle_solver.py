import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


def lower_triangle_solver_ref(A, b):
  return jax.scipy.linalg.solve_triangular(A, b, lower=True, unit_diagonal=True)


def lower_triangle_solver_matmul(A, b):
  return jnp.matmul(jnp.linalg.inv(A), b, precision=jax.lax.Precision.HIGHEST)


def lower_triangle_solver_pallas_kernel(A_ref, b_ref, x_ref):
  A = A_ref[...]
  b = b_ref[...]

  # Matrix dimension
  N = A.shape[1]

  x_list = []

  # Simply do gaussian elimination here
  # Note that all diagnoal is 1 so the x_i is simply:
  # x_i = b_i - (a_i_0 * x_0 + a_i_1 * x_1 ... x_i_x_i-1 * x_i-1)
  for i in range(N):
    if i == 0:
      x_i = b[:, 0, :]
    else:
      stacked_x = jnp.stack(x_list, axis=1)  # (B, i, K)
      all_prev_A = A[:, i, :i]  # (B, i)
      prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)  # (B, K)
      x_i = b[:, i, :] - prev_sum  # (B, K) for the row i

    x_list.append(x_i)

  x = jnp.stack(x_list, axis=1)  # (B, N, K)

  x_ref[...] = x


def lower_triangle_solver_pallas(A, b, *, n_block_size=64):
  # Just solve Ax = b
  # A: (batch_size, chunks, heads, head_dim, head_dim)
  # b: (batch_size, chunks, heads, head_dim, 1)
  # x: (batch_size, chunks, heads, head_dim, 1)
  # n_block_size is block size for chunking up bt_size * chunks * heads

  # Squaysh all the leading dimensions
  A_reshaped = A.reshape(-1, *A.shape[-2:])
  b_reshaped = b.reshape(-1, *b.shape[-2:])

  # A must be square matrix
  A_shape = A_reshaped.shape
  x_shape = b_reshaped.shape
  b_shape = b_reshaped.shape

  N = A_reshaped.shape[0]
  grid_size = pl.cdiv(N, n_block_size)
  x = pl.pallas_call(
      lower_triangle_solver_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x_shape, b.dtype),
      grid=(grid_size,),
      in_specs=[
          pl.BlockSpec(
              (n_block_size, A_shape[-2], A_shape[-1]), lambda idx: (idx, 0, 0)
          ),
          pl.BlockSpec(
              (n_block_size, b_shape[-2], b_shape[-1]), lambda idx: (idx, 0, 0)
          ),
      ],
      out_specs=pl.BlockSpec(
          (n_block_size, x_shape[-2], x_shape[-1]), lambda idx: (idx, 0, 0)
      ),
      name="lower_triangle_solver_kernel",
  )(A_reshaped, b_reshaped)

  # Restore original shape
  return x.reshape(b.shape)


def unit_lower_triangular_matrix_inverse_pallas_kernel(A_ref, x_ref):
  A = A_ref[...]
  # Matrix dimension
  N = A.shape[1]

  x_list = []

  for i in range(N):
    e_i = (jnp.arange(N) == i).astype(A.dtype)
    x_i = jnp.broadcast_to(e_i, (A.shape[0], N))

    if i > 0:
      stacked_x = jnp.stack(x_list, axis=1)  # (B, i, N)
      all_prev_A = A[:, i, :i]  # (B, i)
      prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)  # (B, N)
      x_i = x_i - prev_sum

    x_list.append(x_i)

  x = jnp.stack(x_list, axis=1)  # (B, N, N)
  x_ref[...] = x


def unit_lower_triangular_matrix_inverse_pallas(A, *, n_block_size=64):
  # Solve AX = I
  # A: (batch_size, chunks, heads, head_dim, head_dim)
  # x: (batch_size, chunks, heads, head_dim, head_dim)

  # Squash all the leading dimensions
  A_reshaped = A.reshape(-1, *A.shape[-2:])
  A_shape = A_reshaped.shape
  x_shape = A_shape

  N = A_reshaped.shape[0]
  grid_size = pl.cdiv(N, n_block_size)

  head_dim = A_shape[-1]

  x = pl.pallas_call(
      unit_lower_triangular_matrix_inverse_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x_shape, A.dtype),
      grid=(grid_size,),
      in_specs=[
          pl.BlockSpec(
              (n_block_size, head_dim, head_dim), lambda idx: (idx, 0, 0)
          ),
      ],
      out_specs=pl.BlockSpec(
          (n_block_size, head_dim, head_dim), lambda idx: (idx, 0, 0)
      ),
      name="unit_lower_triangular_matrix_inverse_pallas_kernel",
  )(A_reshaped)

  # Restore original shape
  return x.reshape(A.shape)

