# Copyright 2026 Google LLC
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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def local_forward_substitution(A, b):
    """Solves A X = B for unit lower triangular matrix A using forward substitution.

    Args:
      A: A tensor of shape (B, N, N) representing a batch of unit lower triangular
        matrices.
      b: A tensor of shape (B, N, K) representing the right-hand side.

    Returns:
      A tensor of shape (B, N, K) representing the solution X.
    """
    B, N, K = b.shape
    x_list = []
    for i in range(N):
        b_i = b[:, i, :]
        if i == 0:
            x_i = b_i
        else:
            stacked_x = jnp.stack(x_list, axis=1)  # (B, i, K)
            all_prev_A = A[:, i, :i]  # (B, i)
            prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x,
                               axis=1)  # (B, K)
            x_i = b_i - prev_sum  # (B, K) for the row i
        x_list.append(x_i)
    x = jnp.stack(x_list, axis=1)  # (B, N, K)
    return x


def decompose_triangular_matrix_inverse_pallas_kernel(A_ref,
                                                      x_ref,
                                                      *,
                                                      block_size=16):
    A = A_ref[...]
    # Matrix dimension
    B, N, _ = A.shape
    num_blocks = N // block_size

    # same as lower_triangle_solver_pallas_kernel but 2d block wise
    # AX = I, solve for X block wise. X = I - sum(AX_prev)
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))
        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = x_ref[:, :start, :]
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_substitution(local_A, target_b)
        x_ref[..., start:end, :] = x_block


def decompose_triangular_matrix_inverse_pallas(A,
                                               *,
                                               n_block_size=64,
                                               block_size=16):
    """Inverts unit lower triangular matrices using a block-wise approach in Pallas.

    This function solves A X = I for X, where A is a unit lower triangular matrix.
    It uses a block-wise Gaussian elimination approach to improve performance.

    Args:
      A: A tensor of shape (batch_size, chunks, heads, head_dim, head_dim) where
        the last two dimensions represent unit lower triangular matrices.
      n_block_size: The block size for Pallas grid execution.
      block_size: The block size for the block-wise inversion algorithm.

    Returns:
      A tensor of the same shape as A, representing the inverse of A.
    """

    # Squash all the leading dimensions
    A_reshaped = A.reshape(-1, *A.shape[-2:])
    A_shape = A_reshaped.shape
    x_shape = A_shape

    N = A_reshaped.shape[0]
    grid_size = pl.cdiv(N, n_block_size)

    head_dim = A_shape[-1]
    kernel = functools.partial(
        decompose_triangular_matrix_inverse_pallas_kernel,
        block_size=block_size)
    x = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x_shape, A.dtype),
        grid=(grid_size, ),
        in_specs=[
            pl.BlockSpec((n_block_size, head_dim, head_dim), lambda idx:
                         (idx, 0, 0)),
        ],
        out_specs=pl.BlockSpec((n_block_size, head_dim, head_dim), lambda idx:
                               (idx, 0, 0)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=67108864),
        name=
        f"decompose_triangular_matrix_inverse_pallas_kernel_{n_block_size}_{block_size}",
    )(A_reshaped)

    return x.reshape(A.shape)
