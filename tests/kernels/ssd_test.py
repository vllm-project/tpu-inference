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
"""Correctness tests for Selective State Space (SSD) scan hybrid JAX/Pallas kernel."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.layers.common.ssd_scan import ssd_minimal_discrete_hybrid

# Ensure highest precision for matmuls on TPU to match float32 references.
jax.config.update("jax_default_matmul_precision", "highest")


def _segsum_jax(x):
  T = x.shape[-1]
  x_expanded = jnp.expand_dims(x, axis=-1)
  x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

  mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
  x_masked = jnp.where(mask, x_repeated, 0.0)

  x_segsum = jnp.cumsum(x_masked, axis=-2)

  mask_diag = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
  x_segsum_masked = jnp.where(mask_diag, x_segsum, -jnp.inf)
  return x_segsum_masked


def ssd_minimal_discrete_jax(
    X_c,
    A_c,
    B_c,
    C_c,
    last_chunk_indices,
    initial_states=None,
    reset_mask=None,
):
  n_chunks, n_heads, block_len, d_head = X_c.shape
  d_state = B_c.shape[-1]

  if initial_states is None:
    initial_states = jnp.zeros(
        (n_chunks, n_heads, d_head, d_state), dtype=X_c.dtype
    )
  if reset_mask is None:
    reset_mask = jnp.zeros((n_chunks,), dtype=bool)
    reset_mask = reset_mask.at[0].set(True)

  A_cumsum = jnp.cumsum(A_c, axis=-1)

  L = jnp.exp(_segsum_jax(A_c))
  Y_diag = jnp.einsum("chln,chsn,chls,chsp->chlp", C_c, B_c, L, X_c)

  decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
  states_c = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states)

  decay_chunk = jnp.exp(A_cumsum[..., -1])

  def scan_body(carry, args):
    states_c_c, decay_chunk_c, init_state_c, reset_c = args
    incoming_state = jnp.where(reset_c, init_state_c, carry)
    decay_exp = decay_chunk_c[:, None, None]
    next_state = incoming_state * decay_exp + states_c_c
    return next_state, (incoming_state, next_state)

  scan_args = (states_c, decay_chunk, initial_states, reset_mask)
  init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=X_c.dtype)
  _, (states_incoming, states_outgoing) = jax.lax.scan(
      scan_body, init_carry, scan_args
  )

  state_decay_out = jnp.exp(A_cumsum)
  Y_off = jnp.einsum(
      "chln,chpn,chl->chlp", C_c, states_incoming, state_decay_out
  )

  Y = Y_diag + Y_off
  final_states = states_outgoing[last_chunk_indices]
  return Y, final_states


class SSDScanKernelTest(jtu.JaxTestCase):

  @parameterized.parameters(
      # (batch, seqlen, block_len, n_heads, d_head, d_state)
      (2, 128, 32, 4, 16, 8),
      (4, 256, 64, 8, 32, 16),
  )
  def test_ssd_scan_correctness(
      self, batch, seqlen, block_len, n_heads, d_head, d_state
  ):
    n_chunks_per_seq = seqlen // block_len
    n_chunks = batch * n_chunks_per_seq

    # Generate random inputs in JAX/Pallas v4 layout: (n_chunks, n_heads, block_len, dim)
    rng = jax.random.key(42)
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)

    X_c = jax.random.normal(k1, (n_chunks, n_heads, block_len, d_head))
    # A_c (decay) needs to be negative
    A_c = -jnp.exp(jax.random.normal(k2, (n_chunks, n_heads, block_len)))
    B_c = jax.random.normal(k3, (n_chunks, n_heads, block_len, d_state))
    C_c = jax.random.normal(k4, (n_chunks, n_heads, block_len, d_state))

    # Initial state: shape (batch, n_heads, d_head, d_state)
    init_states_per_batch = jax.random.normal(
        k5, (batch, n_heads, d_head, d_state)
    )

    # Map initial states and reset mask to v4 layout: (n_chunks, n_heads, d_head, d_state)
    initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state))
    reset_mask = np.zeros((n_chunks,), dtype=bool)

    for b in range(batch):
      initial_states = initial_states.at[b * n_chunks_per_seq].set(
          init_states_per_batch[b]
      )
      reset_mask[b * n_chunks_per_seq] = True

    reset_mask = jnp.array(reset_mask)

    last_chunk_indices = jnp.array(
        [b * n_chunks_per_seq + n_chunks_per_seq - 1 for b in range(batch)],
        dtype=jnp.int32,
    )

    # Run JAX Reference
    y_jax, final_state_jax = ssd_minimal_discrete_jax(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )

    # Run Hybrid Pallas Implementation
    y_hybrid, final_state_hybrid = ssd_minimal_discrete_hybrid(
        X_c, A_c, B_c, C_c, last_chunk_indices, initial_states, reset_mask
    )

    # Compare outputs
    self.assertAllClose(y_hybrid, y_jax, rtol=1e-4, atol=1e-4)
    self.assertAllClose(final_state_hybrid, final_state_jax, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
