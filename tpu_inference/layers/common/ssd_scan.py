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
"""JAX/Pallas hybrid implementation of Selective State Space (SSD) scan."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Ensure highest precision for matmuls on TPU to match float32 references.
jax.config.update("jax_default_matmul_precision", "highest")

# ==============================================================================
# Pallas TPU Kernels (Native head batching, grid over chunks only)
# ==============================================================================

def chunk_state_pallas(B_c, decay_states, X_c):
    """Pallas kernel for Step 2 of SSD scan (chunk state computation).

    Computes:
      states = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states)

    This kernel is gridded over (n_chunks, n_heads), so each invocation
    processes a single chunk for a single head. This keeps vectors inside
    the kernel 2D, avoiding compiler bugs in batch matmul lowering.

    Args:
      B_c: (n_chunks, n_heads, block_len, d_state)
      decay_states: (n_chunks, n_heads, block_len, 1) - expanded decay states
        (exp(A_cumsum_last - A_cumsum))
      X_c: (n_chunks, n_heads, block_len, d_head)

    Returns:
      states: (n_chunks, n_heads, d_head, d_state)
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]
    
    def kernel(B_ref, decay_ref, X_ref, out_ref):
        B_val = jnp.squeeze(B_ref[...], axis=(0, 1)).astype(jnp.float32) # (block_len, d_state)
        decay_val = jnp.squeeze(decay_ref[...], axis=(0, 1)).astype(jnp.float32) # (block_len, 1)
        X_val = jnp.squeeze(X_ref[...], axis=(0, 1)).astype(jnp.float32) # (block_len, d_head)
        
        # Scale X by decay
        X_scaled = X_val * decay_val # (block_len, d_head)
        X_scaled_T = jnp.transpose(X_scaled, (1, 0)) # (d_head, block_len)
        
        # Matmul: (d_head, block_len) @ (block_len, d_state) -> (d_head, d_state) in float32
        out = jax.lax.dot_general(
            X_scaled_T,
            B_val,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = out.astype(X_c.dtype) # Cast back to original bf16
        
        out_ref[...] = jnp.expand_dims(out, axis=(0, 1))

    grid = (n_chunks, n_heads)
    
    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, d_state)
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, 1)
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, d_head)
        )
    ]
    
    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h: (c, h, 0, 0),
        block_shape=(1, 1, d_head, d_state)
    )
    
    out_shape = (n_chunks, n_heads, d_head, d_state)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, X_c.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True
        ),
        name="ssd_chunk_state_pallas"
    )(B_c, decay_states, X_c)


def chunk_scan_pallas(C_c, states, decay_states):
    """Pallas kernel for Step 5 of SSD scan (state-to-output conversion).

    Computes:
      Y_off = jnp.einsum('chln,chpn,chl->chlp', C_c, states, state_decay_out)

    This kernel is gridded over (n_chunks, n_heads), processing each head
    individually.

    Args:
      C_c: (n_chunks, n_heads, block_len, d_state)
      states: (n_chunks, n_heads, d_head, d_state)
      decay_states: (n_chunks, n_heads, block_len, 1) - expanded state decay
        output (exp(A_cumsum))

    Returns:
      Y_off: (n_chunks, n_heads, block_len, d_head)
    """
    n_chunks, n_heads, block_len, d_state = C_c.shape
    d_head = states.shape[-2]
    
    def kernel(C_ref, states_ref, decay_ref, out_ref):
        C_val = jnp.squeeze(C_ref[...], axis=(0, 1)).astype(jnp.float32) # (block_len, d_state)
        states_val = jnp.squeeze(states_ref[...], axis=(0, 1)).astype(jnp.float32) # (d_head, d_state)
        decay_val = jnp.squeeze(decay_ref[...], axis=(0, 1)).astype(jnp.float32) # (block_len, 1)
        
        # Transpose states for contraction over d_state
        states_T = jnp.transpose(states_val, (1, 0)) # (d_state, d_head)
        
        # Matmul: (block_len, d_state) @ (d_state, d_head) -> (block_len, d_head) in float32
        prod = jax.lax.dot_general(
            C_val,
            states_T,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        
        # Scale
        out = prod * decay_val
        out = out.astype(C_c.dtype) # Cast back to original bf16
        
        out_ref[...] = jnp.expand_dims(out, axis=(0, 1))

    grid = (n_chunks, n_heads)
    
    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, d_state)
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, d_head, d_state)
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h: (c, h, 0, 0),
            block_shape=(1, 1, block_len, 1)
        )
    ]
    
    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h: (c, h, 0, 0),
        block_shape=(1, 1, block_len, d_head)
    )
    
    out_shape = (n_chunks, n_heads, block_len, d_head)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, C_c.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            disable_bounds_checks=True
        ),
        name="ssd_chunk_scan_pallas"
    )(C_c, states, decay_states)


# ==============================================================================
# Helper Functions
# ==============================================================================

def _segsum_jax(x):
    """Computes JAX cumsum of segment sums."""
    T = x.shape[-1]
    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))
    
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask, x_repeated, 0.0)
    
    x_segsum = jnp.cumsum(x_masked, axis=-2)
    
    mask_diag = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum_masked = jnp.where(mask_diag, x_segsum, -jnp.inf)
    return x_segsum_masked


# ==============================================================================
# Core Scan Operator
# ==============================================================================

def ssd_minimal_discrete_hybrid(
    X_c,
    A_c,
    B_c,
    C_c,
    last_chunk_indices,
    initial_states=None,
    reset_mask=None,
):
    """Hybrid JAX/Pallas implementation of the Selective State Space (SSD) scan.

    This function implements the discrete SSD scan using Pallas TPU kernels
    for memory-intensive step-2 and step-5 computations, and native JAX
    for cumulative sums and sequential inter-chunk recurrence.

    This version assumes the packed library layout: (n_chunks, n_heads,
    block_len, dim).

    Args:
      X_c: (n_chunks, n_heads, block_len, d_head) - Input tensor
      A_c: (n_chunks, n_heads, block_len) - Log decay parameter (dt * A)
      B_c: (n_chunks, n_heads, block_len, d_state) - Input projection
      C_c: (n_chunks, n_heads, block_len, d_state) - Output projection
      last_chunk_indices: (num_seqs,) - Map each sequence to its last chunk
      initial_states: (n_chunks, n_heads, d_head, d_state) - Optional initial
        states (only read at sequence starts)
      reset_mask: (n_chunks,) - Boolean mask indicating where new sequences
        start (aligned to chunks)

    Returns:
      Y: (n_chunks, n_heads, block_len, d_head) - Output tensor
      final_states: (num_seqs, n_heads, d_head, d_state) - Recurrent states
        collected at sequence ends
    """
    n_chunks, n_heads, block_len, d_head = X_c.shape
    d_state = B_c.shape[-1]
    
    if initial_states is None:
        initial_states = jnp.zeros((n_chunks, n_heads, d_head, d_state), dtype=X_c.dtype)
    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool)
        reset_mask = reset_mask.at[0].set(True)

    A_cumsum = jnp.cumsum(A_c, axis=-1)

    # 1. Intra-chunk diagonal output (Y_diag)
    L = jnp.exp(_segsum_jax(A_c))
    Y_diag = jnp.einsum("chln,chsn,chls,chsp->chlp", C_c, B_c, L, X_c)

    # 2. Intra-chunk state generation (Pallas)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[..., None] # (n_chunks, n_heads, block_len, 1)
    states_c = chunk_state_pallas(B_c, decay_states, X_c)

    # 3. Inter-chunk recurrence (JAX Scan)
    decay_chunk = jnp.exp(A_cumsum[..., -1])
    
    def scan_body(carry, args):
        states_c_c, decay_chunk_c, init_state_c, reset_c = args
        incoming_state = jnp.where(reset_c, init_state_c, carry)
        decay_exp = decay_chunk_c[:, None, None]
        next_state = incoming_state * decay_exp + states_c_c
        return next_state, (incoming_state, next_state)

    scan_args = (states_c, decay_chunk, initial_states, reset_mask)
    init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=jnp.float32)
    final_carry, (states_incoming, states_outgoing) = jax.lax.scan(scan_body, init_carry, scan_args)

    # 4. State to output conversion (Pallas)
    state_decay_out = jnp.exp(A_cumsum)[..., None] # (n_chunks, n_heads, block_len, 1)
    Y_off = chunk_scan_pallas(C_c, states_incoming, state_decay_out)

    Y = Y_diag + Y_off
    
    # Extract final states for each sequence using last_chunk_indices
    final_states = states_outgoing[last_chunk_indices]
    return Y, final_states
