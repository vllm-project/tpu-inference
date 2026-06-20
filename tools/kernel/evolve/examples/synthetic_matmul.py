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
"""Synthetic Pallas matmul kernel — the evolution target.

This file is deliberately self-contained and small (≤80 LOC) so the
evolution loop can sweep its tunable literals with real measurable effects
on a TPU.

Tunable knobs live at module scope as literals (``BLOCK_M``, ``BLOCK_N``,
``ACCUM_DTYPE``). Mutations target these strings directly via the
``ProgrammaticMutator``; each candidate is a real diff against this file.

The reference implementation (``matmul_reference``) is a pure JAX matmul.
The verifier (``RpaV3Oracle``-style) compares the kernel's output to the
reference within dtype-aware tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Tunable literals — evolution mutations rewrite these. Keep each on its
# own line with a recognizable assignment so the programmatic mutator can
# target them with simple line-level diffs.
BLOCK_M = 128
BLOCK_N = 128
ACCUM_DTYPE = jnp.float32


def _matmul_kernel(x_ref, y_ref, out_ref):
    """Per-block computation: out[i, j] = x[i, :] @ y[:, j]."""
    x = x_ref[...]
    y = y_ref[...]
    acc = jnp.dot(x, y, preferred_element_type=ACCUM_DTYPE)
    out_ref[...] = acc.astype(out_ref.dtype)


def matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    """Tiled matmul ``(M, K) @ (K, N) -> (M, N)``.

    ``BLOCK_M``, ``BLOCK_N``, and ``ACCUM_DTYPE`` are read from module
    globals so source-level mutations to those names retune the kernel.
    K is not tiled (BLOCK_K = K) — keeps the kernel small and the search
    space tractable.
    """
    M, K = x.shape
    K2, N = y.shape
    assert K == K2, f"matmul shape mismatch: {x.shape} vs {y.shape}"
    return pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid=(M // BLOCK_M, N // BLOCK_N),
        in_specs=[
            pl.BlockSpec((BLOCK_M, K), lambda i, j: (i, 0)),
            pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(x, y)


def matmul_reference(x: jax.Array, y: jax.Array) -> jax.Array:
    """Pure-JAX reference used by the verifier."""
    return jnp.matmul(x, y).astype(x.dtype)
