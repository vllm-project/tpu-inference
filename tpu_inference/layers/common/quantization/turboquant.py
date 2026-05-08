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
"""TurboQuant utilities for TPU Inference."""

import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


def get_hadamard_matrix(n: int) -> jnp.ndarray:
    """Generates an n x n Hadamard matrix scaled by 1/sqrt(n)."""
    if n == 1:
        return jnp.array([[1.0]], dtype=jnp.float32)

    # Recursive construction (iterative for JAX/TPU might be better if n is large)
    # But n is typically 64, 128, or 256.
    h = jnp.array([[1.0, 1.0], [1.0, -1.0]])
    res = h
    curr_n = 2
    while curr_n < n:
        res = jnp.kron(res, h)
        curr_n *= 2

    return res / jnp.sqrt(n)


def _gaussian_pdf(x: jnp.ndarray, sigma2: float) -> jnp.ndarray:
    return (1.0 / jnp.sqrt(2 * jnp.pi * sigma2)) * jnp.exp(-x * x /
                                                           (2 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> jnp.ndarray:
    """Trapezoidal numerical integration."""
    x = jnp.linspace(a, b, n)
    y = f(x)
    return jnp.trapezoid(y, x)


@partial(jax.jit, static_argnums=(0, 1, 2))
def solve_lloyd_max(
    dim: int,
    bits: int,
    max_iter: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/dim) distribution."""
    n_levels = 2**bits
    sigma2 = 1.0 / dim
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -4.0 * sigma, 4.0 * sigma
    centroids = jnp.linspace(lo, hi, n_levels)

    def step(centroids, _):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = jnp.concatenate(
            [jnp.array([lo * 10]), boundaries,
             jnp.array([hi * 10])])

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(
                jnp.where(den > 1e-12, num / den, centroids[i]))

        new_centroids = jnp.stack(new_centroids)
        return new_centroids, None

    centroids, _ = jax.lax.scan(step, centroids, None, length=max_iter)

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return centroids, boundaries


class TurboQuantConfig:

    def __init__(self, cache_dtype: str, head_dim: int, seed: int = 42):
        self.head_dim = head_dim
        self.seed = seed

        if cache_dtype == "turboquant_k8v4":
            self.k_bits = 8
            self.v_bits = 4
        elif cache_dtype == "turboquant_4bit":
            self.k_bits = 4
            self.v_bits = 4
        else:
            raise ValueError(f"Unknown TurboQuant preset: {cache_dtype}")

        # Precompute rotation matrix (Hadamard + random signs)
        self.h_matrix = get_hadamard_matrix(head_dim)

        # Use a fixed random seed for signs to be consistent across shards/layers
        key = jax.random.PRNGKey(seed)
        self.k_signs = jax.random.choice(key,
                                         jnp.array([1.0, -1.0]),
                                         shape=(head_dim, ))
        self.v_signs = jax.random.choice(jax.random.fold_in(key, 1),
                                         jnp.array([1.0, -1.0]),
                                         shape=(head_dim, ))

        # Precompute centroids
        self.k_centroids, self.k_boundaries = solve_lloyd_max(
            head_dim, self.k_bits)
        self.v_centroids, self.v_boundaries = solve_lloyd_max(
            head_dim, self.v_bits)

    def rotate_k(self, k: jnp.ndarray) -> jnp.ndarray:
        return (k * self.k_signs) @ self.h_matrix

    def rotate_v(self, v: jnp.ndarray) -> jnp.ndarray:
        return (v * self.v_signs) @ self.h_matrix

    def rotate_q(self, q: jnp.ndarray) -> jnp.ndarray:
        return (q * self.k_signs) @ self.h_matrix

    def quantize(self, x: jnp.ndarray, boundaries: jnp.ndarray) -> jnp.ndarray:
        """Quantize values by finding their bin in the boundaries and returning as float8 bitcast."""
        # jnp.digitize is like finding the index in a sorted array
        indices = jnp.digitize(x, boundaries).astype(jnp.uint8)
        # Store the raw integer indices disguised as float8_e4m3fn
        return jax.lax.bitcast_convert_type(indices, jnp.float8_e4m3fn)

    def quantize_kv(self, k: jnp.ndarray,
                    v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Rotate and quantize K and V."""
        k_rot = self.rotate_k(k)
        v_rot = self.rotate_v(v)

        k_q = self.quantize(k_rot, self.k_boundaries)
        v_q = self.quantize(v_rot, self.v_boundaries)

        return k_q, v_q

    def rotate_o(self, o: jnp.ndarray) -> jnp.ndarray:
        """Rotate O back to original space."""
        # O_orig = (O_rot @ H) * V_signs
        return (o @ self.h_matrix) * self.v_signs
