# Copyright 2025 Google LLC
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
"""Replay-exactness tests for the indexed-threefry candidate sampler.

These assert that regenerating Gumbel noise at only the top-k candidate
positions (via indexed partitionable threefry) yields the SAME token as the
full-vocabulary top-k-masked jax.random.categorical -- bit-for-bit.

Validated on CPU (jax 0.10.2) and real TPU v7x (jax 0.10.1): 260/260 rows.
"""
import jax
import jax.numpy as jnp
import pytest

from tpu_inference.layers.jax.sample.candidate_sampler import _indexed_uniform

_NEG_INF = -1e12


def _ref_topk_categorical(logits, k, skey):
    """Reference: production top-k mask then full-vocab categorical."""
    vals, _ = jax.lax.top_k(logits, k)
    kth = vals[:, k - 1][:, None]
    masked = jnp.where(logits < kth, _NEG_INF, logits)
    return jax.random.categorical(skey, masked, axis=-1)


def _candidate_tokens(logits, k, skey):
    """Candidate path: top-k, regenerate Gumbel only at candidate indices."""
    B, V = logits.shape
    vals, ids = jax.lax.top_k(logits, k)
    flat = (jnp.arange(B, dtype=jnp.uint32)[:, None] * jnp.uint32(V) +
            ids.astype(jnp.uint32)).reshape(-1)
    u = _indexed_uniform(skey, flat).reshape(B, k)
    gumbel = -jnp.log(-jnp.log(u))
    pos = jnp.argmax(vals + gumbel, axis=-1)
    return jnp.take_along_axis(ids, pos[:, None], axis=1)[:, 0]


@pytest.mark.parametrize("B,V,K,seed", [
    (8, 32000, 64, 1),
    (16, 151936, 50, 2),
    (4, 128256, 100, 3),
    (32, 50000, 32, 4),
    (8, 151936, 200, 5),
    (64, 151936, 50, 6),
    (128, 151936, 40, 7),
])
def test_candidate_matches_full_categorical(B, V, K, seed):
    """The candidate sampler is replay-exact vs full top-k-masked categorical."""
    if not jax.devices():
        pytest.skip("No JAX devices available.")
    logits = jax.random.normal(jax.random.PRNGKey(seed), (B, V)) * 6.0
    skey = jax.random.PRNGKey(seed * 7 + 1)
    truth = _ref_topk_categorical(logits, K, skey)
    cand = _candidate_tokens(logits, K, skey)
    assert bool(
        jnp.all(cand == truth)), (f"replay-exact mismatch B={B} V={V} K={K}: "
                                  f"{int(jnp.sum(cand != truth))} rows differ")


def test_indexed_uniform_matches_jax_uniform():
    """Indexed threefry reproduces jax.random.uniform at arbitrary positions."""
    if not jax.devices():
        pytest.skip("No JAX devices available.")
    dtype = jnp.float32
    tiny = jnp.finfo(dtype).tiny
    key = jax.random.PRNGKey(999)
    N = 8192
    full = jax.random.uniform(key, (N, ), dtype=dtype, minval=tiny, maxval=1.0)
    idx = jnp.array([7, 100, 2049, 8191, 0, 1, 4096], dtype=jnp.uint32)
    indexed = _indexed_uniform(key, idx)
    assert bool(jnp.all(indexed == full[idx.astype(jnp.int32)]))
