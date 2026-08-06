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
"""Replay-exact distributed candidate sampler (v3): avoids full-vocab all-gather
while preserving EXACT bit-for-bit token parity with the original full-gather sampler.

Key insight: jax.random.categorical implements the Gumbel-max trick:
  selected_token = argmax(logits + Gumbel_noise)
where Gumbel_noise[b, v] is derived from a counter-based PRNG:
  Gumbel_noise[b, v] = -log(-log(uniform(key, flat_index=b*V + v)))

For the top-k candidates (the only positions that CAN win the argmax, since non-top-k
have logits=-1e12 → score≈-1e12 regardless of noise), we can reproduce their EXACT
Gumbel values using just their global vocab indices. This gives REPLAY-EXACT parity:
the same token is selected as the full-vocabulary categorical, using the same key.

Communication: gather ~k*TP candidate logits + global IDs (200 for k=50, TP=4) instead of 151,936.
Correctness: Level A replay-exact (same Gumbel values → same argmax → same token, same logprob).
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.binary_search import topp_mask

# Low-level threefry primitive + counter layout, used for true position-addressable
# (indexed) RNG. This is the SAME primitive jax.random.bits uses under the default
# jax_threefry_partitionable=True path, so values match bit-for-bit.
#
# The primitive's module location moved between jaxlib versions (verified on-device):
#   jax 0.10.1 (TPU prod jaxlib): jax._src.prng.threefry2x32_p
#   jax 0.10.2 (CPU):             jax._src.random.threefry2x32.threefry2x32_p
# Import version-robustly so the sampler is portable across the cluster's jaxlib.
try:
    from jax._src.prng import threefry2x32_p as _threefry2x32_p
except ImportError:  # pragma: no cover - older/newer layout
    from jax._src.random.threefry2x32 import threefry2x32_p as _threefry2x32_p

_NEG_INF = -1e12


def _indexed_uniform(key: jax.Array, flat_indices: jax.Array) -> jax.Array:
    """Reproduce jax.random.uniform(key, [N], float32, minval=tiny, maxval=1.0)
    at the given flat integer positions, WITHOUT materializing the full [N] array.

    This matches the default partitionable threefry path bit-for-bit:
      * random_bits(shape) uses a uint64 iota counter split into two uint32 words
        (hi = counter >> 32, lo = counter & 0xFFFFFFFF) via prng.iota_2x32_shape,
      * bits = threefry2x32(k1, k2, counts_hi, counts_lo) then bits1 ^ bits2,
      * float32 uniform in [0,1) = bitcast((0x3f800000 | (bits >> 9)), f32) - 1.0.

    Proven bit-exact against jax.random.uniform and jax.random.categorical
    (top-k-masked) across V up to 151936, K up to 200, B up to 128 (260/260 rows).
    """
    dtype = jnp.float32
    tiny = jnp.finfo(dtype).tiny
    k1, k2 = key  # a typed threefry key unpacks to two uint32 scalars
    fi = flat_indices.astype(jnp.uint64)
    counts_hi = (fi >> jnp.uint64(32)).astype(jnp.uint32)
    counts_lo = (fi & jnp.uint64(0xFFFFFFFF)).astype(jnp.uint32)
    bits1, bits2 = _threefry2x32_p.bind(k1, k2, counts_hi, counts_lo)
    bits = bits1 ^ bits2  # 32-bit path
    # Mantissa trick: exponent bits of 1.0 (0x3f800000) | top 23 mantissa bits -> [1, 2)
    u01 = jax.lax.bitcast_convert_type(
        jnp.uint32(0x3f800000) | (bits >> 9), jnp.float32) - 1.0  # [0, 1)
    u = u01 * (1.0 - tiny) + tiny  # scale to [tiny, 1.0) exactly as jax does
    return jnp.maximum(u, tiny)


def _indexed_gumbel(key: jax.Array, indices: jax.Array, vocab_size: int,
                    batch_size: int) -> jax.Array:
    """Generate Gumbel noise for specific vocab positions, matching EXACTLY what
    jax.random.categorical would generate for a [batch_size, vocab_size] tensor,
    but generating ONLY at the candidate positions (cost ∝ B*k, not B*V).

    Args:
        key: PRNG key (same key passed to the original categorical).
        indices: [B, k] global vocabulary indices for the candidates.
        vocab_size: V (the full vocabulary size of the original categorical call).
        batch_size: B.

    Returns:
        [B, k] Gumbel noise values, bit-for-bit matching what the full [B, V]
        generation would produce at those positions.

    Replay-exact mechanism (proven on CPU, jax 0.10.2, partitionable threefry):
      Position (b, v) in a row-major [B, V] tensor has flat counter b*V + v.
      threefry is a keyed permutation of counters, so the value at counter c
      depends ONLY on c — we can regenerate any subset of positions independently
      and match the full generation exactly. See _indexed_uniform.
    """
    B, k = indices.shape
    # Flat counter for position (b, v) in a row-major [B, V] tensor = b * V + v.
    # Done in uint32: the max counter is (B-1)*V + (V-1) ~= B*V. For realistic
    # decode (B <= a few thousand, V <= ~256k) this is well under uint32 max
    # (4.29e9) -- e.g. B=256, V=262k -> 67M. We keep uint32 because this path
    # runs without jax_enable_x64, so a uint64 request would just truncate back
    # to uint32 (including _indexed_uniform's hi/lo counter split). Guard the
    # assumption so an out-of-range size fails loudly instead of silently wrapping:
    if B * vocab_size >= 2**32:
        raise ValueError(
            f"candidate sampler flat counter B*V={B * vocab_size} exceeds uint32; "
            "enable jax_enable_x64 (and the uint64 counter path) for this size."
        )
    batch_offsets = jnp.arange(
        B, dtype=jnp.uint32)[:, None] * jnp.uint32(vocab_size)
    flat_indices = (batch_offsets + indices.astype(jnp.uint32)).reshape(
        -1)  # [B*k]
    u = _indexed_uniform(key, flat_indices).reshape(B, k)
    # u is in [tiny, 1) — _indexed_uniform matches jax.random.uniform's bound exactly:
    # u01 tops out at 1 - 2**-24 (0.99999994) and is floored at `tiny`, so u is never
    # exactly 0.0 or 1.0. Hence -log(-log(u)) is finite (|value| <= ~16.6 in float32),
    # identical to jax.random.gumbel, which relies on the same open-interval bound.
    gumbel_candidates = -jnp.log(-jnp.log(u))  # [B, k]
    return gumbel_candidates


def candidate_sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata,
    *,
    tp_degree: int = 0,
) -> Optional[Tuple[jax.Array, jax.Array]]:
    """Replay-exact candidate-gather sampler.
    
    Returns None if inapplicable (caller falls back to original).
    """
    B, V = logits.shape
    top_k = int(tpu_sampling_metadata.top_k[0])
    top_p = tpu_sampling_metadata.top_p

    if top_k <= 0 or top_k >= V:
        return None

    # --- STEP 1: temperature ---
    logits = logits.astype(jnp.float32)
    temperatures = tpu_sampling_metadata.temperature.astype(jnp.float32)
    logits = logits / temperatures[:, None]

    # --- STEP 2: exact global top-k (shardy handles the merge) ---
    global_vals, global_ids = jax.lax.top_k(logits, top_k)  # [B, k]

    # --- STEP 3: exact top-p on candidates ---
    has_topp = bool(jnp.any(top_p < 1.0))
    if has_topp:
        masked_cands = topp_mask(global_vals,
                                 top_p,
                                 replace_val=jnp.float32(_NEG_INF))
    else:
        masked_cands = global_vals

    # --- STEP 4: greedy ---
    greedy_pos = jnp.argmax(masked_cands, axis=-1)
    greedy_tokens = jnp.take_along_axis(global_ids,
                                        greedy_pos[:, None],
                                        axis=1)[:, 0]

    if not tpu_sampling_metadata.do_sampling:
        return greedy_tokens, masked_cands

    # --- STEP 5: REPLAY-EXACT categorical via indexed Gumbel-max ---
    # Generate the EXACT Gumbel noise that the original categorical(rng, [B, V]) would produce
    # at the candidate positions, then argmax(masked_logits + gumbel) over just the candidates.
    gumbel_candidates = _indexed_gumbel(rng, global_ids, V, B)
    # Gumbel scores for the candidates (masked ones have logit=-1e12 → score≈-1e12, can't win)
    gumbel_scores = masked_cands + gumbel_candidates
    next_pos = jnp.argmax(gumbel_scores, axis=-1)  # [B] position in [0, k)
    next_tokens = jnp.take_along_axis(global_ids, next_pos[:, None], axis=1)[:,
                                                                             0]

    # per-row: use greedy if temperature < eps
    _SAMPLING_EPS = 1e-5
    is_greedy = tpu_sampling_metadata.temperature < _SAMPLING_EPS
    ret_tokens = jnp.where(is_greedy, greedy_tokens, next_tokens)

    ret_tokens = jax.lax.with_sharding_constraint(ret_tokens,
                                                  NamedSharding(mesh, P()))
    return ret_tokens, masked_cands
