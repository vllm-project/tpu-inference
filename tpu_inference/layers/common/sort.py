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

import jax
import jax.numpy as jnp

# int32 has 32 bits; reserve the sign bit so packed keys stay non-negative.
_USABLE_BITS = 31


def _index_bits(n: int) -> int:
    return max(1, (n - 1).bit_length())


def _key_bits(max_key: int) -> int:
    return max(1, max_key.bit_length())


def can_pack_int32(n: int, max_key: int) -> bool:
    # Static in both args, so callers can branch in Python rather than
    # tracing both the packed and fallback graphs.
    return _key_bits(max_key) + _index_bits(n) <= _USABLE_BITS


def packed_argsort(keys: jax.Array,
                   max_key: int,
                   *,
                   axis: int = -1) -> jax.Array:
    """Faster drop-in replacement for jnp.argsort when keys pack into int32.

    Packs (key, index) into one int32 and runs a single-key unstable sort.
    Stability comes from the index in the low bits, not the sort itself.
    Caller must check can_pack_int32 first since there's no fallback here.
    Keys must be non-negative and at most `max_key`. Ascending order only.
    """
    n = keys.shape[axis]
    if not can_pack_int32(n, max_key):
        raise ValueError(
            f"packing keys up to {max_key} over {n} rows needs "
            f"{_key_bits(max_key) + _index_bits(n)} bits, "
            f"{_USABLE_BITS} available; check can_pack_int32() before calling")

    shift = _index_bits(n)
    index_shape = [1] * keys.ndim
    index_shape[axis] = n
    index = jnp.arange(n, dtype=jnp.int32).reshape(index_shape)
    packed = (keys.astype(jnp.int32) << shift) | index
    sorted_packed = jnp.sort(packed, axis=axis, stable=False)
    return sorted_packed & ((1 << shift) - 1)
