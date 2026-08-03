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
"""The static decode-only selector (AttentionMetadata.decode_only_bucket).

Two contracts:

1. PARITY: on a decode-only batch, `decode_only_bucket=True` (single-token
   path) and `False` (mixed/chunked path) must agree bit-for-bit through the
   full KDA sublayer -- the runner may dispatch either executable shape.

2. MEMORY: with the selector, neither traced variant carries the other
   branch, so the compiled executable's HLO temporaries stay bounded by ONE
   path's needs even when the state pool has hundreds of slots. The old
   `lax.cond` dispatch made XLA budget a pool-sized copy per KDA layer
   (~118G on the pod once the pool ran unpinned); at 256 slots and 8 heads
   one pool is ~100 MiB, and the old cond overhead alone exceeded the bound
   asserted here.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.kda_attention import (KDAParams, KDAState,
                                                       kda_attention)

HID, H, K, W = 32, 4, 128, 4


def _params(rng):

    def rand(*s):
        return jnp.asarray(rng.normal(size=s), jnp.float32) * 0.2

    return KDAParams(q_proj=rand(H * K, HID),
                     k_proj=rand(H * K, HID),
                     v_proj=rand(H * K, HID),
                     q_conv_weight=rand(H * K, 1, W),
                     k_conv_weight=rand(H * K, 1, W),
                     v_conv_weight=rand(H * K, 1, W),
                     A_log=rand(H),
                     dt_bias=rand(H * K),
                     f_a_proj=rand(K, HID),
                     f_b_proj=rand(H * K, K),
                     b_proj=rand(H, HID),
                     o_norm_weight=rand(K),
                     o_proj=rand(HID, H * K),
                     g_proj=rand(H * K, HID))


def _state(num_slots):
    return KDAState(conv_q=jnp.zeros((num_slots, W - 1, H * K)),
                    conv_k=jnp.zeros((num_slots, W - 1, H * K)),
                    conv_v=jnp.zeros((num_slots, W - 1, H * K)),
                    recurrent=jnp.zeros((num_slots, H, K, K)))


def test_decode_bucket_variants_agree_bitwise_on_a_decode_batch():
    rng = np.random.default_rng(3)
    params = _params(rng)
    sidx = jnp.array([1], jnp.int32)
    common = dict(num_heads=H, head_dim=K, gate_lower_bound=-5.0)

    # Prefill 8 tokens to give the slot real state, then one decode token.
    x_pre = jnp.asarray(rng.normal(size=(8, HID)), jnp.float32) * 0.2
    st, _ = kda_attention(x_pre, params, _state(2), sidx,
                          jnp.array([0, 8], jnp.int32),
                          jnp.array([0, 0, 1], jnp.int32),
                          jnp.array([0], jnp.int32), **common)

    x1 = jnp.asarray(rng.normal(size=(1, HID)), jnp.float32) * 0.2
    qsl = jnp.array([0, 1], jnp.int32)
    dist = jnp.array([1, 1, 1], jnp.int32)
    has_init = jnp.array([1], jnp.int32)
    st_d, o_d = kda_attention(x1,
                              params,
                              KDAState(*map(jnp.array, st)),
                              sidx,
                              qsl,
                              dist,
                              has_init,
                              decode_only_bucket=True,
                              **common)
    st_m, o_m = kda_attention(x1,
                              params,
                              KDAState(*map(jnp.array, st)),
                              sidx,
                              qsl,
                              dist,
                              has_init,
                              decode_only_bucket=False,
                              **common)
    np.testing.assert_array_equal(np.asarray(o_d), np.asarray(o_m))
    for a, b in zip(st_d, st_m):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("decode_only_bucket", [True, False])
def test_compiled_temporaries_stay_bounded_at_many_slots(decode_only_bucket):
    if jax.default_backend() == "cpu":
        pytest.skip("memory_analysis needs the TPU backend")
    rng = np.random.default_rng(4)
    params = _params(rng)
    num_slots, num_reqs = 256, 64
    x = jnp.ones((num_reqs, HID), jnp.float32)
    qsl = jnp.arange(num_reqs + 1, dtype=jnp.int32)
    sidx = jnp.arange(num_reqs, dtype=jnp.int32)
    dist = jnp.array([num_reqs] * 3, jnp.int32)
    has_init = jnp.ones((num_reqs, ), jnp.int32)

    fn = jax.jit(
        functools.partial(kda_attention,
                          num_heads=H,
                          head_dim=K,
                          gate_lower_bound=-5.0,
                          decode_only_bucket=decode_only_bucket),
        donate_argnums=(2, ),
    )
    stats = fn.lower(x, params, _state(num_slots), sidx, qsl, dist,
                     has_init).compile().memory_analysis()
    if stats is None:
        pytest.skip("memory_analysis unavailable")
    # One [num_slots, H, K, K] fp32 pool is ~100 MiB here; the old cond
    # dispatch budgeted pool-sized copies on top of the branch bodies. Both
    # single-path variants must stay well under one pool of temporaries.
    pool_bytes = num_slots * H * K * K * 4
    assert stats.temp_size_in_bytes < pool_bytes, (
        f"temporaries {stats.temp_size_in_bytes} >= one state pool "
        f"{pool_bytes}; the state update is no longer aliasing in place")
