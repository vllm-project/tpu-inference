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
"""Multi-request PCP attention wrapper test.

`pcp_ragged_paged_attention` packs R fixed request lanes into one token
buffer, laid out RANK-MAJOR then LANE then head|tail, and loops the proven
single-request head-tail body per lane. This test builds that exact layout for
one or more requests (each with its own strided-cache pages), runs the wrapper,
and compares every real token against an independent full-causal reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    merge_kv, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.attention_interface import \
    pcp_ragged_paged_attention
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)

PAGE = 16  # per-rank block_size; the GLOBAL page_size dim is PAGE * pcp
REF_MAX_SEQ = 8  # padding width for the single-seq reference call
NQ, NKV, HD = 8, 2, 128
DTYPE = jnp.float32
SM_SCALE = HD**-0.5


class PcpAttentionInterfaceTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        # Force the N-D axis names (which carry `pcp`) regardless of the ambient
        # NEW_MODEL_DESIGN env, and restore afterwards.
        self._saved_cls = sharding_mod.ShardingAxisName._cls
        sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase

    def tearDown(self):
        sharding_mod.ShardingAxisName._cls = self._saved_cls
        super().tearDown()

    # ----------------------------- helpers -----------------------------------
    @property
    def _kvp(self):
        return get_dtype_packing(DTYPE)

    @property
    def _nkv2(self):
        return align_to(2 * NKV, self._kvp)

    def _cache_dims(self, pcp):
        return (self._nkv2 // self._kvp, self._kvp, align_to(HD, 128))

    def _rand(self, rng, shape):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(DTYPE)

    def _mesh(self, pcp):
        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        return Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)

    def _ref_cache(self, k, v, ntok, npages):
        """Plain (non-strided) cache of `ntok` tokens, for the reference call."""
        kv = merge_kv(k, v)
        pad = cdiv(ntok, PAGE) * PAGE - ntok
        kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                     constant_values=jnp.nan)
        kv = kv.reshape(-1, PAGE, *self._cache_dims(1))
        cache = jnp.full((npages, PAGE, *self._cache_dims(1)), jnp.nan, DTYPE)
        return cache.at[:kv.shape[0]].set(kv)

    def _strided_shard(self, k, v, ntok, pcp, npages):
        """The GLOBAL strided pcp cache for one request over `npages` pages:
        rank r owns global token g with g % pcp == r at local slot g // pcp,
        placed in columns [r*PAGE, (r+1)*PAGE) -- exactly how KV_CONTEXT
        partitions the page_size dim."""
        dims = self._cache_dims(pcp)
        shards = []
        for r in range(pcp):
            idx = np.arange(r, ntok, pcp)
            kv = np.asarray(merge_kv(k[idx], v[idx])) if len(idx) else None
            shard = np.full((npages, PAGE, *dims), np.nan, np.float32)
            if kv is not None:
                n = kv.shape[0]
                kv = np.pad(kv, ((0, cdiv(n, PAGE) * PAGE - n), (0, 0), (0, 0),
                                 (0, 0)),
                            constant_values=np.nan)
                kv = kv.reshape(-1, PAGE, *dims)
                shard[:kv.shape[0]] = kv
            shards.append(shard)
        return np.concatenate(shards, axis=1)  # (npages, PAGE*pcp, ...)

    def _dest(self, i, p, pcp, C, region):
        """Global buffer position of (lane i, source token p) under the
        RANK-MAJOR / LANE / head|tail layout, matching _prepare_inputs."""
        c = p // C
        o = p % C
        if c < pcp:
            rank, head_col = c, 0
        else:
            rank, head_col = 2 * pcp - 1 - c, C
        return rank * region + i * (2 * C) + head_col + o

    def _run(self, pcp, reqs, padded_s):
        """reqs = list of (num_computed L_i, num_current S_i). Lays out R = len
        lanes each of padded_s tokens, runs the wrapper, returns (out, cache,
        expectations, C, region, ppr) where expectations[i] is the full-causal
        reference for request i."""
        R = len(reqs)
        two_p = 2 * pcp
        C = padded_s // two_p
        region = R * 2 * C
        padded_q = R * padded_s  # = P * region
        rng = np.random.default_rng(4)

        # per-request pages: enough for ceil((L_i+S_i)/pcp) local tokens.
        ppr = max(cdiv(cdiv(L + S, pcp), PAGE) for (L, S) in reqs)
        ppr = max(ppr, 1)
        npages = R * ppr
        cache = np.full((npages, PAGE * pcp, *self._cache_dims(pcp)), np.nan,
                        np.float32)

        q_buf = np.zeros((padded_q, NQ, HD), np.float32)
        k_buf = np.zeros((padded_q, NKV, HD), np.float32)
        v_buf = np.zeros((padded_q, NKV, HD), np.float32)
        page_indices = np.zeros(R * ppr, np.int32)
        kv_lens = np.zeros(R, np.int32)
        kv_cache_lens = np.zeros(R, np.int32)
        exps = []

        for i, (L, S) in enumerate(reqs):
            kv_total = L + S
            k_prev = self._rand(rng, (L, NKV, HD))
            v_prev = self._rand(rng, (L, NKV, HD))
            q_cur = self._rand(rng, (S, NQ, HD))
            k_cur = self._rand(rng, (S, NKV, HD))
            v_cur = self._rand(rng, (S, NKV, HD))

            # reference: plain full-causal prefill over this request's context.
            ref_pps = cdiv(kv_total, PAGE)
            ref_pi = jnp.pad(jnp.arange(ref_pps, dtype=jnp.int32),
                             (0, REF_MAX_SEQ * ref_pps - ref_pps))
            exp, _ = ref_ragged_paged_attention(
                q_cur, k_cur, v_cur,
                self._ref_cache(k_prev, v_prev, L, ref_pps),
                jnp.pad(jnp.array([kv_total], jnp.int32), (0, REF_MAX_SEQ - 1)),
                ref_pi,
                jnp.pad(jnp.array([0, S], jnp.int32), (0, REF_MAX_SEQ - 1)),
                jnp.array([0, 0, 1], jnp.int32), sm_scale=SM_SCALE)
            exps.append(np.asarray(exp[:S]))

            # this request's pages [i*ppr, (i+1)*ppr) and its strided prev-cache.
            page_indices[i * ppr:(i + 1) * ppr] = np.arange(
                i * ppr, (i + 1) * ppr, dtype=np.int32)
            shard = self._strided_shard(k_prev, v_prev, L, pcp, ppr)
            cache[i * ppr:(i + 1) * ppr] = shard

            # scatter the current tokens into the packed token buffer.
            for p in range(S):
                d = self._dest(i, p, pcp, C, region)
                q_buf[d] = np.asarray(q_cur[p], np.float32)
                k_buf[d] = np.asarray(k_cur[p], np.float32)
                v_buf[d] = np.asarray(v_cur[p], np.float32)

            kv_lens[i] = kv_total
            kv_cache_lens[i] = L

        out, new_cache = pcp_ragged_paged_attention(
            self._mesh(pcp),
            jnp.asarray(q_buf, DTYPE),
            jnp.asarray(k_buf, DTYPE),
            jnp.asarray(v_buf, DTYPE),
            jnp.asarray(cache, DTYPE),
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.array([0, 0, R], jnp.int32),  # distribution (unused by wrapper)
            jnp.asarray(kv_cache_lens),
            R,  # num_reqs
            sm_scale=SM_SCALE,
            update_kv_cache=True,
            use_causal_mask=True)
        return (np.asarray(out), np.asarray(new_cache), exps, C, region, ppr)

    def _assert_matches(self, out, exps, pcp, C, region, reqs):
        for i, (_, S) in enumerate(reqs):
            rows = np.array(
                [self._dest(i, p, pcp, C, region) for p in range(S)])
            got = out[rows]
            self.assertTrue(np.all(np.isfinite(got)))
            self.assertGreater(float(np.abs(got).max()), 0.0)
            self.assertAllClose(got, exps[i], atol=2e-2, rtol=2e-2)

    # ------------------------------ tests ------------------------------------
    @parameterized.product(pcp=[2, 4])
    def test_single_request(self, pcp):
        """R=1 reduces to the original single-request head-tail path."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(128, 128)]  # (num_computed, num_current == padded_s)
        out, _, exps, C, region, _ = self._run(pcp, reqs, 128)
        self._assert_matches(out, exps, pcp, C, region, reqs)

    @parameterized.product(pcp=[2, 4])
    def test_single_partial_tail(self, pcp):
        """num_current < padded_s: tail chunks partly padding (per-rank
        tail_real differs)."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(128, 100)]
        out, _, exps, C, region, _ = self._run(pcp, reqs, 128)
        self._assert_matches(out, exps, pcp, C, region, reqs)

    @parameterized.product(pcp=[2, 4])
    def test_multi_request(self, pcp):
        """Several requests of different lengths packed into one buffer, each
        with its own strided-cache pages -- the multi-request path."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(128, 128), (64, 100), (0, 96)]
        out, _, exps, C, region, _ = self._run(pcp, reqs, 128)
        self._assert_matches(out, exps, pcp, C, region, reqs)

    @parameterized.product(pcp=[2, 4])
    def test_multi_request_with_padded_lane(self, pcp):
        """A padded (empty) lane at the end must not corrupt the real lanes."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(128, 128), (0, 64), (0, 0)]  # last lane is all padding
        out, _, exps, C, region, _ = self._run(pcp, reqs, 128)
        # only compare the non-empty lanes
        self._assert_matches(out, exps[:2], pcp, C, region, reqs[:2])

    @parameterized.product(pcp=[2, 4])
    def test_kv_cache_write_multi(self, pcp):
        """Each request's current KV lands in its OWN strided pages: global
        token g at rank g % pcp, local slot g // pcp, in that request's page
        range."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(128, 128), (64, 96)]
        _, cache, _, _, _, ppr = self._run(pcp, reqs, 128)
        rng = np.random.default_rng(4)
        for i, (L, S) in enumerate(reqs):
            _ = self._rand(rng, (L, NKV, HD)), self._rand(rng, (L, NKV, HD))
            _ = self._rand(rng, (S, NQ, HD))
            k_cur = self._rand(rng, (S, NKV, HD))
            v_cur = self._rand(rng, (S, NKV, HD))
            ref = np.asarray(merge_kv(k_cur, v_cur))
            for j in range(S):
                g = L + j
                r, local = g % pcp, g // pcp
                page, off = local // PAGE, local % PAGE
                got = cache[i * ppr + page, r * PAGE + off]
                self.assertAllClose(got, ref[j], atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
