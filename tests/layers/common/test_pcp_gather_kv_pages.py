# SPDX-License-Identifier: Apache-2.0
"""Exercise the gather-KV cache phase when the KV cache holds MORE pages than
the request owns, with the request's pages SCATTERED through the buffer.

The main interface test builds `npages == pages_per_seq`, so the gather-KV
page-compaction path (which only all-gathers the request's own block-table
pages) is never taken there.  This test forces it: the cache is allocated with
`slack x` the request's pages and the block table points at a non-identity,
non-contiguous permutation, so a bug in the compaction indexing shows up as a
numerical mismatch against the plain reference.
"""
import os

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

PAGE = 16
MAX_SEQ = 4
NQ, NKV, HD = 8, 2, 128
DTYPE = jnp.float32
SM = HD**-0.5


def _row_perm(pcp):
    return [c for r in range(pcp) for c in (r, 2 * pcp - 1 - r)]


def _inv_row(pcp):
    inv = np.empty(2 * pcp, np.int64)
    inv[_row_perm(pcp)] = np.arange(2 * pcp)
    return inv


def _to_rank_order(x, pcp, C):
    x = np.asarray(x)
    return jnp.asarray(
        x.reshape(2 * pcp, C, *x.shape[1:])[_row_perm(pcp)].reshape(x.shape))


def _pcp_meta(pcp, C, num_current):
    two_p = 2 * pcp
    cu = np.zeros((pcp, MAX_SEQ + 1), np.int32)
    qpos = np.zeros((pcp, MAX_SEQ), np.int32)
    for r in range(pcp):
        tail_off = (two_p - 1 - r) * C
        tail_real = int(np.clip(num_current - tail_off, 0, C))
        cu[r, 1] = C
        cu[r, 2:] = C + tail_real
        qpos[r, 0] = r * C
        qpos[r, 1] = tail_off
    return jnp.asarray(cu), jnp.asarray(qpos)


class PcpGatherKvPagesTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        self._saved_cls = sharding_mod.ShardingAxisName._cls
        sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase
        self._prev = os.environ.get("PCP_CACHE_PHASE")
        os.environ["PCP_CACHE_PHASE"] = "gather_kv"

    def tearDown(self):
        sharding_mod.ShardingAxisName._cls = self._saved_cls
        if self._prev is None:
            os.environ.pop("PCP_CACHE_PHASE", None)
        else:
            os.environ["PCP_CACHE_PHASE"] = self._prev
        super().tearDown()

    @property
    def _kvp(self):
        return get_dtype_packing(DTYPE)

    @property
    def _nkv2(self):
        return align_to(2 * NKV, self._kvp)

    def _dims(self):
        return (self._nkv2 // self._kvp, self._kvp, align_to(HD, 128))

    def _rand(self, rng, shape):
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(DTYPE)

    def _ref_cache(self, k, v, ntok, npages):
        kv = merge_kv(k, v)
        pad = cdiv(ntok, PAGE) * PAGE - ntok
        kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                     constant_values=jnp.nan)
        kv = kv.reshape(-1, PAGE, *self._dims())
        cache = jnp.full((npages, PAGE, *self._dims()), jnp.nan, DTYPE)
        return cache.at[:kv.shape[0]].set(kv)

    def _scattered_cache(self, k, v, ntok, pcp, pps, npages, phys):
        """Global pcp cache with `npages` pages; the request's logical page i
        lives at PHYSICAL page phys[i]."""
        dims = self._dims()
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
                for i in range(kv.shape[0]):
                    shard[phys[i]] = kv[i]
            shards.append(shard)
        return jnp.asarray(np.concatenate(shards, axis=1), DTYPE)

    def _mesh(self, pcp):
        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        return Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)

    @parameterized.parameters((2, False), (4, False), (2, True), (4, True))
    def test_scattered_pages_with_slack_cache(self, pcp, tight_hint):
        if jax.device_count() < pcp:
            self.skipTest(f"needs {pcp} devices")
        rng = np.random.default_rng(0)
        C = 32
        num_current = 2 * pcp * C
        L = 4 * PAGE * pcp  # cached tokens
        kv_total = L + num_current

        q = self._rand(rng, (num_current, NQ, HD))
        k = self._rand(rng, (num_current, NKV, HD))
        v = self._rand(rng, (num_current, NKV, HD))
        k_prev = self._rand(rng, (L, NKV, HD))
        v_prev = self._rand(rng, (L, NKV, HD))

        # ---- reference: plain non-PCP attention over the same tokens ----
        ref_pps = cdiv(kv_total, PAGE)
        ref_pi = jnp.pad(jnp.arange(ref_pps, dtype=jnp.int32),
                         (0, MAX_SEQ * ref_pps - ref_pps))
        exp, _ = ref_ragged_paged_attention(
            q, k, v, self._ref_cache(k_prev, v_prev, L, ref_pps),
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, MAX_SEQ - 1)),
            ref_pi,
            jnp.pad(jnp.array([0, num_current], jnp.int32), (0, MAX_SEQ - 1)),
            jnp.array([0, 0, 1], jnp.int32), sm_scale=SM)

        # ---- PCP: cache has 4x slack, request pages scattered ----
        pps = cdiv(cdiv(kv_total, pcp), PAGE)
        npages = 4 * pps  # SLACK -> forces the compaction branch
        phys = (np.arange(pps) * 3 + 5) % npages  # scattered, non-identity
        assert len(set(phys.tolist())) == pps, "phys must be a permutation"
        cache = self._scattered_cache(k_prev, v_prev, L, pcp, pps, npages,
                                      phys)

        pi = np.zeros(MAX_SEQ * pps, np.int32)
        pi[:pps] = phys
        pi[pps:2 * pps] = phys
        pi = jnp.asarray(pi)

        def pad1(xs):
            return jnp.pad(jnp.array(xs, jnp.int32), (0, MAX_SEQ - len(xs)))

        # `cache_pages` is the static bound the runner supplies: the number of
        # pages the CACHED tokens occupy (page P of the token-ordered cache
        # holds gpage = PAGE*pcp tokens), rounded up to a power of two.  It is
        # strictly tighter than pages_per_seq here, so it exercises the bound.
        gpage = PAGE * pcp
        live = cdiv(L, gpage)
        hint = (1 << (max(live - 1, 0)).bit_length()) if tight_hint else -1
        if tight_hint:
            self.assertLess(hint, pps, "hint must be tighter than pages_per_seq")
            self.assertGreaterEqual(hint, live, "hint must cover live pages")

        cu, qpos = _pcp_meta(pcp, C, num_current)
        out, _ = pcp_ragged_paged_attention(
            self._mesh(pcp), _to_rank_order(q, pcp, C),
            _to_rank_order(k, pcp, C), _to_rank_order(v, pcp, C), cache,
            pad1([kv_total, kv_total]), pi, cu,
            jnp.array([0, 0, 2], jnp.int32), pad1([L, L]), qpos, SM,
            cache_pages=hint)

        inv = _inv_row(pcp)
        got = np.asarray(out).reshape(2 * pcp, C, NQ, HD)[inv].reshape(
            num_current, NQ, HD)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertGreater(float(np.abs(got).max()), 0.0)
        self.assertAllClose(got, np.asarray(exp), atol=2e-2, rtol=2e-2)


    @parameterized.parameters(2, 4)
    def test_first_chunk_skips_cache_phase(self, pcp):
        """cache_pages=0 elides the cache phase; with no cached tokens the
        result must still match plain attention over the current chunk."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs {pcp} devices")
        rng = np.random.default_rng(1)
        C = 32
        num_current = 2 * pcp * C
        kv_total = num_current  # L == 0 -> first chunk, empty cache

        q = self._rand(rng, (num_current, NQ, HD))
        k = self._rand(rng, (num_current, NKV, HD))
        v = self._rand(rng, (num_current, NKV, HD))

        ref_pps = cdiv(kv_total, PAGE)
        ref_pi = jnp.pad(jnp.arange(ref_pps, dtype=jnp.int32),
                         (0, MAX_SEQ * ref_pps - ref_pps))
        empty = jnp.full((ref_pps, PAGE, *self._dims()), jnp.nan, DTYPE)
        exp, _ = ref_ragged_paged_attention(
            q, k, v, empty,
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, MAX_SEQ - 1)),
            ref_pi,
            jnp.pad(jnp.array([0, num_current], jnp.int32), (0, MAX_SEQ - 1)),
            jnp.array([0, 0, 1], jnp.int32), sm_scale=SM)

        pps = cdiv(cdiv(kv_total, pcp), PAGE)
        npages = 4 * pps
        cache = jnp.full((npages, PAGE * pcp, *self._dims()), jnp.nan, DTYPE)
        pi = np.zeros(MAX_SEQ * pps, np.int32)
        pi[:pps] = np.arange(pps)
        pi[pps:2 * pps] = np.arange(pps)
        pi = jnp.asarray(pi)

        def pad1(xs):
            return jnp.pad(jnp.array(xs, jnp.int32), (0, MAX_SEQ - len(xs)))

        cu, qpos = _pcp_meta(pcp, C, num_current)
        out, _ = pcp_ragged_paged_attention(
            self._mesh(pcp), _to_rank_order(q, pcp, C),
            _to_rank_order(k, pcp, C), _to_rank_order(v, pcp, C), cache,
            pad1([kv_total, kv_total]), pi, cu,
            jnp.array([0, 0, 2], jnp.int32), pad1([0, 0]), qpos, SM,
            cache_pages=0)

        inv = _inv_row(pcp)
        got = np.asarray(out).reshape(2 * pcp, C, NQ, HD)[inv].reshape(
            num_current, NQ, HD)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertAllClose(got, np.asarray(exp), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
