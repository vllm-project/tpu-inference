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
"""Correctness tests for prefill context parallelism (PCP) in RPA v3.

PCP shards a prompt across ``cp_group_size`` ranks in a load-balanced head-tail
layout (the sequence is cut into ``2 * cp_group_size`` equal chunks; rank ``r``
owns chunk ``r`` and chunk ``2P-1-r``). ``kernel.py`` attends the all-gathered
current KV; ``kernel_ring_extern.py`` streams the KV one shard per launch; and
``kernel_ring_fused.py`` streams the shards around the ring inside a single
launch. All are validated against the plain full-causal reference.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    merge_kv, ragged_paged_attention, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_ring_extern import \
    ragged_paged_attention as ragged_paged_attention_ring
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_ring_fused import (
    PADDING_POSITION, ring_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RaggedPagedAttentionPcpTest(jtu.JaxTestCase):

    NUM_PAGES = 512
    MAX_SEQ = 8

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")

    # ----------------------------- helpers -----------------------------------
    def _cfg(self, dtype):
        kvp = get_dtype_packing(dtype)
        return dict(kvp=kvp, phd=128, nkv2=align_to(2 * 2, kvp))

    def _rand(self, rng, shape, dtype):
        return jnp.array(rng.random(size=shape, dtype=np.float32)).astype(dtype)

    def _empty_cache(self, dtype):
        c = self._cfg(dtype)
        return jnp.full((self.NUM_PAGES, self.PAGE, c["nkv2"] // c["kvp"],
                        c["kvp"], c["phd"]), jnp.nan, dtype)

    def _cache_from_kv(self, k, v, ntok, dtype):
        c = self._cfg(dtype)
        kv = merge_kv(k, v)
        pad = cdiv(ntok, self.PAGE) * self.PAGE - ntok
        kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                     constant_values=jnp.nan).reshape(
                         -1, self.PAGE, c["nkv2"] // c["kvp"], c["kvp"],
                         c["phd"])
        cache = self._empty_cache(dtype)
        return cache.at[:kv.shape[0]].set(kv)

    def _pi(self, npages):
        pi = jnp.arange(npages, dtype=jnp.int32)
        return jnp.pad(pi, (0, self.MAX_SEQ * npages - npages))

    def _pad1(self, xs):  # length max_num_seqs (kv_lens, q/kv_pos_offsets)
        return jnp.pad(jnp.array(xs, jnp.int32), (0, self.MAX_SEQ - len(xs)))

    def _padcu(self, xs):  # length max_num_seqs + 1 (cu_q_lens, cu_kv_lens)
        return jnp.pad(jnp.array(xs, jnp.int32), (0, self.MAX_SEQ + 1 - len(xs)))

    def _merge_lse(self, acc_o, acc_l, o, l):
        if acc_o is None:
            return o, l
        m = jnp.maximum(acc_l, l)
        e1 = jnp.exp(acc_l - m)
        e2 = jnp.exp(l - m)
        o = (acc_o * e1[..., None] + o * e2[..., None]) / (e1 + e2)[..., None]
        return o, m + jnp.log(e1 + e2)

    def _tol(self, dtype):
        return 0.05 if dtype == jnp.float32 else 0.2

    # ------------------------------ tests ------------------------------------
    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], P=[1, 2, 4])
    def test_pcp_all_gather_head_tail(self, dtype, P):
        """Head/tail chunks vs full-causal reference (all-gather KV)."""
        self.PAGE = 16
        S, nq, nkv, hd = 256, 8, 2, 128
        C = S // (2 * P)
        rng = np.random.default_rng(0)
        q = self._rand(rng, (S, nq, hd), dtype)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        pps = cdiv(S, self.PAGE)
        exp, _ = ref_ragged_paged_attention(q, k, v, self._empty_cache(dtype),
                                             self._pad1([S]), self._pi(pps),
                                             self._padcu([0, S]),
                                             jnp.array([0, 0, 1], jnp.int32))
        exp = exp[:S]
        for r in range(P):
            for chunk in (r, 2 * P - 1 - r):
                q_buf = jnp.zeros((S, nq, hd), dtype).at[:C].set(q[chunk * C:chunk * C + C])
                out, _ = ragged_paged_attention(
                    q_buf, k, v, self._empty_cache(dtype), self._pad1([S]),
                    self._pi(pps), self._padcu([0, C]),
                    jnp.array([0, 0, 1], jnp.int32),
                    cp_rank=jnp.array([r], jnp.int32), cp_group_size=P,
                    all_gather_kv=True,
                    q_pos_offsets=self._pad1([chunk * C]),
                    update_kv_cache=False, use_causal_mask=True)
                self.assertAllClose(out[:C], exp[chunk * C:chunk * C + C],
                                    atol=self._tol(dtype), rtol=self._tol(dtype))

    @parameterized.product(dtype=[jnp.float32])
    def test_pcp_two_phase_chunked_prefill(self, dtype):
        """Non-causal prev-cache + causal current, merged via LSE.

        Uses cp_group_size=1 (previous cache replicated, so no cross-rank merge
        is needed on a single device); with the derived gather factor this means
        the current tokens split into 2 head-tail chunks (C = Scur / 2).
        """
        self.PAGE = 16
        Lprev, Scur, nq, nkv, hd = 128, 128, 8, 2, 128
        C = Scur // 2  # cp_group_size=1 -> 2*cp_group_size = 2 head-tail chunks
        kv_total = Lprev + Scur
        pps = cdiv(kv_total, self.PAGE)
        rng = np.random.default_rng(3)
        k_all = self._rand(rng, (kv_total, nkv, hd), dtype)
        v_all = self._rand(rng, (kv_total, nkv, hd), dtype)
        q_cur = self._rand(rng, (Scur, nq, hd), dtype)
        exp, _ = ref_ragged_paged_attention(
            q_cur, k_all[Lprev:], v_all[Lprev:],
            self._cache_from_kv(k_all[:Lprev], v_all[:Lprev], Lprev, dtype),
            self._pad1([kv_total]), self._pi(pps), self._padcu([0, Scur]),
            jnp.array([0, 0, 1], jnp.int32))
        exp = exp[:Scur]
        for chunk in (0, 1):
            q_buf = jnp.zeros((Scur, nq, hd), dtype).at[:C].set(
                q_cur[chunk * C:chunk * C + C])
            common = dict(cp_rank=jnp.array([0], jnp.int32),
                          cp_group_size=1, all_gather_kv=True,
                          q_pos_offsets=self._pad1([chunk * C]),
                          update_kv_cache=False, return_lse=True)
            o1, _, l1 = ragged_paged_attention(
                q_buf, k_all[Lprev:], v_all[Lprev:],
                self._cache_from_kv(k_all[:Lprev], v_all[:Lprev], Lprev, dtype),
                self._pad1([kv_total]), self._pi(pps), self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32), use_causal_mask=False,
                skip_current_attn=True, **common)
            o2, _, l2 = ragged_paged_attention(
                q_buf, k_all[Lprev:], v_all[Lprev:],
                self._cache_from_kv(k_all[:Lprev], v_all[:Lprev], Lprev, dtype),
                self._pad1([kv_total]), self._pi(pps), self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32), use_causal_mask=True,
                skip_cache_attn=True, **common)
            o, _ = self._merge_lse(o1[:C], l1[:C], o2[:C], l2[:C])
            self.assertAllClose(o, exp[chunk * C:chunk * C + C],
                                atol=self._tol(dtype), rtol=self._tol(dtype))

    @parameterized.product(P=[2, 3, 4])
    def test_pcp_strided_cache_write(self, P):
        """Interleaved (strided) write of the all-gathered current KV."""
        dtype = jnp.float32
        self.PAGE = 16
        S, nq, nkv, hd = 192, 8, 2, 128
        C = S // (2 * P)  # per-chunk local q; gathered KV len = 2*P*C = S
        pps = cdiv(S, self.PAGE)
        rng = np.random.default_rng(7)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        q = self._rand(rng, (S, nq, hd), dtype)
        c = self._cfg(dtype)
        kv_merged = merge_kv(k, v)
        for r in range(P):
            # Non-causal chunk-sized launch: fetches + strided-writes the full
            # gathered current KV (derived length 2*P*C = S).
            _, cache = ragged_paged_attention(
                q, k, v, self._empty_cache(dtype), self._pad1([S]),
                self._pi(pps), self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=jnp.array([r], jnp.int32), cp_group_size=P,
                all_gather_kv=True, update_kv_cache=True,
                use_causal_mask=False)  # non-causal -> fetch + write full shard
            flat = cache.reshape(-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            local_len = (S + P - 1 - r) // P
            pi = np.arange(pps)
            for m in range(local_len):
                g = r + m * P  # rank r's local slot m holds global token g
                slot = pi[m // self.PAGE] * self.PAGE + m % self.PAGE
                self.assertArraysEqual(flat[slot], kv_merged[g])

    @parameterized.product(P=[2, 3, 4])
    def test_pcp_causal_tail_write_full_coverage(self, P):
        """A *causal* tail launch must still write this rank's whole strided
        share, even for KV tokens beyond the tail chunk's causal range."""
        dtype = jnp.float32
        self.PAGE = 16
        S, nq, nkv, hd = 192, 8, 2, 128
        C = S // (2 * P)
        pps = cdiv(S, self.PAGE)
        rng = np.random.default_rng(9)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        q = self._rand(rng, (S, nq, hd), dtype)
        c = self._cfg(dtype)
        kv_merged = merge_kv(k, v)
        for r in range(P):
            tail_off = (2 * P - 1 - r) * C  # tail chunk global offset
            # Causal + update_kv_cache + all_gather_kv -> kernel extends the
            # fetch/write to the full current KV so the strided share is complete
            # even though the causal attention only spans [0, tail_off + C).
            _, cache = ragged_paged_attention(
                q, k, v, self._empty_cache(dtype), self._pad1([S]),
                self._pi(pps), self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=jnp.array([r], jnp.int32), cp_group_size=P,
                all_gather_kv=True, q_pos_offsets=self._pad1([tail_off]),
                update_kv_cache=True, use_causal_mask=True)
            flat = cache.reshape(-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            local_len = (S + P - 1 - r) // P
            pi = np.arange(pps)
            for m in range(local_len):
                g = r + m * P
                slot = pi[m // self.PAGE] * self.PAGE + m % self.PAGE
                self.assertArraysEqual(flat[slot], kv_merged[g])

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], P=[1, 2])
    def test_pcp_ring_attention(self, dtype, P):
        """Ring: stream KV chunks, merge via LSE, vs full-causal reference."""
        self.PAGE = 16
        S, nq, nkv, hd = 256, 8, 2, 128
        nchunk = 2 * P
        C = S // nchunk
        rng = np.random.default_rng(11)
        q = self._rand(rng, (S, nq, hd), dtype)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        pps_full = cdiv(S, self.PAGE)
        exp, _ = ref_ragged_paged_attention(q, k, v, self._empty_cache(dtype),
                                             self._pad1([S]), self._pi(pps_full),
                                             self._padcu([0, S]),
                                             jnp.array([0, 0, 1], jnp.int32))
        exp = exp[:S]
        pps = cdiv(C, self.PAGE)
        for cq in range(nchunk):
            q_buf = jnp.zeros((C, nq, hd), dtype).at[:C].set(q[cq * C:cq * C + C])
            acc_o = acc_l = None
            for ck in range(cq + 1):  # future chunks are fully masked -> skipped
                o, _, l = ragged_paged_attention_ring(
                    q_buf, k[ck * C:ck * C + C], v[ck * C:ck * C + C],
                    self._empty_cache(dtype), self._pad1([C]), self._pi(pps),
                    self._padcu([0, C]), jnp.array([0, 0, 1], jnp.int32),
                    cp_rank=jnp.array([0], jnp.int32), cp_group_size=1,
                    cu_kv_lens=self._padcu([0, C]),
                    q_pos_offsets=self._pad1([cq * C]),
                    kv_pos_offsets=self._pad1([ck * C]),
                    update_kv_cache=False, use_causal_mask=True, return_lse=True)
                acc_o, acc_l = self._merge_lse(acc_o, acc_l, o[:C], l[:C])
            self.assertAllClose(acc_o, exp[cq * C:cq * C + C],
                                atol=self._tol(dtype), rtol=self._tol(dtype))

    # ----------------------- fused in-kernel ring ----------------------------
    def _full_causal_ref(self, q, k, v, sm_scale):
        S, nq, _ = q.shape
        G = nq // k.shape[1]
        kk = jnp.repeat(k, G, axis=1)
        vv = jnp.repeat(v, G, axis=1)
        s = jnp.einsum("ihd,jhd->hij", q, kk).astype(jnp.float32) * sm_scale
        s = jnp.where(jnp.tril(jnp.ones((S, S), bool))[None], s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1)
        return jnp.einsum("hij,jhd->ihd", p.astype(vv.dtype), vv)

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], P=[1, 2, 4])
    def test_fused_ring_attention(self, dtype, P):
        """Single-launch in-kernel ring vs full-causal reference (head-tail)."""
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        nq, nkv, hd = 8, 2, 128
        C = 64
        S = 2 * P * C
        t_local = 2 * C
        sm_scale = hd**-0.5
        rng = np.random.default_rng(0)
        q = self._rand(rng, (S, nq, hd), dtype)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        exp = np.array(self._full_causal_ref(q, k, v, sm_scale))

        # Head-tail: device r owns chunks {r, 2P-1-r} for both Q and its KV shard.
        gq = np.zeros((P * t_local, nq, hd), np.float32)
        gk = np.zeros((P * t_local, nkv, hd), np.float32)
        gv = np.zeros((P * t_local, nkv, hd), np.float32)
        gpos = np.zeros((P * t_local, ), np.int32)
        gpos_of_row = {}
        for r in range(P):
            for ci, ch in enumerate((r, 2 * P - 1 - r)):
                for j in range(C):
                    g = ch * C + j
                    row = r * t_local + ci * C + j
                    gq[row], gk[row], gv[row] = q[g], k[g], v[g]
                    gpos[row] = g
                    gpos_of_row[row] = g
        mesh = Mesh(np.array(jax.devices()[:P]).reshape(P), ("pcp", ))
        out, lse = ring_attention(
            mesh, "pcp", jnp.array(gq).astype(dtype), jnp.array(gk).astype(dtype),
            jnp.array(gv).astype(dtype), jnp.array(gpos), jnp.array(gpos),
            sm_scale=sm_scale, return_lse=True)
        out = np.array(out)
        for row, g in gpos_of_row.items():
            self.assertAllClose(out[row], exp[g], atol=self._tol(dtype),
                                rtol=self._tol(dtype))
        # LSE must be finite for every (valid) query row.
        self.assertTrue(bool(np.all(np.isfinite(np.array(lse)))))

    def test_fused_ring_padding_masked(self):
        """Padded KV tokens (sentinel position + garbage) change nothing."""
        P = 2
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        dtype = jnp.float32
        nq, nkv, hd = 4, 1, 128
        C = 64
        real, pad = C, 8  # real + padded KV tokens per device (t_q stays `real`)
        sm_scale = hd**-0.5
        rng = np.random.default_rng(5)

        def build(t_kv):
            gq = self._rand(rng, (P * real, nq, hd), dtype)
            gk = self._rand(rng, (P * t_kv, nkv, hd), dtype)
            gv = self._rand(rng, (P * t_kv, nkv, hd), dtype)
            qpos = np.zeros((P * real, ), np.int32)
            kpos = np.zeros((P * t_kv, ), np.int32)
            for r in range(P):
                for j in range(real):
                    qpos[r * real + j] = r * real + j
                for j in range(t_kv):
                    kpos[r * t_kv + j] = (r * real + j
                                          if j < real else PADDING_POSITION)
            return gq, gk, gv, qpos, kpos

        mesh = Mesh(np.array(jax.devices()[:P]).reshape(P), ("pcp", ))
        # Baseline: no padding.
        gq, gk, gv, qpos, kpos = build(real)
        base = np.array(
            ring_attention(mesh, "pcp", jnp.array(gq), jnp.array(gk),
                           jnp.array(gv), jnp.array(qpos), jnp.array(kpos),
                           sm_scale=sm_scale))
        # Same queries/real KV, but each shard has extra garbage padded tokens.
        gk_p = np.concatenate([
            np.concatenate([gk[r * real:(r + 1) * real],
                            1e3 * np.ones((pad, nkv, hd), np.float32)])
            for r in range(P)
        ])
        gv_p = np.concatenate([
            np.concatenate([gv[r * real:(r + 1) * real],
                            1e3 * np.ones((pad, nkv, hd), np.float32)])
            for r in range(P)
        ])
        kpos_p = np.concatenate([
            np.concatenate([qpos[r * real:(r + 1) * real],
                            np.full((pad, ), PADDING_POSITION, np.int32)])
            for r in range(P)
        ])
        padded = np.array(
            ring_attention(mesh, "pcp", jnp.array(gq), jnp.array(gk_p),
                           jnp.array(gv_p), jnp.array(qpos), jnp.array(kpos_p),
                           sm_scale=sm_scale))
        self.assertAllClose(padded, base, atol=1e-4, rtol=1e-4)


    # ----------------- multi-device PCP-vs-TP benchmark ----------------------
    # Correctness half of tests/kernels/ragged_paged_attention_pcp_benchmark.py:
    # on `P` real devices, PCP (all-gather KV + head-tail sequence shard, all
    # heads) must match TP (head shard, full sequence) for a causal prefill, and
    # the PCP strided KV-cache write must round-trip to the current KV.

    def _flat_cache(self, nkv_local, npages, page, hd, dtype):
        kvp = get_dtype_packing(dtype)
        nkv2 = align_to(2 * nkv_local, kvp)
        return jnp.zeros((npages, page, nkv2 // kvp, kvp, hd), dtype)

    @parameterized.product(P=[2, 4])
    def test_pcp_vs_tp_prefill_equivalence(self, P):
        """PCP output (reassembled head-tail chunks) == TP output (head-sharded),
        computed on P devices via shard_map over the same random q/k/v."""
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        dtype = jnp.float32
        page = 16
        # nkv must be divisible by P (TP shards KV heads across the mesh).
        S, nq, nkv, hd = 512, 8, 4, 128
        C = S // (2 * P)
        npages = cdiv(S, page)
        sm = hd**-0.5
        rng = np.random.default_rng(0)
        q = self._rand(rng, (S, nq, hd), dtype)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        mesh = Mesh(np.array(jax.devices()[:P]), ("x", ))
        pi = jnp.arange(npages, dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], jnp.int32)

        # TP: shard heads, full-S causal.
        hs = PS(None, "x", None)

        @partial(shard_map, mesh=mesh, in_specs=(hs, hs, hs), out_specs=hs,
                 check_rep=False)
        def tp(q, k, v):
            out, _ = ragged_paged_attention(
                q, k, v, self._flat_cache(k.shape[1], npages, page, hd, dtype),
                jnp.array([S], jnp.int32), pi, jnp.array([0, S], jnp.int32),
                dist, sm_scale=sm, use_causal_mask=True, update_kv_cache=True)
            return out

        out_tp = np.asarray(jax.jit(tp)(q, k, v), np.float32)

        # PCP: all-gather KV (replicated here), head-tail sequence shard.
        chunks = []
        for r in range(P):
            for ch in (r, 2 * P - 1 - r):
                chunks.append(q[ch * C:ch * C + C])
        q2 = jnp.stack(chunks).reshape(P, 2, C, nq, hd)
        qsp = PS("x", None, None, None, None)

        @partial(shard_map, mesh=mesh, in_specs=(qsp, PS(), PS()),
                 out_specs=qsp, check_rep=False)
        def pcp(q2, k, v):
            r = jax.lax.axis_index("x")
            cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
            cc = self._flat_cache(nkv, npages, page, hd, dtype)
            offs = (r * C, (2 * P - 1 - r) * C)
            outs = []
            for i in range(2):
                qpos = jax.lax.reshape(offs[i], (1, )).astype(jnp.int32)
                qb = jnp.zeros((S, nq, hd), dtype).at[:C].set(q2[0][i])
                o, _ = ragged_paged_attention(
                    qb, k, v, cc, jnp.array([S], jnp.int32), pi,
                    jnp.array([0, C], jnp.int32), dist, cp_rank=cp_rank,
                    cp_group_size=P, all_gather_kv=True, q_pos_offsets=qpos,
                    sm_scale=sm, update_kv_cache=False, use_causal_mask=True)
                outs.append(o[:C])
            return jnp.stack(outs)[None]

        og = np.asarray(jax.jit(pcp)(q2, k, v), np.float32)  # [P, 2, C, nq, hd]
        out_pcp = np.zeros((S, nq, hd), np.float32)
        for r in range(P):
            for i, ch in enumerate((r, 2 * P - 1 - r)):
                out_pcp[ch * C:ch * C + C] = og[r, i]

        # Guard against a trivially-passing all-zero/NaN match.
        self.assertTrue(np.all(np.isfinite(out_tp)))
        self.assertGreater(float(np.abs(out_tp).max()), 0.0)
        self.assertAllClose(out_pcp, out_tp, atol=self._tol(dtype),
                            rtol=self._tol(dtype))

    @parameterized.product(P=[2, 3, 4])
    def test_pcp_kv_cache_write_matches_merged(self, P):
        """De-strided PCP cache write == merge_kv(current KV): whole-sequence
        view of test_pcp_strided_cache_write, run across P devices."""
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        dtype = jnp.float32
        page, nq, nkv, hd, C = 16, 8, 2, 128, 64
        S = 2 * P * C  # gathered current KV length
        npages = cdiv(S, page)
        rng = np.random.default_rng(7)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        ref = np.asarray(merge_kv(k, v))
        mesh = Mesh(np.array(jax.devices()[:P]), ("x", ))

        @partial(shard_map, mesh=mesh, in_specs=(PS(), PS()),
                 out_specs=PS("x", None, None, None, None, None),
                 check_rep=False)
        def fn(k, v):
            r = jax.lax.axis_index("x")
            cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
            cc = self._flat_cache(nkv, npages, page, hd, dtype)
            q = jnp.zeros((S, nq, hd), dtype)
            _, nc = ragged_paged_attention(
                q, k, v, cc, jnp.array([S], jnp.int32),
                jnp.arange(npages, dtype=jnp.int32), jnp.array([0, C], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32), cp_rank=cp_rank,
                cp_group_size=P, all_gather_kv=True, update_kv_cache=True,
                use_causal_mask=False)
            return nc[None]

        caches = np.asarray(jax.jit(fn)(k, v))  # [P, npages, page, h1, h2, hd]
        flat = caches.reshape(P, -1, caches.shape[3], caches.shape[4], hd)
        # DCP-strided: global token g -> rank g%P, local slot g//P.
        g = np.arange(S)
        recon = flat[g % P, g // P]
        # Guard against a silently-zeroed cache (would trivially compare equal
        # only if the reference were also zero, but assert it explicitly).
        self.assertTrue(np.all(np.isfinite(recon)))
        self.assertGreater(float(np.abs(recon).max()), 0.0)
        self.assertLess(float((recon == 0).mean()), 0.5)
        self.assertArraysEqual(recon, ref)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
