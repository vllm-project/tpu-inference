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
"""Correctness tests for prefill context parallelism (PCP) in RPA v3."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS

from tpu_inference.kernels.experimental.rpa_v3_cp.kernel import (
    merge_kv, ragged_paged_attention, ref_ragged_paged_attention)
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
        return jnp.array(rng.random(size=shape,
                                    dtype=np.float32)).astype(dtype)

    def _empty_cache(self, dtype):
        c = self._cfg(dtype)
        return jnp.full((self.NUM_PAGES, self.PAGE, c["nkv2"] // c["kvp"],
                         c["kvp"], c["phd"]), jnp.nan, dtype)

    def _cache_from_kv(self, k, v, ntok, dtype):
        c = self._cfg(dtype)
        kv = merge_kv(k, v)
        pad = cdiv(ntok, self.PAGE) * self.PAGE - ntok
        kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                     constant_values=jnp.nan).reshape(-1, self.PAGE,
                                                      c["nkv2"] // c["kvp"],
                                                      c["kvp"], c["phd"])
        cache = self._empty_cache(dtype)
        return cache.at[:kv.shape[0]].set(kv)

    def _flat_cache(self, nkv_local, npages, page, hd, dtype):
        kvp = get_dtype_packing(dtype)
        nkv2 = align_to(2 * nkv_local, kvp)
        return jnp.zeros((npages, page, nkv2 // kvp, kvp, hd), dtype)

    def _pi(self, npages):
        pi = jnp.arange(npages, dtype=jnp.int32)
        return jnp.pad(pi, (0, self.MAX_SEQ * npages - npages))

    def _pi2(self, npages):
        """page_indices for the fused current phase: seq0 and seq1 are the SAME
        request, so both need the request's pages. The kernel indexes them as
        `seq_idx * pages_per_seq`, and the WRITING seq (the tail, seq1) reads its
        own slice -- so it must be a copy of seq0's, not zeros."""
        pi = jnp.arange(npages, dtype=jnp.int32)
        two = jnp.concatenate([pi, pi])
        return jnp.pad(two, (0, self.MAX_SEQ * npages - 2 * npages))

    def _pad1(self, xs):  # length max_num_seqs (kv_lens, kv_cache_lens, q_pos)
        return jnp.pad(jnp.array(xs, jnp.int32), (0, self.MAX_SEQ - len(xs)))

    def _padcu(self, xs):  # length max_num_seqs + 1 (cu_q_lens)
        return jnp.pad(jnp.array(xs, jnp.int32),
                       (0, self.MAX_SEQ + 1 - len(xs)))

    def _merge_lse(self, acc_o, acc_l, o, lse):
        if acc_o is None:
            return o, lse
        m = jnp.maximum(acc_l, lse)
        e1 = jnp.exp(acc_l - m)
        e2 = jnp.exp(lse - m)
        o = (acc_o * e1[..., None] + o * e2[..., None]) / (e1 + e2)[..., None]
        return o, m + jnp.log(e1 + e2)

    def _tol(self, dtype):
        return 0.05 if dtype == jnp.float32 else 0.2

    # ------------------------------ tests ------------------------------------
    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], P=[1, 2, 4])
    def test_pcp_current_head_tail(self, dtype, P):
        """Current-phase head/tail chunks vs full-causal reference.

        num_computed=0, so only the current phase runs: each head/tail chunk
        attends the (replicated here) current KV causally at its within-current
        position and must match the plain full-causal reference for that chunk.
        """
        self.PAGE = 16
        S, nq, nkv, hd = 256, 8, 2, 128
        C = S // (2 * P)  # S is a multiple of 2P -> every chunk is fully real
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
                q_buf = jnp.zeros((S, nq, hd),
                                  dtype).at[:C].set(q[chunk * C:chunk * C + C])
                out, _ = ragged_paged_attention(
                    q_buf,
                    k,
                    v,
                    self._empty_cache(dtype),
                    self._pad1([S]),
                    self._pi(pps),
                    self._padcu([0, C]),
                    jnp.array([0, 0, 1], jnp.int32),
                    cp_rank=jnp.array([r], jnp.int32),
                    cp_group_size=P,
                    kv_cache_lens=self._pad1([0]),
                    q_pos_offsets=self._pad1([chunk * C]),
                    skip_cache_attn=True,
                    update_kv_cache=False,
                    use_causal_mask=True)
                self.assertAllClose(out[:C],
                                    exp[chunk * C:chunk * C + C],
                                    atol=self._tol(dtype),
                                    rtol=self._tol(dtype))

    @parameterized.product(dtype=[jnp.float32])
    def test_pcp_two_phase_chunked_prefill(self, dtype):
        """Nonzero kv_cache_lens: non-causal prev-cache + causal current, LSE.

        cp_group_size=1 (prev cache replicated, so no cross-rank merge needed on
        one device); the current tokens split into 2 head-tail chunks. Validates
        that ``kv_cache_lens = num_computed`` drives the cache/current split.
        """
        self.PAGE = 16
        Lprev, Scur, nq, nkv, hd = 128, 128, 8, 2, 128
        C = Scur // 2  # cp_group_size=1 -> 2 head-tail chunks
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
            q_buf = jnp.zeros((Scur, nq, hd),
                              dtype).at[:C].set(q_cur[chunk * C:chunk * C + C])
            common = dict(cp_rank=jnp.array([0], jnp.int32),
                          cp_group_size=1,
                          kv_cache_lens=self._pad1([Lprev]),
                          update_kv_cache=False,
                          return_lse=True)
            # kv_cache is donated, so give each phase its own (identical) copy.
            # Cache phase: attend the previous cache (non-causal), no q_pos.
            o1, _, l1 = ragged_paged_attention(
                q_buf,
                k_all[Lprev:],
                v_all[Lprev:],
                self._cache_from_kv(k_all[:Lprev], v_all[:Lprev], Lprev,
                                    dtype),
                self._pad1([kv_total]),
                self._pi(pps),
                self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32),
                use_causal_mask=False,
                skip_current_attn=True,
                **common)
            # Current phase: causal over the current KV (read from HBM).
            o2, _, l2 = ragged_paged_attention(
                q_buf,
                k_all[Lprev:],
                v_all[Lprev:],
                self._cache_from_kv(k_all[:Lprev], v_all[:Lprev], Lprev,
                                    dtype),
                self._pad1([kv_total]),
                self._pi(pps),
                self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32),
                use_causal_mask=True,
                q_pos_offsets=self._pad1([chunk * C]),
                skip_cache_attn=True,
                **common)
            o, _ = self._merge_lse(o1[:C], l1[:C], o2[:C], l2[:C])
            self.assertAllClose(o,
                                exp[chunk * C:chunk * C + C],
                                atol=self._tol(dtype),
                                rtol=self._tol(dtype))

    @parameterized.product(P=[2, 3, 4])
    def test_pcp_strided_cache_write(self, P):
        """Interleaved (strided) write of the current KV, non-causal.

        Each rank writes its 1/P round-robin share (global token g -> rank g%P,
        local slot g//P); de-strided it must equal ``merge_kv`` of the current.
        """
        dtype = jnp.float32
        self.PAGE = 16
        S, nq, nkv, hd = 192, 8, 2, 128
        C = S // (2 * P)
        pps = cdiv(S, self.PAGE)
        rng = np.random.default_rng(7)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        q = self._rand(rng, (S, nq, hd), dtype)
        c = self._cfg(dtype)
        kv_merged = merge_kv(k, v)
        for r in range(P):
            _, cache = ragged_paged_attention(q,
                                              k,
                                              v,
                                              self._empty_cache(dtype),
                                              self._pad1([S]),
                                              self._pi(pps),
                                              self._padcu([0, C]),
                                              jnp.array([0, 0, 1], jnp.int32),
                                              cp_rank=jnp.array([r],
                                                                jnp.int32),
                                              cp_group_size=P,
                                              kv_cache_lens=self._pad1([0]),
                                              update_kv_cache=True,
                                              skip_cache_attn=True,
                                              use_causal_mask=False)
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
        share, even for KV tokens beyond the tail chunk's causal range (the
        ``fetch_kv_len = kv_len`` extension)."""
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
            tail_off = (2 * P - 1 - r) * C  # tail chunk within-current offset
            _, cache = ragged_paged_attention(
                q,
                k,
                v,
                self._empty_cache(dtype),
                self._pad1([S]),
                self._pi(pps),
                self._padcu([0, C]),
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=jnp.array([r], jnp.int32),
                cp_group_size=P,
                kv_cache_lens=self._pad1([0]),
                q_pos_offsets=self._pad1([tail_off]),
                update_kv_cache=True,
                skip_cache_attn=True,
                use_causal_mask=True)
            flat = cache.reshape(-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            local_len = (S + P - 1 - r) // P
            pi = np.arange(pps)
            for m in range(local_len):
                g = r + m * P
                slot = pi[m // self.PAGE] * self.PAGE + m % self.PAGE
                self.assertArraysEqual(flat[slot], kv_merged[g])

    @parameterized.product(P=[2, 4])
    def test_pcp_all_padding_tail_write(self, P):
        """A rank whose tail chunk is entirely padding (q_len=0) must still write
        its full strided KV share. num_current sits low in the next_pow2 bucket
        so the last chunk(s) are all padding; the kernel floors num_bq to >=1 on
        the writing launch so its strided write still runs."""
        dtype = jnp.float32
        self.PAGE = 16
        nq, nkv, hd = 8, 2, 128
        two_p = 2 * P
        # Head chunks (< pcp*C) are all real; put num_current just above pcp*C so
        # the last tail chunk(s) (offset (2P-1-r)*C >= S) are wholly padding.
        C = 64
        S = P * C + 1  # pcp*C < S <= (2P-1)*C -> rank 0's tail chunk is all-pad
        pps = cdiv(S, self.PAGE)
        rng = np.random.default_rng(13)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        q = self._rand(rng, (S, nq, hd), dtype)  # q/k/v must share length
        c = self._cfg(dtype)
        kv_merged = merge_kv(k, v)
        saw_all_pad = False
        for r in range(P):
            tail_off = (two_p - 1 - r) * C
            tail_real = max(0, min(S - tail_off, C))  # clamp -> 0 when all-pad
            saw_all_pad = saw_all_pad or tail_real == 0
            _, cache = ragged_paged_attention(
                q,
                k,
                v,
                self._empty_cache(dtype),
                self._pad1([S]),
                self._pi(pps),
                self._padcu([0, tail_real]),
                jnp.array([0, 0, 1], jnp.int32),
                cp_rank=jnp.array([r], jnp.int32),
                cp_group_size=P,
                kv_cache_lens=self._pad1([0]),
                q_pos_offsets=self._pad1([tail_off]),
                update_kv_cache=True,
                skip_cache_attn=True,
                use_causal_mask=True)
            flat = cache.reshape(-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            local_len = (S + P - 1 - r) // P
            pi = np.arange(pps)
            for m in range(local_len):
                g = r + m * P
                slot = pi[m // self.PAGE] * self.PAGE + m % self.PAGE
                self.assertArraysEqual(flat[slot], kv_merged[g])
        self.assertTrue(
            saw_all_pad, "test config did not exercise an all-pad "
            "tail chunk")

    @parameterized.product(P=[2, 4], aligned=[True, False])
    def test_pcp_fused_current_phase_write(self, P, aligned):
        """FUSED current phase: head+tail as TWO seqs in ONE ragged launch.

        Both seqs are the same request (same kv_lens/kv_cache_lens), so each
        would write the whole strided current KV; `write_last_seq_only` must
        write it exactly once -- on the tail seq. The
        de-strided cache must still equal merge_kv(current) in full, including
        when the tail chunk is wholly padding (aligned=False)."""
        dtype = jnp.float32
        self.PAGE = 16
        nq, nkv, hd = 8, 2, 128
        two_p = 2 * P
        C = 64
        # aligned: every chunk fully real. else: num_current sits just above
        # pcp*C, so the last tail chunk(s) are entirely padding (q_len=0).
        S = two_p * C if aligned else P * C + 1
        pps = cdiv(S, self.PAGE)
        rng = np.random.default_rng(21)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        q = self._rand(rng, (S, nq, hd), dtype)
        c = self._cfg(dtype)
        kv_merged = merge_kv(k, v)
        for r in range(P):
            head_off = r * C
            tail_off = (two_p - 1 - r) * C
            tail_real = max(0, min(S - tail_off, C))
            # one launch, two seqs: cu=[0, C, C+tail_real]
            _, cache = ragged_paged_attention(
                q, k, v, self._empty_cache(dtype),
                self._pad1([S, S]),  # both seqs: same request
                self._pi2(pps), self._padcu([0, C, C + tail_real]),
                jnp.array([0, 0, 2], jnp.int32),
                cp_rank=jnp.array([r], jnp.int32), cp_group_size=P,
                kv_cache_lens=self._pad1([0, 0]),
                q_pos_offsets=self._pad1([head_off, tail_off]),
                update_kv_cache=True, write_last_seq_only=True,
                skip_cache_attn=True, use_causal_mask=True)
            flat = cache.reshape(-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            local_len = (S + P - 1 - r) // P
            pi = np.arange(pps)
            for m in range(local_len):
                g = r + m * P
                slot = pi[m // self.PAGE] * self.PAGE + m % self.PAGE
                self.assertArraysEqual(flat[slot], kv_merged[g])

    # ----------------- multi-device PCP-vs-TP equivalence --------------------
    @parameterized.product(P=[2, 4])
    def test_pcp_vs_tp_prefill_equivalence(self, P):
        """PCP output (reassembled head-tail chunks, all heads) == TP output
        (head-sharded, full sequence), on P devices over the same q/k/v."""
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        dtype = jnp.float32
        page = 16
        S, nq, nkv, hd = 512, 8, 4, 128  # nkv % P == 0 (TP shards KV heads)
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

        hs = PS(None, "x", None)

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(hs, hs, hs),
                 out_specs=hs,
                 check_rep=False)
        def tp(q, k, v):
            out, _ = ragged_paged_attention(q,
                                            k,
                                            v,
                                            self._flat_cache(
                                                k.shape[1], npages, page, hd,
                                                dtype),
                                            jnp.array([S], jnp.int32),
                                            pi,
                                            jnp.array([0, S], jnp.int32),
                                            dist,
                                            sm_scale=sm,
                                            use_causal_mask=True,
                                            update_kv_cache=True)
            return out

        out_tp = np.asarray(jax.jit(tp)(q, k, v), np.float32)

        # PCP: replicated (all-gathered) KV, head-tail sequence shard.
        chunks = []
        for r in range(P):
            for ch in (r, 2 * P - 1 - r):
                chunks.append(q[ch * C:ch * C + C])
        q2 = jnp.stack(chunks).reshape(P, 2, C, nq, hd)
        qsp = PS("x", None, None, None, None)

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(qsp, PS(), PS()),
                 out_specs=qsp,
                 check_rep=False)
        def pcp(q2, k, v):
            r = jax.lax.axis_index("x")
            cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
            cc = self._flat_cache(nkv, npages, page, hd, dtype)
            offs = (r * C, (2 * P - 1 - r) * C)
            outs = []
            for i in range(2):
                qpos = jax.lax.reshape(offs[i], (1, )).astype(jnp.int32)
                qb = jnp.zeros((S, nq, hd), dtype).at[:C].set(q2[0][i])
                o, _ = ragged_paged_attention(qb,
                                              k,
                                              v,
                                              cc,
                                              jnp.array([S], jnp.int32),
                                              pi,
                                              jnp.array([0, C], jnp.int32),
                                              dist,
                                              cp_rank=cp_rank,
                                              cp_group_size=P,
                                              kv_cache_lens=jnp.array(
                                                  [0], jnp.int32),
                                              q_pos_offsets=qpos,
                                              skip_cache_attn=True,
                                              sm_scale=sm,
                                              update_kv_cache=False,
                                              use_causal_mask=True)
                outs.append(o[:C])
            return jnp.stack(outs)[None]

        og = np.asarray(jax.jit(pcp)(q2, k, v),
                        np.float32)  # [P, 2, C, nq, hd]
        out_pcp = np.zeros((S, nq, hd), np.float32)
        for r in range(P):
            for i, ch in enumerate((r, 2 * P - 1 - r)):
                out_pcp[ch * C:ch * C + C] = og[r, i]

        self.assertTrue(np.all(np.isfinite(out_tp)))
        self.assertGreater(float(np.abs(out_tp).max()), 0.0)
        self.assertAllClose(out_pcp,
                            out_tp,
                            atol=self._tol(dtype),
                            rtol=self._tol(dtype))

    @parameterized.product(P=[2, 3, 4])
    def test_pcp_kv_cache_write_matches_merged(self, P):
        """De-strided PCP cache write == merge_kv(current KV), across P devices
        (whole-sequence view of test_pcp_strided_cache_write)."""
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        dtype = jnp.float32
        page, nq, nkv, hd, C = 16, 8, 2, 128, 64
        S = 2 * P * C  # current KV length
        npages = cdiv(S, page)
        rng = np.random.default_rng(7)
        k = self._rand(rng, (S, nkv, hd), dtype)
        v = self._rand(rng, (S, nkv, hd), dtype)
        ref = np.asarray(merge_kv(k, v))
        mesh = Mesh(np.array(jax.devices()[:P]), ("x", ))

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(PS(), PS()),
                 out_specs=PS("x", None, None, None, None, None),
                 check_rep=False)
        def fn(k, v):
            r = jax.lax.axis_index("x")
            cp_rank = jax.lax.reshape(r, (1, )).astype(jnp.int32)
            cc = self._flat_cache(nkv, npages, page, hd, dtype)
            q = jnp.zeros((S, nq, hd), dtype)
            _, nc = ragged_paged_attention(q,
                                           k,
                                           v,
                                           cc,
                                           jnp.array([S], jnp.int32),
                                           jnp.arange(npages, dtype=jnp.int32),
                                           jnp.array([0, C], jnp.int32),
                                           jnp.array([0, 0, 1], jnp.int32),
                                           cp_rank=cp_rank,
                                           cp_group_size=P,
                                           kv_cache_lens=jnp.array([0],
                                                                   jnp.int32),
                                           update_kv_cache=True,
                                           skip_cache_attn=True,
                                           use_causal_mask=False)
            return nc[None]

        caches = np.asarray(jax.jit(fn)(k, v))  # [P, npages, page, h1, h2, hd]
        flat = caches.reshape(P, -1, caches.shape[3], caches.shape[4], hd)
        g = np.arange(S)
        recon = flat[g % P,
                     g // P]  # DCP-strided: token g -> rank g%P, slot g//P
        self.assertTrue(np.all(np.isfinite(recon)))
        self.assertGreater(float(np.abs(recon).max()), 0.0)
        self.assertLess(float((recon == 0).mean()), 0.5)
        self.assertArraysEqual(recon, ref)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
