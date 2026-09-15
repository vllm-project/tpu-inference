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
        would write the whole strided current KV; `kv_write_seq_mask` must
        make it happen exactly once -- on the tail seq. The
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
                q,
                k,
                v,
                self._empty_cache(dtype),
                self._pad1([S, S]),  # both seqs: same request
                self._pi2(pps),
                self._padcu([0, C, C + tail_real]),
                jnp.array([0, 0, 2], jnp.int32),
                cp_rank=jnp.array([r], jnp.int32),
                cp_group_size=P,
                kv_cache_lens=self._pad1([0, 0]),
                q_pos_offsets=self._pad1([head_off, tail_off]),
                update_kv_cache=True,
                # Head+tail are the same request: only the tail writes.
                kv_write_seq_mask=self._pad1([0, 1]),
                skip_cache_attn=True,
                use_causal_mask=True)
            flat = cache.reshape(-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            local_len = (S + P - 1 - r) // P
            pi = np.arange(pps)
            for m in range(local_len):
                g = r + m * P
                slot = pi[m // self.PAGE] * self.PAGE + m % self.PAGE
                self.assertArraysEqual(flat[slot], kv_merged[g])

    # --------------------- multi-request PCP (R > 1) -------------------------
    def _multireq_layout(self, P, reqs):
        """Host-side metadata for R requests fused as 2R seqs.

        Mirrors the layout `_prepare_inputs` builds: request
        i is zigzag-chunked on its own into 2P chunks of C_i, and occupies seq
        2i (head) and 2i+1 (tail). `reqs` is a list of (n_i, L_i).
        """
        two_p = 2 * P
        n = [r[0] for r in reqs]
        L = [r[1] for r in reqs]
        R = len(reqs)
        C = [cdiv(ni, two_p) for ni in n]
        W = [2 * ci for ci in C]
        off = [sum(W[:i]) for i in range(R)]
        S = sum(W)  # live tokens per rank
        # Request-major token-ordered new-KV buffer: request i's 2P*C_i slots
        # start at kv_new_starts[i].
        kv_starts = [two_p * sum(C[:i]) for i in range(R)]
        kv_total = two_p * sum(C)
        # Pages: request i owns a disjoint block so stray writes are detectable.
        pps = max(cdiv(cdiv(Li + ni, P), self.PAGE) for ni, Li in zip(n, L))
        pages = [
            np.arange(i * pps, (i + 1) * pps, dtype=np.int32) for i in range(R)
        ]

        cu = [0]
        for i in range(R):
            cu += [off[i] + C[i], off[i] + 2 * C[i]]
        # Request-level values are duplicated across the request's two seqs.
        per_seq = lambda xs: [x for x in xs for _ in (0, 1)]  # noqa: E731
        return dict(R=R,
                    C=C,
                    off=off,
                    S=S,
                    n=n,
                    L=L,
                    two_p=two_p,
                    kv_starts=kv_starts,
                    kv_total=kv_total,
                    pps=pps,
                    pages=pages,
                    cu_q_lens=self._padcu(cu),
                    kv_lens=self._pad1(
                        per_seq([Li + ni for ni, Li in zip(n, L)])),
                    kv_cache_lens=self._pad1(per_seq(L)),
                    kv_new_starts=self._pad1(per_seq(kv_starts)),
                    kv_write_seq_mask=self._pad1([0, 1] * R),
                    page_indices=self._pi_multi(pages, pps),
                    distribution=jnp.array([0, 0, 2 * R], jnp.int32))

    def _pi_multi(self, pages, pps):
        """page_indices where seqs 2i and 2i+1 both carry request i's pages.

        The writing seq is the tail, and page lookup is `seq_idx *
        pages_per_seq`, so a zero tail row would send every write to page 0.
        """
        rows = []
        for pg in pages:
            row = np.zeros(pps, np.int32)
            row[:len(pg)] = pg
            rows.extend([row, row])
        flat = np.concatenate(rows)
        return jnp.array(np.pad(flat, (0, self.MAX_SEQ * pps - len(flat))),
                         jnp.int32)

    def _qpos_multi(self, lay, r):
        """q_pos_offsets for rank r: head chunk r, tail chunk 2P-1-r."""
        q = []
        for i in range(lay["R"]):
            q += [r * lay["C"][i], (lay["two_p"] - 1 - r) * lay["C"][i]]
        return self._pad1(q)

    @parameterized.product(P=[2, 4])
    def test_pcp_multirequest_fused_write(self, P):
        """R requests in one launch: each writes its OWN strided share, once.

        The whole cache is pre-filled with random values and compared against a
        host-computed expectation, so this catches a write that lands on the
        wrong request's pages, writes twice, or spills outside its range --
        which is the failure mode `kv_write_seq_mask` and `kv_new_starts` exist
        to prevent.
        """
        dtype = jnp.float32
        self.PAGE = 16
        nq, nkv, hd = 8, 2, 128
        # Ragged lengths, mixed cache state, incl. a first-chunk request (L=0).
        reqs = [(70, 0), (33, 48), (16, 96)]
        lay = self._multireq_layout(P, reqs)
        R = lay["R"]
        c = self._cfg(dtype)
        rng = np.random.default_rng(7)

        # Per-request current K/V, laid out request-major in token order.
        kv_buf_k = np.zeros((lay["kv_total"], nkv, hd), np.float32)
        kv_buf_v = np.zeros((lay["kv_total"], nkv, hd), np.float32)
        per_req_kv = []
        for i in range(R):
            ki = rng.random((lay["n"][i], nkv, hd), np.float32)
            vi = rng.random((lay["n"][i], nkv, hd), np.float32)
            s = lay["kv_starts"][i]
            kv_buf_k[s:s + lay["n"][i]] = ki
            kv_buf_v[s:s + lay["n"][i]] = vi
            per_req_kv.append(
                merge_kv(jnp.array(ki, dtype), jnp.array(vi, dtype)))
        k = jnp.array(kv_buf_k, dtype)
        v = jnp.array(kv_buf_v, dtype)
        q = self._rand(rng, (lay["S"], nq, hd), dtype)

        # Pre-fill the entire cache so any stray write shows up. Keep it on the
        # host: kv_cache is a donated argument, so each rank's call consumes
        # its device buffer and the next iteration needs a fresh one.
        cache_shape = (self.NUM_PAGES, self.PAGE, c["nkv2"] // c["kvp"],
                       c["kvp"], c["phd"])
        cache_np = rng.random(cache_shape, np.float32)

        for r in range(P):
            _, cache = ragged_paged_attention(
                q,
                k,
                v,
                jnp.array(cache_np, dtype),
                lay["kv_lens"],
                lay["page_indices"],
                lay["cu_q_lens"],
                lay["distribution"],
                cp_rank=jnp.array([r], jnp.int32),
                cp_group_size=P,
                kv_cache_lens=lay["kv_cache_lens"],
                q_pos_offsets=self._qpos_multi(lay, r),
                kv_new_starts=lay["kv_new_starts"],
                kv_write_seq_mask=lay["kv_write_seq_mask"],
                update_kv_cache=True,
                skip_cache_attn=True,
                use_causal_mask=True)

            # Expected: rank r's strided share of each request's NEW tokens,
            # at global positions [L_i, L_i + n_i), and nothing else.
            flat_shape = (-1, c["nkv2"] // c["kvp"], c["kvp"], c["phd"])
            expected = cache_np.reshape(flat_shape).copy()
            wrote = 0
            for i in range(R):
                Li, ni = lay["L"][i], lay["n"][i]
                for g in range(Li, Li + ni):
                    if g % P != r:
                        continue
                    m = g // P  # local slot for a token this rank owns
                    slot = (lay["pages"][i][m // self.PAGE] * self.PAGE +
                            m % self.PAGE)
                    expected[slot] = np.array(per_req_kv[i][g - Li],
                                              np.float32)
                    wrote += 1
            self.assertGreater(wrote, 0, "test config wrote nothing")
            self.assertAllClose(np.array(cache.reshape(flat_shape),
                                         np.float32),
                                expected,
                                atol=self._tol(dtype),
                                rtol=self._tol(dtype))

    @parameterized.product(P=[2, 4])
    def test_pcp_multirequest_current_phase_output(self, P):
        """R requests in one launch: every chunk's attention output is right.

        Exercises per-seq cu_q_lens, q_pos_offsets, kv_cache_lens and
        kv_new_starts together. A wrong kv_new_starts makes a request attend
        another request's K/V -- shapes stay valid, values do not, which is
        exactly what this catches.
        """
        dtype = jnp.float32
        self.PAGE = 16
        nq, nkv, hd = 8, 2, 128
        reqs = [(70, 0), (33, 48), (16, 96)]
        lay = self._multireq_layout(P, reqs)
        R, C, off, two_p = lay["R"], lay["C"], lay["off"], lay["two_p"]
        rng = np.random.default_rng(11)

        kv_buf_k = np.zeros((lay["kv_total"], nkv, hd), np.float32)
        kv_buf_v = np.zeros((lay["kv_total"], nkv, hd), np.float32)
        q_req, exp = [], []
        for i in range(R):
            ni = lay["n"][i]
            ki = rng.random((ni, nkv, hd), np.float32)
            vi = rng.random((ni, nkv, hd), np.float32)
            qi = rng.random((ni, nq, hd), np.float32)
            s = lay["kv_starts"][i]
            kv_buf_k[s:s + ni] = ki
            kv_buf_v[s:s + ni] = vi
            q_req.append(qi)
            # Reference: plain full-causal attention over this request's own
            # current chunk. With skip_cache_attn the current phase masks to
            # k_span >= kv_cache_len_local, so a query at within-current
            # position p attends current tokens 0..p whatever L_i is.
            e, _ = ref_ragged_paged_attention(jnp.array(qi, dtype),
                                              jnp.array(ki, dtype),
                                              jnp.array(vi, dtype),
                                              self._empty_cache(dtype),
                                              self._pad1([ni]),
                                              self._pi(cdiv(ni, self.PAGE)),
                                              self._padcu([0, ni]),
                                              jnp.array([0, 0, 1], jnp.int32))
            exp.append(e[:ni])
        k = jnp.array(kv_buf_k, dtype)
        v = jnp.array(kv_buf_v, dtype)

        checked = 0
        for r in range(P):
            # Scatter each request's queries into this rank's slots.
            q_buf = np.zeros((lay["S"], nq, hd), np.float32)
            for i in range(R):
                for h in (0, 1):
                    ch = r if h == 0 else two_p - 1 - r
                    for j in range(C[i]):
                        t = ch * C[i] + j
                        if t < lay["n"][i]:
                            q_buf[off[i] + h * C[i] + j] = q_req[i][t]
            out, _ = ragged_paged_attention(
                jnp.array(q_buf, dtype),
                k,
                v,
                self._empty_cache(dtype),
                lay["kv_lens"],
                lay["page_indices"],
                lay["cu_q_lens"],
                lay["distribution"],
                cp_rank=jnp.array([r], jnp.int32),
                cp_group_size=P,
                kv_cache_lens=lay["kv_cache_lens"],
                q_pos_offsets=self._qpos_multi(lay, r),
                kv_new_starts=lay["kv_new_starts"],
                kv_write_seq_mask=lay["kv_write_seq_mask"],
                update_kv_cache=False,
                skip_cache_attn=True,
                use_causal_mask=True)
            for i in range(R):
                for h in (0, 1):
                    ch = r if h == 0 else two_p - 1 - r
                    for j in range(C[i]):
                        t = ch * C[i] + j
                        if t >= lay["n"][i]:
                            continue  # padding row: value is unspecified
                        self.assertAllClose(out[off[i] + h * C[i] + j],
                                            exp[i][t],
                                            atol=self._tol(dtype),
                                            rtol=self._tol(dtype))
                        checked += 1
        # Every real token of every request, on every rank, exactly once.
        self.assertEqual(checked, sum(lay["n"]))

    @parameterized.product(P=[2])
    def test_pcp_zero_length_seq_in_range(self, P):
        """REPRO ONLY -- the kernel HANGS on a zero-length seq. Opt-in.

        A zero-length seq inside the distribution range wedges the kernel's
        per-sequence state machine; this reproduces the stall seen on a real
        8k server. It is skipped by default because it hangs rather than
        fails: the wall-clock assert below never runs, since control never
        returns from the kernel call. Run deliberately, under an external
        timeout:

            PCP_RUN_HANG_REPRO=1 pytest -k zero_length   # expect a hang

        The fix for the server stall was made in the CALLERS (they no longer
        emit zero-length seqs -- see test_multirequest_fewer_reqs_than_bucket
        in the interface suite). Teaching the kernel to tolerate the shape is
        a separate change; until then this documents the limitation.
        """
        import os
        import time
        if os.environ.get("PCP_RUN_HANG_REPRO") != "1":
            self.skipTest("hangs the kernel by construction; opt in with "
                          "PCP_RUN_HANG_REPRO=1 under an external timeout")
        dtype = jnp.float32
        self.PAGE = 16
        nq, nkv, hd = 8, 2, 128
        C, L = 32, 64
        rng = np.random.default_rng(23)
        q = self._rand(rng, (2 * C, nq, hd), dtype)
        k = self._rand(rng, (2 * C, nkv, hd), dtype)
        v = self._rand(rng, (2 * C, nkv, hd), dtype)
        pps = cdiv(cdiv(L + 2 * C, P), self.PAGE)

        # seq0 = [0, C), seq1 = [C, C)  <-- EMPTY, seq2 = [C, 2C)
        cu = self._padcu([0, C, C, 2 * C])
        pi = np.zeros(self.MAX_SEQ * pps, np.int32)
        for s in range(3):
            pi[s * pps:(s + 1) * pps] = np.arange(pps)

        t0 = time.time()
        out, _ = ragged_paged_attention(
            q,
            k,
            v,
            self._empty_cache(dtype),
            self._pad1([L + 2 * C] * 3),
            jnp.asarray(pi),
            cu,
            jnp.array([0, 0, 3], jnp.int32),  # all three seqs in range
            cp_rank=jnp.array([0], jnp.int32),
            cp_group_size=P,
            kv_cache_lens=self._pad1([L] * 3),
            q_pos_offsets=self._pad1([0, C, C]),
            update_kv_cache=False,
            skip_current_attn=True,
            use_causal_mask=False)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 120, "zero-length seq stalled the kernel")
        # The real seqs must still produce finite output.
        self.assertTrue(np.all(np.isfinite(np.asarray(out[:C]))))

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

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], P=[2, 4, 8])
    def test_pcp_ring_cache_phase_matches_full_cache(self, dtype, P):
        self._ring_vs_full_cache(dtype, P)

    def _ring_vs_full_cache(self, dtype, P):
        """In-kernel ring over the striped cache == the same local Q attending
        the whole un-striped cache.

        This is the property that makes the ring a drop-in cache phase: no Q
        all-gather and no output collective, so each rank's result must already
        be the full-cache answer for its own tokens.
        """
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        self.PAGE = 64
        # Lprev % P != 0 so the ranks' stripes have different lengths and a
        # round that uses the wrong originating rank's length is visible.
        Lprev, C, nq, nkv, hd = 1002, 256, 8, 2, 128
        Scur = C * P
        kv_total = Lprev + Scur
        pps = cdiv(kv_total, self.PAGE)
        sm = hd**-0.5
        rng = np.random.default_rng(11)
        k_prev = self._rand(rng, (Lprev, nkv, hd), dtype)
        v_prev = self._rand(rng, (Lprev, nkv, hd), dtype)
        q_all = self._rand(rng, (P, C, nq, hd), dtype)  # rank-major local Q

        kv_lens = self._pad1([kv_total])
        kv_cache_lens = self._pad1([Lprev])
        pi = self._pi(pps)
        dist = jnp.array([0, 0, 1], jnp.int32)
        cu = self._padcu([0, C])
        # Small blocks so both the bq loop and the multi-block ring run.
        blocks = (128, 128, 128, 128)
        dummy_kv = jnp.zeros((C, nkv, hd), dtype)

        # Reference: plain paged attention over the whole cache, per rank.
        # kv_cache is donated, so each call needs its own (identical) copy.
        ref = np.stack([
            np.asarray(
                ragged_paged_attention(q_all[r],
                                       dummy_kv,
                                       dummy_kv,
                                       self._cache_from_kv(
                                           k_prev, v_prev, Lprev, dtype),
                                       kv_lens,
                                       pi,
                                       cu,
                                       dist,
                                       kv_cache_lens=kv_cache_lens,
                                       cp_rank=jnp.array([0], jnp.int32),
                                       cp_group_size=1,
                                       skip_current_attn=True,
                                       use_causal_mask=False,
                                       update_kv_cache=False,
                                       return_lse=True,
                                       sm_scale=sm,
                                       m_block_sizes=blocks)[0], np.float32)
            for r in range(P)
        ])

        # Ring: rank r holds global cache tokens r, r+P, r+2P, ...
        cache_sh = jnp.stack([
            self._cache_from_kv(k_prev[r::P], v_prev[r::P],
                                k_prev[r::P].shape[0], dtype) for r in range(P)
        ])
        mesh = Mesh(np.array(jax.devices()[:P]), ("pcp", ))
        qsp = PS("pcp", None, None, None)
        csp = PS("pcp", None, None, None, None)

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(qsp, csp),
                 out_specs=qsp,
                 check_rep=False)
        def ring(q_l, c_l):
            r = jax.lax.axis_index("pcp")
            o, _, _ = ragged_paged_attention(q_l[0],
                                             dummy_kv,
                                             dummy_kv,
                                             c_l[0],
                                             kv_lens,
                                             pi,
                                             cu,
                                             dist,
                                             kv_cache_lens=kv_cache_lens,
                                             cp_rank=jax.lax.reshape(
                                                 r, (1, )).astype(jnp.int32),
                                             cp_group_size=P,
                                             pcp_ring_axis_name="pcp",
                                             skip_current_attn=True,
                                             use_causal_mask=False,
                                             update_kv_cache=False,
                                             return_lse=True,
                                             sm_scale=sm,
                                             m_block_sizes=blocks)
            return o[None]

        out = np.asarray(jax.jit(ring)(q_all, cache_sh), np.float32)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.abs(out).max()), 0.0)
        tol = 2e-3 if dtype == jnp.float32 else 2e-2
        self.assertAllClose(out, ref, atol=tol, rtol=tol)

    @parameterized.product(dtype=[jnp.float32, jnp.bfloat16], P=[2, 4])
    def test_pcp_ring_multi_seq_matches_full_cache(self, dtype, P):
        """The ring with SEVERAL seqs in one launch, each its own request.

        Multi-request PCP hands the cache phase one seq per request, each with
        its own cached length (one of them zero), its own row count and its
        own page list, all in one launch.  The ring must (a) stream each seq's
        own cache and no other's, (b) keep its cross-rank chain running
        across seq boundaries without a per-seq restart, and (c) stay in
        lock-step across ranks with per-seq block counts.  Reference: plain
        paged attention over each seq's whole un-striped cache.

        Row counts are chosen so that one seq spans several query tiles with
        the boundary mid-tile, one is a single partial tile, and cached lengths
        are not multiples of P so rank stripes differ in length.
        """
        if jax.device_count() < P:
            self.skipTest(f"needs >= {P} devices")
        self.PAGE = 64
        nq, nkv, hd = 8, 2, 128
        sm = hd**-0.5
        # (cached tokens, local query rows) per seq.  Rows are per rank.
        seqs = [(1002, 300), (0, 96), (517, 130)]
        R = len(seqs)
        blocks = (128, 128, 128, 128)  # bq = 128 -> 300 rows = 3 tiles
        rng = np.random.default_rng(23)

        # Each seq owns a disjoint page range of width pps in the cache.
        pps = cdiv(max(L for L, _ in seqs), self.PAGE)
        assert R * pps <= self.NUM_PAGES

        kv_prev = [(self._rand(rng, (L, nkv, hd),
                               dtype), self._rand(rng, (L, nkv, hd), dtype))
                   for L, _ in seqs]
        rows = [n for _, n in seqs]
        # Rank-major local Q: rank r's rows for seq i are
        # q_all[r, cu[i]:cu[i+1]].
        total_rows = sum(rows)
        q_all = self._rand(rng, (P, total_rows, nq, hd), dtype)
        cu_list = [0]
        for n in rows:
            cu_list.append(cu_list[-1] + n)
        cu = self._padcu(cu_list)
        # kv_lens - kv_cache_lens is the "new" length, which the cache phase
        # never reads; keep it > 0 like a real step (rows per rank * P).
        kv_lens = self._pad1([L + n * P for L, n in seqs])
        kv_cache_lens = self._pad1([L for L, _ in seqs])
        pi = jnp.pad(jnp.arange(R * pps, dtype=jnp.int32),
                     (0, self.MAX_SEQ * pps - R * pps))
        dist = jnp.array([0, 0, R], jnp.int32)
        dummy_kv = jnp.zeros((total_rows, nkv, hd), dtype)

        def cache_with(kv_by_seq):
            """Cache whose page range i holds kv_by_seq[i] = (k, v, ntok)."""
            c = self._cfg(dtype)
            cache = self._empty_cache(dtype)
            for i, (k, v, ntok) in enumerate(kv_by_seq):
                if ntok == 0:
                    continue
                kv = merge_kv(k, v)
                pad = cdiv(ntok, self.PAGE) * self.PAGE - ntok
                kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                             constant_values=jnp.nan).reshape(
                                 -1, self.PAGE, c["nkv2"] // c["kvp"],
                                 c["kvp"], c["phd"])
                cache = cache.at[i * pps:i * pps + kv.shape[0]].set(kv)
            return cache

        # Reference, per rank: whole un-striped caches, cp_group_size=1.
        ref = np.stack([
            np.asarray(
                ragged_paged_attention(
                    q_all[r],
                    dummy_kv,
                    dummy_kv,
                    cache_with([(k, v, L)
                                for (k, v), (L, _) in zip(kv_prev, seqs)]),
                    kv_lens,
                    pi,
                    cu,
                    dist,
                    kv_cache_lens=kv_cache_lens,
                    cp_rank=jnp.array([0], jnp.int32),
                    cp_group_size=1,
                    skip_current_attn=True,
                    use_causal_mask=False,
                    update_kv_cache=False,
                    return_lse=True,
                    sm_scale=sm,
                    m_block_sizes=blocks)[0], np.float32) for r in range(P)
        ])

        # Ring: rank r holds tokens r, r+P, ... of every seq's cache.
        cache_sh = jnp.stack([
            cache_with([(k[r::P], v[r::P], k[r::P].shape[0])
                        for (k, v) in kv_prev]) for r in range(P)
        ])
        mesh = Mesh(np.array(jax.devices()[:P]), ("pcp", ))
        qsp = PS("pcp", None, None, None)
        csp = PS("pcp", None, None, None, None)

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(qsp, csp),
                 out_specs=qsp,
                 check_rep=False)
        def ring(q_l, c_l):
            r = jax.lax.axis_index("pcp")
            o, _, _ = ragged_paged_attention(q_l[0],
                                             dummy_kv,
                                             dummy_kv,
                                             c_l[0],
                                             kv_lens,
                                             pi,
                                             cu,
                                             dist,
                                             kv_cache_lens=kv_cache_lens,
                                             cp_rank=jax.lax.reshape(
                                                 r, (1, )).astype(jnp.int32),
                                             cp_group_size=P,
                                             pcp_ring_axis_name="pcp",
                                             skip_current_attn=True,
                                             use_causal_mask=False,
                                             update_kv_cache=False,
                                             return_lse=True,
                                             sm_scale=sm,
                                             m_block_sizes=blocks)
            return o[None]

        out = np.asarray(jax.jit(ring)(q_all, cache_sh), np.float32)
        tol = 2e-3 if dtype == jnp.float32 else 2e-2
        for i, (L, _) in enumerate(seqs):
            got = out[:, cu_list[i]:cu_list[i + 1]]
            exp = ref[:, cu_list[i]:cu_list[i + 1]]
            if L == 0:
                # Empty cache: fully masked, zero output on both sides.
                self.assertTrue(np.all(got == 0.0), f"seq {i}")
                continue
            self.assertTrue(np.all(np.isfinite(got)), f"seq {i}")
            self.assertGreater(float(np.abs(got).max()), 0.0, f"seq {i}")
            self.assertAllClose(got,
                                exp,
                                atol=tol,
                                rtol=tol,
                                err_msg=f"seq {i}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
