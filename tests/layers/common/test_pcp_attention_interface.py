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
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    merge_kv, ref_ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)
from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.attention_metadata import (AttentionMetadata,
                                                            PCPMetadata)
from tpu_inference.layers.common.cp_attention import pcp_forward
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)
from tpu_inference.runner.pcp_utils import pcp_seq_arrays, pcp_token_layout

PAGE = 16  # per-rank block_size; the GLOBAL page_size dim is PAGE * pcp
MAX_SEQ = 8
NQ, NKV, HD = 8, 2, 128
DTYPE = jnp.float32
SM_SCALE = HD**-0.5


def _row_perm(pcp):
    """Rank r owns chunk r (head) and chunk 2P-1-r (tail), laid out per rank."""
    return [c for r in range(pcp) for c in (r, 2 * pcp - 1 - r)]


def _inv_row(pcp):
    """Natural chunk index -> its slot in the rank-order layout."""
    inv = np.empty(2 * pcp, np.int64)
    inv[_row_perm(pcp)] = np.arange(2 * pcp)
    return inv


def _to_rank_order(x, pcp, C):
    """Token order -> rank order (what each rank's local shard must contain)."""
    x = np.asarray(x)
    return jnp.asarray(
        x.reshape(2 * pcp, C, *x.shape[1:])[_row_perm(pcp)].reshape(x.shape))


def _kv_token_order(pcp, C):
    """Single-request `kv_token_order`: token t -> its slot in rank order."""
    order = _inv_row(pcp)[:, None] * C + np.arange(C)[None, :]
    return jnp.asarray(order.reshape(-1), jnp.int32)


def _pcp_meta(pcp, C, num_current):
    """The per-rank fused current-phase metadata, exactly as _prepare_inputs
    builds it: cu = [0, C, C + tail_real] and q_pos_offsets = [head, tail]."""
    two_p = 2 * pcp
    cu = np.zeros((pcp, MAX_SEQ + 1), np.int32)
    qpos = np.zeros((pcp, MAX_SEQ), np.int32)
    for r in range(pcp):
        tail_off = (two_p - 1 - r) * C
        tail_real = int(np.clip(num_current - tail_off, 0, C))
        cu[r, 1] = C  # seq 0 (head) is always fully real
        cu[r, 2:] = C + tail_real  # seq 1 (tail) is clamped
        qpos[r, 0] = r * C
        qpos[r, 1] = tail_off
    return jnp.asarray(cu), jnp.asarray(qpos)


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

    def _ref_cache(self, k, v, ntok, npages):
        """Plain (non-strided) cache of `ntok` tokens, for the reference call."""
        kv = merge_kv(k, v)
        pad = cdiv(ntok, PAGE) * PAGE - ntok
        kv = jnp.pad(kv, ((0, pad), (0, 0), (0, 0), (0, 0)),
                     constant_values=jnp.nan)
        kv = kv.reshape(-1, PAGE, *self._cache_dims(1))
        cache = jnp.full((npages, PAGE, *self._cache_dims(1)), jnp.nan, DTYPE)
        return cache.at[:kv.shape[0]].set(kv)

    def _strided_cache(self, k, v, ntok, pcp, npages):
        """The GLOBAL pcp cache. Build each rank's local shard (its g % pcp
        round-robin share at local slot g // pcp) and concatenate along the
        page_size dim -- which is exactly how KV_CONTEXT partitions it."""
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
        # (npages, PAGE * pcp, ...) -- rank r owns columns [r*PAGE, (r+1)*PAGE)
        return jnp.asarray(np.concatenate(shards, axis=1), DTYPE)

    def _page_indices(self, pps):
        """The fused current phase has TWO seqs that are the SAME request. The
        kernel offsets page_indices by `seq_idx * pages_per_seq` and the WRITING
        seq is the tail (seq 1), so seq 1 must carry a COPY of the request's
        pages -- zeros would send every write to page 0."""
        pi = np.arange(pps, dtype=np.int32)
        out = np.zeros(MAX_SEQ * pps, np.int32)
        out[:pps] = pi
        out[pps:2 * pps] = pi
        return jnp.asarray(out)

    def _cache_at_pages(self, k, v, ntok, pcp, pps, npages, phys):
        """Build an `npages`-page global pcp cache in which the request's
        logical page i is stored at physical page `phys[i]` (so the block
        table is non-contiguous), leaving the other pages unused."""
        dims = self._cache_dims(1)
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

    # ---------------------- multi-request (R > 1) ----------------------------
    def _multi_layout(self, pcp, reqs, t_pad):
        """The layout `_prepare_inputs` builds for R live requests (the
        production helper), checked against the token bucket t_pad."""
        C, off, acc = pcp_token_layout([r[0] for r in reqs], pcp)
        assert t_pad % pcp == 0 and t_pad >= pcp * acc, (t_pad, pcp * acc)
        return C, off, acc, t_pad // pcp

    def _slot(self, pcp, C, off, s_pad, i, t):
        """Global row of token t of request i in the rank-order buffer."""
        c = t // C[i]
        r = c if c < pcp else 2 * pcp - 1 - c
        h = 0 if c < pcp else 1
        return r * s_pad + off[i] + h * C[i] + t % C[i]

    def _assert_multi_matches(self, out, exp, reqs, pcp, C, off, s_pad):
        """Every real token of every request is finite and matches its
        per-request reference; padding rows are not read."""
        checked = 0
        for i, (ni, _) in enumerate(reqs):
            for t in range(ni):
                got = out[self._slot(pcp, C, off, s_pad, i, t)]
                self.assertTrue(np.all(np.isfinite(got)))
                self.assertAllClose(got, exp[i][t], atol=2e-2, rtol=2e-2)
                checked += 1
        self.assertEqual(checked, sum(n for n, _ in reqs))

    def _multi_cache(self, prev_kv, pcp, pps, npages):
        """Global pcp cache with request i's cached tokens on pages
        [i*pps, (i+1)*pps) -- disjoint blocks, as the block table says."""
        dims = self._cache_dims(1)
        shards = []
        for r in range(pcp):
            shard = np.full((npages, PAGE, *dims), np.nan, np.float32)
            for i, (k_prev, v_prev, L) in enumerate(prev_kv):
                idx = np.arange(r, L, pcp)
                if not len(idx):
                    continue
                kv = np.asarray(merge_kv(k_prev[idx], v_prev[idx]))
                m = kv.shape[0]
                kv = np.pad(kv, ((0, cdiv(m, PAGE) * PAGE - m), (0, 0), (0, 0),
                                 (0, 0)),
                            constant_values=np.nan).reshape(-1, PAGE, *dims)
                shard[i * pps:i * pps + kv.shape[0]] = kv
            shards.append(shard)
        return jnp.asarray(np.concatenate(shards, axis=1), DTYPE)

    def _run_multi(self,
                   pcp,
                   reqs,
                   t_pad,
                   num_reqs_bucket=None,
                   with_ref=True):
        """Drive pcp_forward with R requests. `reqs` is [(num_current, L)].

        `num_reqs_bucket` is the STATIC PCPMetadata.num_reqs the runner would
        pass (the live count padded to its ladder); the layout and every
        per-seq array cover the live requests only, as in the runner.
        """
        two_p = 2 * pcp
        R = len(reqs)
        n = [r[0] for r in reqs]
        L = [r[1] for r in reqs]
        C, off, s_live, s_pad = self._multi_layout(pcp, reqs, t_pad)
        rng = np.random.default_rng(17)

        prev_kv, cur, exp = [], [], []
        pps = max(cdiv(cdiv(Li + ni, pcp), PAGE) for ni, Li in zip(n, L))
        for i in range(R):
            k_prev = self._rand(rng, (L[i], NKV, HD))
            v_prev = self._rand(rng, (L[i], NKV, HD))
            q_cur = self._rand(rng, (n[i], NQ, HD))
            k_cur = self._rand(rng, (n[i], NKV, HD))
            v_cur = self._rand(rng, (n[i], NKV, HD))
            prev_kv.append((k_prev, v_prev, L[i]))
            cur.append((q_cur, k_cur, v_cur))
            if not with_ref:
                exp.append(None)
                continue
            # Reference: plain full-causal prefill over this request's context.
            ref_pps = cdiv(L[i] + n[i], PAGE)
            e, _ = ref_ragged_paged_attention(
                q_cur,
                k_cur,
                v_cur,
                self._ref_cache(k_prev, v_prev, L[i], ref_pps),
                jnp.pad(jnp.array([L[i] + n[i]], jnp.int32), (0, MAX_SEQ - 1)),
                jnp.pad(jnp.arange(ref_pps, dtype=jnp.int32),
                        (0, MAX_SEQ * ref_pps - ref_pps)),
                jnp.pad(jnp.array([0, n[i]], jnp.int32), (0, MAX_SEQ - 1)),
                jnp.array([0, 0, 1], jnp.int32),
                sm_scale=SM_SCALE)
            exp.append(np.asarray(e[:n[i]]))

        # Token buffers in rank order, plus the request-major K/V permutation.
        def empty(width):
            return np.zeros((t_pad, width, HD), np.float32)

        q_buf, k_buf, v_buf = empty(NQ), empty(NKV), empty(NKV)
        kv_order = np.zeros(t_pad, np.int32)
        for i in range(R):
            kv_base = pcp * off[i]
            for h in (0, 1):
                for r in range(pcp):
                    c = r if h == 0 else two_p - 1 - r
                    for j in range(C[i]):
                        g = r * s_pad + off[i] + h * C[i] + j
                        t = c * C[i] + j
                        kv_order[kv_base + t] = g
                        if t < n[i]:
                            q_buf[g] = np.asarray(cur[i][0][t], np.float32)
                            k_buf[g] = np.asarray(cur[i][1][t], np.float32)
                            v_buf[g] = np.asarray(cur[i][2][t], np.float32)

        def pad1(xs):
            return jnp.pad(jnp.array(xs, jnp.int32), (0, MAX_SEQ - len(xs)))

        per_seq = lambda xs: [x for x in xs for _ in (0, 1)]  # noqa: E731
        cu_row, qpos, kv_new_starts = pcp_seq_arrays(C, off, pcp, MAX_SEQ)
        cu = np.tile(cu_row, (pcp, 1))

        pi = np.zeros(MAX_SEQ * pps, np.int32)
        for i in range(R):
            row = np.arange(i * pps, (i + 1) * pps, dtype=np.int32)
            pi[2 * i * pps:(2 * i + 1) * pps] = row
            pi[(2 * i + 1) * pps:(2 * i + 2) * pps] = row

        md = AttentionMetadata(
            input_positions=jnp.zeros(1, jnp.int32),
            seq_lens=pad1(per_seq([Li + ni for ni, Li in zip(n, L)])),
            block_tables=jnp.asarray(pi),
            request_distribution=jnp.array([0, 0, 2 * R], jnp.int32),
            pcp=PCPMetadata(
                query_start_loc=jnp.asarray(cu),
                kv_cache_lens=pad1(per_seq(L)),
                q_pos_offsets=jnp.asarray(qpos),
                has_cached_kv=max(L) > 0,
                kv_new_starts=jnp.asarray(kv_new_starts),
                kv_token_order=jnp.asarray(kv_order),
                num_reqs=num_reqs_bucket or R,
            ),
        )
        new_cache, out = pcp_forward(self._mesh(pcp),
                                     jnp.asarray(q_buf, DTYPE),
                                     jnp.asarray(k_buf, DTYPE),
                                     jnp.asarray(v_buf, DTYPE),
                                     self._multi_cache(prev_kv, pcp, pps,
                                                       R * pps),
                                     md,
                                     sm_scale=SM_SCALE,
                                     update_kv_cache=True,
                                     use_causal_mask=True)
        return (np.asarray(out), np.asarray(new_cache), exp, cur, C, off,
                s_pad, pps)

    @parameterized.product(pcp=[2, 4])
    def test_multirequest_ragged(self, pcp):
        """R requests of different lengths, mixed cached/uncached, in one call.

        Each request must see only its own K/V: a wrong kv_new_starts or a
        wrong cache-phase seq split silently mixes requests together.
        """
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(96, 0), (48, 64), (33, 32)]
        out, _, exp, _, C, off, s_pad, _ = self._run_multi(pcp, reqs, 256)
        self._assert_multi_matches(out, exp, reqs, pcp, C, off, s_pad)

    @parameterized.product(pcp=[2, 4])
    def test_multirequest_fewer_reqs_than_bucket(self, pcp):
        """Live request count BELOW the static num_reqs bucket, with cache.

        The static bucket only selects the compiled variant; both phases must
        iterate the LIVE seq count from request_distribution and never touch
        the slots that would pad the count up to the bucket (which have no
        rows in the layout).  This shape hung the gather-Q build, whose cache
        phase iterated the static bucket.  Every request has L > 0 so the
        cache phase actually runs.
        """
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(96, 32), (48, 64)]
        out, _, exp, _, C, off, s_pad, _ = self._run_multi(pcp,
                                                           reqs,
                                                           512,
                                                           num_reqs_bucket=4)
        self._assert_multi_matches(out, exp, reqs, pcp, C, off, s_pad)

    @parameterized.product(pcp=[2, 4])
    def test_multirequest_kv_cache_write(self, pcp):
        """Every request's current KV lands in ITS OWN pages, strided by rank.

        With one shared write gate only the last request's KV would be written
        (and to whichever pages seq 2R-1 addresses), so this is the check that
        kv_write_seq_mask and the per-request block-table rows work together.
        """
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(96, 0), (48, 64), (33, 32)]
        _, cache, _, cur, _, _, _, pps = self._run_multi(pcp, reqs, 256)
        for i, (ni, Li) in enumerate(reqs):
            ref = np.asarray(merge_kv(cur[i][1], cur[i][2]))
            for t in range(ni):
                g = Li + t  # global position within request i
                r, local = g % pcp, g // pcp
                page = i * pps + local // PAGE
                got = cache[page, r * PAGE + local % PAGE]
                self.assertAllClose(got, ref[t], atol=2e-2, rtol=2e-2)

    def test_multirequest_seq_spans_ring_tiles(self):
        """A request whose head+tail run is longer than the ring's query tile.

        The ring cache phase gets ONE seq per request (2*C_i rows), and the
        ring's tile is 8192 / NQ = 1024 rows here, so a request with
        2*C_i > 1024 spans several tiles with the head/tail boundary mid-tile,
        and its cache is streamed once per tile.  A second, short request in
        the same launch keeps the seq count > 1 with different tile counts.
        Row-exact against the per-request reference, as the other tests.
        """
        pcp = 2
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        # C_0 = ceil(2500/4) = 625 -> 1250 rows: two tiles, boundary at 625.
        # C_1 = ceil(700/4) = 175 -> 350 rows: one partial tile.
        reqs = [(2500, 64), (700, 32)]
        out, _, exp, _, C, off, s_pad, _ = self._run_multi(pcp, reqs, 3200)
        self._assert_multi_matches(out, exp, reqs, pcp, C, off, s_pad)

    def test_multirequest_nan_probe_headroom_bucket(self):
        """Shape from the 32k E2E run that produced NaN logits: a batch whose
        layout needs P*S = 16388 rows and so lands in the headroom bucket
        (16512) above max_num_batched_tokens, leaving 62 dead rows per rank.
        Every REAL row must be finite; report where NaN shows up otherwise."""
        pcp = 2
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        reqs = [(13037, 204), (3347, 16384)]
        out, _, _, _, C, off, s_pad, _ = self._run_multi(pcp,
                                                         reqs,
                                                         16512,
                                                         with_ref=False)
        bad = []
        for i, (ni, _) in enumerate(reqs):
            for t in range(ni):
                g = self._slot(pcp, C, off, s_pad, i, t)
                if not np.all(np.isfinite(out[g])):
                    bad.append((i, t, g))
        nan_rows = np.where(~np.isfinite(out).all(axis=(1, 2)))[0]
        self.assertEqual(
            bad, [], f"{len(bad)} non-finite REAL rows; first {bad[:10]}; "
            f"non-finite rows overall {len(nan_rows)}: "
            f"{nan_rows[:20].tolist()} ... {nan_rows[-20:].tolist()}")

    def test_multirequest_1m_cache(self):
        """One request with ~1M CACHED tokens (a mid-prompt chunk of a 1M
        context) alongside a short one, against the per-request reference.
        The current chunks are small so the naive reference stays in memory;
        what is exercised is the ring cache phase streaming a 1M-token stripe
        pair with per-seq block counts (512 blocks vs 1), the merge with a
        very long-context LSE, and the runner-side layout at that scale."""
        pcp = 2
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        # 10^6 rather than 2^20: with this harness's 16-token page and 8
        # padded seq rows, 2^20 cached tokens need 8 x 32776 page indices,
        # just over the kernel's 1 MiB SMEM budget (production runs 1M with
        # a 64-token page).
        reqs = [(256, 1_000_000), (300, 5000)]
        out, _, exp, _, C, off, s_pad, _ = self._run_multi(pcp, reqs, 640)
        self._assert_multi_matches(out, exp, reqs, pcp, C, off, s_pad)

    def _run(self, pcp, L, num_current, padded_s):
        """Drive the wrapper; return (out_rank_order, kv_cache, exp_token_order).

        L = num_computed (already in the strided cache), num_current = the real
        current tokens, padded_s = 2*pcp*C (what the token buffers are sized to).
        """
        C = padded_s // (2 * pcp)
        kv_total = L + num_current  # the REAL kv length
        rng = np.random.default_rng(4)

        k_prev = self._rand(rng, (L, NKV, HD))
        v_prev = self._rand(rng, (L, NKV, HD))
        q_cur = self._rand(rng, (num_current, NQ, HD))
        k_cur = self._rand(rng, (num_current, NKV, HD))
        v_cur = self._rand(rng, (num_current, NKV, HD))

        # --- reference: plain full-causal prefill over the whole context ------
        ref_pps = cdiv(kv_total, PAGE)
        ref_pi = jnp.pad(jnp.arange(ref_pps, dtype=jnp.int32),
                         (0, MAX_SEQ * ref_pps - ref_pps))
        # NOTE: ref_ragged_paged_attention defaults to sm_scale=1.0, so it MUST
        # be passed explicitly to match the wrapper.
        exp, _ = ref_ragged_paged_attention(
            q_cur,
            k_cur,
            v_cur,
            self._ref_cache(k_prev, v_prev, L, ref_pps),
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, MAX_SEQ - 1)),
            ref_pi,
            jnp.pad(jnp.array([0, num_current], jnp.int32), (0, MAX_SEQ - 1)),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=SM_SCALE)
        exp = np.asarray(exp[:num_current])

        # --- the wrapper's inputs, exactly as _prepare_inputs lays them out ---
        # Token buffers are padded to padded_s, padding zeroed, then head-tail
        # rearranged into rank order.
        def pad_and_rank_order(x, width):
            buf = np.zeros((padded_s, width, HD), np.float32)
            buf[:num_current] = np.asarray(x, np.float32)
            return _to_rank_order(jnp.asarray(buf, DTYPE), pcp, C)

        q = pad_and_rank_order(q_cur, NQ)
        k = pad_and_rank_order(k_cur, NKV)
        v = pad_and_rank_order(v_cur, NKV)

        # Per-rank local cache must hold ceil(kv_total / pcp) tokens after the write.
        pps = cdiv(cdiv(kv_total, pcp), PAGE)
        npages = max(pps, 1)
        cache = self._strided_cache(k_prev, v_prev, L, pcp, npages)

        def pad1(xs):
            return jnp.pad(jnp.array(xs, jnp.int32), (0, MAX_SEQ - len(xs)))

        # Both fused seqs are the SAME request -> [T, T] / [P, P].
        kv_lens = pad1([kv_total, kv_total])
        kv_cache_lens = pad1([L, L])
        cu_q_lens, q_pos_offsets = _pcp_meta(pcp, C, num_current)
        distribution = jnp.array([0, 0, 2], jnp.int32)  # head + tail

        md = AttentionMetadata(
            input_positions=jnp.zeros(1, jnp.int32),
            seq_lens=kv_lens,
            block_tables=self._page_indices(pps),
            request_distribution=distribution,
            pcp=PCPMetadata(
                query_start_loc=cu_q_lens,
                kv_cache_lens=kv_cache_lens,
                q_pos_offsets=q_pos_offsets,
                kv_new_starts=jnp.zeros(MAX_SEQ, jnp.int32),
                kv_token_order=_kv_token_order(pcp, C),
                has_cached_kv=L > 0,
            ),
        )
        new_cache, out = pcp_forward(self._mesh(pcp),
                                     q,
                                     k,
                                     v,
                                     cache,
                                     md,
                                     sm_scale=SM_SCALE,
                                     update_kv_cache=True,
                                     use_causal_mask=True)
        return np.asarray(out), np.asarray(new_cache), exp, C

    def _assert_matches(self, out, exp, pcp, C, num_current):
        """Compare only the REAL tokens: global token g sits in chunk g // C,
        which the rank-order layout places at slot inv_row[g // C]."""
        inv = _inv_row(pcp)
        rows = np.array(
            [inv[g // C] * C + (g % C) for g in range(num_current)])
        got = out[rows]
        # Guard against a trivially-passing all-zero / NaN "match".
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertGreater(float(np.abs(got).max()), 0.0)
        self.assertAllClose(got, exp, atol=2e-2, rtol=2e-2)

    # ------------------------------ tests ------------------------------------
    @parameterized.product(pcp=[2, 4])
    def test_ring_cache_phase_matches_reference(self, pcp):
        """The ring cache phase must reproduce the full-causal reference.

        The ring streams each rank's KV shard around the pcp axis instead of
        materializing the cache, so any divergence is a synchronization or
        masking bug in the rotation, not a modelling choice.
        """
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 512, 128
        out, _, exp, C = self._run(pcp, L, S, S)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_chunked_prefill(self, pcp):
        """Wrapper output == full-causal reference, for a chunked prefill: L
        previously-computed tokens in the strided cache + a full current chunk."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 128, 128  # S == padded_s: every chunk is fully real
        out, _, exp, C = self._run(pcp, L, S, S)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_partial_tail(self, pcp):
        """num_current < padded_s: the tail chunks are partly (or wholly)
        padding, so `tail_real` differs per rank -- the case that forces
        query_start_loc to be pcp-sharded in the first place."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S, padded_s = 128, 100, 128
        out, _, exp, C = self._run(pcp, L, S, padded_s)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_first_chunk_no_cache(self, pcp):
        """num_computed == 0: the cache phase has nothing to attend, so every
        cache-term LSE is -inf and the combine must fall back cleanly to the
        current term (no NaNs)."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        S = 128
        out, _, exp, C = self._run(pcp, 0, S, S)
        self._assert_matches(out, exp, pcp, C, S)

    @parameterized.product(pcp=[2, 4])
    def test_kv_cache_write(self, pcp):
        """The current KV must land in the strided cache: global token g at
        rank g % pcp, local slot g // pcp -- i.e. global column
        (g % pcp) * PAGE + (g // pcp) % PAGE of page (g // pcp) // PAGE.

        This is what `kv_write_seq_mask` + the duplicated page-index row buy;
        a stale page_indices slice would send every write to page 0."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, S = 128, 128
        rng = np.random.default_rng(4)  # same seed as _run -> same tensors
        _ = self._rand(rng, (L, NKV, HD)), self._rand(rng, (L, NKV, HD))
        _ = self._rand(rng, (S, NQ, HD))
        k_cur, v_cur = self._rand(rng,
                                  (S, NKV, HD)), self._rand(rng, (S, NKV, HD))
        ref = np.asarray(merge_kv(k_cur, v_cur))  # [S, nkv2//kvp, kvp, phd]

        _, cache, _, _ = self._run(pcp, L, S, S)
        for i in range(S):
            g = L + i  # global position of current token i
            r, local = g % pcp, g // pcp
            page, off = local // PAGE, local % PAGE
            got = cache[page, r * PAGE + off]
            self.assertAllClose(got, ref[i], atol=2e-2, rtol=2e-2)

    @parameterized.parameters(2, 4)
    def test_noncontiguous_block_table(self, pcp):
        """The cache phase must be correct when the request's pages are an
        arbitrary, non-contiguous subset of a cache that is bigger than the
        request -- i.e. what a real server looks like, where one KV cache holds
        many sequences (in production num_blocks is ~9x the per-request block
        table width).  The ring fetches each round-0 block through the block
        table, so an indexing bug shows up as a numerical mismatch against the
        plain-attention reference.
        """
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
            q,
            k,
            v,
            self._ref_cache(k_prev, v_prev, L, ref_pps),
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, MAX_SEQ - 1)),
            ref_pi,
            jnp.pad(jnp.array([0, num_current], jnp.int32), (0, MAX_SEQ - 1)),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=SM_SCALE)

        # ---- PCP: cache is 4x the request, pages non-contiguous ----
        pps = cdiv(cdiv(kv_total, pcp), PAGE)
        npages = 4 * pps  # cache bigger than the request
        # arbitrary non-identity, non-contiguous physical placement
        phys = (np.arange(pps) * 3 + 5) % npages
        assert len(set(phys.tolist())) == pps, "phys must be a permutation"
        cache = self._cache_at_pages(k_prev, v_prev, L, pcp, pps, npages, phys)

        pi = np.zeros(MAX_SEQ * pps, np.int32)
        pi[:pps] = phys
        pi[pps:2 * pps] = phys
        pi = jnp.asarray(pi)

        def pad1(xs):
            return jnp.pad(jnp.array(xs, jnp.int32), (0, MAX_SEQ - len(xs)))

        cu, qpos = _pcp_meta(pcp, C, num_current)
        md = AttentionMetadata(
            input_positions=jnp.zeros(1, jnp.int32),
            seq_lens=pad1([kv_total, kv_total]),
            block_tables=pi,
            request_distribution=jnp.array([0, 0, 2], jnp.int32),
            pcp=PCPMetadata(
                query_start_loc=cu,
                kv_cache_lens=pad1([L, L]),
                q_pos_offsets=qpos,
                kv_new_starts=jnp.zeros(MAX_SEQ, jnp.int32),
                kv_token_order=_kv_token_order(pcp, C),
                has_cached_kv=True,
            ),
        )
        _, out = pcp_forward(self._mesh(pcp), _to_rank_order(q, pcp, C),
                             _to_rank_order(k, pcp, C),
                             _to_rank_order(v, pcp, C), cache, md, SM_SCALE)

        inv = _inv_row(pcp)
        got = np.asarray(out).reshape(2 * pcp, C, NQ,
                                      HD)[inv].reshape(num_current, NQ, HD)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertGreater(float(np.abs(got).max()), 0.0)
        self.assertAllClose(got, np.asarray(exp), atol=2e-2, rtol=2e-2)

    @parameterized.parameters(2, 4)
    def test_first_chunk_skips_cache_phase(self, pcp):
        """has_cached_kv=False elides the cache phase; with no cached tokens the
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
        empty = jnp.full((ref_pps, PAGE, *self._cache_dims(1)), jnp.nan, DTYPE)
        exp, _ = ref_ragged_paged_attention(
            q,
            k,
            v,
            empty,
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, MAX_SEQ - 1)),
            ref_pi,
            jnp.pad(jnp.array([0, num_current], jnp.int32), (0, MAX_SEQ - 1)),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=SM_SCALE)

        pps = cdiv(cdiv(kv_total, pcp), PAGE)
        npages = 4 * pps
        cache = jnp.full((npages, PAGE * pcp, *self._cache_dims(1)), jnp.nan,
                         DTYPE)
        pi = np.zeros(MAX_SEQ * pps, np.int32)
        pi[:pps] = np.arange(pps)
        pi[pps:2 * pps] = np.arange(pps)
        pi = jnp.asarray(pi)

        def pad1(xs):
            return jnp.pad(jnp.array(xs, jnp.int32), (0, MAX_SEQ - len(xs)))

        cu, qpos = _pcp_meta(pcp, C, num_current)
        md = AttentionMetadata(
            input_positions=jnp.zeros(1, jnp.int32),
            seq_lens=pad1([kv_total, kv_total]),
            block_tables=pi,
            request_distribution=jnp.array([0, 0, 2], jnp.int32),
            pcp=PCPMetadata(
                query_start_loc=cu,
                kv_cache_lens=pad1([0, 0]),
                q_pos_offsets=qpos,
                kv_new_starts=jnp.zeros(MAX_SEQ, jnp.int32),
                kv_token_order=_kv_token_order(pcp, C),
                has_cached_kv=False,
            ),
        )
        _, out = pcp_forward(self._mesh(pcp), _to_rank_order(q, pcp, C),
                             _to_rank_order(k, pcp, C),
                             _to_rank_order(v, pcp, C), cache, md, SM_SCALE)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
