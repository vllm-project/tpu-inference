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
from tpu_inference.layers.common.attention_interface import \
    pcp_ragged_paged_attention
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)

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

    def _mesh(self, pcp):
        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        return Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)

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

        pad1 = lambda xs: jnp.pad(jnp.array(xs, jnp.int32),
                                  (0, MAX_SEQ - len(xs)))
        # Both fused seqs are the SAME request -> [T, T] / [P, P].
        kv_lens = pad1([kv_total, kv_total])
        kv_cache_lens = pad1([L, L])
        cu_q_lens, q_pos_offsets = _pcp_meta(pcp, C, num_current)
        distribution = jnp.array([0, 0, 2], jnp.int32)  # head + tail

        out, new_cache = pcp_ragged_paged_attention(self._mesh(pcp),
                                                    q,
                                                    k,
                                                    v,
                                                    cache,
                                                    kv_lens,
                                                    self._page_indices(pps),
                                                    cu_q_lens,
                                                    distribution,
                                                    kv_cache_lens,
                                                    q_pos_offsets,
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

        This is what `write_last_seq_only` + the duplicated page-index row buy;
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


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
