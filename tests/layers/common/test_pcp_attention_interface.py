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
"""End-to-end (interface-level) test for prefill context parallel attention.

Exercises the real `pcp_ragged_paged_attention` from attention_interface against
a chunked-prefill reference (new queries attend the strided previous cache +
causal new KV, merged via LSE).
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


class PcpAttentionInterfaceTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        # Force the N-D axis names (which carry the `pcp` axis) regardless of the
        # ambient NEW_MODEL_DESIGN env, and restore afterwards.
        self._saved_cls = sharding_mod.ShardingAxisName._cls
        sharding_mod.ShardingAxisName._cls = ShardingAxisNameBase

    def tearDown(self):
        sharding_mod.ShardingAxisName._cls = self._saved_cls
        super().tearDown()

    @parameterized.product(pcp=[2], dtype=[jnp.float32])
    def test_chunked_prefill(self, pcp, dtype):
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, Snew, nq, nkv, D = 128, 128, 8, 2, 128
        C = Snew // (2 * pcp)
        kv_total = L + Snew
        page_size, num_pages, max_seq = 16, 512, 8
        sm_scale = D**-0.5
        kvp = get_dtype_packing(dtype)
        phd, nkv2 = align_to(D, 128), align_to(nkv * 2, kvp)
        npr = num_pages // pcp
        rng = np.random.default_rng(4)
        gen = lambda s: jnp.array(rng.random(size=s, dtype=np.float32)).astype(
            dtype)
        k_prev, v_prev = gen((L, nkv, D)), gen((L, nkv, D))
        q_new, k_new, v_new = gen((Snew, nq, D)), gen((Snew, nkv, D)), gen(
            (Snew, nkv, D))

        def cache_from_kv(k, v, ntok):
            kv = merge_kv(k, v)
            kv = jnp.pad(kv, ((0, cdiv(ntok, page_size) * page_size - ntok),
                              (0, 0), (0, 0), (0, 0)),
                         constant_values=jnp.nan)
            kv = kv.reshape(-1, page_size, nkv2 // kvp, kvp, phd)
            c = jnp.full((num_pages, page_size, nkv2 // kvp, kvp, phd),
                         jnp.nan, dtype)
            return c.at[:kv.shape[0]].set(kv)

        pps_full = cdiv(kv_total, page_size)
        pi_full = jnp.pad(jnp.arange(pps_full, dtype=jnp.int32),
                          (0, max_seq * pps_full - pps_full))
        exp, _ = ref_ragged_paged_attention(
            q_new,
            k_new,
            v_new,
            cache_from_kv(k_prev, v_prev, L),
            jnp.pad(jnp.array([kv_total], jnp.int32), (0, max_seq - 1)),
            pi_full,
            jnp.pad(jnp.array([0, Snew], jnp.int32), (0, max_seq - 1)),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=sm_scale)
        exp = np.array(exp[:Snew])

        def to_rank_order(x):
            out = np.zeros_like(np.array(x))
            for r in range(pcp):
                out[r * 2 * C:r * 2 * C + C] = np.array(x)[r * C:(r + 1) * C]
                out[r * 2 * C + C:(r + 1) * 2 *
                    C] = np.array(x)[(2 * pcp - 1 - r) * C:(2 * pcp - r) * C]
            return jnp.array(out)

        q_ro, k_ro, v_ro = to_rank_order(q_new), to_rank_order(
            k_new), to_rank_order(v_new)

        gcache = np.full((num_pages, page_size, nkv2 // kvp, kvp, phd), np.nan,
                         np.float32)
        for r in range(pcp):
            idx = np.arange(r, L, pcp)
            kv = np.array(merge_kv(k_prev[idx], v_prev[idx]))
            n = kv.shape[0]
            kv = np.pad(kv, ((0, cdiv(n, page_size) * page_size - n), (0, 0),
                             (0, 0), (0, 0)),
                        constant_values=np.nan)
            kv = kv.reshape(-1, page_size, nkv2 // kvp, kvp, phd)
            gcache[r * npr:r * npr + kv.shape[0]] = kv
        gcache = jnp.array(gcache)

        pps = cdiv(L, page_size)
        page_indices = jnp.pad(jnp.arange(pps, dtype=jnp.int32),
                               (0, max_seq * pps - pps))
        kv_lens = jnp.pad(jnp.array([kv_total], jnp.int32), (0, max_seq - 1))
        cu_q = jnp.pad(jnp.array([0, 2 * C], jnp.int32), (0, max_seq + 1 - 2))
        dist = jnp.array([0, 0, 1], jnp.int32)

        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        mesh = Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)
        # Single sequence => one head-tail chunk per rank; sort perm is the
        # whole-buffer perm; cu_q_lens holds the chunk size C.
        C = Snew // (2 * pcp)
        perm = _single_seq_perm(pcp, C)
        cu_chunk = jnp.pad(jnp.array([0, C], jnp.int32), (0, max_seq + 1 - 2))
        out, _ = pcp_ragged_paged_attention(mesh,
                                            q_ro,
                                            k_ro,
                                            v_ro,
                                            gcache,
                                            kv_lens,
                                            page_indices,
                                            cu_chunk,
                                            dist,
                                            jnp.asarray(perm),
                                            sm_scale=sm_scale,
                                            update_kv_cache=True,
                                            use_causal_mask=True)
        self.assertAllClose(np.array(out),
                            np.array(to_rank_order(exp)),
                            atol=0.05,
                            rtol=0.05)

    def test_ragged_prefill_from_scratch(self):
        pcp = 2
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        dtype = jnp.float32
        nq, nkv, D = 8, 2, 128
        Cs = [32, 16]
        Ss = [2 * pcp * c for c in Cs]
        max_seq, page_size, num_pages = 8, 16, 512
        sm_scale = D**-0.5
        kvp = get_dtype_packing(dtype)
        phd, nkv2 = align_to(D, 128), align_to(nkv * 2, kvp)
        rng = np.random.default_rng(1)
        gen = lambda s: np.asarray(rng.random(size=s, dtype=np.float32), np.
                                   float32)
        off = np.cumsum([0] + Ss)
        Stot = off[-1]
        q_tok, k_tok, v_tok = gen((Stot, nq, D)), gen((Stot, nkv, D)), gen(
            (Stot, nkv, D))

        def ref(q, k, v):
            G = nq // nkv
            kk, vv = np.repeat(k, G, 1), np.repeat(v, G, 1)
            S = q.shape[0]
            s = np.einsum("ihd,jhd->hij", q, kk) * sm_scale
            s = np.where(np.tril(np.ones((S, S), bool))[None], s, -np.inf)
            p = np.array(jax.nn.softmax(jnp.array(s), -1))
            return np.einsum("hij,jhd->ihd", p, vv)

        exp = np.zeros((Stot, nq, D), np.float32)
        for i, S in enumerate(Ss):
            exp[off[i]:off[i + 1]] = ref(q_tok[off[i]:off[i + 1]],
                                         k_tok[off[i]:off[i + 1]],
                                         v_tok[off[i]:off[i + 1]])
        ro_gids = []
        for r in range(pcp):
            heads, tails = [], []
            for i, C in enumerate(Cs):
                b = off[i]
                heads += [b + p for p in range(r * C, (r + 1) * C)]
                tails += [
                    b + p
                    for p in range((2 * pcp - 1 - r) * C, (2 * pcp - r) * C)
                ]
            ro_gids += heads + tails
        ro_gids = np.array(ro_gids)
        pos_of = {g: i for i, g in enumerate(ro_gids)}
        perm = np.array([pos_of[j] for j in range(Stot)], np.int32)
        to_ro = lambda x: x[ro_gids]
        cu_chunk = np.pad(
            np.cumsum([0] + Cs).astype(np.int32),
            (0, max_seq + 1 - (len(Cs) + 1)))
        kv_lens = np.pad(np.array(Ss, np.int32), (0, max_seq - len(Ss)))
        pps = cdiv(max(Ss), page_size)
        page_indices = np.pad(np.arange(pps, dtype=np.int32),
                              (0, max_seq * pps - pps))
        dist = np.array([0, 0, len(Cs)], np.int32)
        cache = jnp.full((num_pages, page_size, nkv2 // kvp, kvp, phd),
                         jnp.nan, dtype)
        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        mesh = Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)
        out, _ = pcp_ragged_paged_attention(
            mesh,
            jnp.asarray(to_ro(q_tok)).astype(dtype),
            jnp.asarray(to_ro(k_tok)).astype(dtype),
            jnp.asarray(to_ro(v_tok)).astype(dtype),
            cache,
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu_chunk),
            jnp.asarray(dist),
            jnp.asarray(perm),
            sm_scale=sm_scale,
            update_kv_cache=False,
            use_causal_mask=True)
        self.assertAllClose(np.array(out), to_ro(exp), atol=0.05, rtol=0.05)


def _single_seq_perm(pcp, C):
    S = 2 * pcp * C
    perm = np.empty(S, np.int32)
    for p in range(S):
        c, w = p // C, p % C
        perm[p] = (c * 2 * C + w) if c < pcp else ((2 * pcp - 1 - c) * 2 * C +
                                                   C + w)
    return perm


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
