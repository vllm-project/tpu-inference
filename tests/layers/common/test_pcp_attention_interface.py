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
"""End-to-end test for the PCP attention wrapper.

Exercises the real `pcp_ragged_paged_attention` from attention_interface on a
live `pcp` mesh -- the production path the runner dispatches to. Unlike the
kernel tests (which call `ragged_paged_attention` directly), this covers the
wrapper's own logic: the head-tail chunk split, the current-KV all-gather into
token order, the cache/current two-phase split with its LSE all-reduce + combine,
and the pcp-sharded per-(half, rank) launch metadata built in `_prepare_inputs`.

PCP is single-request; a chunked prefill (num_computed = L > 0) is the case that
exercises both phases.
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


def _pcp_launch_meta(pcp, C, num_current, max_seq):
    """Per-(half, rank) cu_q_lens / q_pos_offsets.

    Mirrors what `TPUModelRunner._prepare_inputs` builds host-side: head (half=0)
    is always fully real (C tokens) at within-current offset rank*C; tail
    (half=1) sits at (2*pcp-1-rank)*C, clamped so padding past num_current is
    excluded. Shapes are [2, pcp, ...], sharded on `pcp` by the wrapper.
    """
    two_p = 2 * pcp
    cu = np.zeros((2, pcp, max_seq + 1), np.int32)
    qpos = np.zeros((2, pcp, max_seq), np.int32)
    for r in range(pcp):
        tail_off = (two_p - 1 - r) * C
        cu[0, r, 1:] = C
        qpos[0, r, 0] = r * C
        cu[1, r, 1:] = int(np.clip(num_current - tail_off, 0, C))
        qpos[1, r, 0] = tail_off
    return jnp.asarray(cu), jnp.asarray(qpos)


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
        """Wrapper output == full-causal reference, for a chunked prefill
        (L previously-computed tokens in a pcp-strided cache + Snew current)."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        L, Snew, nq, nkv, D = 128, 128, 8, 2, 128
        C = Snew // (2 * pcp)  # head-tail chunk size
        kv_total = L + Snew
        page_size, num_pages, max_seq = 16, 512, 8
        sm_scale = D**-0.5
        kvp = get_dtype_packing(dtype)
        phd, nkv2 = align_to(D, 128), align_to(nkv * 2, kvp)
        npr = num_pages // pcp
        rng = np.random.default_rng(4)

        def gen(s):
            return jnp.array(rng.random(size=s,
                                        dtype=np.float32)).astype(dtype)

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

        # --- reference: plain full-causal prefill over the whole context ---
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
            """Token order -> rank order: rank r holds [chunk r | chunk 2P-1-r]."""
            out = np.zeros_like(np.array(x))
            for r in range(pcp):
                out[r * 2 * C:r * 2 * C + C] = np.array(x)[r * C:(r + 1) * C]
                out[r * 2 * C + C:(r + 1) * 2 *
                    C] = np.array(x)[(2 * pcp - 1 - r) * C:(2 * pcp - r) * C]
            return jnp.array(out)

        q_ro, k_ro, v_ro = to_rank_order(q_new), to_rank_order(
            k_new), to_rank_order(v_new)

        # --- pcp-strided previous cache: global token g -> rank g%pcp ---
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
        # seq_lens = REAL total (num_computed + num_current); kv_cache_lens =
        # num_computed. The kernel derives the current length as their difference.
        kv_lens = jnp.pad(jnp.array([kv_total], jnp.int32), (0, max_seq - 1))
        kv_cache_lens = jnp.pad(jnp.array([L], jnp.int32), (0, max_seq - 1))
        cu_chunk = jnp.pad(jnp.array([0, C], jnp.int32), (0, max_seq + 1 - 2))
        dist = jnp.array([0, 0, 1], jnp.int32)
        pcp_cu, pcp_qpos = _pcp_launch_meta(pcp, C, Snew, max_seq)

        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        mesh = Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)
        out, _ = pcp_ragged_paged_attention(mesh,
                                            q_ro,
                                            k_ro,
                                            v_ro,
                                            gcache,
                                            kv_lens,
                                            page_indices,
                                            cu_chunk,
                                            dist,
                                            kv_cache_lens,
                                            pcp_cu,
                                            pcp_qpos,
                                            sm_scale=sm_scale,
                                            update_kv_cache=True,
                                            use_causal_mask=True)
        out = np.array(out)
        # Guard against a trivially-passing all-zero/NaN match.
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(float(np.abs(out).max()), 0.0)
        self.assertAllClose(out,
                            np.array(to_rank_order(exp)),
                            atol=0.05,
                            rtol=0.05)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
