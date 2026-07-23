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
"""Prefill context parallelism (PCP) for the GDN (gated deltanet) kernel.

GDN under PCP all-gathers the head-tail token shards so every device sees the
full sequence, then distributes the linear-attention heads over ``tp * pcp``
devices -- each device runs the full sequence for ``1/pcp`` of its TP heads, so
there is no redundant compute. This test drives :func:`run_jax_gdn_attention`
over a real pcp mesh and asserts the output and both updated state caches match
a single-device run of the same Pallas kernel over the whole sequence.
"""

import jax
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh

from tpu_inference.kernels.gdn.v3 import wrapper
from tpu_inference.layers.common import sharding as sharding_mod
from tpu_inference.layers.common.gdn_attention import run_jax_gdn_attention
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)

N_KQ, N_V, D_K, D_V, KS = 4, 8, 128, 128, 4


def _row_perm(pcp):
    """Rank r owns global chunk r (head) and chunk 2P-1-r (tail)."""
    return np.asarray([c for r in range(pcp) for c in (r, 2 * pcp - 1 - r)])


def _inv_row(pcp):
    inv = np.empty(2 * pcp, np.int64)
    inv[_row_perm(pcp)] = np.arange(2 * pcp)
    return inv


class GdnPcpAttentionTest(jtu.JaxTestCase):

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

    def _mesh(self, pcp):
        shape = tuple(pcp if a == "pcp" else 1 for a in MESH_AXIS_NAMES)
        return Mesh(
            np.array(jax.devices()[:pcp]).reshape(shape), MESH_AXIS_NAMES)

    def _inputs(self, num_current, padded_s, seed):
        rng = np.random.default_rng(seed)
        dim = N_KQ * D_K * 2 + N_V * D_V
        nb, slot = 2, 1
        f32 = np.float32

        def buf(width):
            b = np.zeros((padded_s, width), f32)
            b[:num_current] = rng.standard_normal((num_current, width))
            return b

        return dict(
            qkv=buf(dim),
            b=buf(N_V),
            a=buf(N_V),
            conv_state=np.zeros((nb, KS - 1, dim), f32),
            recurrent_state=np.zeros((nb, N_V, D_K, D_V), f32),
            conv_weight=rng.standard_normal((dim, 1, KS)).astype(f32),
            conv_bias=rng.standard_normal((dim, )).astype(f32),
            a_log=rng.standard_normal((N_V, )).astype(f32),
            dt_bias=rng.standard_normal((N_V, )).astype(f32),
            qsl=np.array([0, num_current], np.int32),
            dist=np.array([0, 0, 1], np.int32),
            sidx=np.array([slot], np.int32),
            slens=np.array([num_current], np.int32),
        )

    def _reference(self, x):
        """Single-device full-sequence run of the real Pallas kernel."""
        (conv, rec), out = wrapper.fused_conv1d_gdn(x["qkv"],
                                                    x["b"],
                                                    x["a"],
                                                    x["conv_state"],
                                                    x["recurrent_state"],
                                                    x["conv_weight"],
                                                    x["conv_bias"],
                                                    x["a_log"],
                                                    x["dt_bias"],
                                                    x["qsl"],
                                                    x["sidx"],
                                                    x["dist"],
                                                    x["slens"],
                                                    n_kq=N_KQ,
                                                    n_v=N_V,
                                                    d_k=D_K,
                                                    d_v=D_V,
                                                    kernel_size=KS)
        return np.asarray(out), np.asarray(conv), np.asarray(rec)

    def _run_pcp(self, x, pcp, padded_s):
        C = padded_s // (2 * pcp)
        rp = _row_perm(pcp)

        def to_rank(v):
            v = np.asarray(v)
            return v.reshape(2 * pcp, C, *v.shape[1:])[rp].reshape(v.shape)

        (conv, rec), out = run_jax_gdn_attention(to_rank(x["qkv"]),
                                                 to_rank(x["b"]),
                                                 to_rank(x["a"]),
                                                 x["conv_state"],
                                                 x["recurrent_state"],
                                                 x["conv_weight"],
                                                 x["conv_bias"],
                                                 x["a_log"],
                                                 x["dt_bias"],
                                                 x["sidx"],
                                                 x["qsl"],
                                                 x["dist"],
                                                 x["slens"],
                                                 N_KQ,
                                                 N_V,
                                                 D_K,
                                                 D_V,
                                                 KS,
                                                 mesh=self._mesh(pcp))
        # Output returns in rank order; un-permute to natural token order.
        out = np.asarray(out)
        W = N_V * D_V
        out = out.reshape(2 * pcp, C, W)[_inv_row(pcp)].reshape(padded_s, W)
        return out, np.asarray(conv), np.asarray(rec)

    @parameterized.product(pcp=[2, 4])
    def test_full_chunk(self, pcp):
        """Every chunk fully real (num_current == padded_s)."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        S = 256
        x = self._inputs(S, S, seed=pcp)
        eo, ec, er = self._reference(x)
        po, pc, pr = self._run_pcp(x, pcp, S)
        self.assertAllClose(po, eo, atol=2e-2, rtol=2e-2)
        self.assertAllClose(pc, ec, atol=2e-2, rtol=2e-2)
        self.assertAllClose(pr, er, atol=2e-2, rtol=2e-2)

    @parameterized.product(pcp=[2, 4])
    def test_partial_tail(self, pcp):
        """num_current < padded_s: tail chunks are partly/entirely padding."""
        if jax.device_count() < pcp:
            self.skipTest(f"needs >= {pcp} devices")
        S, padded_s = 200, 256
        x = self._inputs(S, padded_s, seed=pcp + 10)
        eo, ec, er = self._reference(x)
        po, pc, pr = self._run_pcp(x, pcp, padded_s)
        # Compare only the real tokens for the output; padding rows are
        # meaningless and the head-tail reshuffle scatters them.
        self.assertAllClose(po[:S], eo[:S], atol=2e-2, rtol=2e-2)
        self.assertAllClose(pc, ec, atol=2e-2, rtol=2e-2)
        self.assertAllClose(pr, er, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
