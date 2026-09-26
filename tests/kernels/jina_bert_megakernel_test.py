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
"""Kernel-level tests for the JinaBert v2 encoder megakernel.

Covers input validation (including the hard T <= 2048 limit), the per-step
tile metadata, weight packing, the VMEM budget, and numerics against the dense
pure-JAX reference (`reference.py`). Numerics run the compiled kernel on TPU
and the Pallas TPU interpreter elsewhere (small sizes only).

The model-level kernel-vs-XLA comparison lives in
tests/models/jax/test_jina_bert_megakernel.py.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.jina_bert_megakernel import kernel as mk
from tpu_inference.kernels.jina_bert_megakernel import reference as ref

jax.config.parse_flags_with_absl()

D, H, F, L = 512, 8, 2048, 4
SLOPES = tuple(2.0**-(i + 1) for i in range(H))


def _random_layers(seed: int = 0, num_layers: int = L):
    """Random per-layer weights at realistic scales, as float32."""
    key = jax.random.key(seed)
    layers = []
    for _ in range(num_layers):
        ks = list(jax.random.split(key, 16))
        key = ks.pop()

        def n(shape, std, ks=ks):
            return std * jax.random.normal(ks.pop(), shape, jnp.float32)

        layers.append(
            dict(q_w=n((D, H, D // H), 0.05),
                 q_b=n((H, D // H), 0.1),
                 k_w=n((D, H, D // H), 0.05),
                 k_b=n((H, D // H), 0.1),
                 v_w=n((D, H, D // H), 0.05),
                 v_b=n((H, D // H), 0.1),
                 o_w=n((H, D // H, D), 0.05),
                 o_b=n((D, ), 0.1),
                 ln1_g=1.0 + n((D, ), 0.1),
                 ln1_b=n((D, ), 0.1),
                 gate_w=n((D, 2 * F), 0.05),
                 down_w=n((F, D), 0.03),
                 down_b=n((D, ), 0.1),
                 ln2_g=1.0 + n((D, ), 0.1),
                 ln2_b=n((D, ), 0.1)))
    return layers


def _effective_lens(seq_lens, num_tokens):
    """Request lengths clipped to the step's token count."""
    eff, off = [], 0
    for n in seq_lens:
        n = max(0, min(int(n), num_tokens - off))
        eff.append(n)
        off += n
    return eff


def _errors(got, want, seq_lens):
    """(max abs error, min per-token cosine, min pooled cosine), real rows."""
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    lens = _effective_lens(seq_lens, got.shape[0])
    total = sum(lens)
    g, w = got[:total], want[:total]
    cos = np.sum(
        g * w, -1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    pooled, off = [], 0
    for n in lens:
        if n:
            a, b = g[off:off + n].mean(0), w[off:off + n].mean(0)
            pooled.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            off += n
    return float(np.max(np.abs(g - w))), float(cos.min()), float(min(pooled))


def _on_tpu() -> bool:
    return jax.devices()[0].platform == "tpu"


class JinaBertMegakernelTest(jtu.JaxTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.layers = _random_layers()

    def _packed(self, precision):
        return mk.pack_jina_bert_weights(self.layers, precision=precision)

    # ------------------------------------------------------------------
    # Validation.

    def test_rejects_more_than_max_tokens(self):
        packed = self._packed("default")
        x = jnp.zeros((mk.MAX_TOKENS + 1, D), jnp.float32)
        seq_lens = jnp.array([mk.MAX_TOKENS + 1], jnp.int32)
        with self.assertRaisesRegex(ValueError, "at most 2048 tokens"):
            mk.jina_bert_encoder_megakernel(x,
                                            seq_lens,
                                            packed,
                                            alibi_slopes=SLOPES)
        # Also when traced under an outer jit (as in the model runner).
        with self.assertRaisesRegex(ValueError, "at most 2048 tokens"):
            jax.jit(lambda x, s: mk.jina_bert_encoder_megakernel(
                x, s, packed, alibi_slopes=SLOPES))(x, seq_lens)
        # The limit itself is accepted.
        mk._check_inputs(x[:mk.MAX_TOKENS], seq_lens, packed, SLOPES,
                         "default", None)

    def test_rejects_malformed_inputs(self):
        packed = self._packed("default")
        x = jnp.zeros((16, D), jnp.float32)
        seq_lens = jnp.array([16], jnp.int32)
        call = mk.jina_bert_encoder_megakernel
        with self.assertRaises(TypeError):  # bf16 activations.
            call(x.astype(jnp.bfloat16), seq_lens, packed, alibi_slopes=SLOPES)
        with self.assertRaises(TypeError):  # Packed for the other precision.
            call(x, seq_lens, packed, alibi_slopes=SLOPES, precision="highest")
        with self.assertRaises(ValueError):
            call(x, seq_lens, packed, alibi_slopes=SLOPES, precision="bf16")
        with self.assertRaises(ValueError):
            call(x, seq_lens, packed, alibi_slopes=SLOPES, tile=200)
        with self.assertRaises(NotImplementedError):  # head_dim != 64.
            call(x, seq_lens, packed, alibi_slopes=SLOPES[:4])

    def test_unsupported_geometry_reason(self):
        self.assertIsNone(
            mk.unsupported_geometry_reason(hidden_size=512,
                                           num_heads=8,
                                           intermediate_size=2048))
        self.assertIsNotNone(  # Odd number of heads.
            mk.unsupported_geometry_reason(hidden_size=448,
                                           num_heads=7,
                                           intermediate_size=2048))
        self.assertIsNotNone(  # head_dim 32.
            mk.unsupported_geometry_reason(hidden_size=512,
                                           num_heads=16,
                                           intermediate_size=2048))
        self.assertIsNotNone(
            mk.unsupported_geometry_reason(hidden_size=512,
                                           num_heads=8,
                                           intermediate_size=1000))

    @mock.patch.object(mk, "_vmem_capacity_bytes", lambda: 128 * 1024 * 1024)
    def test_unsupported_vmem_reason(self):
        # On v6e (128 MiB of VMEM), jina-embeddings-v2-small-en fits at the
        # step limit in both modes.
        for precision in mk.PRECISIONS:
            self.assertIsNone(
                mk.unsupported_vmem_reason(mk.MAX_TOKENS, precision=precision))
        # A 12-layer 768-wide model (jina-embeddings-v2-base geometry) with
        # fp32 weights does not.
        base = dict(hidden_size=768,
                    num_heads=12,
                    intermediate_size=3072,
                    num_layers=12)
        self.assertIsNotNone(
            mk.unsupported_vmem_reason(mk.MAX_TOKENS,
                                       precision="highest",
                                       **base))

    # ------------------------------------------------------------------
    # Metadata, packing, VMEM.

    @parameterized.parameters(
        ([5], 16, 128),
        ([3, 0, 9, 1], 16, 128),
        ([300, 5, 150], 600, 256),
        ([256, 256], 512, 256),
        ([1009, 1009], 2048, 256),
        ([7, 33, 128, 1, 250, 64, 3, 19], 512, 128),
        ([40, 900, 7], 128, 128),  # sum(seq_lens) > T.
        ([0, 0], 32, 128),
    )
    def test_tile_metadata(self, seq_lens, num_tokens, tile):
        padded = -(-num_tokens // tile) * tile
        n_tiles = padded // tile
        meta, seg_q, seg_k = mk.build_tile_metadata(
            jnp.array(seq_lens, jnp.int32), num_tokens, padded, tile)
        meta, seg_q, seg_k = map(np.asarray, (meta, seg_q, seg_k))
        # Segment ids agree with the XLA path's definition on [0, T) ...
        seg = seg_q[:, :, 0].reshape(-1)
        want = np.asarray(
            ref.segment_ids(jnp.array(seq_lens, jnp.int32), num_tokens))
        np.testing.assert_array_equal(seg[:num_tokens], want)
        # ... the kernel's own padding joins the trailing padding segment,
        np.testing.assert_array_equal(seg[num_tokens:], len(seq_lens))
        np.testing.assert_array_equal(seg_k[:, 0, :].reshape(-1), seg)
        np.testing.assert_array_equal(seg_k, np.repeat(seg_k[:, :1], 8, 1))
        # ... and exactly the tiles holding real tokens are active.
        total = sum(_effective_lens(seq_lens, num_tokens))
        n_active = int(meta[0])
        self.assertEqual(n_active, -(-total // tile))
        # Every KV tile an active query tile needs (same segment, active) is
        # inside its [kv_lo, kv_hi) range.
        for qi in range(n_active):
            lo, hi = int(meta[1 + qi]), int(meta[1 + n_tiles + qi])
            self.assertLessEqual(hi, n_active)
            q_segs = set(seg[qi * tile:(qi + 1) * tile])
            for kj in range(n_active):
                k_segs = set(seg[kj * tile:(kj + 1) * tile])
                if q_segs & k_segs:
                    self.assertTrue(lo <= kj < hi, (qi, kj, lo, hi))

    @parameterized.parameters("default", "highest")
    def test_pack_weights(self, precision):
        packed = self._packed(precision)
        wdt = mk.mxu_dtype_for(precision)
        self.assertEqual(packed.wqkv.shape, (L, D, 3 * D))
        self.assertEqual(packed.wo.shape, (L, D, D))
        self.assertEqual(packed.wg.shape, (L, D, 2 * F))
        self.assertEqual(packed.wd.shape, (L, F, D))
        self.assertEqual(packed.vecs.shape, (L, mk.NUM_VEC_ROWS, D))
        for w in (packed.wqkv, packed.wo, packed.wg, packed.wd):
            self.assertEqual(w.dtype, wdt)
        self.assertEqual(packed.vecs.dtype, jnp.float32)
        p = self.layers[1]
        # Q columns / bias carry the (exact) 1/sqrt(64) softmax scale.
        np.testing.assert_array_equal(packed.wqkv[1, :, :D],
                                      (p["q_w"].reshape(D, D) *
                                       0.125).astype(wdt))
        np.testing.assert_array_equal(packed.wqkv[1, :, D:2 * D],
                                      p["k_w"].reshape(D, D).astype(wdt))
        np.testing.assert_array_equal(packed.vecs[1, mk.ROW_BQ],
                                      p["q_b"].reshape(D) * 0.125)
        np.testing.assert_array_equal(packed.vecs[1, mk.ROW_LN2_G], p["ln2_g"])

    def test_vmem_budget(self):
        for precision in mk.PRECISIONS:
            sizes = [
                mk.estimate_vmem_bytes(t, precision=precision)
                for t in (16, 512, 2048)
            ]
            self.assertEqual(sizes, sorted(sizes))
            # Must leave headroom below v6e's 128 MiB of VMEM.
            self.assertLess(sizes[-1], 96 * 1024 * 1024)

    # ------------------------------------------------------------------
    # Numerics vs the dense reference.

    @parameterized.named_parameters([
        (f"{name}_{precision}", seq_lens, num_tokens, tile, interpretable,
         precision)
        for name, seq_lens, num_tokens, tile, interpretable in (
            # (name, seq_lens, num_tokens, tile override, runs in the
            # interpreter off-TPU)
            ("one_token", [1], 1, None, True),
            ("short", [5], 16, None, True),
            ("unused_slots", [3, 0, 9, 1], 16, None, True),
            ("three_seqs_tile128", [130, 70, 40], 256, 128, True),
            ("trailing_padding", [300, 5, 150], 600, None, True),
            ("lens_exceed_tokens", [40, 900, 7], 128, None, True),
            ("len129", [129], 256, None, True),
            ("eight_seqs", [7, 33, 128, 1, 250, 64, 3, 19], 512, None, True),
            ("len1009", [1009], 1024, None, False),
            ("len2016", [2016], 2048, None, False),
            ("len1009_len1000", [1009, 1000], 2048, None, False),
            ("len2048", [2048], 2048, None, False),
        ) for precision in mk.PRECISIONS
    ])
    def test_matches_reference(self, seq_lens, num_tokens, tile, interpretable,
                               precision):
        interpret = not _on_tpu()
        if interpret and not interpretable:
            self.skipTest("Large case: TPU only (too slow for the "
                          "interpreter).")
        x = jax.random.normal(jax.random.key(num_tokens), (num_tokens, D),
                              jnp.float32)
        sl = jnp.array(seq_lens, jnp.int32)
        try:
            got = mk.jina_bert_encoder_megakernel(x,
                                                  sl,
                                                  self._packed(precision),
                                                  alibi_slopes=SLOPES,
                                                  precision=precision,
                                                  tile=tile,
                                                  interpret=interpret)
            got = np.asarray(got)
        finally:
            if interpret:
                pltpu.reset_tpu_interpret_mode_state()
        self.assertEqual(got.shape, (num_tokens, D))
        total = sum(_effective_lens(seq_lens, num_tokens))
        self.assertTrue(np.all(np.isfinite(got[:total])))
        want32 = ref.reference_jina_bert_encoder(x,
                                                 sl,
                                                 self.layers,
                                                 alibi_slopes=SLOPES)
        max_abs, cos, pooled = _errors(got, want32, seq_lens)
        print(f"{self._testMethodName}: vs fp32 reference: max_abs="
              f"{max_abs:.3e} min_token_cos={cos:.7f} "
              f"min_pooled_cos={pooled:.8f}")
        if precision == "highest":
            # fp32 matmuls: agrees with the fp32 reference to rounding.
            self.assertLess(max_abs, 2e-3)
            self.assertGreater(cos, 0.99999)
        else:
            # bf16 MXU operands: compare against the reference rounding at
            # the same points, and bound the error vs fp32.
            want16 = ref.reference_jina_bert_encoder(x,
                                                     sl,
                                                     self.layers,
                                                     alibi_slopes=SLOPES,
                                                     mxu_dtype=jnp.bfloat16)
            _, cos16, pooled16 = _errors(got, want16, seq_lens)
            self.assertGreater(cos16, 0.9995)
            self.assertGreater(pooled16, 0.9999)
            self.assertGreater(cos, 0.999)
            self.assertGreater(pooled, 0.9995)

    def test_no_real_tokens(self):
        # E.g. precompilation/warmup steps: every request slot is empty.
        x = jnp.ones((32, D), jnp.float32)
        interpret = not _on_tpu()
        try:
            got = mk.jina_bert_encoder_megakernel(x,
                                                  jnp.zeros((4, ), jnp.int32),
                                                  self._packed("default"),
                                                  alibi_slopes=SLOPES,
                                                  interpret=interpret)
            got = np.asarray(got)
        finally:
            if interpret:
                pltpu.reset_tpu_interpret_mode_state()
        self.assertEqual(got.shape, (32, D))


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
