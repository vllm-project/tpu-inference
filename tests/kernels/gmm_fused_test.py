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
"""Tests for the fused MoE FFN kernel (gmm_fused).

Host tests ask shape and dtype questions under `pinned_tpu()`; execution
tests compare against the two-gmm_v2 pair and need a real TPU."""

import contextlib
import unittest.mock as mock

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import mesh as mesh_lib
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.gmm_test import get_group_sizes, quantize_tensor
from tpu_inference.kernels.megablox import gmm_fused as gf
from tpu_inference.kernels.megablox.gmm_v2 import TileSizes as GmmV2TileSizes
from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.kernels.megablox.gmm_v2_fused_support import (
    LHS_QUANT_BLOCK_SIZE, TileSizes, make_gmm_configs)

jax.config.parse_flags_with_absl()

# The generation the host tests answer for.
PINNED_DEVICE_KIND = "TPU7x"

# The MoE layer's own operand shapes, per expert-parallel shard.
SERVED_HIDDEN = 4096
SERVED_INTER = 1024
SERVED_LOCAL_EXPERTS = 16


def pinned_mesh(axis_sizes=(1, ), axis_names=("d", )):
    """A mesh of the served generation's devices, without needing any."""
    device = jax.sharding.AbstractDevice(PINNED_DEVICE_KIND, 1, "tpu")
    return jax.sharding.AbstractMesh(axis_sizes,
                                     axis_names,
                                     abstract_device=device)


@contextlib.contextmanager
def pinned_tpu(axis_sizes=(1, ), axis_names=("d", )):
    """Answer shape questions for the served generation, off TPU included."""
    mesh = pinned_mesh(axis_sizes, axis_names)
    with mesh_lib.use_abstract_mesh(mesh):
        yield mesh


def ffn_shapes(num_groups=SERVED_LOCAL_EXPERTS,
               num_rows=128,
               hidden=SERVED_HIDDEN,
               inter=SERVED_INTER,
               weight_dtype=jnp.float8_e4m3fn,
               quant_block=None,
               w2_inter=None):
    """Shapes and dtypes of one expert-parallel shard's FFN operands;
    quant_block None is the channelwise (fp8) layout, a size is fp4."""
    w1_blocks = 1 if quant_block is None else hidden // quant_block
    w2_blocks = 1 if quant_block is None else inter // quant_block
    return dict(
        lhs=jax.ShapeDtypeStruct((num_rows, hidden), jnp.bfloat16),
        w1=jax.ShapeDtypeStruct((num_groups, hidden, 2 * inter), weight_dtype),
        w2=jax.ShapeDtypeStruct(
            (num_groups, inter if w2_inter is None else w2_inter, hidden),
            weight_dtype),
        w1_scale=jax.ShapeDtypeStruct((num_groups, w1_blocks, 1, 2 * inter),
                                      jnp.float32),
        w2_scale=jax.ShapeDtypeStruct((num_groups, w2_blocks, 1, hidden),
                                      jnp.float32),
    )


def reason_for(shapes, w1_bias=None, w2_bias=None):
    return gf.unsupported_reason(shapes["lhs"], shapes["w1"], shapes["w2"],
                                 shapes["w1_scale"], shapes["w2_scale"],
                                 w1_bias, w2_bias)


def trace_gmm_fused(shapes, **kwargs):
    """Build the kernel for these shapes without running it."""
    group_sizes = jax.ShapeDtypeStruct((shapes["w1"].shape[0], ), jnp.int32)
    return jax.make_jaxpr(lambda *operands: gf.gmm_fused(*operands, **kwargs))(
        shapes["lhs"], shapes["w1"], shapes["w2"], group_sizes,
        shapes["w1_scale"], shapes["w2_scale"])


def stage_configs(shapes, **kwargs):
    group_sizes = jax.ShapeDtypeStruct((shapes["w1"].shape[0], ), jnp.int32)
    group_offset = jax.ShapeDtypeStruct((1, ), jnp.int32)
    return gf.build_stage_configs(shapes["lhs"], shapes["w1"], shapes["w2"],
                                  shapes["w1_scale"], shapes["w2_scale"],
                                  group_sizes, group_offset, **kwargs)


def quantized_ffn_operands(key, num_groups, num_rows, hidden, inter,
                           weight_dtype, quant_block):
    """Real operands for an execution test: bf16 rows, quantized experts."""
    lhs_key, w1_key, w2_key = jax.random.split(key, 3)
    lhs = jax.random.uniform(lhs_key, (num_rows, hidden), jnp.bfloat16, -1, 1)
    w1 = jax.random.uniform(w1_key, (num_groups, hidden, 2 * inter),
                            jnp.bfloat16, -1, 1)
    w2 = jax.random.uniform(w2_key, (num_groups, inter, hidden), jnp.bfloat16,
                            -1, 1)
    w1_q, w1_scale = quantize_tensor(w1,
                                     weight_dtype,
                                     axis=1,
                                     block_size=min(quant_block, hidden))
    w2_q, w2_scale = quantize_tensor(w2,
                                     weight_dtype,
                                     axis=1,
                                     block_size=min(quant_block, inter))
    return (lhs, w1_q, w2_q, jnp.expand_dims(w1_scale, axis=2),
            jnp.expand_dims(w2_scale, axis=2))


def expert_biases(key, num_groups, hidden, inter, dtype=jnp.bfloat16):
    """One gate+up bias and one down bias per expert, in the concatenated
    layout both kernels read: [g, 1, 2 * inter] and [g, 1, hidden]."""
    w1_key, w2_key = jax.random.split(key, 2)
    return (jax.random.uniform(w1_key, (num_groups, 1, 2 * inter), dtype, -1,
                               1),
            jax.random.uniform(w2_key, (num_groups, 1, hidden), dtype, -1, 1))


def sequential_pair(lhs,
                    w1,
                    w2,
                    group_sizes,
                    w1_scale,
                    w2_scale,
                    tile_m,
                    activation="silu",
                    w1_bias=None,
                    w2_bias=None):
    """The two-gmm_v2 composition gmm_fused stands in for, tiled to the
    fused kernel's own contract so the two are comparable bit for bit.
    The biases go in the same operand each kernel adds on its last
    contraction step, so a biased pair is comparable the same way."""
    hidden = lhs.shape[1]
    inter = w1.shape[2] // 2
    mid = gmm_v2(
        lhs,
        w1,
        group_sizes,
        rhs_scale=w1_scale,
        rhs_bias=w1_bias,
        tile_info=GmmV2TileSizes(tile_m=tile_m,
                                 tile_k=hidden,
                                 tile_n=inter,
                                 bucket_base=tile_m),
        preferred_element_type=lhs.dtype,
        zero_initialize=False,
        fuse_act=activation,
    )
    return gmm_v2(
        mid,
        w2,
        group_sizes,
        rhs_scale=w2_scale,
        rhs_bias=w2_bias,
        tile_info=GmmV2TileSizes(tile_m=tile_m,
                                 tile_k=inter,
                                 tile_n=hidden,
                                 bucket_base=tile_m),
    )


class BucketBaseTest(jtu.JaxTestCase):
    """_derive_bucket_base: which row-count ladder a tile height can host."""

    @parameterized.product(
        tile_m=[16, 32, 48, 64, 80, 96, 112, 128, 192, 256],
        sublane=[8, 16, 32],
        requested=[1, 8, 16, 32, 64, 128],
    )
    def test_derived_base_is_a_rung_the_tile_can_host(self, tile_m, sublane,
                                                      requested):
        """Every answer divides the tile, is whole sublanes, fits the arms."""
        if tile_m % sublane != 0:
            self.skipTest("tile_m is a multiple of the sublane by the time "
                          "_derive_bucket_base is called")
        base = gf._derive_bucket_base(tile_m, sublane, requested)
        self.assertEqual(tile_m % base, 0)
        self.assertEqual(base % sublane, 0)
        self.assertBetween(tile_m // base, 1, gf.MAX_BUCKET_ARMS)

    @parameterized.product(
        tile_m=[16, 32, 48, 64, 80, 96, 112, 128, 192, 256],
        sublane=[8, 16, 32],
        requested=[1, 8, 16, 32, 64, 128],
    )
    def test_bucketing_stays_on_wherever_a_finer_rung_exists(
            self, tile_m, sublane, requested):
        """A tile with a legal 2-arm rung never answers "no bucketing"."""
        if tile_m % sublane != 0:
            self.skipTest("tile_m is a multiple of the sublane by the time "
                          "_derive_bucket_base is called")
        finer_rung_exists = any(
            tile_m % arms == 0 and (tile_m // arms) % sublane == 0
            for arms in range(2, gf.MAX_BUCKET_ARMS + 1))
        base = gf._derive_bucket_base(tile_m, sublane, requested)
        if finer_rung_exists:
            self.assertLess(base, tile_m)
        else:
            self.assertEqual(base, tile_m)

    @parameterized.parameters(16, 32)
    def test_served_tile_derives_the_requested_base_unchanged(self, sublane):
        """tile_m 128, request 32 -> 32, the base the MoE layer asks for."""
        self.assertEqual(
            gf._derive_bucket_base(128, sublane, gf.GMM_FUSED_BUCKET_BASE), 32)

    @parameterized.parameters(
        # tile_m, sublane, requested, expected
        (48, 16, 32, 16),  # request too coarse for the ladder -> finest rung
        (64, 16, 32, 32),
        (128, 16, 100, 32),  # only tile_m is >= the request -> finest rung
        (256, 32, 32, 64),  # request finer than the ladder -> snapped up
        (80, 16, 32, 80),  # ladder holds nothing else: bucketing off
        (112, 16, 32, 112),
        (16, 16, 32, 16),
    )
    def test_derivation_grid(self, tile_m, sublane, requested, expected):
        self.assertEqual(gf._derive_bucket_base(tile_m, sublane, requested),
                         expected)

    def test_a_non_positive_base_is_refused(self):
        with pinned_tpu():
            with self.assertRaisesRegex(ValueError, "bucket_base"):
                trace_gmm_fused(ffn_shapes(), bucket_base=0)

    def test_only_a_bucketed_program_carries_the_bucket_suffix(self):
        """The unbucketed program keeps its unsuffixed scope name."""
        with pinned_tpu():
            cfgs1, cfgs2, _, _, _ = stage_configs(ffn_shapes())
            self.assertNotIn("bb_", gf.get_fused_scope_name(cfgs1, cfgs2))
            self.assertIn("bb_32", gf.get_fused_scope_name(cfgs1, cfgs2, 32))


class UnsupportedReasonTest(jtu.JaxTestCase):
    """The gate the MoE layer branches on: what the fused kernel refuses."""

    @parameterized.product(
        num_rows=[16, 32, 48, 128, 144, 1024],
        num_groups=[1, 8, 16],
        quant_block=[None, 512],
    )
    def test_served_shapes_are_supported(self, num_rows, num_groups,
                                         quant_block):
        """fp8 channelwise and fp4-e2m1 block 512, one expert and many."""
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        shapes = ffn_shapes(num_groups=num_groups,
                            num_rows=num_rows,
                            weight_dtype=weight_dtype,
                            quant_block=quant_block)
        with pinned_tpu():
            self.assertIsNone(reason_for(shapes))
            _, _, _, _, tile_m = stage_configs(shapes)
        self.assertEqual(tile_m, min(128, num_rows))

    @parameterized.named_parameters(
        # Per expert-parallel or tensor-parallel shard.
        ("deepseek_v3_ep8", 32, 7168, 2048),
        ("llama4_tp4", 16, 5120, 2048),
        ("mixtral_tp8", 8, 4096, 1792),
    )
    def test_wide_deployment_shapes_are_refused_by_name(
            self, num_groups, hidden, inter):
        """An expert pair too wide for VMEM gets a reason, not a build crash."""
        shapes = ffn_shapes(num_groups=num_groups,
                            num_rows=1024,
                            hidden=hidden,
                            inter=inter)
        with pinned_tpu():
            reason = reason_for(shapes)
        self.assertIsNotNone(reason)
        self.assertIn("VMEM", reason)

    def test_expert_biases_are_supported(self):
        shapes = ffn_shapes()
        w1_bias = jax.ShapeDtypeStruct(
            (SERVED_LOCAL_EXPERTS, 1, 2 * SERVED_INTER), jnp.bfloat16)
        w2_bias = jax.ShapeDtypeStruct(
            (SERVED_LOCAL_EXPERTS, 1, SERVED_HIDDEN), jnp.bfloat16)
        with pinned_tpu():
            self.assertIsNone(reason_for(shapes, w1_bias=w1_bias))
            self.assertIsNone(reason_for(shapes, w2_bias=w2_bias))
            self.assertIsNone(
                reason_for(shapes, w1_bias=w1_bias, w2_bias=w2_bias))

    def test_a_bias_of_the_wrong_width_is_refused_by_name(self):
        """The gate/up bias spans both halves; the down bias spans hidden."""
        shapes = ffn_shapes()
        half = jax.ShapeDtypeStruct((SERVED_LOCAL_EXPERTS, 1, SERVED_INTER),
                                    jnp.bfloat16)
        with pinned_tpu():
            self.assertIn("w1_bias layout", reason_for(shapes, w1_bias=half))
            self.assertIn("w2_bias layout", reason_for(shapes, w2_bias=half))
            flat = jax.ShapeDtypeStruct((SERVED_LOCAL_EXPERTS, SERVED_HIDDEN),
                                        jnp.bfloat16)
            self.assertIn("rank", reason_for(shapes, w2_bias=flat))

    def test_only_a_biased_program_carries_the_bias_suffix(self):
        """The unbiased program keeps its unsuffixed scope name."""
        shapes = ffn_shapes()
        biases = dict(w1_bias=jax.ShapeDtypeStruct(
            (SERVED_LOCAL_EXPERTS, 1, 2 * SERVED_INTER), jnp.float32),
                      w2_bias=jax.ShapeDtypeStruct(
                          (SERVED_LOCAL_EXPERTS, 1, SERVED_HIDDEN),
                          jnp.float32))
        with pinned_tpu():
            plain = gf.get_fused_scope_name(*stage_configs(shapes)[:2])
            biased = gf.get_fused_scope_name(
                *stage_configs(shapes, **biases)[:2])
        self.assertNotIn("bias", plain)
        self.assertEqual(biased, plain + "-bias")

    def test_a_bias_costs_vmem_but_not_the_tile_height(self):
        shapes = ffn_shapes()
        biases = dict(w1_bias=jax.ShapeDtypeStruct(
            (SERVED_LOCAL_EXPERTS, 1, 2 * SERVED_INTER), jnp.float32),
                      w2_bias=jax.ShapeDtypeStruct(
                          (SERVED_LOCAL_EXPERTS, 1, SERVED_HIDDEN),
                          jnp.float32))
        with pinned_tpu():
            plain = stage_configs(shapes)
            biased = stage_configs(shapes, **biases)
        self.assertEqual(plain[4], biased[4])
        self.assertGreater(gf.fused_vmem_estimate(*biased[:2]),
                           gf.fused_vmem_estimate(*plain[:2]))

    def test_unquantized_weights_are_refused_by_name(self):
        shapes = ffn_shapes()
        shapes["w1_scale"] = None
        with pinned_tpu():
            self.assertIn("scales", reason_for(shapes))
        shapes = ffn_shapes()
        shapes["w2_scale"] = None
        with pinned_tpu():
            self.assertIn("scales", reason_for(shapes))

    def test_padded_intermediate_is_refused_by_name(self):
        """There is no HBM intermediate to trim between the two matmuls."""
        shapes = ffn_shapes(w2_inter=SERVED_INTER - 128)
        with pinned_tpu():
            reason = reason_for(shapes)
        self.assertIn("padded", reason)

    def test_empty_lhs_is_refused_by_name(self):
        """Zero tokens: a named refusal at the gate and at the kernel."""
        shapes = ffn_shapes(num_rows=0)
        with pinned_tpu():
            self.assertIn("empty", reason_for(shapes))
            with self.assertRaisesRegex(ValueError, "at least one row"):
                trace_gmm_fused(shapes)

    def test_mismatched_expert_shapes_are_refused_by_name(self):
        with pinned_tpu():
            wrong_hidden = ffn_shapes()
            wrong_hidden["w1"] = jax.ShapeDtypeStruct(
                (SERVED_LOCAL_EXPERTS, SERVED_HIDDEN // 2, 2 * SERVED_INTER),
                jnp.float8_e4m3fn)
            self.assertIn("contracts", reason_for(wrong_hidden))

            odd_width = ffn_shapes()
            odd_width["w1"] = jax.ShapeDtypeStruct(
                (SERVED_LOCAL_EXPERTS, SERVED_HIDDEN, 2 * SERVED_INTER + 1),
                jnp.float8_e4m3fn)
            self.assertIn("odd", reason_for(odd_width))

    @parameterized.product(
        num_rows=[16, 32, 48, 128, 144, 192, 256, 1024],
        num_groups=[1, 8],
        quant_block=[None, 512],
    )
    def test_a_reason_of_none_means_the_kernel_builds(self, num_rows,
                                                      num_groups, quant_block):
        """The gate answers for the builder: every shape in this product is
        accepted, and every accepted shape builds."""
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        shapes = ffn_shapes(num_groups=num_groups,
                            num_rows=num_rows,
                            weight_dtype=weight_dtype,
                            quant_block=quant_block)
        with pinned_tpu():
            self.assertIsNone(reason_for(shapes))
            trace_gmm_fused(shapes, bucket_base=gf.GMM_FUSED_BUCKET_BASE)

    @parameterized.parameters(24, 40, 56)
    def test_a_row_count_that_has_no_legal_tile_is_refused_by_name(
            self, num_rows):
        """Below 128 rows the tile height IS the row count, and a row count
        that is not whole sublanes has no legal tile."""
        shapes = ffn_shapes(num_rows=num_rows)
        with pinned_tpu():
            reason = reason_for(shapes)
        self.assertIsNotNone(reason)
        self.assertIn("sublane", reason)

    @parameterized.parameters(129, 200, 257)
    def test_a_row_count_that_is_not_whole_sublanes_is_named(
            self, num_rows):
        """Above the tile height such a row count has no layout. The gate
        names it, so the layer falls back instead of crashing."""
        shapes = ffn_shapes(num_rows=num_rows)
        with pinned_tpu():
            reason = reason_for(shapes)
        self.assertIsNotNone(reason)
        self.assertIn("sublane", reason)


class ScaleBlockGuardTest(jtu.JaxTestCase):
    """Which rhs scale block widths each dequant path may carry."""

    def test_pinned_generation_constants(self):
        """The widths below are read against these two numbers."""
        with pinned_tpu():
            self.assertEqual(pltpu.get_tpu_info().mxu_column_size, 256)
        self.assertEqual(LHS_QUANT_BLOCK_SIZE, 512)

    @parameterized.parameters(32, 64, 128, 256, 512, 1024, 4096)
    def test_scale_block_widths_through_the_fused_gate(self, quant_block):
        """The fused kernel carries the post-matmul dequant path only, and
        a multi-block width there must be whole lhs quantization blocks."""
        hidden, inter = 4096, 1024
        shapes = ffn_shapes(hidden=hidden,
                            inter=inter,
                            weight_dtype=jnp.float4_e2m1fn,
                            quant_block=min(quant_block, inter))
        # w1 carries the width under test; w2 carries a width it accepts.
        with pinned_tpu():
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            shapes["w1_scale"] = jax.ShapeDtypeStruct(
                (SERVED_LOCAL_EXPERTS, hidden // quant_block, 1, 2 * inter),
                jnp.float32)
            reason = reason_for(shapes)
        blocks = hidden // quant_block
        if quant_block < mxu_size:
            self.assertIn("narrower than the MXU column", reason)
        elif blocks > 1 and quant_block % LHS_QUANT_BLOCK_SIZE != 0:
            self.assertIn("lhs quantization block", reason)
        else:
            self.assertIsNone(reason)

    @parameterized.product(
        quant_block=[32, 64, 128, 256, 512, 1024, 4096],
        weight_dtype=[jnp.float8_e4m3fn, jnp.float4_e2m1fn],
    )
    def test_scale_block_guard_fires_on_the_post_matmul_path_only(
            self, quant_block, weight_dtype):
        """The pair's own builder keeps its narrow-block layouts, which the
        pre-matmul path applies in full."""
        size_k, size_n = 4096, 2048
        blocks = size_k // quant_block
        lhs = jax.ShapeDtypeStruct((128, size_k), jnp.bfloat16)
        rhs = jax.ShapeDtypeStruct((8, size_k, size_n), weight_dtype)
        rhs_scale = jax.ShapeDtypeStruct((8, blocks, 1, size_n), jnp.float32)

        def build():
            return make_gmm_configs(
                lhs,
                rhs,
                rhs_scale,
                None,
                jax.ShapeDtypeStruct((8, ), jnp.int32),
                jax.ShapeDtypeStruct((1, ), jnp.int32),
                tile_info=TileSizes(tile_m=128, tile_k=size_k, tile_n=size_n),
                vmem_limit_bytes=60 * 2**20,
                out_dtype=None,
                acc_dtype=None,
                maybe_quantize_lhs=True,
                zero_initialize=False,
            )

        with pinned_tpu():
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            post_matmul = quant_block >= mxu_size
            if (post_matmul and blocks > 1
                    and quant_block % LHS_QUANT_BLOCK_SIZE != 0):
                with self.assertRaisesRegex(ValueError,
                                            "lhs quantization block size"):
                    build()
            else:
                cfgs = build()
                self.assertEqual(cfgs.rhs_cfgs.should_dequantize_after_matmul,
                                 post_matmul)


class SingleTileContractTest(jtu.JaxTestCase):
    """num_k == num_n == 1, which the scale-layout acceptance rests on."""

    @parameterized.product(
        num_rows=[16, 128, 1024],
        quant_block=[None, 512, 1024],
    )
    def test_accepted_shapes_keep_the_single_tile_contract(
            self, num_rows, quant_block):
        """Every accepted shape puts the whole contraction in one k tile,
        which is what makes the multi-block (fp4) scale layout safe."""
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        shapes = ffn_shapes(num_rows=num_rows,
                            weight_dtype=weight_dtype,
                            quant_block=quant_block)
        with pinned_tpu():
            self.assertIsNone(reason_for(shapes))
            cfgs1, cfgs2, tiles1, tiles2, _ = stage_configs(shapes)
        for cfgs, tiles in ((cfgs1, tiles1), (cfgs2, tiles2)):
            self.assertEqual(pl.cdiv(cfgs.dims.size_k, tiles.tile_k), 1)
            self.assertEqual(pl.cdiv(cfgs.out_size_n, tiles.tile_n), 1)

    def test_a_second_k_step_is_refused(self):
        """Halving the builder's k tile must fire the single-tile check."""

        def halve_k(tile_m, tile_k, tile_n):
            return TileSizes(tile_m=tile_m, tile_k=tile_k // 2, tile_n=tile_n)

        with pinned_tpu(), mock.patch.object(gf, "TileSizes", halve_k):
            with self.assertRaisesRegex(ValueError, "must fit one k tile"):
                stage_configs(ffn_shapes())

    def test_a_second_n_step_is_refused(self):

        def halve_n(tile_m, tile_k, tile_n):
            return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n // 2)

        with pinned_tpu(), mock.patch.object(gf, "TileSizes", halve_n):
            with self.assertRaisesRegex(ValueError, "must fit one n tile"):
                stage_configs(ffn_shapes())

    def test_a_tile_height_off_the_sublane_is_refused(self):
        with pinned_tpu():
            with self.assertRaisesRegex(ValueError, "multiple of the sublane"):
                stage_configs(ffn_shapes(), tile_m=24)


def _fused_versus_pair_cases():
    """The (rows, groups, scale layout, pipeline mode) grid, minus the
    fp4 unconditional_pipeline arm, whose compile does not return."""
    cases = []
    for num_rows in (16, 32, 48, 144, 192, 256):
        for num_groups in (1, 8, 16):
            for quant_block in (None, 512):
                for unconditional_pipeline in (True, False):
                    if quant_block is not None and unconditional_pipeline:
                        continue
                    scales = ("channelwise" if quant_block is None else
                              f"block{quant_block}")
                    pipeline = ("one_block"
                                if unconditional_pipeline else "pipelined")
                    cases.append(
                        dict(testcase_name=(f"_rows{num_rows}"
                                            f"_groups{num_groups}"
                                            f"_{scales}_{pipeline}"),
                             num_rows=num_rows,
                             num_groups=num_groups,
                             quant_block=quant_block,
                             unconditional_pipeline=unconditional_pipeline))
    return cases


def _biased_fused_versus_pair_cases():
    """The biased grid: row counts that fill tiles and row counts that leave
    remainders, both weight formats, both activations. The fp4 arm skips the
    unconditional pipeline for the same reason the unbiased grid does."""
    cases = []
    for num_rows in (16, 48, 144, 256):
        for quant_block in (None, 512):
            for activation in ("silu", "swigluoai"):
                for unconditional_pipeline in (True, False):
                    if quant_block is not None and unconditional_pipeline:
                        continue
                    scales = ("channelwise" if quant_block is None else
                              f"block{quant_block}")
                    pipeline = ("one_block"
                                if unconditional_pipeline else "pipelined")
                    cases.append(
                        dict(testcase_name=(f"_rows{num_rows}_{scales}"
                                            f"_{activation}_{pipeline}"),
                             num_rows=num_rows,
                             quant_block=quant_block,
                             activation=activation,
                             unconditional_pipeline=unconditional_pipeline))
    return cases


class FusedVersusPairTest(jtu.JaxTestCase):
    """The kernel contract: same result as the two-gmm_v2 pair, on TPU."""

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")

    @parameterized.named_parameters(*_fused_versus_pair_cases())
    def test_fused_matches_the_pair_bitwise(self, num_rows, num_groups,
                                            quant_block,
                                            unconditional_pipeline):
        """Row counts that fill tiles and row counts that leave remainders,
        with uneven group sizes so boundaries land inside sublanes."""
        hidden, inter = 1024, 512
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), num_groups, num_rows, hidden, inter,
            weight_dtype, quant_block or hidden)
        group_sizes = get_group_sizes(num_rows, num_groups)

        _, _, _, _, tile_m = gf.build_stage_configs(
            lhs, w1, w2, w1_scale, w2_scale, group_sizes,
            jnp.array([0], dtype=jnp.int32))
        expected = sequential_pair(lhs, w1, w2, group_sizes, w1_scale,
                                   w2_scale, tile_m)
        actual = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              unconditional_pipeline=unconditional_pipeline)

        self.assertEqual(actual.shape, (num_rows, hidden))
        self.assertArraysEqual(actual, expected)

    @parameterized.product(quant_block=[None, 512])
    def test_fused_matches_the_pair_bitwise_at_the_served_shape(
            self, quant_block):
        hidden, inter, num_rows = SERVED_HIDDEN, SERVED_INTER, 512
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), SERVED_LOCAL_EXPERTS, num_rows, hidden, inter,
            weight_dtype, quant_block or hidden)
        group_sizes = get_group_sizes(num_rows, SERVED_LOCAL_EXPERTS)

        _, _, _, _, tile_m = gf.build_stage_configs(
            lhs, w1, w2, w1_scale, w2_scale, group_sizes,
            jnp.array([0], dtype=jnp.int32))
        expected = sequential_pair(lhs, w1, w2, group_sizes, w1_scale,
                                   w2_scale, tile_m)
        actual = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              bucket_base=gf.GMM_FUSED_BUCKET_BASE,
                              unconditional_pipeline=False)

        self.assertArraysEqual(actual, expected)

    @parameterized.named_parameters(*_biased_fused_versus_pair_cases())
    def test_biased_fused_matches_the_pair_bitwise(self, num_rows, quant_block,
                                                   activation,
                                                   unconditional_pipeline):
        """The kernel contract with expert biases present: the same operands
        into both programs must give the same bits out."""
        hidden, inter, num_groups = 1024, 512, 8
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), num_groups, num_rows, hidden, inter,
            weight_dtype, quant_block or hidden)
        w1_bias, w2_bias = expert_biases(jax.random.key(1), num_groups, hidden,
                                         inter)
        group_sizes = get_group_sizes(num_rows, num_groups)

        _, _, _, _, tile_m = gf.build_stage_configs(lhs,
                                                    w1,
                                                    w2,
                                                    w1_scale,
                                                    w2_scale,
                                                    group_sizes,
                                                    jnp.array([0],
                                                              dtype=jnp.int32),
                                                    w1_bias=w1_bias,
                                                    w2_bias=w2_bias,
                                                    fuse_act=activation)
        expected = sequential_pair(lhs,
                                   w1,
                                   w2,
                                   group_sizes,
                                   w1_scale,
                                   w2_scale,
                                   tile_m,
                                   activation=activation,
                                   w1_bias=w1_bias,
                                   w2_bias=w2_bias)
        actual = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              w1_bias=w1_bias,
                              w2_bias=w2_bias,
                              fuse_act=activation,
                              unconditional_pipeline=unconditional_pipeline)

        self.assertEqual(actual.shape, (num_rows, hidden))
        self.assertArraysEqual(actual, expected)

    @parameterized.product(num_rows=[64, 512], quant_block=[None, 512])
    def test_biased_fused_matches_the_pair_bitwise_at_the_gptoss_shape(
            self, num_rows, quant_block):
        """The wide-and-square expert pair the biases were built for, at the
        row counts a decode step presents, with the clamped activation."""
        hidden, inter, num_groups = 3072, 3072, 4
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), num_groups, num_rows, hidden, inter,
            weight_dtype, quant_block or hidden)
        w1_bias, w2_bias = expert_biases(jax.random.key(1), num_groups, hidden,
                                         inter)
        group_sizes = get_group_sizes(num_rows, num_groups)

        _, _, _, _, tile_m = gf.build_stage_configs(lhs,
                                                    w1,
                                                    w2,
                                                    w1_scale,
                                                    w2_scale,
                                                    group_sizes,
                                                    jnp.array([0],
                                                              dtype=jnp.int32),
                                                    w1_bias=w1_bias,
                                                    w2_bias=w2_bias,
                                                    fuse_act="swigluoai")
        expected = sequential_pair(lhs,
                                   w1,
                                   w2,
                                   group_sizes,
                                   w1_scale,
                                   w2_scale,
                                   tile_m,
                                   activation="swigluoai",
                                   w1_bias=w1_bias,
                                   w2_bias=w2_bias)
        actual = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              w1_bias=w1_bias,
                              w2_bias=w2_bias,
                              fuse_act="swigluoai",
                              unconditional_pipeline=False)

        self.assertArraysEqual(actual, expected)

    @parameterized.product(quant_block=[None, 512],
                           activation=["silu", "swigluoai"])
    def test_a_bias_of_zeros_reproduces_the_unbiased_program(
            self, quant_block, activation):
        """The control for the operand itself: passing zeros must leave the
        no-bias answer untouched, bit for bit."""
        hidden, inter, num_rows, num_groups = 1024, 512, 144, 8
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), num_groups, num_rows, hidden, inter,
            weight_dtype, quant_block or hidden)
        group_sizes = get_group_sizes(num_rows, num_groups)
        zeros1 = jnp.zeros((num_groups, 1, 2 * inter), jnp.bfloat16)
        zeros2 = jnp.zeros((num_groups, 1, hidden), jnp.bfloat16)

        unbiased = gf.gmm_fused(lhs,
                                w1,
                                w2,
                                group_sizes,
                                w1_scale,
                                w2_scale,
                                fuse_act=activation,
                                unconditional_pipeline=False)
        biased = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              w1_bias=zeros1,
                              w2_bias=zeros2,
                              fuse_act=activation,
                              unconditional_pipeline=False)

        self.assertArraysEqual(biased, unbiased)

    @parameterized.product(num_rows=[48, 144], quant_block=[None, 512])
    def test_fused_matches_the_pair_the_layer_would_have_run(
            self, num_rows, quant_block):
        """The pair at its own tiling, as the MoE layer calls it: a
        different k accumulation order, so allclose rather than bitwise."""
        hidden, inter = 1024, 512
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), 8, num_rows, hidden, inter, weight_dtype,
            quant_block or hidden)
        group_sizes = get_group_sizes(num_rows, 8)

        mid = gmm_v2(lhs,
                     w1,
                     group_sizes,
                     rhs_scale=w1_scale,
                     zero_initialize=False,
                     fuse_act="silu",
                     preferred_element_type=lhs.dtype)
        expected = gmm_v2(mid,
                          w2,
                          group_sizes,
                          rhs_scale=w2_scale,
                          zero_initialize=False)
        actual = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              bucket_base=gf.GMM_FUSED_BUCKET_BASE,
                              unconditional_pipeline=False)

        self.assertArraysAllClose(actual.astype(jnp.float32),
                                  expected.astype(jnp.float32),
                                  atol=5e-2,
                                  rtol=5e-2)

    @parameterized.product(
        num_rows=[16, 48, 144, 256],
        quant_block=[None, 512],
        bucket_base=[32, 64],
    )
    def test_bucketing_does_not_change_the_output(self, num_rows, quant_block,
                                                  bucket_base):
        """Bucketing narrows each grid step's compute extent and nothing
        else: same rows in, same bits out."""
        hidden, inter = 1024, 512
        weight_dtype = (jnp.float8_e4m3fn
                        if quant_block is None else jnp.float4_e2m1fn)
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), 8, num_rows, hidden, inter, weight_dtype,
            quant_block or hidden)
        group_sizes = get_group_sizes(num_rows, 8)

        # Both calls take the conditional pipeline: the unconditional
        # default does not finish compiling on the fp4 path.
        unbucketed = gf.gmm_fused(lhs,
                                  w1,
                                  w2,
                                  group_sizes,
                                  w1_scale,
                                  w2_scale,
                                  unconditional_pipeline=False)
        bucketed = gf.gmm_fused(lhs,
                                w1,
                                w2,
                                group_sizes,
                                w1_scale,
                                w2_scale,
                                bucket_base=bucket_base,
                                unconditional_pipeline=False)

        self.assertArraysEqual(bucketed, unbucketed)

    def test_a_single_expert_owning_every_row(self):
        """One group, so one expert owns the whole batch."""
        hidden, inter, num_rows = 1024, 512, 128
        lhs, w1, w2, w1_scale, w2_scale = quantized_ffn_operands(
            jax.random.key(0), 1, num_rows, hidden, inter, jnp.float8_e4m3fn,
            hidden)
        group_sizes = jnp.array([num_rows], dtype=jnp.int32)

        _, _, _, _, tile_m = gf.build_stage_configs(
            lhs, w1, w2, w1_scale, w2_scale, group_sizes,
            jnp.array([0], dtype=jnp.int32))
        expected = sequential_pair(lhs, w1, w2, group_sizes, w1_scale,
                                   w2_scale, tile_m)
        actual = gf.gmm_fused(lhs,
                              w1,
                              w2,
                              group_sizes,
                              w1_scale,
                              w2_scale,
                              bucket_base=gf.GMM_FUSED_BUCKET_BASE)

        self.assertArraysEqual(actual, expected)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
