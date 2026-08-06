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
"""Tests for the fused expert-parallel MoE serving adapter: the acceptance
envelope, the VMEM accounting a caller answers "will this model fit" from,
and the layer's numerics against a plain jax reference."""

import contextlib
import dataclasses
import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tests.kernels.gmm_fused_test import pinned_tpu
from tests.layers.common.conftest import (EightDeviceTestCase, FakeMoELayer,
                                          FakeWeights, mesh_devices,
                                          moe_activations, quantize_weight,
                                          relative_l2, serving_mesh)
from tpu_inference import envs
from tpu_inference.kernels.fused_moe.v2 import host
from tpu_inference.kernels.fused_moe.v2.host import NBUF, VMEM_FRACTION
from tpu_inference.layers.common import moe_fused_ep
from tpu_inference.layers.common.moe_fused_ep import (moe_fused_ep_apply,
                                                      unsupported_batch_reason,
                                                      unsupported_reason)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)

# The served MoE layer.
EXPERTS, HIDDEN, INTER, TOPK, TOKENS, EP, FP4_QB = (512, 4096, 1024, 10, 8192,
                                                    8, 512)

FakeLayer = functools.partial(FakeMoELayer, top_k=TOPK)

# One core's VMEM times the fraction the kernel may use.
V7_VMEM_BUDGET = int(64 * 1024 * 1024 * VMEM_FRACTION)

# The tile height the adapter passes the kernel as its capacity.
TILE_M = moe_fused_ep._TILE_M

# A hidden the kernel's row staging cannot hold (DeepSeek-class), and
# GPT-OSS's width beside what the four-bit requantization pads it to: 2880
# is 22.5 blocks of 128 lanes, 3072 is 24 of them.
WIDE_HIDDEN, PADDED_HIDDEN, PADDED_W_HIDDEN = 7168, 2880, 3072

# How far the layer's output may sit from the same MoE block computed with a
# bfloat16 wire, as a relative difference, and the same for the worst single
# token. Two observations on the fixed-seed reference at the DEV_ geometry
# below read 0.0535/0.0537 (batch) and 0.0644/0.0627 (worst token) on the two
# weight formats; the bounds are the larger of each plus margin. The formats
# do not agree to the digit, so a bound covers a small spread rather than one
# reproducible number. A routing corruption puts one token near 0.39.
WIRE_RELATIVE_DELTA_BOUND = 0.06
WIRE_TOKEN_MAX_DELTA_BOUND = 0.075

# Distinguishes "passed nothing" from "passed None".
_UNSET = object()


def abstract_weights(experts=EXPERTS,
                     hidden=HIDDEN,
                     inter=INTER,
                     dtype=jnp.float8_e4m3fn,
                     qb=None,
                     w_hidden=None,
                     biased=False,
                     scaled=True):
    """The served weight bundle as shapes and dtypes. qb None is the
    per-channel scale layout, an integer the block layout at that size;
    scaled False is the unquantized form, which carries no scales at all;
    w_hidden is the WEIGHT width, which a requantizing loader pads above the
    activation width, and every weight axis carrying hidden follows it."""
    w_hidden = hidden if w_hidden is None else w_hidden
    struct = jax.ShapeDtypeStruct
    blocks1 = 1 if qb is None else w_hidden // qb
    blocks2 = 1 if qb is None else inter // qb
    bias1 = struct((experts, 1, 2 * inter), jnp.float32) if biased else None
    bias2 = struct((experts, 1, w_hidden), jnp.float32) if biased else None
    scale1 = (struct(
        (experts, blocks1, 1, 2 * inter), jnp.float32) if scaled else None)
    scale2 = (struct(
        (experts, blocks2, 1, w_hidden), jnp.float32) if scaled else None)
    return FakeWeights(w13_weight=struct((experts, w_hidden, 2 * inter),
                                         dtype),
                       w2_weight=struct((experts, inter, w_hidden), dtype),
                       w13_weight_scale=scale1,
                       w2_weight_scale=scale2,
                       w13_bias=bias1,
                       w2_bias=bias2)


def gate(mesh,
         weights=None,
         layer=None,
         tokens=TOKENS,
         hidden=HIDDEN,
         activation="silu",
         scatter_results=True,
         extra_backend_kwargs=None,
         defer_all_reduce=False,
         gating_width=None,
         gating_rows=None,
         gating_dtype=jnp.float32,
         x_shape=None,
         gating_output=_UNSET,
         pin=True):
    """unsupported_reason on abstract inputs at the served shapes; pin names
    the served chip, pin=False asks on a host that names none."""
    weights = abstract_weights() if weights is None else weights
    layer = FakeLayer() if layer is None else layer
    if gating_width is None:
        w13 = weights.w13_weight
        gating_width = w13.shape[0] if w13 is not None else EXPERTS
    leaves, treedef = jax.tree.flatten(
        (jax.ShapeDtypeStruct(x_shape or (tokens, hidden), jnp.bfloat16),
         jax.ShapeDtypeStruct(
             (tokens if gating_rows is None else gating_rows, gating_width),
             gating_dtype), weights))
    reason = {}

    def probe(*flat):
        x, gating, w = jax.tree.unflatten(treedef, flat)
        reason["value"] = unsupported_reason(
            layer=layer,
            x=x,
            gating_output=gating if gating_output is _UNSET else gating_output,
            weights=w,
            mesh=mesh,
            activation=activation,
            scatter_results=scatter_results,
            extra_backend_kwargs=extra_backend_kwargs,
            defer_all_reduce=defer_all_reduce)
        return jnp.zeros((1, ), jnp.float32)

    with pinned_tpu() if pin else contextlib.nullcontext():
        jax.eval_shape(probe, *leaves)
    return reason["value"]


def batch_reason(mesh, layer=None, tokens=TOKENS):
    """unsupported_batch_reason at the served shapes."""
    return unsupported_batch_reason(FakeLayer() if layer is None else layer,
                                    jax.ShapeDtypeStruct((tokens, HIDDEN),
                                                         jnp.bfloat16),
                                    mesh,
                                    num_experts=EXPERTS)


def _case(name, expect=(), *, absent=(), mesh=None, axis_set=None, **call):
    """One row of the tables below: what is wrong with the call, and the
    phrases the answer has to carry. mesh is (shape, names) where the served
    one will not do; axis_set names a sharding axis-set override."""
    return dict(testcase_name=name,
                expect=expect,
                absent=absent,
                mesh=mesh,
                axis_set=axis_set,
                call=call)


# Every layer-level refusal, and the phrases it has to say. The reason string
# is what an operator reads out of a stopped build, so the phrase is held
# with the condition rather than only the fact of a refusal, and every
# condition keeps a case of its own. The refusals that need something patched
# out from under the gate, and the ones that read the device record, keep
# tests of their own below.
# yapf: disable
REFUSALS = (
    # Geometry the kernel's transport cannot stage. The staging ceiling is
    # 32 blocks of 128 lanes, and the question is asked of the WEIGHT width.
    _case("_a_hidden_the_row_staging_cannot_hold",
          ("7168", "row staging holds"),
          weights=abstract_weights(hidden=WIDE_HIDDEN, w_hidden=WIDE_HIDDEN),
          hidden=WIDE_HIDDEN),
    _case("_a_hidden_that_is_not_whole_lane_blocks", ("128-lane blocks", ),
          weights=abstract_weights(hidden=4032, w_hidden=4032), hidden=4032),
    _case("_the_unpadded_form_of_a_padded_width", ("128-lane blocks", ),
          weights=abstract_weights(hidden=PADDED_HIDDEN,
                                   w_hidden=PADDED_HIDDEN),
          hidden=PADDED_HIDDEN),
    _case("_a_padded_hidden_that_is_not_whole_lane_blocks",
          ("128-lane blocks", ),
          weights=abstract_weights(hidden=PADDED_HIDDEN,
                                   w_hidden=PADDED_HIDDEN + 32),
          hidden=PADDED_HIDDEN),
    # The other direction is not padding: the kernel contracts the whole
    # activation row and there is nothing to contract the tail against.
    _case("_weights_narrower_than_the_activations",
          ("narrower than activation hidden", ),
          weights=abstract_weights(hidden=HIDDEN, w_hidden=HIDDEN - 128)),
    _case("_a_zero_hidden_width", ("narrower than one", ),
          weights=abstract_weights(hidden=0, w_hidden=0), hidden=0),
    # The hidden axis had a floor and the intermediate had none, so an FFN
    # with no intermediate channels was accepted, built a program and cached
    # it: zero is a whole number of every block size below it, and the
    # estimate for such a build clears the budget.
    _case("_an_intermediate_of_zero", ("narrower than one", "intermediate"),
          weights=abstract_weights(inter=0)),
    _case("_an_intermediate_of_one", ("narrower than one", "intermediate"),
          weights=abstract_weights(inter=1)),
    _case("_an_intermediate_under_a_lane_block",
          ("narrower than one", "intermediate"),
          weights=abstract_weights(inter=64)),
    _case("_an_odd_gate_and_up_width", ("is odd", ),
          weights=dataclasses.replace(
              abstract_weights(),
              w13_weight=jax.ShapeDtypeStruct((EXPERTS, HIDDEN,
                                               2 * INTER + 1),
                                              jnp.float8_e4m3fn))),

    # Routing the kernel cannot honour.
    _case("_sigmoid_scoring", ("softmax + top_k", "sigmoid"),
          layer=FakeLayer(scoring_func="sigmoid")),
    _case("_an_activation_the_kernel_does_not_fuse", ("FFN activation",
                                                      "gelu"),
          activation="gelu"),
    _case("_a_tensor_parallel_layer", ("expert-parallel", ),
          layer=FakeLayer(use_ep=False)),
    _case("_the_all_reduced_output_form", ("scatter_results", ),
          scatter_results=False),
    _case("_defer_all_reduce", ("defer_all_reduce", ), defer_all_reduce=True),
    # Each of these three changes which experts a token goes to.
    _case("_hash_based_topk_indices", ("hash_based_topk_indices", ),
          extra_backend_kwargs={"hash_based_topk_indices": jnp.zeros((4, ))}),
    _case("_e_score_correction_bias", ("e_score_correction_bias", ),
          extra_backend_kwargs={"e_score_correction_bias": jnp.zeros((4, ))}),
    _case("_num_valid_tokens", ("num_valid_tokens", ),
          extra_backend_kwargs={"num_valid_tokens": jnp.zeros((4, ))}),
    # Half the NaN-score guard is that one slot cannot sum two sentinels;
    # the other half is that the guard zeroes the row by renormalizing it.
    _case("_top_k_below_two", ("top_k is 1", ), layer=FakeLayer(top_k=1)),
    _case("_a_layer_that_does_not_renormalize", ("does not renormalize", ),
          layer=FakeLayer(renormalize=False)),
    # Past the last expert the selector re-picks expert zero at the sentinel
    # weight, zeroing a good token the way a NaN row is zeroed.
    _case("_top_k_over_the_expert_count", ("over the 8 experts", ),
          layer=FakeLayer(top_k=16), weights=abstract_weights(experts=8)),
    # A float or a string top_k would serve at the truncated width while the
    # layer's own attribute read the original everywhere else.
    _case("_a_fractional_top_k", ("selection width is an integer", ),
          layer=FakeLayer(top_k=2.9)),
    _case("_a_top_k_written_as_a_string", ("selection width is an integer", ),
          layer=FakeLayer(top_k="10")),
    _case("_an_expert_count_the_shards_do_not_divide",
          ("expert-parallel width", "width that divides the expert count"),
          weights=abstract_weights(experts=EXPERTS + 4)),
    # Refused only by accident before: the top_k check fired first and said
    # "top_k is 10, over the 0 experts to choose from", which sends a loader
    # debugging an empty bundle to look at its router config.
    _case("_a_bundle_with_no_experts", ("carries no experts", ),
          weights=abstract_weights(experts=0)),

    # The gating array.
    _case("_a_gating_output_that_is_not_one_array", ("one logits array", ),
          gating_output=(None, None)),
    _case("_a_three_dimensional_activation", ("two-dimensional", ),
          x_shape=(4, TOKENS, HIDDEN)),
    # One row per token: the routing tables are built per routed pair.
    _case("_a_gating_array_with_the_wrong_row_count", ("one row per token", ),
          gating_rows=TOKENS // 2),
    _case("_an_integer_gating_array", ("float32", ), gating_dtype=jnp.int32),
    # float8_e4m3fn IS a floating point type, so "is it floating" accepted
    # it. At ordinary logit magnitudes neighbouring experts tie exactly under
    # the cast and the selection falls to the lowest index -- a routing
    # decision made by the quantizer -- and past about 448 the cast produces
    # NaN and the mask deletes the token.
    _case("_an_eight_bit_float_gating_array", ("float8_e4m3fn", "float32"),
          gating_dtype=jnp.float8_e4m3fn),
    # The router's index range is this width, and the plan has no sink for
    # an id past the expert count.
    _case("_a_gating_width_past_the_expert_count",
          ("no sink for an expert id", ), gating_width=EXPERTS + 4),

    # The weight element types and their scale layouts. int4 is one of the
    # two four-bit forms deliberately outside the table, and a refusal has
    # to say what IS accepted rather than only that this is not.
    _case("_an_int4_weight", ("not one of the accepted weight element types",
                              "float8_e4m3fn, per_channel",
                              "float4_e2m1fn, per_contraction_block",
                              "int8, per_channel", "bfloat16, none"),
          weights=abstract_weights(dtype=jnp.int4)),
    _case("_an_e5m2_weight", ("not one of the accepted weight element types",
                              "float8_e4m3fn, per_channel"),
          weights=abstract_weights(dtype=jnp.float8_e5m2)),
    # The other four-bit form outside the envelope: the common mixed-format
    # layout blocks at 32, and eight values packed per 32-bit word against
    # the unsigned-32 sublane tiling makes 64 rows the smallest addressable
    # block. A geometry refusal, not a table one, and it stays.
    _case("_a_native_block32_four_bit_layout", ("packed-weight row tile", ),
          weights=abstract_weights(dtype=jnp.float4_e2m1fn, qb=32)),
    # The table's four (dtype, layout) pairs were reported as if exclusive
    # and were not: an (E, 1, 1, N) per-channel table satisfies the block
    # check too, because that check leaves the block axis free. Wherever
    # hidden == inter the block count of one divided through and the bundle
    # was ACCEPTED, the whole contraction read as one block; where the widths
    # differed it was refused for divisibility, telling a loader debugging a
    # scale layout about arithmetic. Equal widths here, so divisibility
    # cannot be what refuses it.
    _case("_fp4_weights_carrying_the_per_channel_layout",
          ("per-channel layout", "per_contraction_block"),
          weights=abstract_weights(qb=None, dtype=jnp.float4_e2m1fn,
                                   hidden=1024, inter=1024), hidden=1024),
    # Scales beside an unquantized weight came from a quantizing loader, and
    # ignoring them would run the model at the wrong magnitudes. Either
    # table alone is enough to say the bundle was quantized, so each side of
    # the test is asked on its own as well as together.
    _case("_bf16_weights_carrying_both_scale_tables",
          ("unquantized and carry no scales", ),
          weights=abstract_weights(dtype=jnp.bfloat16, inter=256)),
    _case("_bf16_weights_carrying_only_a_gate_scale_table",
          ("unquantized and carry no scales", ),
          weights=dataclasses.replace(
              abstract_weights(dtype=jnp.bfloat16, inter=256),
              w2_weight_scale=None)),
    _case("_bf16_weights_carrying_only_a_down_scale_table",
          ("unquantized and carry no scales", ),
          weights=dataclasses.replace(
              abstract_weights(dtype=jnp.bfloat16, inter=256),
              w13_weight_scale=None)),
    # Integer weights reach the matrix unit one widened contraction chunk at
    # a time, and that chunk has to tile both matmuls exactly.
    _case("_an_int8_contraction_the_widening_chunk_does_not_divide",
          ("contraction chunk at a time", ),
          weights=abstract_weights(dtype=jnp.int8, inter=INTER + 128)),
    _case("_mixed_weight_dtypes",
          ("both expert weights must carry one dtype", ),
          weights=dataclasses.replace(
              abstract_weights(),
              w2_weight=jax.ShapeDtypeStruct((EXPERTS, INTER, HIDDEN),
                                             jnp.bfloat16))),
    _case("_two_fp4_block_sizes_between_the_matmuls", ("same block size", ),
          weights=dataclasses.replace(
              abstract_weights(dtype=jnp.float4_e2m1fn, qb=FP4_QB),
              w2_weight_scale=jax.ShapeDtypeStruct((EXPERTS, INTER // 256, 1,
                                                    HIDDEN), jnp.float32))),
    # The refusal names the layout the format wants and the accepted set,
    # where it used to say only that scales were required.
    _case("_a_quantized_bundle_with_no_scales",
          ("per_channel scales and this bundle supplies none", ),
          weights=dataclasses.replace(abstract_weights(),
                                      w13_weight_scale=None,
                                      w2_weight_scale=None)),
    _case("_a_scale_table_narrower_than_float32", ("descales in float32", ),
          weights=dataclasses.replace(
              abstract_weights(),
              w13_weight_scale=jax.ShapeDtypeStruct((EXPERTS, 1, 1,
                                                     2 * INTER),
                                                    jnp.bfloat16))),
    # Deriving the block size from this table divides by its block count,
    # and the guard above it leaves that axis free on purpose, so the answer
    # has to be a refusal rather than an escaped exception.
    _case("_an_fp4_scale_table_with_no_blocks", (),
          absent=("could not be evaluated", ),
          weights=dataclasses.replace(
              abstract_weights(qb=512),
              w13_weight_scale=jax.ShapeDtypeStruct((EXPERTS, 0, 1,
                                                     2 * INTER),
                                                    jnp.float32))),
    _case("_a_weight_bundle_that_did_not_finish_loading",
          ("carries no w13_weight", ),
          weights=dataclasses.replace(abstract_weights(), w13_weight=None)),
    _case("_a_weight_that_is_not_three_dimensional", ("three-dimensional", ),
          weights=dataclasses.replace(
              abstract_weights(),
              w2_weight=jax.ShapeDtypeStruct((EXPERTS, INTER),
                                             jnp.float8_e4m3fn))),
    # The kernel sizes the second weight's buffer from the first, so this is
    # asked of both weight formats rather than only of fp4.
    _case("_the_two_fp8_weights_disagree_on_the_intermediate",
          ("disagree on the intermediate", ),
          weights=dataclasses.replace(
              abstract_weights(),
              w2_weight=jax.ShapeDtypeStruct((EXPERTS, INTER - 128, HIDDEN),
                                             jnp.float8_e4m3fn))),
    _case("_the_two_fp4_weights_disagree_on_the_intermediate",
          ("disagree on the intermediate", ),
          weights=dataclasses.replace(
              abstract_weights(dtype=jnp.float4_e2m1fn, qb=FP4_QB),
              w2_weight=jax.ShapeDtypeStruct((EXPERTS, INTER - 128, HIDDEN),
                                             jnp.float4_e2m1fn))),

    # The expert biases. A loader that padded the weights and not the bias
    # beside them: the down bias reaches the row before the wire
    # quantization, so a wrong-width one is refused rather than mixed in.
    _case("_a_bias_without_its_middle_axis", ("w13_bias layout", ),
          weights=dataclasses.replace(
              abstract_weights(biased=True),
              w13_bias=jax.ShapeDtypeStruct((EXPERTS, 2 * INTER),
                                            jnp.float32))),
    _case("_a_down_bias_left_at_the_unpadded_hidden", ("w2_bias layout", ),
          weights=dataclasses.replace(
              abstract_weights(hidden=HIDDEN - 512, w_hidden=HIDDEN,
                               biased=True),
              w2_bias=jax.ShapeDtypeStruct((EXPERTS, 1, HIDDEN - 512),
                                           jnp.float32)),
          hidden=HIDDEN - 512),
    _case("_an_integer_bias_table", ("floating point table", ),
          weights=dataclasses.replace(
              abstract_weights(biased=True),
              w13_bias=jax.ShapeDtypeStruct((EXPERTS, 1, 2 * INTER),
                                            jnp.int32))),

    # The model does not fit the VMEM budget. The weight slabs dominate and
    # do not shrink with the batch, so the gate must refuse before the build
    # assert fires inside the traced layer.
    _case("_a_layer_whose_buffers_do_not_fit_vmem", ("MiB of VMEM", "budget"),
          weights=abstract_weights(inter=INTER * 4)),

    # The mesh. data > 1 permutes shards under the single-axis re-wrap; the
    # apply function's closing sharding constraint names the full axis set,
    # so an axis the mesh does not carry has to be a reason and not a raise;
    # and the axis-set refusal names the selection rather than the mesh.
    _case("_data_parallel_replicas", ("mesh data axis size 2 != 1", ),
          mesh=((2, 4, 1, 1, 1, 1, 1), MESH_AXIS_NAMES)),
    _case("_a_mesh_whose_axes_are_out_of_order", ("do not contain", ),
          mesh=((8, 1, 1, 1, 1, 1, 1),
                ("attn_dp", "data", "attn_dp_expert", "expert", "model",
                 "dcp", "pcp"))),
    _case("_attention_that_is_not_pure_data_parallel",
          ("attention is not pure DP over all devices", ),
          mesh=((1, 1, 1, 8, 1, 1, 1), MESH_AXIS_NAMES)),
    _case("_a_non_degenerate_axis_outside_the_proven_set", ("'stage'",
                                                            "degenerate"),
          mesh=((1, 4, 1, 1, 1, 1, 1, 2), MESH_AXIS_NAMES + ("stage", ))),
    _case("_a_mesh_missing_the_attention_data_axis", ("'pcp'",
                                                      "does not carry"),
          mesh=((1, 8, 1, 1, 1, 1),
                tuple(n for n in MESH_AXIS_NAMES if n != "pcp"))),
    _case("_the_two_dimensional_axis_set", ("the replica axis",
                                            "NEW_MODEL_DESIGN"),
          axis_set="data"),
    # On one device there are no shards to combine across, which is the
    # whole of what this kernel does. It is answered before the
    # scatter_results question, which has no content on a single rank: the
    # per-rank and all-reduced forms are the same array there, so a caller
    # who asked for the all-reduced form used to be sent looking at a
    # distinction that does not exist on their mesh.
    _case("_a_single_device_mesh", ("one device", ),
          absent=("scatter_results", ),
          mesh=((1, 1, 1, 1, 1, 1, 1), MESH_AXIS_NAMES),
          weights=abstract_weights(experts=8, hidden=512, inter=512),
          layer=FakeLayer(top_k=4), tokens=1024, hidden=512),
    _case("_a_single_device_mesh_asked_for_the_all_reduced_form",
          ("one device", ), absent=("scatter_results", ),
          mesh=((1, 1, 1, 1, 1, 1, 1), MESH_AXIS_NAMES),
          weights=abstract_weights(experts=8, hidden=512, inter=512),
          layer=FakeLayer(top_k=4), tokens=1024, hidden=512,
          scatter_results=False),
)

# Everything the envelope has to take. The accept side is a boot failure if
# it is wrong, so what the routers and loaders in the tree actually produce
# is held case by case: the served configuration on both weight formats, the
# clamped GPT-OSS activation the kernel calls the grouped-matmul kernel's
# swigluoai for, the padded-hidden form a requantizing loader emits, each
# expert bias on its own because the two are independent operands, the
# unquantized and integer weight forms, and the gating dtypes a router emits.
ACCEPTANCES = (
    _case("_the_served_fp8_configuration"),
    _case("_the_served_fp4_block_configuration",
          weights=abstract_weights(dtype=jnp.float4_e2m1fn, qb=FP4_QB)),
    _case("_the_clamped_gpt_oss_activation", activation="swigluoai"),
    _case("_padded_hidden_weights",
          weights=abstract_weights(hidden=PADDED_HIDDEN,
                                   w_hidden=PADDED_W_HIDDEN),
          hidden=PADDED_HIDDEN),
    # The caller forwards the whole kwarg set; only a value refuses.
    _case("_routing_modifiers_passed_as_none",
          extra_backend_kwargs={"hash_based_topk_indices": None,
                                "e_score_correction_bias": None,
                                "num_valid_tokens": None}),
    _case("_bias_carrying_weights", weights=abstract_weights(biased=True)),
    _case("_a_gate_bias_alone",
          weights=dataclasses.replace(abstract_weights(biased=True),
                                      w2_bias=None)),
    _case("_a_down_bias_alone",
          weights=dataclasses.replace(abstract_weights(biased=True),
                                      w13_bias=None)),
    _case("_unquantized_bf16_weights",
          weights=abstract_weights(dtype=jnp.bfloat16, scaled=False,
                                   inter=256)),
    _case("_int8_weights_with_per_channel_scales",
          weights=abstract_weights(dtype=jnp.int8, inter=512)),
    _case("_a_float32_gating_array", gating_dtype=jnp.float32),
    _case("_a_bfloat16_gating_array", gating_dtype=jnp.bfloat16),
)

# The device record, which the gate reads three questions off. The newer
# chip's capacity is readable and its sublane tiling is not, so both sides of
# the VMEM comparison have to sit under one guard or the first fleet to get
# newer chips gets a traceback out of the middle of the model build -- with
# the switch on OR off, because the gate is asked before the switch decides.
# The older chips answer BOTH of the other questions, which is how they were
# being accepted onto a kernel never correctness-checked or timed there. The
# floor is a floor and not a pin, so the served generation passes. The
# validated generation is written out rather than read from the module, so
# that lowering the floor fails here instead of moving both sides together.
DEVICE_QUESTIONS = (
    dict(testcase_name="_no_chip_to_read_a_capacity_from", chip=None,
         expect=("VMEM budget cannot be read", )),
    dict(testcase_name="_a_generation_with_no_layout_rules",
         chip=pltpu.ChipVersion.TPU_8I,
         expect=("VMEM budget cannot be read", "not been built for")),
    dict(testcase_name="_a_v5p_below_the_validated_generation",
         chip=pltpu.ChipVersion.TPU_V5P,
         expect=("generation 5", "validated on TPU v7",
                 "USE_MOE_FUSED_EP_KERNEL")),
    dict(testcase_name="_a_v6e_below_the_validated_generation",
         chip=pltpu.ChipVersion.TPU_V6E,
         expect=("generation 6", "validated on TPU v7",
                 "USE_MOE_FUSED_EP_KERNEL")),
    dict(testcase_name="_the_served_generation",
         chip=pltpu.ChipVersion.TPU_7X, expect=None),
)

# The batch-shape conditions, which describe the batch rather than the layer
# and so route away instead of raising. Zero tokens passes every divisibility
# check and the gather table it builds has no in-bounds row to clamp to: the
# upper clamp IS -1 there. 8200 tokens over 8 shards at top-10 admits no
# routing block of at least 8.
BATCH_SHAPES = (
    dict(testcase_name="_the_served_batch_shape", tokens=TOKENS, expect=None),
    dict(testcase_name="_a_count_the_expert_parallel_width_misses",
         tokens=TOKENS + 4, layer_gate_silent=True,
         expect=("not divisible by the expert-parallel width", )),
    dict(testcase_name="_a_batch_that_admits_no_routing_block", tokens=8200,
         layer_gate_silent=True, expect=("block of at least 8", )),
    dict(testcase_name="_an_empty_batch", tokens=0, expect=("no tokens", )),
)
# yapf: enable


class AcceptanceEnvelopeTest(EightDeviceTestCase, parameterized.TestCase):
    """unsupported_reason: what the kernel takes, what it refuses, and what
    each refusal says."""

    def ask(self, mesh, axis_set, call):
        if axis_set is not None:
            ShardingAxisName.override(ATTN_DATA=axis_set)
        return gate(serving_mesh() if mesh is None else serving_mesh(*mesh),
                    **call)

    @parameterized.named_parameters(*REFUSALS)
    def test_the_gate_refuses_and_says_why(self, expect, absent, mesh,
                                           axis_set, call):
        reason = self.ask(mesh, axis_set, call)
        self.assertIsNotNone(reason)
        for phrase in expect:
            self.assertIn(phrase, reason)
        for phrase in absent:
            self.assertNotIn(phrase, reason)

    @parameterized.named_parameters(*ACCEPTANCES)
    def test_the_gate_accepts(self, expect, absent, mesh, axis_set, call):
        del expect, absent
        self.assertIsNone(self.ask(mesh, axis_set, call))

    def test_approximate_top_k_is_refused(self):
        with mock.patch.dict("os.environ", {"MOE_APPROX_TOPK": "1"}):
            self.assertIn("MOE_APPROX_TOPK", gate(serving_mesh()) or "")

    def test_random_routing_for_benchmarking_is_refused(self):
        """The general path honours it and the kernel has no operand for it,
        so ignoring it would leave a benchmark measuring real routing and
        reporting that it measured random."""
        with mock.patch.object(envs, "FORCE_MOE_RANDOM_ROUTING", True):
            self.assertIn("FORCE_MOE_RANDOM_ROUTING",
                          gate(serving_mesh()) or "")

    def test_an_expert_parallel_width_over_the_slot_field_is_refused(self):
        """Eight devices cannot reach the overflow, so the row block stands
        in for the width the packed field would not hold."""
        moe_fused_ep._import_kernel()
        with mock.patch.object(moe_fused_ep, "_row_block", 16):
            self.assertIn("alignment slots", gate(serving_mesh()) or "")

    def test_an_unimportable_kernel_is_refused(self):
        """The no-fallback path: the gate answers, so the caller still
        chooses, and the answer says whose defect it is."""

        def unimportable():
            raise ImportError("no kernel package here")

        with mock.patch.object(moe_fused_ep, "_import_kernel", unimportable):
            reason = gate(serving_mesh())
        self.assertIn("not importable in this tree", reason)
        self.assertIn("defect in the install", reason)

    def test_the_general_path_remedy_is_offered_only_where_it_works(self):
        """The layer refusal tells an operator to unset the switch and run
        on fused_moe_func. For a weight element type or a mesh that is true.
        For the expert-count divisibility it is not -- fused_moe_func shards
        the experts the same way and refuses the same condition -- so the
        promise would send them round a loop and back to the same sentence
        under another program's name."""
        survives = gate(serving_mesh(),
                        weights=abstract_weights(experts=EXPERTS + 4))
        self.assertTrue(
            moe_fused_ep.reason_survives_the_general_path(survives))
        for call in (dict(defer_all_reduce=True), dict(gating_dtype=jnp.int32),
                     dict(layer=FakeLayer(scoring_func="sigmoid"))):
            reason = gate(serving_mesh(), **call)
            self.assertIsNotNone(reason)
            self.assertFalse(
                moe_fused_ep.reason_survives_the_general_path(reason), call)

    def test_a_defect_in_the_adapter_is_answered_rather_than_raised(self):
        """The never-raises contract, on both gates. The caller turns a
        reason into "this layer cannot: ...", so an escape framed as a
        property of the model sends an operator to look at their checkpoint,
        and an escape at all surfaces as an unhandled build failure for a
        condition the contract promises is an answer."""

        class LayerWithoutUseEp:
            top_k, renormalize, scoring_func = TOPK, True, "softmax"

        reason = gate(serving_mesh(), layer=LayerWithoutUseEp())
        self.assertIn("could not be evaluated", reason)
        self.assertIn("use_ep", reason)
        self.assertIn("defect in this kernel adapter", reason)

        class ExplodingLayer:

            @property
            def top_k(self):
                raise RuntimeError("a defect inside the adapter")

        reason = batch_reason(serving_mesh(), layer=ExplodingLayer())
        self.assertIn("a defect inside the adapter", reason)
        self.assertIn("defect in this kernel adapter", reason)

    @parameterized.named_parameters(*DEVICE_QUESTIONS)
    def test_the_gate_answers_the_device_questions(self, chip, expect):
        if chip is None:
            if jtu.test_device_matches(["tpu"]):
                self.skipTest("a chip is attached, so its capacity is read")
            reason = gate(serving_mesh(), pin=False)
        else:
            info = pltpu.get_tpu_info_for_chip(chip, 1)
            with mock.patch.object(pltpu, "get_tpu_info", lambda: info):
                reason = gate(serving_mesh(), pin=False)
        if expect is None:
            self.assertIsNone(reason)
            return
        self.assertIsNotNone(reason)
        for phrase in expect:
            self.assertIn(phrase, reason)

    def test_each_device_question_is_asked_of_a_chip_the_others_admit(self):
        """The premise of the cases above, without which each would pass on
        a gate that asked nothing, and the floor itself, which the cases
        quote as a literal so that lowering it fails here."""
        self.assertEqual(host.MIN_GENERATION, 7)
        newer = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_8I, 1)
        self.assertGreater(newer.vmem_capacity_bytes, 0)
        with self.assertRaises(NotImplementedError):
            newer.get_sublane_tiling(jnp.float8_e4m3fn)
        for chip in (pltpu.ChipVersion.TPU_V5P, pltpu.ChipVersion.TPU_V6E):
            older = pltpu.get_tpu_info_for_chip(chip, 1)
            self.assertGreater(older.vmem_capacity_bytes, 0)
            self.assertGreater(older.get_sublane_tiling(jnp.float8_e4m3fn), 0)
        served = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        self.assertGreaterEqual(host.chip_generation(served),
                                host.MIN_GENERATION)


class BatchShapeTest(EightDeviceTestCase, parameterized.TestCase):
    """unsupported_batch_reason: what routes away rather than raising."""

    @parameterized.named_parameters(*BATCH_SHAPES)
    def test_the_batch_gate_answers(self,
                                    tokens,
                                    expect,
                                    layer_gate_silent=False):
        reason = batch_reason(serving_mesh(), tokens=tokens)
        if expect is None:
            self.assertIsNone(reason)
            return
        self.assertIsNotNone(reason)
        for phrase in expect:
            self.assertIn(phrase, reason)
        if layer_gate_silent:
            # The batch check answers it; the gate, which a caller raises
            # on, does not.
            self.assertIsNone(gate(serving_mesh(), tokens=tokens))

    def test_a_batch_past_the_arrival_row_field_is_refused(self):
        """The routing tables pack an arrival position and an alignment slot
        into one 32-bit word. The mesh-width check bounds the slot; this
        bounds the other factor of the same word, and it is a statement
        about the batch, so an oversized bucket routes away rather than
        serving a plan whose two fields have corrupted each other."""
        from tpu_inference.kernels.fused_moe.v2.host import (
            ALIGNMENT_SLOT_FIELD, ROWBLK)

        # The smallest power of two past the field, so the routing block the
        # check above it needs is still available and this reason answers.
        tokens = EP
        while (tokens * TOPK +
               (ROWBLK - 1) * EXPERTS) * ALIGNMENT_SLOT_FIELD < 2**31:
            tokens *= 2
        self.assertIn("arrival rows",
                      batch_reason(serving_mesh(), tokens=tokens) or "")

    def test_the_batch_gate_reads_top_k_rather_than_coercing_it(self):
        """The layer gate stopped converting it and the batch gate beside it
        kept int(layer.top_k): "10" became 10 and 2.9 became 2, in a
        function exported next to the apply entry point, so a direct caller
        got the coercion the layer gate exists to refuse."""
        for bad in ("10", 2.9, True, None):
            reason = batch_reason(serving_mesh(), layer=FakeLayer(top_k=bad))
            self.assertIsNotNone(reason, f"top_k={bad!r} accepted")
            self.assertIn("read rather than", reason)


class MeshReconciliationTest(EightDeviceTestCase):
    """The single-axis mesh the kernel transport is entered on."""

    def test_the_single_axis_rewrap_keeps_the_devices_and_the_axis_name(self):
        """The adapter names the mesh 'd', the transport reads that name,
        and the device order is what the shard cuts assume. A mesh the gate
        refuses raises here rather than being re-wrapped."""
        from tpu_inference.kernels.fused_moe import v2 as kernel_package
        self.assertEqual(kernel_package.AXIS, "d")

        mesh = serving_mesh()
        rewrapped = moe_fused_ep._single_axis_mesh(mesh)
        self.assertEqual(rewrapped.axis_names, ("d", ))
        self.assertEqual(list(rewrapped.devices.reshape(-1)),
                         list(mesh.devices.reshape(-1)))

        with self.assertRaisesRegex(ValueError,
                                    "fused EP MoE: mesh data axis size"):
            moe_fused_ep._single_axis_mesh(
                serving_mesh((2, 4, 1, 1, 1, 1, 1), MESH_AXIS_NAMES))


class ApplyEntryPointTest(EightDeviceTestCase):
    """moe_fused_ep_apply re-asks the acceptance questions, so a refused
    configuration cannot reach the kernel by another door."""

    def probe(self, acceptance=_UNSET):
        mesh = serving_mesh()
        leaves, treedef = jax.tree.flatten((jax.ShapeDtypeStruct(
            (TOKENS, HIDDEN),
            jnp.bfloat16), jax.ShapeDtypeStruct(
                (TOKENS, EXPERTS), jnp.float32), abstract_weights()))
        extra = {} if acceptance is _UNSET else {"acceptance": acceptance}

        def run(*flat):
            x, gating, weights = jax.tree.unflatten(treedef, flat)
            return moe_fused_ep_apply(layer=FakeLayer(),
                                      x=x,
                                      gating_output=gating,
                                      weights=weights,
                                      mesh=mesh,
                                      activation="silu",
                                      scatter_results=True,
                                      **extra)

        return jax.eval_shape(run, *leaves)

    def test_apply_asks_both_questions_when_the_caller_has_not_asked(self):
        """A direct caller passes no acceptance pair, so the apply function
        asks -- as it did before the pair existed, and in the same order."""
        asked = []

        with mock.patch.object(moe_fused_ep, "unsupported_reason",
                               lambda *a, **k: asked.append("layer")):
            with mock.patch.object(
                    moe_fused_ep, "unsupported_batch_reason", lambda *a, **k:
                (asked.append("batch"), "this batch shape, no")[1]):
                with self.assertRaisesRegex(ValueError, "this batch shape"):
                    self.probe()
        self.assertEqual(asked, ["layer", "batch"])

    def test_apply_trusts_an_acceptance_pair_the_caller_already_holds(self):
        """moe_apply asks both questions and hands the answers back, so an
        accepted call pays for the check once rather than twice -- two
        thirds of it is the VMEM estimate, which depends only on the model.
        The pair is raised on the way a locally computed one is."""

        def refuse(*a, **k):
            raise AssertionError("the acceptance pair was not trusted")

        with mock.patch.object(moe_fused_ep, "unsupported_reason", refuse):
            with mock.patch.object(moe_fused_ep, "unsupported_batch_reason",
                                   refuse):
                with self.assertRaisesRegex(ValueError, "a caller's reason"):
                    self.probe(acceptance=(None, "a caller's reason"))
                with self.assertRaisesRegex(ValueError, "a layer reason"):
                    self.probe(acceptance=("a layer reason", None))


# The VMEM accounting
#
# A caller uses the kernel's VMEM figure to answer "will this model fit", so
# it has to be an accounting of the buffers the kernel declares rather than
# an approximation of them. The geometries below pin it against arithmetic
# done by hand from those declared shapes: each array padded in its minor
# dimension to the chip's 128 lanes and in its second-minor to its dtype's
# sublane tiling (32 rows for eight-bit, 16 for bf16, 8 for f32 and u32),
# then summed. The tile height is 128 and the four-bit block size 512
# throughout.
#
# They are the served layer, the two GPT-OSS shapes with hidden and
# intermediate padded from 2880 up to whole lane blocks (the form that fits
# at all), and the two formats that quantize no activations. The largest
# GPT-OSS shape is the tightest case in the envelope and the reason the
# figure has to be exact: it clears the budget by about two and a half
# megabytes, which is less than the padding an approximation leaves out.
#
# bias_buffers holds the two resident bias tables the same way, for a build
# that asks for both, so each geometry's buffers total is the figure for a
# build with neither and the two totals together the figure for a build with
# both. The tables are f32 [local experts, output channels] and the local
# expert count is the second-minor dimension: at four local experts it pads
# to the f32 tiling of eight, so those tables cost twice their element count.
#
# The two unquantized-activation formats hold the token rows and the wire
# buffer in bf16 rather than fp8, so those cost twice their eight-bit form;
# neither declares the activation row scale (ls_vm) or the intermediate's
# requantized copy (mid_q), because there is no quantization in their bodies
# to need either. They pay for the concatenation instead (mid_concat), and
# neither charges a wire row, because a sixteen-bit wire ships down_bf16
# itself. The unquantized weight declares no scale tables at all; the integer
# weight declares the same per-channel tables the eight-bit float form does
# and adds one widened contraction chunk, double-buffered, because the served
# generation's matrix unit takes no integer input.
SERVED_FP8 = dict(g_local=64,
                  hidden=4096,
                  inter=1024,
                  weight_format=host.WeightFormat.FP8,
                  rhs_qb=4096,
                  buffers={
                      "lhs_vm": 1048576,
                      "w1_vm": 25165824,
                      "w2_vm": 12582912,
                      "w1s_vm": 524288,
                      "w2s_vm": 1048576,
                      "ls_vm": 8192,
                      "out_vm": 1048576,
                      "oscl_vm": 131072,
                      "acc1": 1048576,
                      "mid_chunks": 262144,
                      "mid_q": 131072,
                      "acc2": 2097152,
                      "down_bf16": 1048576,
                      "wire_rows": 524288,
                  },
                  bias_buffers={
                      "w1b_vm": 524288,
                      "w2b_vm": 1048576,
                  })

GPT_OSS_20B_PADDED_FP4 = dict(g_local=4,
                              hidden=3072,
                              inter=3072,
                              weight_format=host.WeightFormat.FP4,
                              rhs_qb=512,
                              buffers={
                                  "lhs_vm": 1048576,
                                  "w1_vm": 28311552,
                                  "w2_vm": 14155776,
                                  "w1s_vm": 786432,
                                  "w2s_vm": 393216,
                                  "ls_vm": 8192,
                                  "out_vm": 1048576,
                                  "oscl_vm": 131072,
                                  "acc1": 3145728,
                                  "mid_chunks": 786432,
                                  "mid_q": 393216,
                                  "acc2": 1572864,
                                  "down_bf16": 786432,
                                  "wire_rows": 393216,
                                  "widened_weight_block": 6291456,
                              },
                              bias_buffers={
                                  "w1b_vm": 196608,
                                  "w2b_vm": 98304,
                              })

GPT_OSS_120B_PADDED_FP4 = dict(g_local=16,
                               hidden=3072,
                               inter=3072,
                               weight_format=host.WeightFormat.FP4,
                               rhs_qb=512,
                               buffers={
                                   "lhs_vm": 1048576,
                                   "w1_vm": 28311552,
                                   "w2_vm": 14155776,
                                   "w1s_vm": 3145728,
                                   "w2s_vm": 1572864,
                                   "ls_vm": 8192,
                                   "out_vm": 1048576,
                                   "oscl_vm": 131072,
                                   "acc1": 3145728,
                                   "mid_chunks": 786432,
                                   "mid_q": 393216,
                                   "acc2": 1572864,
                                   "down_bf16": 786432,
                                   "wire_rows": 393216,
                                   "widened_weight_block": 6291456,
                               },
                               bias_buffers={
                                   "w1b_vm": 393216,
                                   "w2b_vm": 196608,
                               })

BF16_WEIGHTS = dict(g_local=4,
                    hidden=1024,
                    inter=512,
                    weight_format=host.WeightFormat.BF16,
                    rhs_qb=1024,
                    buffers={
                        "lhs_vm": 1048576,
                        "w1_vm": 6291456,
                        "w2_vm": 3145728,
                        "out_vm": 1048576,
                        "oscl_vm": 131072,
                        "acc1": 524288,
                        "mid_chunks": 131072,
                        "mid_concat": 131072,
                        "acc2": 524288,
                        "down_bf16": 262144,
                    },
                    bias_buffers={
                        "w1b_vm": 32768,
                        "w2b_vm": 32768,
                    })

INT8_WEIGHTS = dict(g_local=8,
                    hidden=2048,
                    inter=1024,
                    weight_format=host.WeightFormat.INT8,
                    rhs_qb=2048,
                    buffers={
                        "lhs_vm": 1048576,
                        "w1_vm": 12582912,
                        "w2_vm": 6291456,
                        "w1s_vm": 65536,
                        "w2s_vm": 65536,
                        "out_vm": 1048576,
                        "oscl_vm": 131072,
                        "acc1": 1048576,
                        "mid_chunks": 262144,
                        "mid_concat": 262144,
                        "acc2": 1048576,
                        "down_bf16": 524288,
                        "widened_weight_block": 4194304,
                    },
                    bias_buffers={
                        "w1b_vm": 65536,
                        "w2b_vm": 65536,
                    })

HAND_ARITHMETIC = {
    "the served layer, fp8 weights": SERVED_FP8,
    "a bf16 layer, unquantized weights": BF16_WEIGHTS,
    "an int8 layer, per-channel scales": INT8_WEIGHTS,
    "GPT-OSS 20B padded, fp4 weights": GPT_OSS_20B_PADDED_FP4,
    "GPT-OSS 120B padded, fp4 weights": GPT_OSS_120B_PADDED_FP4,
}

TIGHTEST = "GPT-OSS 120B padded, fp4 weights"

# The tightest geometry's margin, for a build with neither bias and one with
# both. Ranges, so that losing the margin or gaining a suspiciously large one
# both fail here. The GPT-OSS shapes do carry both biases, so the second is
# the one that matters for them; it comes out of the first and still clears.
MARGIN_RANGE_MIB = {False: (2.0, 3.0), True: (1.5, 2.5)}


def estimate_for(case, biased=False):
    """The kernel's VMEM accounting for one hand-arithmetic geometry."""
    with pinned_tpu():
        return host.vmem_estimate_bytes(case["g_local"],
                                        TILE_M,
                                        case["hidden"],
                                        case["inter"],
                                        nbuf=NBUF,
                                        weight_format=case["weight_format"],
                                        rhs_qb=case["rhs_qb"],
                                        has_w1_bias=biased,
                                        has_w2_bias=biased)


def declared_arrays(case, biased=False, weight_format=None, hidden=None):
    """{name: bytes} for every VMEM buffer one build declares."""
    weight_format = weight_format or case["weight_format"]
    rhs_qb = case["rhs_qb"] if weight_format == case["weight_format"] else None
    hidden = case["hidden"] if hidden is None else hidden
    with pinned_tpu():
        info = pltpu.get_tpu_info()
        arrays = (host.vmem_scratch_arrays(case["g_local"],
                                           TILE_M,
                                           hidden,
                                           case["inter"],
                                           nbuf=NBUF,
                                           weight_format=weight_format,
                                           rhs_qb=rhs_qb,
                                           has_w1_bias=biased,
                                           has_w2_bias=biased) +
                  host.vmem_tile_body_arrays(TILE_M,
                                             hidden,
                                             case["inter"],
                                             weight_format=weight_format,
                                             rhs_qb=rhs_qb))
        return {
            name: host.array_vmem_bytes(shape, dtype, info)
            for name, shape, dtype in arrays
        }


class VmemAccountingTest(EightDeviceTestCase, parameterized.TestCase):
    """The figure a caller compares against the budget."""

    @parameterized.named_parameters(*[
        dict(testcase_name=f"_{label}_{'biased' if biased else 'plain'}",
             label=label,
             biased=biased) for label in sorted(HAND_ARITHMETIC)
        for biased in (False, True)
    ])
    def test_the_accounting_matches_the_hand_arithmetic(self, label, biased):
        """Buffer by buffer, because the total is only trustworthy that way,
        and then as the total the gate quotes. The dict equality is exact,
        so a format that declares a buffer it should not -- a row scale on a
        body that quantizes nothing, say -- fails here too."""
        case = HAND_ARITHMETIC[label]
        expected = dict(case["buffers"])
        if biased:
            expected.update(case["bias_buffers"])
        self.assertEqual(declared_arrays(case, biased=biased), expected)
        self.assertEqual(
            estimate_for(case, biased=biased), sum(expected.values()),
            f"{label}: the accounting disagrees with its own buffers")

    def test_the_staging_and_the_wire_follow_the_activation_format(self):
        """Both buffers hold a token row as (hidden // 128, 128) and it is
        the SECOND-MINOR axis that pads: to 32 sublanes for an eight-bit
        element and 16 for a sixteen-bit one. So an unquantized layer holds
        them at two bytes an element where a quantized one holds them at one
        -- but only above a hidden of 2048, below which that axis is under
        32 either way and the eight-bit form pays for padding exactly where
        the sixteen-bit form pays for data. A geometry that reads the
        difference as a flat doubling is reading it wrong."""
        case = dict(BF16_WEIGHTS, g_local=4, inter=512)
        for hidden, ratio in ((1024, 1), (2048, 1), (3072, 2), (HIDDEN, 2)):
            quantized = declared_arrays(case,
                                        weight_format=host.WeightFormat.FP8,
                                        hidden=hidden)
            unquantized = declared_arrays(case,
                                          weight_format=host.WeightFormat.BF16,
                                          hidden=hidden)
            for name in ("lhs_vm", "out_vm"):
                self.assertEqual(unquantized[name], ratio * quantized[name],
                                 f"{name} at hidden {hidden}")

    @parameterized.named_parameters(("_plain", False), ("_biased", True))
    def test_the_tightest_geometry_that_fits_keeps_its_margin(self, biased):
        """The margin is smaller than the tile padding is, which is why the
        padding has to be counted."""
        with pinned_tpu():
            budget = host.vmem_limit()
        margin = (budget -
                  estimate_for(HAND_ARITHMETIC[TIGHTEST], biased=biased))
        margin /= 2**20
        low, high = MARGIN_RANGE_MIB[biased]
        self.assertTrue(low < margin < high,
                        f"margin {margin:.2f}MiB outside {low}-{high}")

    def test_the_same_geometry_on_eight_bit_weights_does_not_fit(self):
        """Eight-bit weights hold the same slabs at twice the width: the
        four-bit form is what fits, not the shape."""
        case = dict(HAND_ARITHMETIC[TIGHTEST],
                    weight_format=host.WeightFormat.FP8,
                    rhs_qb=3072)
        with pinned_tpu():
            self.assertGreater(estimate_for(case), host.vmem_limit())

    def test_the_accounting_needs_a_chip_to_read_the_tile_shape_from(self):
        """The padding comes off the device record, so a host that can name
        no chip gets an error rather than an answer computed from a guess."""
        if jtu.test_device_matches(["tpu"]):
            self.skipTest("a chip is attached, so the record is readable")
        case = HAND_ARITHMETIC["the served layer, fp8 weights"]
        with self.assertRaises(Exception):
            host.vmem_estimate_bytes(case["g_local"],
                                     TILE_M,
                                     case["hidden"],
                                     case["inter"],
                                     nbuf=NBUF)

    def test_the_refusal_quotes_the_build_this_call_would_get(self):
        """It carries what the layer needs and what the budget is, or the
        reader cannot tell how far over it is; the figure is the one for the
        kernel this call would build, so a model carrying biases is quoted
        the biased accounting; and the comparison is `over the budget`, not
        `far over it`."""
        wide = INTER * 4
        reason = gate(serving_mesh(), weights=abstract_weights(inter=wide))
        with pinned_tpu():
            budget = host.vmem_limit()
            args = (EXPERTS // EP, TILE_M, HIDDEN)
            plain = host.vmem_estimate_bytes(*args, wide, nbuf=NBUF)
            biased = host.vmem_estimate_bytes(*args,
                                              wide,
                                              nbuf=NBUF,
                                              has_w1_bias=True,
                                              has_w2_bias=True)
            just_over = host.vmem_estimate_bytes(*args, 1536, nbuf=NBUF)
        self.assertIn(f"{plain / 2**20:.1f}MiB of VMEM", reason)
        self.assertIn(f"{budget / 2**20:.1f}MiB budget", reason)

        self.assertGreater(biased, plain)
        self.assertIn(
            f"{biased / 2**20:.1f}MiB of VMEM",
            gate(serving_mesh(),
                 weights=abstract_weights(inter=wide, biased=True)))

        self.assertTrue(budget < just_over < 1.1 * budget)
        self.assertIn(
            "MiB of VMEM",
            gate(serving_mesh(), weights=abstract_weights(inter=1536)) or "")


class PaddedHiddenTest(EightDeviceTestCase):
    """The padded-hidden path executed, with the kernel replaced by a
    recorder, so the pad, the down-bias re-zeroing and the trim are what is
    under test rather than the kernel."""

    EXPERTS, INTER, TOKENS, TOPK = 8, 512, 64, 2

    def padded_call(self):
        """(output, operands the kernel was handed). The recorder returns a
        row whose every column is its own index, so which columns the trim
        keeps is visible in the output."""
        seen = {}

        def recorder(x_in, w13, w2, s13, s2, gating_in, w13_bias, w2_bias,
                     **kwargs):
            seen.update(x=x_in, w2_bias=w2_bias)
            w_hidden = w13.shape[1]
            # f32 so every column index is exactly representable; the trim
            # does not touch the dtype.
            return jnp.broadcast_to(jnp.arange(w_hidden, dtype=jnp.float32),
                                    (x_in.shape[0], w_hidden))

        moe_fused_ep._import_kernel()
        # The served chip answers the VMEM questions. pinned_tpu cannot: it
        # pins a one-device abstract mesh and the apply function enters its
        # own eight-device one. The operands are built on a mesh device, so
        # this runs the same way on a host with an accelerator attached.
        self.pin_served_chip()
        zeros = jnp.zeros
        with jax.default_device(mesh_devices(8)[0]):
            weights = FakeWeights(
                w13_weight=zeros(
                    (self.EXPERTS, PADDED_W_HIDDEN, 2 * self.INTER),
                    jnp.float8_e4m3fn),
                w2_weight=zeros((self.EXPERTS, self.INTER, PADDED_W_HIDDEN),
                                jnp.float8_e4m3fn),
                w13_weight_scale=zeros((self.EXPERTS, 1, 1, 2 * self.INTER),
                                       jnp.float32),
                w2_weight_scale=zeros((self.EXPERTS, 1, 1, PADDED_W_HIDDEN),
                                      jnp.float32),
                w13_bias=zeros((self.EXPERTS, 1, 2 * self.INTER), jnp.float32),
                # Every column set, so a padded tail left alone is visible.
                w2_bias=jnp.ones((self.EXPERTS, 1, PADDED_W_HIDDEN),
                                 jnp.float32))
            with mock.patch.object(moe_fused_ep, "_kernel", recorder):
                out = moe_fused_ep_apply(layer=FakeLayer(top_k=self.TOPK),
                                         x=jnp.ones(
                                             (self.TOKENS, PADDED_HIDDEN),
                                             jnp.bfloat16),
                                         gating_output=zeros(
                                             (self.TOKENS, self.EXPERTS),
                                             jnp.float32),
                                         weights=weights,
                                         mesh=serving_mesh(),
                                         activation="silu",
                                         scatter_results=True)
        return out, seen

    def test_the_pad_the_bias_and_the_trim(self):
        """The activation is padded and zero-filled to the weight width; the
        down bias's padded columns are re-zeroed, because they reach the row
        before the wire quantization and a nonzero one would move the row's
        scale; and the trim keeps the LEADING columns, where trimming the
        wrong end gives the same shape and the wrong data."""
        out, seen = self.padded_call()
        self.assertEqual(seen["x"].shape, (self.TOKENS, PADDED_W_HIDDEN))
        self.assertFalse(np.any(np.asarray(seen["x"][:, PADDED_HIDDEN:])))

        bias = np.asarray(seen["w2_bias"])
        self.assertFalse(np.any(bias[..., PADDED_HIDDEN:]))
        self.assertTrue(np.all(bias[..., :PADDED_HIDDEN] == 1.0))

        self.assertEqual(out.shape, (self.TOKENS, PADDED_HIDDEN))
        np.testing.assert_array_equal(
            np.asarray(out[0], np.float32),
            np.arange(PADDED_HIDDEN, dtype=np.float32))


# Numerics against a plain jax reference (device). A reduced but structurally
# identical layer: expert count divisible by the shard count, hidden whole
# 128-lane blocks, fp4 block dividing both axes.
DEV_EXPERTS, DEV_HIDDEN, DEV_INTER, DEV_TOPK, DEV_TOKENS = (16, 1024, 512, 4,
                                                            1024)


def dequantize_weight(q, scale):
    """The f32 weight the kernel's matmuls are meant to be computing with."""
    experts, contract, out = q.shape
    blocks = scale.shape[1]
    span = contract // blocks
    return (q.astype(jnp.float32).reshape(experts, blocks, span, out) *
            scale.reshape(experts, blocks, 1, out)).reshape(
                experts, contract, out)


def ref_moe(x, weights, gating, biased=False):
    """A plain f32 jax MoE over the same dequantized weights, with no fp8
    transport anywhere: the comparison the wire tolerance is stated against.
    The biases enter where the kernel puts them -- the gate and up halves
    before the activation, the down bias on the expert's output row, which
    the router weight then scales, once per selected expert."""
    w13 = dequantize_weight(weights.w13_weight, weights.w13_weight_scale)
    w2 = dequantize_weight(weights.w2_weight, weights.w2_weight_scale)
    experts, inter = w13.shape[0], w2.shape[1]
    b13 = weights.w13_bias if biased else None
    b2 = weights.w2_bias if biased else None
    scores = jax.nn.softmax(gating.astype(jnp.float32), axis=-1)
    tw, ti = jax.lax.top_k(scores, DEV_TOPK)
    tw = tw / tw.sum(axis=-1, keepdims=True)
    x32 = x.astype(jnp.float32)
    out = jnp.zeros_like(x32)
    for e in range(experts):
        # A token's gate for this expert is the sum of the slots that chose
        # it, which is zero for a token that did not.
        gate_w = jnp.sum(jnp.where(ti == e, tw, 0.0), axis=-1)[:, None]
        acc1 = x32 @ w13[e]
        if b13 is not None:
            acc1 = acc1 + b13[e]
        row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2[e]
        if b2 is not None:
            row = row + b2[e]
        out = out + gate_w * row
    return out


def device_layer_inputs(seed, dtype, qb=None, biased=False):
    kx, k1, k2, kb1, kb2 = jax.random.split(jax.random.key(seed), 5)
    x, gating = moe_activations(kx, DEV_TOKENS, DEV_HIDDEN, DEV_EXPERTS)
    q13, s13 = quantize_weight(
        jax.random.normal(k1, (DEV_EXPERTS, DEV_HIDDEN, 2 * DEV_INTER),
                          jnp.float32) / 10, dtype, qb)
    q2, s2 = quantize_weight(
        jax.random.normal(k2,
                          (DEV_EXPERTS, DEV_INTER, DEV_HIDDEN), jnp.float32) /
        10, dtype, qb)
    # The loader's layout: one row per expert on each matmul's outputs.
    b13 = (jax.random.normal(kb1,
                             (DEV_EXPERTS, 1, 2 * DEV_INTER), jnp.float32) /
           10) if biased else None
    b2 = (jax.random.normal(kb2, (DEV_EXPERTS, 1, DEV_HIDDEN), jnp.float32) /
          10) if biased else None
    return x, FakeWeights(w13_weight=q13,
                          w2_weight=q2,
                          w13_weight_scale=s13,
                          w2_weight_scale=s2,
                          w13_bias=b13,
                          w2_bias=b2), gating


def worst_token_relative_l2(actual, want):
    """The largest per-token relative error. Routing failures are per token,
    and a batch-wide norm divides one bad row by every other row's size."""
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    per_token = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    return float(np.max(per_token / np.where(scale == 0, 1.0, scale)))


class LayerNumericsTest(EightDeviceTestCase, parameterized.TestCase):
    """The layer against a plain jax reference, on the chip it is served on."""

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+ (the kernel reads its VMEM budget "
                          "off the device)")

    def apply(self, x, weights, gating):
        return moe_fused_ep_apply(layer=FakeLayer(top_k=DEV_TOPK),
                                  x=x,
                                  gating_output=gating,
                                  weights=weights,
                                  mesh=serving_mesh(),
                                  activation="silu",
                                  scatter_results=True)

    # yapf: disable
    @parameterized.named_parameters(
        dict(testcase_name="_fp8", dtype=jnp.float8_e4m3fn, qb=None,
             biased=False),
        dict(testcase_name="_fp8_biased", dtype=jnp.float8_e4m3fn, qb=None,
             biased=True),
        dict(testcase_name="_fp4_qb512", dtype=jnp.float4_e2m1fn, qb=FP4_QB,
             biased=False),
        dict(testcase_name="_fp4_qb512_biased", dtype=jnp.float4_e2m1fn,
             qb=FP4_QB, biased=True),
    )
    # yapf: enable
    def test_layer_output_tracks_a_plain_jax_reference(self, dtype, qb,
                                                       biased):
        """How far the layer may sit from an unquantized reference. The
        bound is the fp8 wire tolerance and the rotated-expert control makes
        it meaningful. The biased case is the whole add-once argument end to
        end: the down bias is added inside the pallas body, on one shard,
        and the combine weights it once per selected expert; adding it
        twice, or once per shard, moves the batch norm well past the band."""
        x, weights, gating = device_layer_inputs(0, dtype, qb, biased=biased)
        out = self.apply(x, weights, gating)
        want = ref_moe(x, weights, gating, biased=biased)
        error = relative_l2(out, want)
        self.assertLess(error, WIRE_RELATIVE_DELTA_BOUND,
                        f"relative L2 {error:.4f} past the wire band")
        worst = worst_token_relative_l2(out, want)
        self.assertLess(
            worst, WIRE_TOKEN_MAX_DELTA_BOUND,
            f"worst token's relative error {worst:.4f} past the per-token "
            f"band; the batch norm was {error:.4f}, so this is a few tokens "
            f"rather than the whole batch")
        roll = lambda a: jnp.roll(a, 1, axis=0)  # noqa: E731
        rotated = ref_moe(
            x,
            dataclasses.replace(
                weights,
                w13_weight=roll(weights.w13_weight),
                w13_weight_scale=roll(weights.w13_weight_scale),
                w2_weight=roll(weights.w2_weight),
                w2_weight_scale=roll(weights.w2_weight_scale)), gating)
        self.assertGreater(relative_l2(out, rotated), 10 * error)

    def test_the_negative_controls_move_the_measures_they_guard(self):
        """A doubled down bias has to be visible in the batch norm, or the
        biased case above would pass on a body that added it any number of
        times. And one token corrupted by a quarter has to slip PAST the
        batch norm and be caught by the per-token one, or the per-token
        measure is not seeing what the batch norm cannot; the quarter-way
        mix is sized to sit in that gap, since replacing the token outright
        the batch norm also sees."""
        x, weights, gating = device_layer_inputs(0,
                                                 jnp.float8_e4m3fn,
                                                 biased=True)
        out = self.apply(x, weights, gating)
        doubled = ref_moe(x,
                          dataclasses.replace(weights,
                                              w2_bias=2 * weights.w2_bias),
                          gating,
                          biased=True)
        self.assertGreater(relative_l2(out, doubled),
                           WIRE_RELATIVE_DELTA_BOUND)

        x, weights, gating = device_layer_inputs(2, jnp.float8_e4m3fn)
        out = self.apply(x, weights, gating)
        want = np.asarray(ref_moe(x, weights, gating), np.float64)
        corrupted = np.array(want)
        bad_token, mix = 41, 0.25
        corrupted[bad_token] = ((1.0 - mix) * want[bad_token] +
                                mix * want[(bad_token + 1) % DEV_TOKENS])
        clean = relative_l2(out, want)
        batch = relative_l2(out, corrupted)
        self.assertLess(
            batch, WIRE_RELATIVE_DELTA_BOUND,
            f"one corrupted token is meant to slip past the batch-wide "
            f"norm: it took a clean {clean:.4f} to {batch:.4f}")
        self.assertGreater(
            worst_token_relative_l2(out, corrupted), WIRE_RELATIVE_DELTA_BOUND,
            "the per-token measure did not see a token corrupted by a "
            "quarter, so it is not seeing what the batch norm cannot")

    def test_a_nan_row_leaves_every_other_token_bitwise_unchanged(self):
        """Token locality end to end: one inf logit gives the router a NaN
        row, and every token but that one comes back with identical bits."""
        x, weights, gating = device_layer_inputs(1, jnp.float8_e4m3fn)
        nan_token = 137
        poisoned = gating.at[nan_token, 3].set(jnp.inf)
        self.assertTrue(
            bool(
                jnp.isnan(jax.nn.softmax(poisoned, axis=-1)[nan_token]).all()))
        clean = np.asarray(self.apply(x, weights, gating).astype(jnp.float32))
        dirty = np.asarray(
            self.apply(x, weights, poisoned).astype(jnp.float32))
        keep = [t for t in range(DEV_TOKENS) if t != nan_token]
        np.testing.assert_array_equal(dirty[keep], clean[keep])
        self.assertTrue(np.isfinite(dirty[nan_token]).all())

    def test_the_pinned_vmem_budget_matches_the_device(self):
        """The envelope answers from V7_VMEM_BUDGET; hold it to the truth."""
        self.assertEqual(host.vmem_limit(), V7_VMEM_BUDGET)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
