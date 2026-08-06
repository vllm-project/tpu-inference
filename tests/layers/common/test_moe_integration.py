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
"""Tests for which MoE program an expert-parallel call gets: the
USE_MOE_FUSED_EP_KERNEL engagement switch, the token threshold underneath
it, what the fused kernel's refusals do in each switch state, and what the
two kernels compute for the same batch."""

import os
import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tests.layers.common.conftest import (EightDeviceTestCase, FakeMoELayer,
                                          moe_activations, quantize_weight,
                                          serving_mesh)
from tpu_inference import envs

# moe.py imports RoutedExperts from vllm. Probe that one symbol rather than
# guarding the import of the module under test, so a real breakage in moe.py
# fails here instead of reporting a green skip.
_vllm_fused_moe = pytest.importorskip("vllm.model_executor.layers.fused_moe")
if not hasattr(_vllm_fused_moe, "RoutedExperts"):
    import sys

    import vllm

    _FLOOR = (
        f"the installed vllm {vllm.__version__} predates "
        "vllm.model_executor.layers.fused_moe.RoutedExperts, which "
        "tpu_inference.layers.common.moe imports at module level, so "
        "nothing in this suite can run. This is the only suite that "
        "exercises moe_apply routing into the fused expert-parallel kernel "
        "(the switch, the threshold, the refusal-versus-fallback split), "
        "so a skipped run leaves that path unexercised. RoutedExperts is "
        "present from vllm 0.25.1 onwards; upgrade vllm to run this suite.")
    # A module-scope raise would be a collection error, and CI runs pytest
    # over the whole tests/ tree with -x, so an environment floor raised here
    # would take down unrelated suites. Reported as a module-level skip with
    # the reason printed once to stderr.
    print(f"\nSKIPPING {__name__}: {_FLOOR}\n", file=sys.stderr, flush=True)
    pytest.skip(_FLOOR, allow_module_level=True)

from tpu_inference.layers.common import moe as moe_module  # noqa: E402
from tpu_inference.layers.common import moe_fused_ep  # noqa: E402
from tpu_inference.layers.common.moe import MoEBackend  # noqa: E402
from tpu_inference.layers.common.moe import moe_apply  # noqa: E402
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights  # noqa: E402

# A served-shaped expert layer, small enough to build on CPU: one local
# expert per device, hidden a whole number of the kernel's lane blocks.
NUM_EXPERTS, HIDDEN, INTER, TOP_K = 8, 512, 512, 8

# Token counts either side of the default threshold. All are acceptable to
# the fused kernel, so only the threshold separates them.
BELOW_THRESHOLD, AT_THRESHOLD, ABOVE_THRESHOLD = 1016, 1024, 2048

# The EP path carries expert output rows over the wire as fp8 e4m3 where the
# general path uses bfloat16; a zero delta would mean that stopped happening.
# The ceiling is how far the fused layer sits from the same MoE block on a
# bfloat16 wire: measured 0.0535 on both weight formats at the served shape,
# deterministic across independent runs, plus margin.
WIRE_DELTA_FLOOR = 1e-4
WIRE_DELTA_CEILING = 0.06

THRESHOLD_SETTING = "MOE_FUSED_EP_KERNEL_MIN_TOKENS"


def refused_layer():
    """A layer the fused kernel cannot take: its routing is not softmax."""
    return FakeMoELayer(top_k=TOP_K, scoring_func="sigmoid")


class MoEWeightsMixin:
    """A served-shaped quantized expert layer and a batch to send through."""

    def build_weights(self, key, with_bias=False):
        w13_key, w2_key = jax.random.split(key)
        w13_q, w13_scale = quantize_weight(
            jax.random.normal(w13_key,
                              (NUM_EXPERTS, HIDDEN, 2 * INTER), jnp.float32) /
            10, jnp.float8_e4m3fn)
        w2_q, w2_scale = quantize_weight(
            jax.random.normal(w2_key,
                              (NUM_EXPERTS, INTER, HIDDEN), jnp.float32) / 10,
            jnp.float8_e4m3fn)
        zeros = jnp.zeros
        return FusedMoEWeights(
            w13_weight=w13_q,
            w13_weight_scale=w13_scale,
            w13_bias=(zeros((NUM_EXPERTS, 1,
                             2 * INTER), jnp.float32) if with_bias else None),
            w2_weight=w2_q,
            w2_weight_scale=w2_scale,
            w2_bias=(zeros(
                (NUM_EXPERTS, 1, HIDDEN), jnp.float32) if with_bias else None),
        )

    def build_activations(self, key, num_tokens):
        return moe_activations(key, num_tokens, HIDDEN, NUM_EXPERTS)


class MoERouteTestBase(MoEWeightsMixin, EightDeviceTestCase):
    """An eight-device CPU mesh, and the hardware answers the route needs."""

    def setUp(self):
        super().setUp()
        self.mesh = serving_mesh(devices=self.cpu_devices)
        self.pin_served_chip()

    def programs_reached(self,
                         num_tokens,
                         *,
                         fused_ep,
                         weights=None,
                         extra=None,
                         layer=None,
                         backend=MoEBackend.GMM_EP):
        """The MoE programs one moe_apply call reaches, in order. Both are
        stubbed, so only the choice between them is under test."""
        taken = []

        def fake_fused_ep_apply(**kwargs):
            taken.append("fused_ep")
            return jnp.zeros((kwargs["x"].shape[0], HIDDEN), jnp.bfloat16)

        def fake_fused_moe_func(**kwargs):
            taken.append("fused_moe_func")
            return jnp.zeros((kwargs["hidden_states"].shape[0], HIDDEN),
                             jnp.bfloat16)

        with jax.default_device(self.cpu_devices[0]):
            if weights is None:
                weights = self.build_weights(jax.random.key(0))
            x, gating = self.build_activations(jax.random.key(1), num_tokens)
            self.enter_context(
                mock.patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", fused_ep))
            self.enter_context(
                mock.patch.object(moe_fused_ep, "moe_fused_ep_apply",
                                  fake_fused_ep_apply))
            self.enter_context(
                mock.patch.object(moe_module, "fused_moe_func",
                                  fake_fused_moe_func))
            moe_apply(
                layer=layer or FakeMoELayer(top_k=TOP_K),
                x=x,
                gating_output=gating,
                weights=weights,
                moe_backend=backend,
                mesh=self.mesh,
                extra_backend_kwargs={
                    "scatter_results": True,
                    **(extra or {})
                },
            )
        return taken

    def route_taken(self, num_tokens, *, fused_ep=True, **kwargs):
        """The single MoE program the call was handed."""
        taken = self.programs_reached(num_tokens, fused_ep=fused_ep, **kwargs)
        self.assertLen(taken, 1)
        return taken[0]


# Where every call ends up. Off, the general path serves everything and a
# stale threshold beside the unset switch changes nothing -- "0" is what a
# launcher writes for "no threshold" and what a configuration written
# against an older spelling carries forward. On, the threshold decides, the
# comparison is at-or-above, the switch only ever selects for the
# expert-parallel backend, and a token count the expert-parallel width does
# not divide describes the batch, so it routes rather than raising.
# yapf: disable
ROUTES = (
    dict(testcase_name="_off_above_the_threshold", fused_ep=False,
         tokens=ABOVE_THRESHOLD, program="fused_moe_func"),
    dict(testcase_name="_off_with_a_stale_threshold_beside_it",
         fused_ep=False, tokens=ABOVE_THRESHOLD, environ={THRESHOLD_SETTING:
                                                          "0"},
         program="fused_moe_func"),
    dict(testcase_name="_off_for_a_layer_the_kernel_refuses", fused_ep=False,
         tokens=ABOVE_THRESHOLD, refused=True, program="fused_moe_func"),
    dict(testcase_name="_on_below_the_threshold", tokens=BELOW_THRESHOLD,
         program="fused_moe_func"),
    dict(testcase_name="_on_at_the_threshold", tokens=AT_THRESHOLD,
         program="fused_ep"),
    dict(testcase_name="_on_above_the_threshold", tokens=ABOVE_THRESHOLD,
         program="fused_ep"),
    dict(testcase_name="_on_with_the_tensor_parallel_backend",
         tokens=ABOVE_THRESHOLD, backend=MoEBackend.GMM_TP,
         program="fused_moe_func"),
    dict(testcase_name="_on_with_the_tensor_parallel_backend_and_a_refusal",
         tokens=ABOVE_THRESHOLD, backend=MoEBackend.GMM_TP, refused=True,
         program="fused_moe_func"),
    dict(testcase_name="_on_with_a_count_the_width_does_not_divide",
         tokens=ABOVE_THRESHOLD + 4, program="fused_moe_func"),
)
# yapf: enable


class RouteSelectionTest(MoERouteTestBase, parameterized.TestCase):
    """Which program a call is handed. Both branches are stubbed, so only
    the choice is under test."""

    @parameterized.named_parameters(*ROUTES)
    def test_the_route_taken(self,
                             tokens,
                             program,
                             fused_ep=True,
                             backend=MoEBackend.GMM_EP,
                             refused=False,
                             environ=None):
        with mock.patch.dict(os.environ, environ or {}):
            self.assertEqual(
                self.route_taken(tokens,
                                 fused_ep=fused_ep,
                                 backend=backend,
                                 layer=refused_layer() if refused else None),
                program)

    def test_the_threshold_is_the_only_thing_moving_the_boundary(self):
        """The default is the served one, and a deployment that sets another
        gets that one: raised through the module the boundary moves with it,
        and a value written into the environment the way a launcher writes
        it is read as that value rather than as the default."""
        self.assertEqual(envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS, 1024)
        with mock.patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                               ABOVE_THRESHOLD):
            self.assertEqual(self.route_taken(AT_THRESHOLD), "fused_moe_func")
            self.assertEqual(self.route_taken(ABOVE_THRESHOLD), "fused_ep")
        with mock.patch.dict(os.environ,
                             {THRESHOLD_SETTING: str(ABOVE_THRESHOLD)}):
            self.assertEqual(envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
                             ABOVE_THRESHOLD)
            self.assertEqual(self.route_taken(AT_THRESHOLD), "fused_moe_func")

    def test_the_threshold_is_still_validated_once_the_switch_is_on(self):
        """It is only the non-user who is spared a stale value: with the
        switch on, a bad one is an error rather than a silent default."""
        with mock.patch.dict(os.environ, {THRESHOLD_SETTING: "0"}):
            with self.assertRaises(ValueError) as caught:
                self.programs_reached(ABOVE_THRESHOLD, fused_ep=True)
        self.assertIn(THRESHOLD_SETTING, str(caught.exception))


class EngagementSwitchTest(MoERouteTestBase):
    """USE_MOE_FUSED_EP_KERNEL decides whether the fused expert-parallel MoE
    kernel is considered at all. Off, the general path serves every call and
    the kernel cannot affect the program. On, the kernel serves every call it
    can take and a LAYER it cannot take stops the build."""

    def test_the_switch_defaults_to_off(self):
        """Nothing set means the kernel never runs, which is what makes the
        general path unchanged for a tree that merely carries it."""
        self.assertFalse(envs.USE_MOE_FUSED_EP_KERNEL)

    def test_off_never_asks_the_kernel_whether_it_could_serve(self):
        """The off-state guarantee is structural: the adapter is not imported
        and its acceptance check is never consulted, so it is not a fallback
        that happens to agree."""

        def must_not_run(**kwargs):
            raise AssertionError(
                "the acceptance check ran with the switch off")

        with mock.patch.object(moe_fused_ep, "unsupported_reason",
                               must_not_run):
            self.assertEqual(self.route_taken(ABOVE_THRESHOLD, fused_ep=False),
                             "fused_moe_func")

    def test_on_stops_the_build_for_a_layer_the_kernel_refuses(self):
        """On means the kernel serves the call, so a layer it cannot take is
        an error naming the reason, never a quiet change of program. The
        refusal is about the layer, so it does not wait for a shape the
        threshold lets through, and it carries the acceptance check's reason
        string verbatim so the error says exactly what was wrong."""
        for tokens in (ABOVE_THRESHOLD, BELOW_THRESHOLD):
            with self.assertRaises(ValueError) as caught:
                self.programs_reached(tokens,
                                      fused_ep=True,
                                      layer=refused_layer())
            message = str(caught.exception)
            self.assertIn("USE_MOE_FUSED_EP_KERNEL is on", message)
            self.assertIn("softmax", message)

        extra = {"num_valid_tokens": jnp.zeros((4, ), jnp.int32)}
        with jax.default_device(self.cpu_devices[0]):
            weights = self.build_weights(jax.random.key(0))
            x, gating = self.build_activations(jax.random.key(1),
                                               ABOVE_THRESHOLD)
            reason = moe_fused_ep.unsupported_reason(
                layer=FakeMoELayer(top_k=TOP_K),
                x=x,
                gating_output=gating,
                weights=weights,
                mesh=self.mesh,
                activation="silu",
                scatter_results=True,
                extra_backend_kwargs={
                    "scatter_results": True,
                    **extra
                },
            )
        self.assertIsNotNone(reason)
        with self.assertRaises(ValueError) as caught:
            self.programs_reached(ABOVE_THRESHOLD, fused_ep=True, extra=extra)
        self.assertIn(reason, str(caught.exception))

    def test_on_refuses_a_conflicting_selector(self):
        """USE_MOE_EP_KERNEL wins the backend selection, so the two together
        are refused rather than one of them silently ignored. The refusal is
        answered where the backend is chosen, which is the only place that
        knows which selector won."""
        with mock.patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            with mock.patch.object(envs, "USE_MOE_EP_KERNEL", True):
                with self.assertRaises(ValueError) as caught:
                    moe_module.announce_moe_backend(MoEBackend.FUSED_MOE)
        message = str(caught.exception)
        self.assertIn("USE_MOE_EP_KERNEL", message)
        self.assertIn("USE_MOE_FUSED_EP_KERNEL", message)

    def test_announce_refuses_something_that_is_not_a_backend(self):
        """It is the build-time surface that says which program ran, and it
        runs before anything else in the feature, so a caller handing it a
        string or a None is told that rather than handed an attribute error
        out of a logging line."""
        for not_a_backend in (None, "gmm_ep", object()):
            with self.assertRaises(ValueError) as caught:
                moe_module.announce_moe_backend(not_a_backend)
            self.assertIn("not a MoEBackend", str(caught.exception))

    def test_on_without_expert_parallelism_warns_rather_than_being_silent(
            self):
        """The most likely first attempt is the switch on and expert
        parallelism off. That selects GMM_TP, which the switch cannot reach,
        and nothing else in the path would say a word about it."""
        with mock.patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            with mock.patch.object(moe_module.logger, "warning_once") as warn:
                moe_module.announce_moe_backend(MoEBackend.GMM_TP)
        warn.assert_called_once()
        message = warn.call_args[0][0] % warn.call_args[0][1:]
        self.assertIn("USE_MOE_FUSED_EP_KERNEL is on", message)
        self.assertIn("expert parallelism", message)

    def test_both_engagement_directions_are_reported_at_info(self):
        """The only channel that says what actually happened, rather than
        what the switch says, is one INFO line per batch shape in each
        direction. The verification drivers grep for these.

        Patched on the adapter's own logger, which is where the line is
        emitted. Patching moe.py's logger passed for the wrong reason: the
        assertion is over the calls that were recorded, an empty recording
        makes any(...) false, and it was only ever true here because the
        adapter's real logger was left running and the recorder saw
        nothing."""
        for num_tokens, expected in ((ABOVE_THRESHOLD,
                                      "runs on the fused expert-parallel "
                                      "MoE kernel"),
                                     (BELOW_THRESHOLD,
                                      "runs on fused_moe_func")):
            with mock.patch.object(moe_fused_ep.logger, "info_once") as info:
                self.route_taken(num_tokens, fused_ep=True)
            lines = [call[0][0] % call[0][1:] for call in info.call_args_list]
            self.assertTrue(
                any(expected in line and str(num_tokens) in line
                    for line in lines), lines)


class CrossThresholdNumericsTest(MoEWeightsMixin, jtu.JaxTestCase):
    """What the two kernels compute for the same batch. These run the
    kernels, so they need a TPU mesh and no substituted hardware answers."""

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        devices = jax.devices()[:8]
        if len(devices) < 8:
            self.skipTest("Expect eight devices")
        from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                          ShardingAxisNameBase)
        ShardingAxisName.override(ATTN_DATA=ShardingAxisNameBase.ATTN_DATA)
        self.addCleanup(ShardingAxisName.reset)
        self.mesh = serving_mesh(devices=devices)

    def test_the_two_kernels_agree_within_the_wire_quantization(self):
        """The same call routed each way, by moving the threshold with the
        switch left on, so the threshold is the only difference."""
        weights = self.build_weights(jax.random.key(2))
        x, gating = self.build_activations(jax.random.key(3), AT_THRESHOLD)
        call = dict(layer=FakeMoELayer(top_k=TOP_K),
                    x=x,
                    gating_output=gating,
                    weights=weights,
                    moe_backend=MoEBackend.GMM_EP,
                    mesh=self.mesh,
                    extra_backend_kwargs={"scatter_results": True})
        with mock.patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", True):
            with mock.patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                                   AT_THRESHOLD):
                fused = moe_apply(**call)
            with mock.patch.object(envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
                                   AT_THRESHOLD + 1):
                general = moe_apply(**call)

        fused = np.asarray(fused, np.float32)
        general = np.asarray(general, np.float32)
        delta = float(
            np.linalg.norm(fused - general) / np.linalg.norm(general))
        self.assertGreater(delta, WIRE_DELTA_FLOOR)
        self.assertLess(delta, WIRE_DELTA_CEILING)

    def test_the_switch_off_gets_exactly_the_general_path(self):
        """With the switch off the call runs the general path itself, bit for
        bit, not an imitation of it -- for a layer the kernel could not have
        taken anyway."""
        weights = self.build_weights(jax.random.key(2), with_bias=True)
        x, gating = self.build_activations(jax.random.key(3), ABOVE_THRESHOLD)
        layer = refused_layer()

        through_moe_apply = moe_apply(
            layer=layer,
            x=x,
            gating_output=gating,
            weights=weights,
            moe_backend=MoEBackend.GMM_EP,
            mesh=self.mesh,
            extra_backend_kwargs={"scatter_results": True},
        )
        direct = moe_module.fused_moe_func(
            hidden_states=x,
            w1=weights.w13_weight,
            w2=weights.w2_weight,
            w1_scale=weights.w13_weight_scale,
            w2_scale=weights.w2_weight_scale,
            w1_bias=weights.w13_bias,
            w2_bias=weights.w2_bias,
            gating_output=gating,
            topk=layer.top_k,
            renormalize=layer.renormalize,
            mesh=self.mesh,
            use_ep=layer.use_ep,
            activation="silu",
            scoring_fn=layer.scoring_func,
            all_gather_fp8=False,
            enable_rs_kernel=envs.ENABLE_RS_KERNEL,
            onehot_moe_permute_threshold=envs.ONEHOT_MOE_PERMUTE_THRESHOLD,
            scatter_results=True,
            defer_all_reduce=False,
            hash_based_topk_indices=None,
            expert_score_correction_bias=None,
            moe_chunk_size=0,
            num_valid_tokens=None,
        )
        self.assertArraysEqual(through_moe_apply, direct)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
