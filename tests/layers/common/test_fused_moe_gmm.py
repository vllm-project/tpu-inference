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
"""Kernel selection in the MoE GMM layer, decided from shapes and dtypes
at trace time: these tests build programs and read which kernel was
reached for, none runs one.
"""

import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from tests.kernels.gmm_fused_test import (SERVED_HIDDEN, SERVED_INTER,
                                          SERVED_LOCAL_EXPERTS, pinned_tpu)
from tpu_inference.layers.common import fused_moe_gmm as fmg

jax.config.parse_flags_with_absl()

# One expert-parallel shard of the served MoE layer: 8-way expert
# parallelism over 128 experts, hidden 4096, intermediate 1024.
EP_WIDTH = 8
TOPK = 8


class _KernelReached(Exception):
    """Raised by the stubs below once a kernel call has been recorded."""


def record_calls(module, name, calls):
    """Patch a kernel out of `module`, recording the call and stopping there."""

    def stub(*args, **kwargs):
        calls.append((name, args, kwargs))
        raise _KernelReached(name)

    return mock.patch.object(module, name, stub)


def layer_operands(num_experts=SERVED_LOCAL_EXPERTS,
                   num_rows=1024,
                   hidden=SERVED_HIDDEN,
                   inter=SERVED_INTER,
                   weight_dtype=jnp.float8_e4m3fn,
                   quant_block=None,
                   w2_inter=None,
                   with_scales=True,
                   with_biases=False):
    """The operands moe_gmm_local sees inside one shard."""
    w1_blocks = 1 if quant_block is None else hidden // quant_block
    w2_blocks = 1 if quant_block is None else inter // quant_block
    sds = jax.ShapeDtypeStruct
    return dict(
        x=sds((num_rows, hidden), jnp.bfloat16),
        w1=sds((num_experts, hidden, 2 * inter), weight_dtype),
        w1_scale=(sds((num_experts, w1_blocks, 1,
                       2 * inter), jnp.float32) if with_scales else None),
        w1_bias=(sds((num_experts, 1,
                      2 * inter), jnp.bfloat16) if with_biases else None),
        w2=sds((num_experts, inter if w2_inter is None else w2_inter, hidden),
               weight_dtype),
        w2_scale=(sds((num_experts, w2_blocks, 1,
                       hidden), jnp.float32) if with_scales else None),
        w2_bias=(sds(
            (num_experts, 1, hidden), jnp.bfloat16) if with_biases else None),
        group_sizes=sds((num_experts, ), jnp.int32),
        # Indexed by the layer (group_offset[0]), so it is a real array.
        group_offset=jnp.array([0], dtype=jnp.int32),
        topk_argsort_revert_indices=sds((num_rows, ), jnp.int32),
        topk_weights=sds((num_rows // TOPK, TOPK), jnp.bfloat16),
    )


class KernelSelectionTest(jtu.JaxTestCase):
    """Which of the two FFN programs the layer builds, and with what."""

    def select(self, operands, fused_flag=True):
        """Return the kernel name the layer reached for, and its kwargs."""
        calls = []
        with pinned_tpu(), record_calls(fmg, "gmm_fused", calls), \
                record_calls(fmg, "gmm_v2", calls), \
                mock.patch.object(fmg.envs, "USE_MOE_FUSED_GMM_KERNEL",
                                  fused_flag, create=True):
            with self.assertRaises(_KernelReached):
                fmg.moe_gmm_local(activation="silu",
                                  topk=TOPK,
                                  parallelism="ep",
                                  **operands)
        name, _, kwargs = calls[0]
        return name, kwargs

    def test_the_flag_off_takes_the_grouped_matmul_pair(self):
        name, _ = self.select(layer_operands(), fused_flag=False)
        self.assertEqual(name, "gmm_v2")

    @parameterized.named_parameters(
        ("fp8", None, jnp.float8_e4m3fn),
        ("fp4_block_512", 512, jnp.float4_e2m1fn),
    )
    def test_served_weights_take_the_fused_kernel(self, quant_block,
                                                  weight_dtype):
        name, _ = self.select(
            layer_operands(quant_block=quant_block, weight_dtype=weight_dtype))
        self.assertEqual(name, "gmm_fused")

    def test_the_fused_call_keeps_the_arguments_the_layer_depends_on(self):
        _, kwargs = self.select(layer_operands())
        self.assertIs(kwargs["zero_initialize"], True)
        self.assertIs(kwargs["unconditional_pipeline"], False)
        self.assertEqual(kwargs["bucket_base"], fmg.GMM_FUSED_BUCKET_BASE)
        self.assertEqual(kwargs["fuse_act"], "silu")

    def test_biased_experts_take_the_fused_kernel(self):
        name, kwargs = self.select(layer_operands(with_biases=True))
        self.assertEqual(name, "gmm_fused")
        self.assertIsNotNone(kwargs["w1_bias"])
        self.assertIsNotNone(kwargs["w2_bias"])

    def test_the_down_bias_is_held_by_one_tensor_parallel_shard(self):
        """Under TP the shards' partial results are summed, so a bias each
        shard held would land once per shard; under EP it is already split
        on the expert axis and must be left alone."""
        bias = jax.ShapeDtypeStruct((4, 1, 8), jnp.float32)
        self.assertIs(fmg.down_bias_added_once(bias, "ep"), bias)
        self.assertIsNone(fmg.down_bias_added_once(None, "tp"))
        # On a one-shard mesh the only shard is shard 0, which keeps it.
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model"))
        ones = jnp.ones((4, 1, 8), jnp.float32)
        kept = jax.shard_map(lambda b: fmg.down_bias_added_once(b, "tp"),
                             mesh=mesh,
                             in_specs=(P(None, None, "model"), ),
                             out_specs=P(None, None, "model"),
                             check_vma=False)(ones)
        self.assertArraysEqual(kept, ones)

    def test_unquantized_experts_run_the_pair(self):
        name, _ = self.select(layer_operands(with_scales=False))
        self.assertEqual(name, "gmm_v2")

    def test_padded_intermediate_runs_the_pair(self):
        """The pair trims the padded intermediate; the fused kernel cannot."""
        name, _ = self.select(layer_operands(w2_inter=SERVED_INTER - 128))
        self.assertEqual(name, "gmm_v2")

    @parameterized.named_parameters(
        ("deepseek_v3_ep8", 32, 7168, 2048),
        ("llama4_tp4", 16, 5120, 2048),
        ("mixtral_tp8", 8, 4096, 1792),
    )
    def test_wide_deployments_run_the_pair(self, num_experts, hidden, inter):
        """Models whose expert pair does not fit VMEM keep working."""
        name, _ = self.select(
            layer_operands(num_experts=num_experts, hidden=hidden,
                           inter=inter))
        self.assertEqual(name, "gmm_v2")

    def test_zero_tokens_run_the_pair(self):
        name, _ = self.select(layer_operands(num_rows=0))
        self.assertEqual(name, "gmm_v2")

    def test_a_single_expert_per_shard_takes_the_fused_kernel(self):
        name, _ = self.select(layer_operands(num_experts=1, num_rows=128))
        self.assertEqual(name, "gmm_fused")

    def test_the_bucket_base_is_whole_sublanes_of_the_fp8_rows(self):
        with pinned_tpu():
            sublane = pltpu.get_tpu_info().get_sublane_tiling(
                jnp.float8_e4m3fn)
        self.assertEqual(fmg.GMM_FUSED_BUCKET_BASE % sublane, 0)


class ExpertParallelLayerTest(jtu.JaxTestCase):
    """The whole expert-parallel MoE layer, built but not run."""

    def build(self, num_experts=128, num_tokens=512, **kwargs):
        """Trace expert_parallel_gmm at one served batch shape."""
        sds = jax.ShapeDtypeStruct
        rows = num_tokens * TOPK
        operands = (
            sds((rows, SERVED_HIDDEN), jnp.bfloat16),
            sds((num_experts, SERVED_HIDDEN, 2 * SERVED_INTER),
                jnp.float8_e4m3fn),
            sds((num_experts, 1, 1, 2 * SERVED_INTER), jnp.float32),
            None,
            sds((num_experts, SERVED_INTER, SERVED_HIDDEN), jnp.float8_e4m3fn),
            sds((num_experts, 1, 1, SERVED_HIDDEN), jnp.float32),
            None,
            sds((num_experts, ), jnp.int32),
            sds((rows, ), jnp.int32),
            sds((num_tokens, TOPK), jnp.bfloat16),
        )
        with pinned_tpu(axis_sizes=(1, EP_WIDTH),
                        axis_names=("data", "model")) as mesh, \
                mock.patch.object(fmg.envs, "USE_MOE_FUSED_GMM_KERNEL",
                                  True, create=True):
            return jax.eval_shape(
                lambda *args: fmg.expert_parallel_gmm(
                    *args, activation="silu", topk=TOPK, mesh=mesh, **kwargs),
                *operands)

    def test_the_served_layer_builds_on_the_fused_kernel(self):
        out = self.build(num_tokens=512)
        self.assertEqual(out.shape, (512, SERVED_HIDDEN))

    def test_the_shards_reach_the_fused_kernel(self):
        calls = []
        with record_calls(fmg, "gmm_fused", calls):
            with self.assertRaises(_KernelReached):
                self.build()
        self.assertEqual(calls[0][0], "gmm_fused")

    def test_an_expert_count_off_the_ep_width_fails_loudly(self):
        with self.assertRaisesRegex(
                ValueError, "not divisible by the expert-parallel width"):
            self.build(num_experts=EP_WIDTH - 1)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
