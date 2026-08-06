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
"""The rhs scale block sizes gmm_v2 accepts.

The quantized-lhs matmul walks the contracting dimension in lhs
quantization blocks but reads the rhs scale by rhs block, so an rhs block
that is not a whole number of lhs blocks has scales that are never read.
The unquantized matmul beside it walks in rhs blocks and reads every
scale. Which of the two a call takes depends on the chip, so these tests
ask both generations. They build programs; none runs one.
"""

import contextlib

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import mesh as mesh_lib
from jax._src import test_util as jtu
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.megablox.gmm_v2 import LHS_QUANT_BLOCK_SIZE, gmm_v2

jax.config.parse_flags_with_absl()

# A generation that quantizes the lhs against an fp8 rhs and one that does
# not, so both sides of the split are covered without a device.
QUANTIZING_CHIP = "TPU7x"
NON_QUANTIZING_CHIP = "TPU v5e"

NUM_GROUPS = 4
SIZE_M = 512
SIZE_K = 3072
SIZE_N = 512

# Blocks the lhs block does not divide, and blocks it does. Both lists are
# at or above the quantizing chip's MXU column, so the scale is applied
# after the matmul in every case.
MISALIGNED_BLOCKS = (256, 768)
ALIGNED_BLOCKS = (512, 1536)


@contextlib.contextmanager
def pinned_tpu(device_kind):
    """Answer shape questions for one generation, off TPU included."""
    device = jax.sharding.AbstractDevice(device_kind, 1, "tpu")
    mesh = jax.sharding.AbstractMesh((1, ), ("d", ), abstract_device=device)
    with mesh_lib.use_abstract_mesh(mesh):
        yield mesh


def build(device_kind, rhs_block, quantize_lhs=True):
    """Trace gmm_v2 with an rhs scale carrying the given block size."""
    sds = jax.ShapeDtypeStruct
    operands = dict(
        lhs=sds((SIZE_M, SIZE_K), jnp.bfloat16),
        rhs=sds((NUM_GROUPS, SIZE_K, SIZE_N), jnp.float8_e4m3fn),
        group_sizes=sds((NUM_GROUPS, ), jnp.int32),
        rhs_scale=sds((NUM_GROUPS, SIZE_K // rhs_block, 1, SIZE_N),
                      jnp.float32),
    )
    with pinned_tpu(device_kind):
        return jax.eval_shape(
            lambda **kw: gmm_v2(**kw, maybe_quantize_lhs=quantize_lhs),
            **operands)


class ScaleBlockGuardTest(jtu.JaxTestCase):
    """Which rhs scale block sizes the quantized k loop can read."""

    @parameterized.parameters(*MISALIGNED_BLOCKS)
    def test_a_block_the_quantized_k_loop_would_skip_is_refused(
            self, rhs_block):
        with self.assertRaisesRegex(ValueError,
                                    "would otherwise skip scale blocks"):
            build(QUANTIZING_CHIP, rhs_block)

    @parameterized.parameters(*ALIGNED_BLOCKS)
    def test_a_block_the_quantized_k_loop_reads_whole_is_accepted(
            self, rhs_block):
        out = build(QUANTIZING_CHIP, rhs_block)
        self.assertEqual(out.shape, (SIZE_M, SIZE_N))

    def test_one_block_over_the_whole_contraction_is_accepted(self):
        """A single scale is read once per step whatever the step is."""
        out = build(QUANTIZING_CHIP, SIZE_K)
        self.assertEqual(out.shape, (SIZE_M, SIZE_N))

    @parameterized.parameters(*MISALIGNED_BLOCKS)
    def test_the_unquantized_matmul_takes_the_same_block(self, rhs_block):
        """It walks in rhs blocks and reads every scale, so the block the
        quantized path is refused is one this path computes correctly."""
        out = build(QUANTIZING_CHIP, rhs_block, quantize_lhs=False)
        self.assertEqual(out.shape, (SIZE_M, SIZE_N))

    @parameterized.parameters(*MISALIGNED_BLOCKS)
    def test_a_chip_that_does_not_quantize_the_lhs_takes_the_same_block(
            self, rhs_block):
        """Nothing on this generation reaches the quantized k loop."""
        out = build(NON_QUANTIZING_CHIP, rhs_block)
        self.assertEqual(out.shape, (SIZE_M, SIZE_N))

    def test_a_block_narrower_than_the_mxu_column_is_dequantized_first(self):
        """Narrow blocks never reach either k loop as scales."""
        with pinned_tpu(QUANTIZING_CHIP):
            mxu = pltpu.get_tpu_info().mxu_column_size
        self.assertLess(128, mxu)
        out = build(QUANTIZING_CHIP, 128)
        self.assertEqual(out.shape, (SIZE_M, SIZE_N))

    def test_the_refusal_names_both_block_sizes(self):
        with self.assertRaisesRegex(
                ValueError, f"{MISALIGNED_BLOCKS[0]}.*{LHS_QUANT_BLOCK_SIZE}"):
            build(QUANTIZING_CHIP, MISALIGNED_BLOCKS[0])

    def test_the_refusal_names_the_way_out(self):
        with self.assertRaisesRegex(ValueError, "maybe_quantize_lhs=False"):
            build(QUANTIZING_CHIP, MISALIGNED_BLOCKS[0])


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
