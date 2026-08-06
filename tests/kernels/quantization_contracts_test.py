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
"""Contracts between the quantized weights on disk and the kernels reading
them: four-bit packing order, weight-scale block layouts, and bfloat16
accumulator drift. Shape and bit arithmetic runs on CPU with the served
generation's hardware answers; cases needing a real matmul skip off TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.fused_moe.v2 import PACK4
from tpu_inference.kernels.megablox.gmm_fused import gmm_fused
from tpu_inference.kernels.megablox.gmm_v2_fused_support import \
    LHS_QUANT_BLOCK_SIZE
from tpu_inference.layers.common.quantization import quantize_tensor

jax.config.parse_flags_with_absl()

# e2m1 code i is the value at index i; the top eight are the bottom eight
# negated.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
               -2.0, -3.0, -4.0, -6.0)


def pack_fp4_to_u32(values: jax.Array) -> jax.Array:
    """Pack fp4 [K, N] into uint32 [K // PACK4, N]; offset j along K
    lands in bits [4j, 4j+3], the order the kernel's bitcast undoes."""
    num_k, num_n = values.shape
    if num_k % PACK4:
        raise ValueError(f"{num_k} rows is not a whole number of {PACK4}")
    # [K, N] -> [K / 8, 8, N] -> [K / 8, N, 8], so the eight values that
    # share a word are the minor axis the bitcast folds into one uint32.
    grouped = values.reshape(num_k // PACK4, PACK4, num_n)
    grouped = jnp.swapaxes(grouped, 1, 2)
    return jax.lax.bitcast_convert_type(grouped, jnp.uint32)


def kernel_unpack_fp4(packed: jax.Array, num_k: int) -> jax.Array:
    """The kernel's widening bitcast, uint32 [K / 8, N] to fp4 [K, N]."""

    def body(packed_ref, out_ref):
        out_ref[...] = pltpu.bitcast(packed_ref[...], jnp.float4_e2m1fn)

    out_shape = jax.ShapeDtypeStruct((num_k, packed.shape[1]),
                                     jnp.float4_e2m1fn)
    return pl.pallas_call(body, out_shape=out_shape, interpret=True)(packed)


def kernel_unpack_fp4_on_device(values: jax.Array, num_k: int) -> jax.Array:
    """The same widening bitcast on the backend, off the kernel's own HBM
    ref view, so the order under test is the hardware's not the packer's."""
    num_n = values.shape[1]

    def body(weight_hbm, out_ref, packed_vm, sem):
        packed_hbm = weight_hbm.bitcast(jnp.uint32)
        copy = pltpu.make_async_copy(packed_hbm, packed_vm, sem)
        copy.start()
        copy.wait()
        out_ref[...] = pltpu.bitcast(packed_vm[...],
                                     jnp.float4_e2m1fn).astype(jnp.float32)

    return pl.pallas_call(
        body,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
        out_shape=jax.ShapeDtypeStruct((num_k, num_n), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((num_k // PACK4, num_n), jnp.uint32),
            pltpu.SemaphoreType.DMA,
        ],
    )(values)


def fp4_from_codes(codes: np.ndarray) -> jax.Array:
    table = np.asarray(E2M1_VALUES, dtype=np.float32)
    return jnp.asarray(table[codes]).astype(jnp.float4_e2m1fn)


class Fp4PackingContractTest(jtu.JaxTestCase):
    """The packing order the checkpoint and the kernel have to agree on."""

    def test_code_table_is_the_one_the_dtype_implements(self):
        codes = np.arange(16, dtype=np.int32).reshape(2, 8)
        values = fp4_from_codes(codes)
        packed = pack_fp4_to_u32(jnp.swapaxes(values, 0, 1))
        expected = np.array(
            [sum(c << (4 * j) for j, c in enumerate(row)) for row in codes],
            dtype=np.uint32)
        self.assertArraysEqual(np.asarray(packed).reshape(-1), expected)

    def test_packer_and_kernel_unpack_round_trip_known_values(self):
        num_k, num_n = 32, 128
        codes = np.random.default_rng(0).integers(0, 16, size=(num_k, num_n))
        values = fp4_from_codes(codes)

        packed = pack_fp4_to_u32(values)
        self.assertEqual(packed.shape, (num_k // PACK4, num_n))
        self.assertEqual(packed.dtype, jnp.uint32)

        recovered = kernel_unpack_fp4(packed, num_k)
        self.assertArraysEqual(recovered.astype(jnp.float32),
                               values.astype(jnp.float32))

    def test_a_disagreeing_order_does_not_round_trip(self):
        num_k, num_n = 32, 128
        codes = np.random.default_rng(1).integers(1, 16, size=(num_k, num_n))
        values = fp4_from_codes(codes)

        reversed_groups = values.reshape(num_k // PACK4, PACK4,
                                         num_n)[:, ::-1, :]
        mispacked = jax.lax.bitcast_convert_type(
            jnp.swapaxes(reversed_groups, 1, 2), jnp.uint32)

        recovered = kernel_unpack_fp4(mispacked, num_k)
        self.assertFalse(
            np.array_equal(np.asarray(recovered.astype(jnp.float32)),
                           np.asarray(values.astype(jnp.float32))))

    def test_the_backend_reads_the_same_nibble_order(self):
        """The nibble order the hardware reads, not the test packer's."""
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")

        num_k, num_n = 32, 128
        codes = np.random.default_rng(6).integers(0, 16, size=(num_k, num_n))
        values = fp4_from_codes(codes)

        recovered = kernel_unpack_fp4_on_device(values, num_k)
        self.assertArraysEqual(recovered, values.astype(jnp.float32))

    def test_quantizer_output_survives_the_round_trip(self):
        hidden, size_n, block = 64, 128, 32
        weight = jax.random.normal(jax.random.key(2),
                                   (hidden, size_n), jnp.float32) / 4
        quantized, scale = quantize_tensor(jnp.float4_e2m1fn, weight, 0, block)
        self.assertEqual(quantized.dtype, jnp.float4_e2m1fn)
        self.assertEqual(scale.shape, (hidden // block, size_n))

        recovered = kernel_unpack_fp4(pack_fp4_to_u32(quantized), hidden)
        self.assertArraysEqual(recovered.astype(jnp.float32),
                               quantized.astype(jnp.float32))

        dequantized = (recovered.astype(jnp.float32).reshape(
            hidden // block, block, size_n) * scale[:, None, :]).reshape(
                hidden, size_n)
        reference = (quantized.astype(jnp.float32).reshape(
            hidden // block, block, size_n) * scale[:, None, :]).reshape(
                hidden, size_n)
        self.assertArraysEqual(dequantized, reference)


def postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale, acc_dtype):
    """One quantized matmul the way the fused kernel's k loop runs it: steps
    by LHS_QUANT_BLOCK_SIZE, reads rhs scale block start_k // rhs_block."""
    size_m, size_k = lhs_q.shape
    size_n = rhs_q.shape[1]
    num_rhs_blocks = rhs_scale.shape[0]
    rhs_block = size_k // num_rhs_blocks

    def body(lhs_ref, lhs_scale_ref, rhs_ref, rhs_scale_ref, out_ref):
        lhs_f32 = lhs_ref[...].astype(jnp.float32)
        rhs_f32 = rhs_ref[...].astype(jnp.float32)
        acc = jnp.zeros((size_m, size_n), acc_dtype)
        for start_k in range(0, size_k, LHS_QUANT_BLOCK_SIZE):
            window = slice(start_k, start_k + LHS_QUANT_BLOCK_SIZE)
            product = jax.lax.dot_general(lhs_f32[:, window],
                                          rhs_f32[window, :],
                                          (((1, ), (0, )), ((), ())),
                                          preferred_element_type=jnp.float32)
            lhs_block_id = start_k // LHS_QUANT_BLOCK_SIZE
            product = product * lhs_scale_ref[:, lhs_block_id][:, None]
            product = product * rhs_scale_ref[start_k // rhs_block][None, :]
            acc = (acc.astype(jnp.float32) + product).astype(acc_dtype)
        out_ref[...] = acc

    out_shape = jax.ShapeDtypeStruct((size_m, size_n), acc_dtype)
    return pl.pallas_call(body, out_shape=out_shape,
                          interpret=True)(lhs_q, lhs_scale, rhs_q, rhs_scale)


def quantize_rows(x, num_blocks):
    """Per-row, per-block fp8 quantization, the lhs form the kernel uses."""
    size_m, size_k = x.shape
    blocked = x.astype(jnp.float32).reshape(size_m, num_blocks,
                                            size_k // num_blocks)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.max(jnp.abs(blocked), axis=-1, keepdims=True)
    scale = amax / fp8_max
    inverse = jnp.where(scale == 0.0, 0.0,
                        1.0 / jnp.where(scale == 0.0, 1.0, scale))
    quantized = jnp.clip(blocked * inverse, -fp8_max,
                         fp8_max).astype(jnp.float8_e4m3fn)
    return (quantized.reshape(size_m, size_k),
            scale.reshape(size_m, num_blocks).astype(jnp.float32))


def quantize_columns(w, num_blocks):
    """Per-column fp8 quantization over num_blocks contraction blocks."""
    size_k, size_n = w.shape
    blocked = w.astype(jnp.float32).reshape(num_blocks, size_k // num_blocks,
                                            size_n)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.max(jnp.abs(blocked), axis=1, keepdims=True)
    scale = amax / fp8_max
    inverse = jnp.where(scale == 0.0, 0.0,
                        1.0 / jnp.where(scale == 0.0, 1.0, scale))
    quantized = jnp.clip(blocked * inverse, -fp8_max,
                         fp8_max).astype(jnp.float8_e4m3fn)
    return (quantized.reshape(size_k, size_n),
            scale.reshape(num_blocks, size_n).astype(jnp.float32))


class ScaleLayoutContractTest(jtu.JaxTestCase):
    """That both rhs scale layouts reconstruct the unquantized product."""

    @parameterized.named_parameters(("channelwise", 1), ("block_512", 4))
    def test_postscale_dequantization_matches_a_full_precision_reference(
            self, rhs_blocks):
        """Both scale layouts reconstruct the unquantized product."""
        size_m, size_k, size_n = 8, 2048, 128
        lhs_key, rhs_key = jax.random.split(jax.random.key(3))
        lhs = jax.random.normal(lhs_key, (size_m, size_k), jnp.float32) / 8
        rhs = jax.random.normal(rhs_key, (size_k, size_n), jnp.float32) / 8

        lhs_q, lhs_scale = quantize_rows(lhs, size_k // LHS_QUANT_BLOCK_SIZE)
        rhs_q, rhs_scale = quantize_columns(rhs, rhs_blocks)

        actual = postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale,
                                  jnp.float32)

        lhs_deq = (lhs_q.astype(jnp.float32).reshape(
            size_m, size_k // LHS_QUANT_BLOCK_SIZE, LHS_QUANT_BLOCK_SIZE) *
                   lhs_scale[:, :, None]).reshape(size_m, size_k)
        rhs_deq = (rhs_q.astype(jnp.float32).reshape(
            rhs_blocks, size_k // rhs_blocks, size_n) *
                   rhs_scale[:, None, :]).reshape(size_k, size_n)
        reference = lhs_deq @ rhs_deq

        self.assertAllClose(actual, reference, atol=2e-3, rtol=2e-3)


class AccumulationDivergenceTest(jtu.JaxTestCase):
    """How far the quantized path's bfloat16 accumulator drifts."""

    # The emulation lands between 0.00383 and 0.00390 relative Frobenius norm
    # at hidden 4096 across seeds; this bracket is set just outside that.
    MIN_DIVERGENCE = 3e-3
    MAX_DIVERGENCE = 5e-3

    # The kernel's own drift on the device is a different measurement and
    # is not characterized across seeds, so it keeps a wide bracket.
    MIN_DEVICE_DIVERGENCE = 1e-4
    MAX_DEVICE_DIVERGENCE = 2e-2

    def divergence(self, actual, reference):
        actual = np.asarray(actual, np.float32)
        reference = np.asarray(reference, np.float32)
        return float(
            np.linalg.norm(actual - reference) / np.linalg.norm(reference))

    def test_bfloat16_accumulation_diverges_and_stays_bounded(self):
        """The accumulator dtype, isolated from everything else."""
        size_m, size_k, size_n = 64, 4096, 256
        lhs_key, rhs_key = jax.random.split(jax.random.key(4))
        lhs = jax.random.normal(lhs_key, (size_m, size_k), jnp.float32) / 10
        rhs = jax.random.normal(rhs_key, (size_k, size_n), jnp.float32) / 10

        lhs_q, lhs_scale = quantize_rows(lhs, size_k // LHS_QUANT_BLOCK_SIZE)
        rhs_q, rhs_scale = quantize_columns(rhs, 1)

        in_bf16 = postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale,
                                   jnp.bfloat16)
        in_f32 = postscale_matmul(lhs_q, lhs_scale, rhs_q, rhs_scale,
                                  jnp.float32)

        drift = self.divergence(in_bf16.astype(jnp.float32), in_f32)
        self.assertGreater(drift, self.MIN_DIVERGENCE)
        self.assertLess(drift, self.MAX_DIVERGENCE)

    def test_fused_kernel_carries_the_same_divergence(self):
        """The same measurement on the kernel itself, which needs a TPU."""
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")

        groups, rows, hidden, inter = 2, 64, 4096, 512
        lhs_key, w1_key, w2_key = jax.random.split(jax.random.key(5), 3)
        lhs = (jax.random.normal(lhs_key, (rows, hidden), jnp.float32) /
               10).astype(jnp.bfloat16)
        w1 = jax.random.normal(w1_key,
                               (groups, hidden, 2 * inter), jnp.float32) / 10
        w2 = jax.random.normal(w2_key,
                               (groups, inter, hidden), jnp.float32) / 10

        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)

        def channelwise(w):
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = amax / fp8_max
            quantized = jnp.clip(w / jnp.where(scale == 0, 1, scale), -fp8_max,
                                 fp8_max).astype(jnp.float8_e4m3fn)
            return quantized, scale

        w1_q, w1_scale = channelwise(w1)
        w2_q, w2_scale = channelwise(w2)
        w1_scale = w1_scale.reshape(groups, 1, 1, 2 * inter)
        w2_scale = w2_scale.reshape(groups, 1, 1, hidden)
        group_sizes = jnp.array([rows // 2, rows - rows // 2], jnp.int32)

        served = gmm_fused(lhs, w1_q, w2_q, group_sizes, w1_scale, w2_scale)
        in_f32 = gmm_fused(lhs,
                           w1_q,
                           w2_q,
                           group_sizes,
                           w1_scale,
                           w2_scale,
                           acc_dtype=jnp.float32)

        drift = self.divergence(served.astype(jnp.float32),
                                in_f32.astype(jnp.float32))
        self.assertGreater(drift, self.MIN_DEVICE_DIVERGENCE)
        self.assertLess(drift, self.MAX_DEVICE_DIVERGENCE)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
