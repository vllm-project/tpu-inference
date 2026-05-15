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

import timeit
from functools import partial
from unittest.mock import patch

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.mla.v2.transpose import (prev_closest_valid_divisor,
                                                    xpose_full, xpose_pipeline)


def benchmark_op(name, op_func, input_data, number=10):
    """Utility to benchmark a JAX operation and print results."""
    # Warmup
    res = op_func(input_data)
    if isinstance(res, (list, tuple)):
        res[0].block_until_ready()
    else:
        res.block_until_ready()

    def sync_op():
        out = op_func(input_data)
        if isinstance(out, (list, tuple)):
            out[0].block_until_ready()
        else:
            out.block_until_ready()

    t = timeit.timeit(sync_op, number=number)
    avg_time = t / number
    print(
        f"\n{name} (shape {input_data.shape}): Mean execution time: {avg_time:.6f}s"
    )
    return res


def xpose_full_wrapper(x, transpose_axes):
    """Helper to extract the first output from xpose_full."""
    return xpose_full(x, transpose_axes=transpose_axes)[0]


class TransposeTest(parameterized.TestCase):

    @parameterized.parameters(
        dict(shape=(1024, 1024), transpose_axes=(1, 0)),
        dict(shape=(32, 64, 128), transpose_axes=(2, 0, 1)),
        dict(shape=(8, 16, 32, 64), transpose_axes=(3, 2, 1, 0)),
        dict(shape=(128, 256), transpose_axes=(1, 0)),
        dict(shape=(16, 32, 64), transpose_axes=(2, 0, 1)),
    )
    def test_xpose_full(self, shape, transpose_axes):
        key = jax.random.PRNGKey(42)
        input_data = jax.random.normal(key, shape, dtype=jnp.float8_e4m3fn)

        name = f"xpose_full_{len(shape)}d"
        result = benchmark_op(
            name, lambda x: xpose_full(x, transpose_axes=transpose_axes),
            input_data)

        expected = jnp.transpose(input_data, transpose_axes)

        # Validation
        self.assertEqual(result[0].shape, expected.shape)
        self.assertTrue(jnp.allclose(result[0], expected))

    @parameterized.parameters(
        dict(shape=(1024, 2048), transpose_axes=(1, 0), n_tile=128,
             m_tile=128),
        dict(shape=(2048, 1024), transpose_axes=(1, 0), n_tile=256,
             m_tile=256),
        dict(shape=(512, 1024, 16),
             transpose_axes=(1, 0, 2),
             n_tile=64,
             m_tile=128),
        dict(shape=(256, 512, 128),
             transpose_axes=(1, 0, 2),
             n_tile=128,
             m_tile=128,
             parallel_axis=1,
             pipeline_axis=0),
        # ~64MB (would OOM loading everything into VMEM)
        dict(shape=(128, 2048, 256),
             transpose_axes=(1, 0, 2),
             n_tile=128,
             m_tile=128,
             parallel_axis=1,
             pipeline_axis=0),
        # index maps that put grid-dim-0 (parallel, axis 2) into output axis 1 and
        # grid-dim-1 (pipeline, axis 1) into output axis 2.
        dict(shape=(4, 256, 512, 128),
             transpose_axes=(0, 2, 1, 3),
             n_tile=64,
             m_tile=64,
             parallel_axis=2,
             pipeline_axis=1),
        dict(shape=(192, 256, 64),
             transpose_axes=(1, 0, 2),
             n_tile=160,
             m_tile=64),
        dict(shape=(320, 256, 64),
             transpose_axes=(1, 0, 2),
             n_tile=160,
             m_tile=64),
    )
    def test_xpose_pipeline(self,
                            shape,
                            transpose_axes,
                            n_tile,
                            m_tile,
                            parallel_axis=0,
                            pipeline_axis=1):
        key = jax.random.PRNGKey(42)
        input_data = jax.random.normal(key, shape, dtype=jnp.float8_e4m3fn)

        name = f"xpose_pipeline_{len(shape)}d"
        result = benchmark_op(
            name, lambda x: xpose_pipeline(x,
                                           transpose_axes=transpose_axes,
                                           n_tile=n_tile,
                                           m_tile=m_tile,
                                           parallel_axis=parallel_axis,
                                           pipeline_axis=pipeline_axis),
            input_data)

        expected = jnp.transpose(input_data, transpose_axes)

        self.assertEqual(result[0].shape, expected.shape)
        self.assertTrue(jnp.allclose(result[0], expected, atol=1e-5))

    def test_xpose_sharded_mla(self):
        # Mimic q_nope scenario from flash_attn_mla.py
        # q_nope shape (N, B, L) where N=Heads, B=Batch, L=LoraRank
        num_devices = len(jax.devices())
        N, B, L = 16, 128 * num_devices, 512
        shape = (N, B, L)
        key = jax.random.PRNGKey(42)
        input_data = jax.random.normal(key, shape, dtype=jnp.float8_e4m3fn)

        mesh = Mesh(jax.devices(), ('model', ))
        transpose_axes = (1, 0, 2)

        sharded_xpose_fn = shard_map(partial(xpose_full_wrapper,
                                             transpose_axes=transpose_axes),
                                     mesh=mesh,
                                     in_specs=P(None, 'model', None),
                                     out_specs=P('model', None, None),
                                     check_vma=False)

        @jax.jit
        def run_sharded_xpose(x):
            return sharded_xpose_fn(x)

        result = benchmark_op("xpose_sharded_mla", run_sharded_xpose,
                              input_data)

        expected = jnp.transpose(input_data, transpose_axes)
        self.assertEqual(result.shape, expected.shape)
        self.assertTrue(jnp.allclose(result, expected))


class TestXposePipelineTiling(parameterized.TestCase):
    """Verifies that xpose_pipeline uses the expected tile sizes at runtime."""

    def test_non_divisible_n_tile(self):
        # shape[0]=448, n_tile=160: 448 % 160 != 0.
        # For fp8, n_tile needs to be divisible by:
        # sublane_multiple = get_dtype_packing(fp8) * 8 = 32.
        # prev_closest_valid_divisor(448, 160, multiple_of=32) = 64 (largest divisor
        # of 448 that is <= 160 and % 32 == 0
        shape = (448, 192, 128)
        input_data = jax.random.normal(jax.random.PRNGKey(0),
                                       shape,
                                       dtype=jnp.float8_e4m3fn)

        # We patch the module-level logger and force a JIT retrace so the
        # warning fires regardless of absltest's logging capture behavior.
        with patch('tpu_inference.kernels.mla.v2.transpose.logger'
                   ) as mock_logger:
            result = xpose_pipeline(input_data,
                                    transpose_axes=(1, 0, 2),
                                    n_tile=160,
                                    m_tile=64)[0]

        # Warning should name the requested tile (160) and the chosen tile (64).
        mock_logger.warning.assert_called()
        warning_text = mock_logger.warning.call_args[0][0]
        self.assertIn('160', warning_text)
        self.assertIn('64', warning_text)

        # Output should still be numerically correct.
        expected = jnp.transpose(input_data, (1, 0, 2))
        self.assertTrue(jnp.allclose(result, expected))

    def test_no_aligned_divisor_raises(self):
        # shape[0]=300: no divisor of 300 is both <= 160 and % 32 == 0
        # xpose_pipeline should raise ValueError rather than silently using a
        # non-divisor tile that would leave rows unprocessed or cause VMEM OOM.
        shape = (300, 192, 128)
        input_data = jax.random.normal(jax.random.PRNGKey(0),
                                       shape,
                                       dtype=jnp.float8_e4m3fn)
        with self.assertRaises(ValueError):
            xpose_pipeline(input_data,
                           transpose_axes=(1, 0, 2),
                           n_tile=160,
                           m_tile=64)

    def test_clamped_n_tile(self):
        # shape[0]=128, n_tile=160: 160 > 128, so n_tile is clamped to 128.
        # For fp8, sublane_multiple=32; prev_closest_valid_divisor(128, 128, multiple_of=32) = 128,
        # which evenly divides 128, so no warning should be emitted.
        # NOTE: need to keep shape distinct from other test cases to emit
        # logging at compilation time.
        shape = (128, 192, 128)
        input_data = jax.random.normal(jax.random.PRNGKey(0),
                                       shape,
                                       dtype=jnp.float8_e4m3fn)
        with self.assertNoLogs('tpu_inference.kernels.mla.v2.transpose',
                               level='WARNING'):
            result = xpose_pipeline(input_data,
                                    transpose_axes=(1, 0, 2),
                                    n_tile=160,
                                    m_tile=64)[0]

        expected = jnp.transpose(input_data, (1, 0, 2))
        self.assertTrue(jnp.allclose(result, expected))


class TestPrevClosestDivisor(parameterized.TestCase):

    @parameterized.parameters(
        dict(number=112, divider=160, expected=112),
        # Largest divisor of 300 <= 160 is 150.
        dict(number=300, divider=160, expected=150),
        # Exact match: divider equals the number itself.
        dict(number=160, divider=160, expected=160),
        # Common TPU tile sizes (powers of 2)
        dict(number=256, divider=160, expected=128),
        # Non-power-of-2: largets divisor of 160 less than 128 is 80.
        dict(number=160, divider=128, expected=80),
        # divider=1 always returns 1
        dict(number=128, divider=1, expected=1),
        # divider larger than number: returns the number itself.
        dict(number=64, divider=1000, expected=64),
    )
    def test_prev_closest_valid_divisor(self, number, divider, expected):
        self.assertEqual(prev_closest_valid_divisor(number, divider), expected)

    @parameterized.parameters(
        # Exact match: 128 is a divisor and % 8.
        dict(number=128, divider=160, multiple_of=8, expected=128),
        # largest divisor <=160 that is %8 is 128.
        dict(number=512, divider=160, multiple_of=8, expected=128),
        # largest divisor <=100 that is %8 is 40.
        dict(number=120, divider=100, multiple_of=8, expected=40),
        # largest divisor <=128 that is %8 is 128 itself.
        dict(number=256, divider=128, multiple_of=8, expected=128),
        # divider larger than number: 64 is a divisor of itself and %8.
        dict(number=64, divider=1000, multiple_of=8, expected=64),
        # multiple_of=1 behaves identically to no multiple_of argument.
        dict(number=300, divider=160, multiple_of=1, expected=150),
        # number < multiple_of and divider >= number: returns number.
        dict(number=4, divider=10, multiple_of=8, expected=4),
    )
    def test_prev_closest_valid_divisor_multiple_of(self, number, divider,
                                                    multiple_of, expected):
        self.assertEqual(
            prev_closest_valid_divisor(number,
                                       divider,
                                       multiple_of=multiple_of), expected)

    @parameterized.parameters(
        # divider=2 < number=4, both < multiple_of=8: no valid tile.
        dict(number=4, divider=2, multiple_of=8),
        # 300 = 2^2 * 3 * 5^2: no divisor of 300 is both <= 160 and % 8 == 0.
        dict(number=300, divider=160, multiple_of=8),
        # 150 = 2 * 3 * 5^2: no divisor of 150 is both <= 100 and % 8 == 0.
        dict(number=150, divider=100, multiple_of=8),
    )
    def test_prev_closest_valid_divisor_multiple_of_raises(
            self, number, divider, multiple_of):
        with self.assertRaises(ValueError):
            prev_closest_valid_divisor(number,
                                       divider,
                                       multiple_of=multiple_of)


if __name__ == "__main__":
    absltest.main()
