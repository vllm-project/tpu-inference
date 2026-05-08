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

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.custom_calls.kernel import (xpose_full,
                                                       xpose_pipeline)


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


class CustomCallsTest(parameterized.TestCase):

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
        # Grid iterates axis 1 (parallel, 512//128=4 steps) and axis 0 (pipeline, 256//128=2 steps).
        dict(shape=(256, 512, 128),
             transpose_axes=(1, 0, 2),
             n_tile=128,
             m_tile=128,
             parallel_axis=1,
             pipeline_axis=0),
        # 64MB (would OOM loading everything into VMEM)
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

        # Validation
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


if __name__ == "__main__":
    absltest.main()
