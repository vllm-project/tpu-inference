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

import functools
import os

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import shard_map
from jax.sharding import Mesh, NamedSharding

from tpu_inference.kernels.collectives import reduce_scatter_matmul

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

SpongeDir: str | None = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', None)

# NB(xw32): To run the test:
# pytest -s -v tests/kernels/collectives/reduce_scatter_matmul_kernel_test.py -k test_basic_ref_impl_reduce_scatter_matmul_on_bs
# pytest -s -v tests/kernels/collectives/reduce_scatter_matmul_kernel_test.py -k test_basic_reduce_scatter_matmul_kernel_on_bs


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class ReduceScatterMatmulTest(jtu.JaxTestCase):

  def test_basic_ref_impl_reduce_scatter_matmul_on_bs(self):
    dtype = jnp.float32
    # let's limit to 2 devices to make it simple.
    # TODO(xw32): should add a more general test that uses more devices.
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ('x', ))

    def S(p):
      return NamedSharding(mesh, p)

    # In tpu-inference SP,
    # lhs: [bs, in_features@TP], rhs: [out_features, in_features@TP]) ->
    # [bs@TP, out_features]
    def shard(x, spec):
      return jax.device_put(x, NamedSharding(mesh, spec))

    x_pspec = P(None, 'x')
    y_pspec = P(None, 'x')
    o_pspec = P('x', None)

    m = 768
    k = 256
    n = 512

    x = jax.random.uniform(jax.random.key(0), (m, k), dtype=dtype)
    x = shard(x, x_pspec)
    y = jax.random.uniform(jax.random.key(1), (n, k), dtype=dtype)
    y = shard(y, y_pspec)

    @functools.partial(jax.jit,
                       in_shardings=(S(x_pspec), S(y_pspec)),
                       out_shardings=S(o_pspec))
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(x_pspec, y_pspec),
        out_specs=o_pspec,
        check_rep=False,
    )
    def rs_matmul(x, y):
      return reduce_scatter_matmul.reduce_scatter_matmul_ref_impl(
          x,
          y,
          axis_name='x',
          scatter_dim=0,
          rhs_transpose=False,
      )

    @functools.partial(jax.jit,
                       in_shardings=(S(x_pspec), S(y_pspec)),
                       out_shardings=S(o_pspec))
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(x_pspec, y_pspec),
        out_specs=o_pspec,
        check_rep=False,
    )
    def rs_matmul_reference(x, y):
      return jax.lax.psum_scatter(
          jnp.dot(x, y.T, preferred_element_type=jnp.float32),
          axis_name='x',
          tiled=True,
          scatter_dimension=0,
      )

    out = rs_matmul(x, y)
    out_ref = rs_matmul_reference(x, y)
    self.assertAllClose(out, out_ref, atol=1e-5, rtol=1e-5)
    print('The test passed.')

  # Kernel test.
  # blaze test -c opt --test_output=errors //experimental/users/jevinjiang/ullm:tests/reduce_scatter_matmul_test --test_filter=test_basic_reduce_scatter_matmul_on_bs
  # blaze test -c opt --test_output=errors //experimental/users/jevinjiang/ullm:tests/reduce_scatter_matmul_test --test_filter=test_basic_reduce_scatter_matmul_kernel_on_bs --test_arg=--xla_tpu_enable_log_recorder
  # Minimum test with 2 devices.
  @parameterized.product(
      num_devices=[2],  # change to [1, 2, 8]
      grid_m=[1],  # change to [1, 2, 3]
      grid_k=[1],  # change to [1, 2, 3]
      grid_n=[1],  # change to [1, 2, 3]
  )
  def test_basic_reduce_scatter_matmul_kernel_on_bs(self, num_devices,
                                                    grid_m, grid_k, grid_n):
    dtype = jnp.float32
    # let's limit to 2 devices to make it simple.
    # TODO(xw32): should add a more general test that uses more devices.
    devices = jax.devices()[:num_devices]
    mesh = Mesh(devices, ('x', ))

    def S(p):
      return NamedSharding(mesh, p)

    # In tpu-inference SP,
    # lhs: [bs, in_features@TP], rhs: [out_features, in_features@TP]) ->
    # [bs@TP, out_features]
    def shard(x, spec):
      return jax.device_put(x, NamedSharding(mesh, spec))

    x_pspec = P(None, 'x')
    y_pspec = P(None, 'x')
    o_pspec = P('x', None)

    bm = 384
    bk = 128
    bn = 256
    # global shapes:
    m = bm * grid_m * num_devices  # *2 because the algorithm split x by num_devices on m-dim at the output.
    k = bk * grid_k * num_devices  # * devices because the input is sharded on k-dim.
    n = bn * grid_n * 2  # *2 because the algorithm split y by 2 on n-dim.
    # If grid_k==1, bk=256, then k=256, assuming num_devices=2, then
    # k_per_dev=128 due to the input is sharded on k dimension. Then the block size should be bk=128 (aka bk/num_devices)
    # bk = bk//num_devices
    # So on each device, lhs shape should be [bm*grid_m*num_devices, bk*grid_k]=[768, 128]
    # On each device, rhs shape should be [bn*grid_n*2, bk*grid_k]=[512, 128]
    # On each device, output shape should be [bm*grid_m, bn*grid_n*2]=[384, 512]

    x = jax.random.uniform(jax.random.key(0), (m, k), dtype=dtype)
    x = shard(x, x_pspec)
    y = jax.random.uniform(jax.random.key(1), (n, k), dtype=dtype)
    y = shard(y, y_pspec)
    print(
        f'xw32 in test line278 Global shape: {x.shape=}, {y.shape=}, {x.sharding=} {y.sharding=}'
    )

    @functools.partial(
        jax.jit,
        in_shardings=(S(x_pspec), S(y_pspec)),
        out_shardings=S(o_pspec),
    )
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(x_pspec, y_pspec),
        out_specs=o_pspec,
        check_rep=False,
    )
    def rs_matmul(x, y):
      return reduce_scatter_matmul.reduce_scatter_matmul(
          x,
          y,
          mesh=mesh,
          collective_id=0,
          axis_name='x',
          scatter_dim=0,
          rhs_transpose=False,
          debug_mode=True,
          bm=bm,
          bk=bk,
          bn=bn,
      )

    @functools.partial(jax.jit,
                       in_shardings=(S(x_pspec), S(y_pspec)),
                       out_shardings=S(o_pspec))
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(x_pspec, y_pspec),
        out_specs=o_pspec,
        check_rep=False,
    )
    def rs_matmul_reference(x, y):
      return jax.lax.psum_scatter(
          jnp.dot(x, y.T, preferred_element_type=jnp.float32),
          axis_name='x',
          tiled=True,
          scatter_dimension=0,
      )

    out = rs_matmul(x, y)
    out_ref = rs_matmul_reference(x, y)
    # print(f'xw32 out[0, 0] = {out[0, 0]}')
    # print(f'xw32 out_ref[0, 0] = {out_ref[0, 0]}')
    self.assertAllClose(out, out_ref, atol=1e-5, rtol=1e-5)
    print('The test passed.')


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
