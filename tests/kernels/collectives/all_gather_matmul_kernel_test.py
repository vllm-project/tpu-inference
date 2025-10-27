# SPDX-License-Identifier: Apache-2.0

import os

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference import utils
from tpu_inference.kernels.collectives import all_gather_matmul

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec

SpongeDir: str | None = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', None)


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class AllGatherMatmulTest(jtu.JaxTestCase):

    @parameterized.product(
        grid_k=[1, 2, 3],
        grid_n=[1, 2, 3],
        rhs_transpose=[True, False],
    )
    def test_all_gather_matmul(self, grid_k, grid_n, rhs_transpose):
        if jax.device_count() != 8:
            self.skipTest('Not enough devices for test')

        axis_name = 'x'
        num_devices = jax.device_count()
        mesh = utils.make_optimized_mesh((num_devices, ), (axis_name, ))
        bk, bn = 1024, 1024
        m, k, n = 1024, bk * grid_k, bn * grid_n * num_devices

        # Run the test 10 times to expose race conditions as much as possible.
        for i in range(10):
            # Create input data
            prng_key = jax.random.key(1234 + i)
            k0, k1 = jax.random.split(prng_key, 2)
            x = jax.random.normal(k0, (m, k), dtype=jnp.bfloat16)
            y_shape = (n, k) if rhs_transpose else (k, n)
            y_sharding = P(axis_name, None) if rhs_transpose else P(
                None, axis_name)
            y = jax.random.normal(k1, y_shape, dtype=jnp.bfloat16)
            sharded_x = jax.device_put(
                x, jax.sharding.NamedSharding(mesh, P(axis_name, None)))
            sharded_y = jax.device_put(
                y, jax.sharding.NamedSharding(mesh, y_sharding))

            # Run the all_gather_matmul function
            output = all_gather_matmul.all_gather_matmul(
                sharded_x,
                sharded_y,
                mesh,
                axis_name,
                bk=bk,
                bn=bn,
                rhs_transpose=rhs_transpose,
            )
            y_for_dot = sharded_y.T if rhs_transpose else sharded_y
            expected_output = jnp.dot(sharded_x, y_for_dot)
            self.assertAllClose(output, expected_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
