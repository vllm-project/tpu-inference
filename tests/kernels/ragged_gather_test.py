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

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core.ragged_gather import ragged_gather

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GatherTest(jtu.JaxTestCase):

    @parameterized.product(
        in_out_size=[(512, 400), (512, 1024)],
        start_end=[(3, 338), (10, 422)],
        hidden_size=[128, 512, 8192],
        dtype=[jnp.int4, jnp.int8, jnp.bfloat16, jnp.float32],
    )
    def test_sc_gather(self, in_out_size, hidden_size, start_end, dtype):
        in_size, out_size = in_out_size
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (in_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.randint(key, (out_size, ), 0, in_size, jnp.int32)

        start_arr = jnp.array([start], jnp.int32)
        end_arr = jnp.array([end], jnp.int32)

        actual = ragged_gather(x, indices, start_arr, end_arr)

        # Correctness check.
        actual = actual[start:end]
        desired = x[indices][start:end]

        self.assertArraysEqual(actual, desired)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
