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

import itertools

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core.ragged_scatter import ragged_scatter

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScatterTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o, start_end=se, hidden_size=h, dtype=d)
        for o, se, h, d in itertools.chain(
            itertools.product(
                [400, 1024],
                [(3, 338), (10, 422)],
                [128, 512, 8192],
                [jnp.int4, jnp.int8, jnp.bfloat16, jnp.float32],
            ),
            itertools.product(
                [16384],
                [(13, 1030)],
                [7168],
                [jnp.bfloat16],
            ),
            itertools.product(
                [4096],
                [(13, 500)],
                [2048],
                [jnp.bfloat16],
            ),
        )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_gather(self, out_size, hidden_size, start_end, dtype):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)

        start_arr = jnp.array([start], jnp.int32)
        end_arr = jnp.array([end], jnp.int32)

        actual = ragged_scatter(x, indices, start_arr, end_arr)

        # Correctness check.
        desired = x[indices]
        mask = jnp.where(jnp.logical_and(indices >= start, indices < end), 1,
                         0)
        desired = jnp.where(mask[:, None], desired, 0)
        actual = jnp.where(mask[:, None], actual, 0)

        self.assertArraysEqual(actual, desired)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
