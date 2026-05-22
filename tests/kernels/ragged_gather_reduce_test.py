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
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce

jax.config.parse_flags_with_absl()


def reference_ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScatterTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o,
             start_end=se,
             hidden_size=h,
             dtype=d,
             reduce_group_size=rg) for o, se, h, d, rg in itertools.chain(
                 itertools.product(
                     [400, 840],
                     [(3, 338), (10, 255)],
                     [128, 512, 8192],
                     [jnp.bfloat16, jnp.float32],
                     [8, 5],
                 ),
                 itertools.product(
                     [16384],
                     [(99, 1120)],
                     [7168],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [16384],
                     [(300, 2358)],
                     [6144],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [20480],
                     [(300, 2850)],
                     [4096],
                     [jnp.bfloat16],
                     [10],
                 ),
             )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_ragged_gather_reduce(self, out_size, hidden_size, start_end,
                                     dtype, reduce_group_size):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )
        actual = ragged_gather_reduce(x, indices, topk_weights,
                                      valid_rows_mask, reduce_group_size)
        # Correctness check.
        desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                 valid_rows_mask,
                                                 reduce_group_size)
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
