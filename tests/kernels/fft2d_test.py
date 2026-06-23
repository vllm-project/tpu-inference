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

from absl.testing import absltest, parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

from tpu_inference.kernels.experimental.fft.fft2d import fft2d


@absltest.skipUnless(jax.devices()[0].platform == "tpu", "TPU-only Pallas kernel")
class Fft2dTest(jtu.JaxTestCase):
    @parameterized.parameters(
        dict(shape=(1, 128, 128), col_tile=128),
        dict(shape=(2, 256, 256), col_tile=128),
        dict(shape=(1, 128, 256), col_tile=128),
    )
    def test_matches_jax_fft2(self, shape, col_tile):
        key = jax.random.key(0)
        x = jax.random.normal(key, shape, dtype=jnp.float32).astype(jnp.complex64)

        actual = fft2d(
            x,
            col_tile=col_tile,
            row_k_tile=shape[2],
            col_k_tile=shape[1],
        )
        expected = jnp.fft.fft2(x)

        self.assertArraysAllClose(actual, expected, rtol=1e-3, atol=1e-3)


class Fft2dValidationTest(jtu.JaxTestCase):
    def test_rejects_non_complex_input(self):
        x = jnp.ones((1, 128, 128), dtype=jnp.float32)
        with self.assertRaisesRegex(TypeError, "complex64"):
            fft2d(x, col_tile=128, row_k_tile=128, col_k_tile=128)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
