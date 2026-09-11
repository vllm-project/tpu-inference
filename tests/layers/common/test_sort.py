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

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.layers.common.sort import can_pack_int32, packed_argsort


class CanPackInt32Test(jtu.JaxTestCase):

    def test_fits_exactly(self):
        # 21 key bits + 10 index bits == 31 usable bits.
        self.assertTrue(can_pack_int32(n=1024, max_key=2**21 - 1))

    def test_one_bit_too_wide(self):
        self.assertFalse(can_pack_int32(n=1024, max_key=2**21))


class PackedArgsortTest(jtu.JaxTestCase):

    @parameterized.named_parameters(
        ("random", np.random.default_rng(0).integers(0, 64, size=512), 63),
        # All keys tied: order can only come from the packed index.
        ("all_ties", np.zeros(128, dtype=np.int32), 0),
        # The shape ragged_gather_reduce_v2 passes: 2-D, sorted on axis=-1.
        ("boolean_2d", np.array([[1, 0, 1, 0], [0, 0, 1, 1]]), 1),
    )
    def test_matches_argsort(self, keys, max_key):
        keys = jnp.asarray(keys)
        expected = jnp.argsort(keys, axis=-1, stable=True)
        self.assertArraysEqual(packed_argsort(keys, max_key=max_key), expected)

    def test_raises_when_packing_does_not_fit(self):
        keys = jnp.zeros(1024, dtype=jnp.int32)
        with self.assertRaises(ValueError):
            packed_argsort(keys, max_key=2**21)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
