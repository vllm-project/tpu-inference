# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common.fused_moe_gmm import _invert_permutation


def test_invert_permutation_matches_argsort():
    rng = np.random.default_rng(1234)

    for size in (1, 16, 256, 1024):
        permutation = jnp.asarray(rng.permutation(size), dtype=jnp.int32)

        actual = _invert_permutation(permutation)
        expected = jnp.argsort(permutation)

        np.testing.assert_array_equal(actual, expected)


def test_invert_permutation_restores_original_order():
    permutation = jnp.array([1, 3, 0, 2], dtype=jnp.int32)
    original = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    reordered = original[permutation]

    restored = reordered[_invert_permutation(permutation)]

    np.testing.assert_array_equal(restored, original)
