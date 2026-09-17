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

from tpu_inference.layers.common.fused_moe_gmm import \
    _tokens_from_sorted_assignments


def test_sorted_assignments_map_to_source_tokens_without_gather():
    rng = np.random.default_rng(1234)

    for num_tokens, topk in ((1, 1), (16, 2), (32, 4), (256, 8)):
        num_assignments = num_tokens * topk
        sorted_assignments = jnp.asarray(rng.permutation(num_assignments),
                                         dtype=jnp.int32)

        token_indices = jnp.arange(num_tokens, dtype=jnp.int32).repeat(topk)
        expected = token_indices[sorted_assignments]
        actual = _tokens_from_sorted_assignments(sorted_assignments, topk)

        np.testing.assert_array_equal(actual, expected)
