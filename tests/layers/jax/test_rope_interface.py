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
from absl.testing import absltest
from jax._src import test_util as jtu

from tpu_inference.layers.jax.rope_interface import apply_rope


class RopeInterfaceTest(jtu.JaxTestCase):

    def test_apply_rope_standard(self):
        seq_len = 2
        num_heads = 1
        head_dim = 8
        rope_theta = 10000.0

        inputs = jnp.ones((seq_len, num_heads, head_dim), dtype=jnp.float32)
        positions = jnp.arange(seq_len, dtype=jnp.float32)

        outputs = apply_rope(inputs,
                             positions=positions,
                             head_dim=head_dim,
                             rope_theta=rope_theta,
                             rope_proportion=1.0,
                             rope_input_ordering="split")

        # Check pos 0: should be identity (1.0)
        self.assertArraysAllClose(outputs[0], inputs[0])

        # Check pos 1: all elements should be rotated
        pos1_output = outputs[1, 0, :]
        for idx in range(head_dim):
            self.assertNotAlmostEqual(pos1_output[idx], 1.0, delta=1e-5)

    def test_apply_rope_partial_interleaved(self):
        seq_len = 2
        num_heads = 1
        head_dim = 8
        rope_theta = 10000.0
        rope_proportion = 0.5

        inputs = jnp.ones((seq_len, num_heads, head_dim), dtype=jnp.float32)
        positions = jnp.arange(seq_len, dtype=jnp.float32)

        outputs = apply_rope(inputs,
                             positions=positions,
                             head_dim=head_dim,
                             rope_theta=rope_theta,
                             rope_proportion=rope_proportion,
                             rope_input_ordering="interleaved")

        # In interleaved ordering:
        # pairs are (0,1), (2,3), (4,5), (6,7)
        # rope_angles = 2.
        # First 2 pairs are rotated: (0,1) and (2,3).
        # i.e., indices 0, 1, 2, 3.
        # Unrotated: 4, 5, 6, 7.

        pos1_output = outputs[1, 0, :]

        # Unrotated
        unrotated_indices = [4, 5, 6, 7]
        for idx in unrotated_indices:
            self.assertAlmostEqual(pos1_output[idx], 1.0, delta=1e-5)

        # Rotated
        rotated_indices = [0, 1, 2, 3]
        for idx in rotated_indices:
            self.assertNotAlmostEqual(pos1_output[idx], 1.0, delta=1e-5)


if __name__ == "__main__":
    absltest.main()
