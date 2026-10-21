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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     apply_rope_scaling,
                                                     get_rope_scaling,
                                                     normalize_rope_scaling)


class RopeInterfaceTest(jtu.JaxTestCase):

    @parameterized.product(
        factor=[2.0, 8.0],
        ordering=["split", "interleaved"],
        config_field=["rope_parameters", "rope_scaling"],
    )
    def test_apply_rope_linear_scaling(self, factor, ordering, config_field):
        head_dim = 8
        positions = np.array([0, 1, 127, 4096], dtype=np.int32)
        inputs = np.random.default_rng(0).normal(size=(4, 2, head_dim))
        inputs = inputs.astype(np.float32)
        config = SimpleNamespace(
            **{config_field: {
                "rope_type": "linear",
                "factor": factor
            }})
        rope_scaling = get_rope_scaling(config)
        rotate = jax.jit(
            functools.partial(apply_rope,
                              head_dim=head_dim,
                              rope_scaling=rope_scaling,
                              rope_input_ordering=ordering))
        padded_inputs = np.pad(inputs, ((0, 0), (0, 0), (0, 4)))
        output = rotate(jnp.array(padded_inputs), jnp.array(positions))

        # Linear RoPE rotates at position / factor for every frequency.
        frequencies = 10000.0**(-np.arange(0, head_dim, 2) / head_dim)
        angles = positions[:, None, None] / factor * frequencies[None, None, :]
        cos, sin = np.cos(angles), np.sin(angles)
        if ordering == "split":
            real, imag = np.split(inputs, 2, axis=-1)
            expected = np.concatenate(
                [real * cos - imag * sin, imag * cos + real * sin], axis=-1)
        else:
            real, imag = inputs[..., 0::2], inputs[..., 1::2]
            expected = np.stack(
                [real * cos - imag * sin, imag * cos + real * sin], axis=-1)
            expected = expected.reshape(inputs.shape)
        expected = np.pad(expected, ((0, 0), (0, 0), (0, 4)))
        # Float32 frequency errors accumulate at large positions on TPU.
        self.assertAllClose(output,
                            expected.astype(np.float32),
                            atol=1e-4,
                            rtol=1e-5)

    @parameterized.parameters("rope_type", "type")
    def test_linear_rope_scaling_legacy_keys(self, type_key):
        scaling = normalize_rope_scaling({type_key: "linear", "factor": 8.0})
        frequencies = jnp.array([1.0, 0.1, 0.01, 0.001])
        self.assertAllClose(apply_rope_scaling(frequencies, scaling),
                            frequencies / 8.0)

    @parameterized.parameters(None, "llama3")
    def test_llama3_rope_scaling_preserved(self, rope_type):
        scaling = {"scale_factor": 8.0}
        if rope_type is not None:
            scaling["rope_type"] = rope_type
        frequencies = jnp.array([1.0, 0.001, 0.0001])
        self.assertAllClose(apply_rope_scaling(frequencies, scaling),
                            jnp.array([1.0, 0.00021360798, 0.0000125]),
                            atol=1e-8,
                            rtol=1e-6)

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
