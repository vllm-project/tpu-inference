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

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, shard_linear_weights)
from tpu_inference.layers.common.process_weights.moe_weights import (
    W13PaddingConfig, process_w13_for_gmm)


class TestProcessWeights(unittest.TestCase):

    def setUp(self):
        devices = jax.devices()
        if len(devices) < 1:
            self.skipTest("Need at least 1 device")
        self.mesh = Mesh(np.array(devices).reshape(-1, 1), ('data', 'model'))

    def test_shard_linear_weights_2d_scale(self):
        """Test shard_linear_weights with 2D block-wise scales."""
        mesh = self.mesh
        weight_p_spec = P('model', 'data')
        bias_p_spec = P('model')

        # 2D scale [out_blocks, in_blocks]
        weight_scale = jnp.ones((8, 4))
        weights = LinearWeights(
            weight=jnp.ones((1024, 512)),
            weight_scale=weight_scale,
            zero_point=None,
            bias=jnp.ones((1024, )),
        )

        sharded_weights = shard_linear_weights(
            weights,
            mesh,
            weight_p_spec,
            bias_p_spec,
            per_tensor=False,
        )

        # 2D block-wise scale should follow weight sharding
        self.assertEqual(sharded_weights.weight_scale.sharding,
                         NamedSharding(mesh, weight_p_spec))

    def test_shard_linear_weights_3d_scale(self):
        """Test shard_linear_weights with 3D scales (legacy block-wise)."""
        mesh = self.mesh
        weight_p_spec = P('model', 'data')
        bias_p_spec = P('model')

        # 3D scale [num_blocks, 1, out_features]
        weight_scale = jnp.ones((4, 1, 1024))
        weights = LinearWeights(
            weight=jnp.ones((1024, 512)),
            weight_scale=weight_scale,
            zero_point=None,
            bias=jnp.ones((1024, )),
        )

        sharded_weights = shard_linear_weights(
            weights,
            mesh,
            weight_p_spec,
            bias_p_spec,
            per_tensor=False,
        )

        # 3D scale sharding: P(in_axis, None, out_axis)
        expected_scale_spec = P('data', None, 'model')
        self.assertEqual(sharded_weights.weight_scale.sharding,
                         NamedSharding(mesh, expected_scale_spec))

    @patch("tpu_inference.envs.DISABLE_WEIGHT_REQUANTIZATION", True)
    def test_process_w13_for_gmm_with_scale_ratio(self):
        """Test process_w13_for_gmm with scale_ratio > 1 (block-quantized scales)."""
        # config for weights: intermediate=128, local=64, pad=0, padded=128
        # total width = 256
        config = W13PaddingConfig(
            intermediate_size=128,
            local_intermediate_size=64,
            pad_amount=0,
            padded_intermediate_size=128,
            w13_reorder_size=2,
        )

        # If scale_ratio=2, the scale tensor dimensions are half
        # Input tensor (scale) has width 128 (64 for W1, 64 for W3)
        scale_ratio = 2
        tensor = jnp.arange(128).reshape(1, 128)

        processed = process_w13_for_gmm(
            tensor,
            concat_dim=-1,
            config=config,
            scale_ratio=scale_ratio,
        )

        # Expected behavior:
        # 1. Split into w1 (64) and w3 (64)
        # 2. _pad_tensor(w1):
        #    - reshape to (1, 2, 32)
        #    - pad (0) -> stays (1, 2, 32)
        #    - reshape back to (1, 64)
        # 3. Concatenate w1, w3 -> (1, 128)
        # 4. Repeat by scale_ratio=2 along concat_dim -> (1, 256)

        self.assertEqual(processed.shape, (1, 256))
        # Check first few elements (repeated)
        self.assertEqual(processed[0, 0], 0)
        self.assertEqual(processed[0, 1], 0)
        self.assertEqual(processed[0, 2], 1)
        self.assertEqual(processed[0, 3], 1)


if __name__ == "__main__":
    unittest.main()
