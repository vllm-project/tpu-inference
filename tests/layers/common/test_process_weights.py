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

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, shard_linear_weights)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, W13PaddingConfig, _process_quantized_moe_weights_impl,
    get_w13_padding_config, process_w13_for_gmm)
from tpu_inference.layers.common.sharding import ShardingAxisName


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

        # 2D scale[out_blocks, in_blocks]
        # Multiply size by mesh dimension to ensure divisibility across multiple chips.
        weight_scale = jnp.ones((8, 4 * mesh.shape['data']))
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

        # 3D scale[num_blocks, 1, out_features]
        # Multiply size by mesh dimension to ensure divisibility across multiple chips.
        weight_scale = jnp.ones((4 * mesh.shape['data'], 1, 1024))
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
        expected_scale_spec = P('model', None, 'data')
        self.assertEqual(sharded_weights.weight_scale.sharding,
                         NamedSharding(mesh, expected_scale_spec))

    def test_get_w13_padding_config_with_outer_block_size(self):
        """Test get_w13_padding_config scales down dimensions based on outer_block_size."""
        # intermediate_size = 128, reorder_size = 2 -> local_intermediate_size = 64
        # align = 128 -> padded_local_intermediate_size = 128, pad_amount = 64
        # padded_intermediate_size = 256
        # With outer_block_size = 2, all these should be halved.
        config = get_w13_padding_config(intermediate_size=128,
                                        reorder_size=2,
                                        align=128,
                                        outer_block_size=2)

        self.assertEqual(config.intermediate_size, 64)
        self.assertEqual(config.w13_reorder_size, 2)
        self.assertEqual(config.local_intermediate_size, 32)
        self.assertEqual(config.pad_amount, 32)
        self.assertEqual(config.padded_intermediate_size, 128)

    def test_process_w13_for_gmm_with_scaled_config(self):
        """Test process_w13_for_gmm with config scaled for block-quantized scales."""
        # For outer_block_size=2, the config should be pre-scaled by get_w13_padding_config.
        # Original intermediate=128, total width=256.
        # Scaled config: intermediate=64, local=32, pad=0, padded=64.
        config = W13PaddingConfig(
            intermediate_size=64,
            local_intermediate_size=32,
            pad_amount=0,
            padded_intermediate_size=64,
            w13_reorder_size=2,
        )

        # Input tensor (scale) has width 128 (64 for W1, 64 for W3)
        tensor = jnp.arange(128).reshape(1, 128)

        processed = process_w13_for_gmm(
            tensor,
            concat_dim=-1,
            config=config,
        )

        # Expected behavior:
        # 1. Split into w1 (64) and w3 (64)
        # 2. _pad_tensor(w1):
        #    - reshape to (1, 2, 32)
        #    - pad (0) -> stays (1, 2, 32)
        #    - reshape back to (1, 64)
        # 3. Concatenate w1, w3 -> (1, 128)
        # Note: Repeat by outer_block_size is now handled outside process_w13_for_gmm.

        self.assertEqual(processed.shape, (1, 128))
        # Check first few elements (not repeated anymore)
        self.assertEqual(processed[0, 0], 0)
        self.assertEqual(processed[0, 1], 1)
        self.assertEqual(processed[0, 2], 2)
        self.assertEqual(processed[0, 3], 3)

    def test_process_weights_requant_512(self):
        """Test process_quantized_moe_weights with MOE_REQUANTIZE_BLOCK_SIZE = 512."""
        # Dynamically adapt the mesh size to the physical devices available (CPU or TPU)
        devices = jax.devices()
        tp_size = len(devices)
        mock_mesh = Mesh(
            np.array(devices).reshape(1, tp_size), ('data', 'model'))

        # Create FusedMoEWeights with dynamically adapted shapes:
        # hidden_size = 7168, intermediate_size = 256 * tp_size
        # block_size = 128
        input_weights = FusedMoEWeights(
            w13_weight=jnp.ones((128, 512 * tp_size, 7168),
                                dtype=jnp.float8_e4m3fn),
            w13_weight_scale=jnp.ones((128, 4 * tp_size, 56),
                                      dtype=jnp.float32),
            w13_bias=None,
            w2_weight=jnp.ones((128, 7168, 256 * tp_size),
                               dtype=jnp.float8_e4m3fn),
            w2_weight_scale=jnp.ones((128, 56, 2 * tp_size),
                                     dtype=jnp.float32),
            w2_bias=None,
        )

        # Process the weights under GMM_TP by calling the JIT implementation directly
        processed_weights = _process_quantized_moe_weights_impl(
            input_weights,
            moe_backend=MoEBackend.GMM_TP,
            mesh=mock_mesh,
            activation="silu",
            weight_block_size=(128, ),
            desired_quant_dtype=jnp.float8_e4m3fn,
            requant_block_size=512,
        )

        # Assertions:
        # w2 scale: global shape
        self.assertEqual(processed_weights.w2_weight_scale.shape,
                         (128, tp_size, 1, 7168))

        # w2 scale: sharding spec and local shard shapes
        if tp_size == 1:
            expected_w2_sharding = NamedSharding(mock_mesh, P())
        else:
            expected_w2_sharding = NamedSharding(
                mock_mesh, P(None, ShardingAxisName.MLP_TENSOR))
        self.assertEqual(processed_weights.w2_weight_scale.sharding,
                         expected_w2_sharding)
        w2_local_shapes = [
            s.data.shape[1]
            for s in processed_weights.w2_weight_scale.addressable_shards
        ]
        self.assertEqual(sum(w2_local_shapes), tp_size)
        for s_size in w2_local_shapes:
            self.assertEqual(s_size, 1)


if __name__ == "__main__":
    unittest.main()
