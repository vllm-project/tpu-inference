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

import math
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.linear import sharded_matmul
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)


class TestShardedMatmulPadding(unittest.TestCase):

    def test_divisor_computation_and_padding_calculation(self):
        # Case 1: DP attention with attn_dp = 8
        mesh = MagicMock()
        mesh.shape = {"data": 1, "attn_dp": 8, "model": 1}

        attn_data_axes = (ShardingAxisNameBase.ATTN_DATA if isinstance(
            ShardingAxisNameBase.ATTN_DATA, (tuple, list)) else
                          (ShardingAxisNameBase.ATTN_DATA, ))
        divisor = math.prod(
            mesh.shape.get(ax, 1) for ax in attn_data_axes if ax)

        self.assertEqual(divisor, 8)

        # Unaligned sequence length: 634
        seq_len = 634
        pad_len = -seq_len % divisor
        self.assertEqual(pad_len, 6)
        self.assertEqual((seq_len + pad_len) % divisor, 0)
        self.assertEqual(seq_len + pad_len, 640)

        # Aligned sequence length: 640
        seq_len_aligned = 640
        pad_len_aligned = -seq_len_aligned % divisor
        self.assertEqual(pad_len_aligned, 0)

    def test_hybrid_mesh_divisor(self):
        # Case 2: Hybrid PCP=2 and attn_dp=4
        mesh = MagicMock()
        mesh.shape = {"data": 1, "pcp": 2, "attn_dp": 4, "model": 1}

        attn_data_axes = (ShardingAxisNameBase.ATTN_DATA if isinstance(
            ShardingAxisNameBase.ATTN_DATA, (tuple, list)) else
                          (ShardingAxisNameBase.ATTN_DATA, ))
        divisor = math.prod(
            mesh.shape.get(ax, 1) for ax in attn_data_axes if ax)

        self.assertEqual(divisor, 8)

        # Sequence length 8980 (MMMU batch)
        seq_len = 8980
        pad_len = -seq_len % divisor
        self.assertEqual(pad_len, 4)
        self.assertEqual((seq_len + pad_len) % 8, 0)

    @patch("tpu_inference.envs.NEW_MODEL_DESIGN", True)
    def test_sharded_matmul_numerical_equivalence_on_device(self):
        devices = jax.devices()

        # Create standard mesh with all required axis names
        mesh_shape = (len(devices), 1, 1, 1, 1, 1, 1)
        mesh = Mesh(np.array(devices).reshape(mesh_shape), MESH_AXIS_NAMES)

        hidden_in = 32
        hidden_out = 64

        # Test unaligned lengths
        for seq_len in [5, 7, 13, 17, 33, 634]:
            x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, hidden_in),
                                  dtype=jnp.bfloat16)
            w = jax.random.normal(jax.random.PRNGKey(1),
                                  (hidden_in, hidden_out),
                                  dtype=jnp.bfloat16)

            weight_sharding = P(None, "model")
            out = sharded_matmul(x, w, weight_sharding, mesh=mesh)

            expected = x @ w
            self.assertEqual(out.shape, (seq_len, hidden_out))
            np.testing.assert_allclose(
                np.array(out, dtype=np.float32),
                np.array(expected, dtype=np.float32),
                rtol=1e-2,
                atol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()
