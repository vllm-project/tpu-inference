# Copyright 2025 Google LLC
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

import os
import unittest
from unittest.mock import patch

# Configure 8 simulated CPU devices for multi-device mesh testing before importing JAX
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.quantized_matmul.util import \
    xla_quantized_batched_matmul
from tpu_inference.layers.common.linear import (
    _pad_sharded_activation, _parse_einsum_dims, _unpad_sharded_activation,
    sharded_matmul, sharded_quantized_batched_matmul, sharded_quantized_matmul,
    xla_quantized_matmul)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)


@patch("tpu_inference.envs.NEW_MODEL_DESIGN", True)
class TestShardedMatmulPadding(unittest.TestCase):

    def setUp(self):
        ShardingAxisName.reset()
        ShardingAxisName._cls = None
        devices = jax.devices()
        num_devices = len(devices)
        # Create an 8-device (or available device count) multi-axis mesh
        # MESH_AXIS_NAMES: ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp", "pcp")
        if num_devices >= 8:
            mesh_shape = (1, 2, 1, 1, 2, 1, 2)
        else:
            mesh_shape = (num_devices, 1, 1, 1, 1, 1, 1)
        self.mesh = Mesh(
            np.array(devices[:np.prod(mesh_shape)]).reshape(mesh_shape),
            MESH_AXIS_NAMES)

    def tearDown(self):
        ShardingAxisName.reset()
        ShardingAxisName._cls = None

    def test_pad_and_unpad_sharded_activation_direct(self):
        # Test boundary and unaligned lengths
        hidden_dim = 32
        for seq_len in [1, 5, 7, 13, 17, 33, 64, 128, 634]:
            x = jax.random.normal(jax.random.PRNGKey(seq_len),
                                  (seq_len, hidden_dim),
                                  dtype=jnp.bfloat16)

            # Test with no sharding axis -> identity
            padded_no_shard, orig_len_no_shard = _pad_sharded_activation(
                x, self.mesh, None, axis_idx=0)
            self.assertEqual(orig_len_no_shard, seq_len)
            self.assertEqual(padded_no_shard.shape, x.shape)
            np.testing.assert_array_equal(np.array(padded_no_shard),
                                          np.array(x))

            # Test with ATTN_DATA sharding axis
            padded, orig_len = _pad_sharded_activation(
                x, self.mesh, ShardingAxisName.ATTN_DATA, axis_idx=0)
            self.assertEqual(orig_len, seq_len)
            self.assertGreaterEqual(padded.shape[0], seq_len)

            # Verify that padding region (if any) is zero-padded
            if padded.shape[0] > seq_len:
                np.testing.assert_array_equal(
                    np.array(padded[seq_len:]),
                    np.zeros((padded.shape[0] - seq_len, hidden_dim),
                             dtype=np.float32),
                )

            # Verify that unpadding perfectly restores original array
            unpadded = _unpad_sharded_activation(padded, orig_len, axis_idx=0)
            self.assertEqual(unpadded.shape, (seq_len, hidden_dim))
            np.testing.assert_array_equal(np.array(unpadded), np.array(x))

    def test_pad_and_unpad_multidimensional_axis(self):
        num_heads = 4
        head_dim = 16
        for seq_len in [1, 5, 7, 13, 17, 33, 64, 128, 634]:
            x = jax.random.normal(
                jax.random.PRNGKey(seq_len),
                (num_heads, seq_len, head_dim),
                dtype=jnp.bfloat16,
            )

            padded, orig_len = _pad_sharded_activation(
                x, self.mesh, ShardingAxisName.ATTN_DATA, axis_idx=1)
            self.assertEqual(orig_len, seq_len)
            self.assertEqual(padded.shape[0], num_heads)
            self.assertEqual(padded.shape[2], head_dim)
            self.assertGreaterEqual(padded.shape[1], seq_len)

            if padded.shape[1] > seq_len:
                np.testing.assert_array_equal(
                    np.array(padded[:, seq_len:, :]),
                    np.zeros((num_heads, padded.shape[1] - seq_len, head_dim),
                             dtype=np.float32),
                )

            unpadded = _unpad_sharded_activation(padded, orig_len, axis_idx=1)
            self.assertEqual(unpadded.shape, (num_heads, seq_len, head_dim))
            np.testing.assert_array_equal(np.array(unpadded), np.array(x))

    def test_sharded_matmul_numerical_equivalence_on_device(self):
        hidden_in = 32
        hidden_out = 64

        # 2D, 3D, and 4D unaligned/aligned sequence lengths
        for seq_len in [1, 5, 7, 13, 17, 33, 64, 128, 634]:
            # 2D Input: (seq_len, hidden_in)
            x_2d = jax.random.normal(jax.random.PRNGKey(0),
                                     (seq_len, hidden_in),
                                     dtype=jnp.bfloat16)
            w = jax.random.normal(
                jax.random.PRNGKey(1),
                (hidden_in, hidden_out),
                dtype=jnp.bfloat16,
            )

            # Column parallel sharding
            weight_sharding_col = P(None, "model")
            out_2d_col = sharded_matmul(x_2d,
                                        w,
                                        weight_sharding_col,
                                        mesh=self.mesh)
            expected_2d = x_2d @ w

            self.assertEqual(out_2d_col.shape, (seq_len, hidden_out))
            np.testing.assert_allclose(
                np.array(out_2d_col, dtype=np.float32),
                np.array(expected_2d, dtype=np.float32),
                rtol=1e-2,
                atol=1e-2,
            )

            # Row parallel sharding with all-reduce
            weight_sharding_row = P("model", None)
            out_2d_row = sharded_matmul(x_2d,
                                        w,
                                        weight_sharding_row,
                                        mesh=self.mesh)
            self.assertEqual(out_2d_row.shape, (seq_len, hidden_out))
            np.testing.assert_allclose(
                np.array(out_2d_row, dtype=np.float32),
                np.array(expected_2d, dtype=np.float32),
                rtol=1e-2,
                atol=1e-2,
            )

            # 3D Batched Input: (batch_size, seq_len, hidden_in)
            batch_size = 2
            x_3d = jax.random.normal(
                jax.random.PRNGKey(2),
                (batch_size, seq_len, hidden_in),
                dtype=jnp.bfloat16,
            )
            out_3d = sharded_matmul(x_3d,
                                    w,
                                    weight_sharding_col,
                                    mesh=self.mesh)
            expected_3d = x_3d @ w

            self.assertEqual(out_3d.shape, (batch_size, seq_len, hidden_out))
            np.testing.assert_allclose(
                np.array(out_3d, dtype=np.float32),
                np.array(expected_3d, dtype=np.float32),
                rtol=1e-2,
                atol=1e-2,
            )

    def test_sharded_quantized_matmul_numerical_parity_on_device(self):
        hidden_in = 32
        hidden_out = 64

        for seq_len in [1, 5, 7, 13, 17, 33, 64, 128, 634]:
            x = jax.random.normal(jax.random.PRNGKey(0), (seq_len, hidden_in),
                                  dtype=jnp.bfloat16)
            w_q = jax.random.randint(
                jax.random.PRNGKey(1),
                (hidden_in, hidden_out),
                -8,
                7,
                dtype=jnp.int8,
            )
            w_s = jax.random.uniform(jax.random.PRNGKey(2), (hidden_out, ),
                                     dtype=jnp.bfloat16)

            # Column parallel sharding
            weight_sharding = P(None, "model")
            out = sharded_quantized_matmul(x,
                                           w_q,
                                           w_s,
                                           weight_sharding,
                                           mesh=self.mesh)

            # Compare against unpadded reference kernel
            expected = xla_quantized_matmul(x,
                                            w_q,
                                            w_s,
                                            quantize_activation=True)
            self.assertEqual(out.shape, (seq_len, hidden_out))
            np.testing.assert_allclose(
                np.array(out, dtype=np.float32),
                np.array(expected, dtype=np.float32),
                rtol=0.05,
                atol=0.5,
            )

    def test_sharded_quantized_batched_matmul_numerical_parity_on_device(self):
        num_heads = 4
        head_dim = 16
        out_dim = 32
        einsum_str = "TNH,ANH->NTA"

        (
            contract_dims_x,
            contract_dims_w,
            batch_dims_x,
            batch_dims_w,
            output_perm,
        ) = _parse_einsum_dims(einsum_str)
        dimension_numbers = (
            (contract_dims_x, contract_dims_w),
            (batch_dims_x, batch_dims_w),
        )

        for seq_len in [1, 5, 7, 13, 17, 33, 64, 128, 634]:
            # Einsum: TNH,ANH->NTA
            x = jax.random.normal(
                jax.random.PRNGKey(0),
                (seq_len, num_heads, head_dim),
                dtype=jnp.bfloat16,
            )
            w_q = jax.random.randint(
                jax.random.PRNGKey(1),
                (out_dim, num_heads, head_dim),
                -8,
                7,
                dtype=jnp.int8,
            )
            w_s = jax.random.uniform(jax.random.PRNGKey(2), (out_dim, ),
                                     dtype=jnp.bfloat16)

            weight_sharding = P("model", None, None)
            out = sharded_quantized_batched_matmul(x,
                                                   w_q,
                                                   w_s,
                                                   einsum_str,
                                                   weight_sharding,
                                                   mesh=self.mesh)

            # Compare against unpadded reference kernel
            expected = xla_quantized_batched_matmul(
                x,
                w_q,
                w_s,
                dimension_numbers=dimension_numbers,
                quantize_activation=True,
            )
            if output_perm != tuple(range(len(output_perm))):
                expected = jnp.transpose(expected, output_perm)

            self.assertEqual(out.shape, (num_heads, seq_len, out_dim))
            np.testing.assert_allclose(
                np.array(out, dtype=np.float32),
                np.array(expected, dtype=np.float32),
                rtol=0.05,
                atol=0.5,
            )


if __name__ == "__main__":
    unittest.main()
