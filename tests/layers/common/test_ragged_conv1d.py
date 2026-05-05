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
from absl.testing import parameterized

import tpu_inference.layers.common.ragged_conv1d_jax as ragged_conv1d_jax


class RaggedConv1dJaxTest(parameterized.TestCase):

    def test_single_sequence(self):
        dim = 2
        kernel_size = 3
        num_tokens = 5
        num_reqs = 1

        x = jnp.arange(num_tokens * dim, dtype=jnp.float32).reshape(
            (num_tokens, dim))
        conv_state = jnp.array([[[-0.5, 0.5], [1.0, -1.0]]], dtype=jnp.float32)
        conv_weight = jnp.array([[[0.3, -0.6, 1.2]], [[-0.8, 0.4, 0.9]]],
                                dtype=jnp.float32)
        conv_bias = None

        query_start_loc = jnp.array([0, 5], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)
        distribution = jnp.array([0, 0, num_reqs], dtype=jnp.int32)

        has_initial_state = jnp.array([True], dtype=bool)
        output, updated_state = ragged_conv1d_jax.ragged_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )

        self.assertEqual(output.shape, (num_tokens, dim))
        self.assertEqual(updated_state.shape, (1, kernel_size - 1, dim))

        expected_output = np.array(
            [
                [-0.75, 0.09999996423721313],
                [2.700000047683716, 3.8999998569488525],
                [3.6000001430511475, 4.9],
                [5.400000095367432, 5.899999141693115],
                [7.199999809265137, 6.899999618530273],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(output, expected_output, atol=1e-6)

        expected_state = np.array([[[6.0, 7.0], [8.0, 9.0]]], dtype=np.float32)
        np.testing.assert_allclose(updated_state[0],
                                   expected_state[0],
                                   atol=1e-6)

    def test_multiple_sequences(self):
        dtype = jnp.float32
        dim = 2
        kernel_size = 4
        num_tokens = 12
        max_reqs = 6
        num_valid_reqs = 4

        x = jnp.arange(num_tokens * dim, dtype=dtype).reshape(
            (num_tokens, dim))
        # 2 requests, kernel size 4 -> 3 state tokens
        # shape: (2, 3, 2)
        conv_state = jnp.array(
            [
                [[-0.1, 0.4], [0.8, -0.3], [0.2, -0.9]],
                [[0.5, -0.2], [-0.6, 0.7], [0.1, -0.5]],
                [[-0.3, 0.6], [0.4, -0.8], [0.9, -0.1]],
                [[0.2, -0.4], [-0.7, 0.3], [0.6, -0.2]],
                [[-0.9, 0.1], [0.5, -0.6], [0.3, -0.7]],
                [[0.7, -0.5], [-0.2, 0.9], [0.4, -0.3]],
            ],
            dtype=dtype,
        )
        # shape: (dim, 1, kernel_size)
        conv_weight = jnp.array(
            [[[0.3, -0.6, 1.2, -0.5]], [[-0.8, 0.4, 0.9, 0.1]]], dtype=dtype)
        conv_bias = jnp.array([1.0, 2.0], dtype=dtype)

        # Lengths: 2, 1, 4, 3
        query_start_loc = jnp.array([0, 2, 3, 7, 10, 1, 1], dtype=jnp.int32)

        # max_reqs = 4
        state_indices = jnp.arange(max_reqs, dtype=jnp.int32)
        distribution = jnp.array([1, 1, num_valid_reqs], dtype=jnp.int32)
        has_initial_state = jnp.array([True] * max_reqs, dtype=bool)
        output, updated_state = ragged_conv1d_jax.ragged_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )
        self.assertEqual(output.shape, (num_tokens, dim))
        expected_output = np.array(
            [
                [0.7300000190734863, 0.8500000238418579],
                [0.12000000476837158, 3.0799999237060547],
                [-0.3700000047683716, 2.490000009536743],
                [-1.249999761581421, 1.809999942779541],
                [3.7800002098083496, 9.799999237060547],
                [2.2700002193450928, 14.079999923706055],
                [4.0, 11.200000762939453],
                [-4.799999713897705, 3.760000228881836],
                [9.230001449584961, 16.880001068115234],
                [2.9800002574920654, 25.35999870300293],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=dtype,
        )
        np.testing.assert_allclose(output, expected_output, atol=1e-6)
        expected_state = np.array(
            [
                [[0.2, -0.9], [0.0, 1.0], [2.0, 3.0]],
                [[-0.6, 0.7], [0.1, -0.5], [4.0, 5.0]],
                [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]],
                [[14.0, 15.0], [16.0, 17.0], [18.0, 19.0]],
                [[-0.9, 0.1], [0.5, -0.6], [0.3, -0.7]],
                [[0.7, -0.5], [-0.2, 0.9], [0.4, -0.3]],
            ],
            dtype=dtype,
        )
        np.testing.assert_allclose(updated_state, expected_state, atol=1e-6)

    def test_multiple_sequences_with_no_initial_state(self):
        dtype = jnp.float32
        dim = 2
        kernel_size = 4
        num_tokens = 12
        max_reqs = 6
        num_valid_reqs = 4

        x = jnp.arange(num_tokens * dim, dtype=dtype).reshape(
            (num_tokens, dim))
        # 2 requests, kernel size 4 -> 3 state tokens
        # shape: (2, 3, 2)
        conv_state = jnp.array(
            [
                [[-0.1, 0.4], [0.8, -0.3], [0.2, -0.9]],
                [[0.5, -0.2], [-0.6, 0.7], [0.1, -0.5]],
                [[-0.3, 0.6], [0.4, -0.8], [0.9, -0.1]],
                [[0.2, -0.4], [-0.7, 0.3], [0.6, -0.2]],
                [[-0.9, 0.1], [0.5, -0.6], [0.3, -0.7]],
                [[0.7, -0.5], [-0.2, 0.9], [0.4, -0.3]],
            ],
            dtype=dtype,
        )
        # shape: (dim, 1, kernel_size)
        conv_weight = jnp.array(
            [[[0.3, -0.6, 1.2, -0.5]], [[-0.8, 0.4, 0.9, 0.1]]], dtype=dtype)
        conv_bias = jnp.array([1.0, 2.0], dtype=dtype)

        # Lengths: 2, 1, 4, 3
        query_start_loc = jnp.array([0, 2, 3, 7, 10, 1, 1], dtype=jnp.int32)

        # max_reqs = 4
        state_indices = jnp.arange(max_reqs, dtype=jnp.int32)
        distribution = jnp.array([1, 1, num_valid_reqs], dtype=jnp.int32)
        has_initial_state = jnp.array([False] * max_reqs, dtype=bool)
        output, updated_state = ragged_conv1d_jax.ragged_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )
        self.assertEqual(output.shape, (num_tokens, dim))
        expected_output = np.array(
            [
                [1.0, 2.1],
                [0.0, 3.2],
                [-1.0, 2.5],
                [-2.0, 2.7],
                [4.2000003, 9.2],
                [2.0, 14.0],
                [4.0, 11.2],
                [-6.0, 3.5],
                [9.800001, 17.2],
                [2.8000002, 25.199999],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=dtype,
        )
        np.testing.assert_allclose(output, expected_output, atol=1e-6)
        expected_state = np.array(
            [
                [[0.0, 0.0], [0.0, 1.0], [2.0, 3.0]],
                [[0.0, 0.0], [0.0, 0.0], [4.0, 5.0]],
                [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]],
                [[14.0, 15.0], [16.0, 17.0], [18.0, 19.0]],
                [[-0.9, 0.1], [0.5, -0.6], [0.3, -0.7]],
                [[0.7, -0.5], [-0.2, 0.9], [0.4, -0.3]],
            ],
            dtype=dtype,
        )
        np.testing.assert_allclose(updated_state, expected_state, atol=1e-6)

    def test_multiple_sequences_all_length_1(self):
        dtype = jnp.float32
        dim = 2
        kernel_size = 4
        num_tokens = 6
        max_reqs = 6
        num_valid_reqs = 4

        x = jnp.arange(num_tokens * dim, dtype=dtype).reshape(
            (num_tokens, dim))
        conv_state = jnp.array(
            [
                [[-0.1, 0.4], [0.8, -0.3], [0.2, -0.9]],
                [[0.5, -0.2], [-0.6, 0.7], [0.1, -0.5]],
                [[-0.3, 0.6], [0.4, -0.8], [0.9, -0.1]],
                [[0.2, -0.4], [-0.7, 0.3], [0.6, -0.2]],
                [[-0.9, 0.1], [0.5, -0.6], [0.3, -0.7]],
                [[0.7, -0.5], [-0.2, 0.9], [0.4, -0.3]],
            ],
            dtype=dtype,
        )
        conv_weight = jnp.array(
            [[[0.3, -0.6, 1.2, -0.5]], [[-0.8, 0.4, 0.9, 0.1]]], dtype=dtype)
        conv_bias = jnp.array([1.0, 2.0], dtype=dtype)

        # Lengths: 1, 1, 1, 1, 0, 0
        query_start_loc = jnp.array([0, 1, 2, 3, 4, 1, 1], dtype=jnp.int32)

        state_indices = jnp.arange(max_reqs, dtype=jnp.int32)
        distribution = jnp.array(
            [num_valid_reqs, num_valid_reqs, num_valid_reqs], dtype=jnp.int32)

        has_initial_state = jnp.array([True] * max_reqs, dtype=bool)
        output, updated_state = ragged_conv1d_jax.ragged_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )
        self.assertEqual(output.shape, (num_tokens, dim))
        expected_output = np.array(
            [
                [0.73, 0.85],
                [0.63, 2.29],
                [-0.25, 1.61],
                [-0.80, 2.96],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=dtype,
        )
        np.testing.assert_allclose(output, expected_output, atol=1e-5)

        expected_state = np.array(
            [
                [[0.8, -0.3], [0.2, -0.9], [0.0, 1.0]],
                [[-0.6, 0.7], [0.1, -0.5], [2.0, 3.0]],
                [[0.4, -0.8], [0.9, -0.1], [4.0, 5.0]],
                [[-0.7, 0.3], [0.6, -0.2], [6.0, 7.0]],
                [[-0.9, 0.1], [0.5, -0.6], [0.3, -0.7]],
                [[0.7, -0.5], [-0.2, 0.9], [0.4, -0.3]],
            ],
            dtype=dtype,
        )
        np.testing.assert_allclose(updated_state, expected_state, atol=1e-6)
