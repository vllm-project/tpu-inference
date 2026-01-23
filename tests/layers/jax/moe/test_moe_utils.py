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
from unittest import mock

import jax
import jax.numpy as jnp

# Adjust the import path to match your project structure
# yapf: disable
from tpu_inference import envs
from tpu_inference.layers.jax.moe.utils import (MoEBackend,
                                                get_all_to_all_params_fn,
                                                global_permute_fn, gmm_fn,
                                                local_permute_fn,
                                                select_moe_backend,
                                                sort_activations_fn,
                                                unpermute_fn)

# yapf: enable


class TestMoEUtils(unittest.TestCase):

    def setUp(self):
        self.key = jax.random.PRNGKey(0)

    def test_select_moe_backend_defaults(self):
        """Test default backend selection (Dense)."""
        # Ensure all flags are False
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False), \
             mock.patch.object(envs, 'USE_VLLM_MOE_KERNEL', False), \
             mock.patch.object(envs, 'USE_MEGABLOCKS', False), \
             mock.patch.object(envs, 'USE_RAGGED_DOT', False):

            backend = select_moe_backend()
            self.assertEqual(backend, MoEBackend.DENSE_MAT)

    def test_select_moe_backend_priority(self):
        """Test priority logic for backend selection."""
        # Case 1: Fused MoE has highest priority
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', True), \
             mock.patch.object(envs, 'USE_VLLM_MOE_KERNEL', False):
            self.assertEqual(select_moe_backend(), MoEBackend.FUSED_MOE)

        # Case 2: VLLM MoE
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False), \
             mock.patch.object(envs, 'USE_VLLM_MOE_KERNEL', True):
            self.assertEqual(select_moe_backend(), MoEBackend.VLLM_MOE)

        # Case 3: MegaBlocks
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False), \
             mock.patch.object(envs, 'USE_VLLM_MOE_KERNEL', False), \
             mock.patch.object(envs, 'USE_MEGABLOCKS', True):
            self.assertEqual(select_moe_backend(), MoEBackend.MEGABLX_GMM)

        # Case 4: Ragged Dot
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', False), \
             mock.patch.object(envs, 'USE_VLLM_MOE_KERNEL', False), \
             mock.patch.object(envs, 'USE_MEGABLOCKS', False), \
             mock.patch.object(envs, 'USE_RAGGED_DOT', True):
            self.assertEqual(select_moe_backend(), MoEBackend.RAGGED_DOT)

    def test_select_moe_backend_conflict(self):
        """Test conflict detection."""
        with mock.patch.object(envs, 'USE_MOE_EP_KERNEL', True), \
             mock.patch.object(envs, 'USE_VLLM_MOE_KERNEL', True):
            with self.assertRaises(ValueError):
                select_moe_backend()

    def test_sort_activations_fn(self):
        """Test stateless sorting of activations."""
        inputs = jnp.array([[10], [20], [30], [40]])
        indices = jnp.array([3, 0, 1, 2])
        sorted_out = sort_activations_fn(inputs, indices)
        expected = jnp.array([[40], [10], [20], [30]])
        self.assertTrue(jnp.array_equal(sorted_out, expected))

    def test_global_permute_fn(self):
        """Test global permutation logic."""
        # Setup: 2 tokens, embedding dim 1.
        # Expert assignments: Token 0 -> Expert 1, Token 1 -> Expert 0.
        # num_experts_per_tok = 1, num_local_experts = 2.
        inputs = jnp.array([[10.0], [20.0]])
        selected_experts = jnp.array([[1], [0]])

        # Expected behavior:
        # Flattened experts: [1, 0]
        # Sorted indices should bring expert 0 first, then expert 1.
        # Indices: [1, 0] (since expert 0 is at index 1, expert 1 is at index 0)

        sorted_inputs, sort_indices, group_sizes, expert_assignments = \
            global_permute_fn(inputs, selected_experts,
                              num_experts_per_tok=1,
                              num_local_experts=2)

        # Check group sizes (1 token for expert 0, 1 token for expert 1)
        self.assertTrue(jnp.array_equal(group_sizes, jnp.array([1, 1])))

        # Check sorted inputs: Expert 0's token (20.0) first, then Expert 1's (10.0)
        expected_inputs = jnp.array([[20.0], [10.0]])
        self.assertTrue(jnp.array_equal(sorted_inputs, expected_inputs))

        # Check sort indices
        expected_indices = jnp.array([1, 0])
        self.assertTrue(jnp.array_equal(sort_indices, expected_indices))

    def test_unpermute_fn(self):
        """Test unpermute logic (recombining sorted expert outputs)."""
        # 2 tokens, 2 experts per token. Flattened = 4 computations.
        # Let's say inputs were originally T0, T1.
        # Sorted order processed was: T0_e1, T1_e0, T0_e2, T1_e2 (just hypothetical).
        # We test that it puts them back into [T0_combined, T1_combined].

        num_tokens = 2
        k = 2
        dim = 4

        # Processed tokens (T*K, D)
        processed = jnp.ones((num_tokens * k, dim))

        # Indices that were used to sort them.
        # If we provide identity indices, unpermute should just reshape and sum weighted.
        sort_indices = jnp.arange(num_tokens * k)

        # Router weights: (T, K)
        weights = jnp.full((num_tokens, k), 0.5)

        output = unpermute_fn(processed,
                              sort_indices,
                              weights,
                              num_experts_per_tok=k,
                              hidden_size=dim,
                              output_dtype=jnp.float32)

        # Expected: sum(token_val * 0.5) over k=2 -> 1.0 * 0.5 + 1.0 * 0.5 = 1.0
        self.assertEqual(output.shape, (num_tokens, dim))
        self.assertTrue(jnp.allclose(output, jnp.ones((num_tokens, dim))))

    def test_local_permute_fn(self):
        """Test local permutation slicing."""
        # 4 tokens total across all experts
        # Global Group Sizes: [[1, 1, 1, 1]] (1 token per expert, 4 experts)
        # Local expert size: 2.
        # Shard 0 owns experts [0, 1]. Shard 1 owns experts [2, 3].

        inputs = jnp.arange(4)[:, None]  # [0, 1, 2, 3]
        global_group_sizes = jnp.array([[1, 1, 1, 1]])

        # Test Shard 0 (Experts 0, 1)
        # It should slice the first 2 counts from group sizes -> [1, 1]. Total 2.
        # is_offset=False (Token sharded usually implies standard modulo or repetition,
        # but here we test the logic flow).

        # Mocking logic: effectively testing the dynamic slice and argsort
        # For this test, we construct inputs to match the logic.

        _, _, size, expert_ids = local_permute_fn(inputs,
                                                  global_group_sizes,
                                                  local_expert_size=2,
                                                  shard_index=0,
                                                  is_offset=False)

        # We expect it to process the local experts
        self.assertEqual(list(size), [1, 1])
        print(expert_ids)
        self.assertTrue(jnp.array_equal(expert_ids, jnp.array([0, 1, 1, 1])))

    def test_get_all_to_all_params_fn(self):
        """Test parameter generation for all-to-all communication."""
        # 2 Shards.
        # Group sizes matrix (conceptually):
        # Shard 0 sends 10 to Shard 0, 20 to Shard 1
        # Shard 1 sends 30 to Shard 0, 40 to Shard 1
        # Passed as flat array or specific shape depending on usage,
        # but the function expects array access logic.

        # Let's assume input is (2, 2) for (senders, receivers)
        group_sizes = jnp.array([[10, 20], [30, 40]])

        # Test Shard 0 params
        shard_id = 0

        # Strategy: INPUT_OFFSET (cumsum of what I send)
        # I am Shard 0. I send [10, 20]. offsets: [0, 10]
        input_offsets, send_sizes, output_offsets, recv_sizes = \
            get_all_to_all_params_fn(group_sizes, shard_id,
                                     num_expert_parallelism=2,
                                     is_batch_sharded=True)

        self.assertTrue(jnp.array_equal(input_offsets, jnp.array([0, 10])))
        self.assertTrue(jnp.array_equal(send_sizes, jnp.array([10, 20])))

        # RECV_SIZE: I am Shard 0. I receive column 0: [10, 30]
        self.assertTrue(jnp.array_equal(recv_sizes, jnp.array([10, 30])))

    @mock.patch('jax.lax.ragged_dot')
    @mock.patch(
        'tpu_inference.layers.jax.moe.utils.round_up_to_multiple_of_128_within_limit'
    )
    def test_gmm_fn_ragged_dot(self, mock_round, mock_ragged_dot):
        """Test GMM function calls Ragged Dot correctly."""
        # Setup
        inputs = jnp.ones((128, 7168))
        kernel = jnp.ones((1, 7168, 2048))
        group_sizes = jnp.array([32])
        tile_size = (128, 128, 128)

        # Mock return
        mock_ragged_dot.return_value = jnp.zeros((128, 128))
        mock_round.side_effect = lambda x, y: x  # Identity for test

        # Execute
        out = gmm_fn(inputs,
                     kernel,
                     group_sizes,
                     tile_size,
                     moe_backend=MoEBackend.RAGGED_DOT,
                     dtype=jnp.bfloat16,
                     quantized_dtype=False)

        # Verify call
        mock_ragged_dot.assert_called_once()
        self.assertEqual(out.shape, (128, 128))

    @mock.patch('tpu_inference.layers.jax.moe.utils.megablox_gmm')
    @mock.patch(
        'tpu_inference.layers.jax.moe.utils.round_up_to_multiple_of_128_within_limit'
    )
    def test_gmm_fn_megablox(self, mock_round, mock_megablox):
        """Test GMM function calls MegaBlocks correctly."""
        inputs = jnp.ones((128, 64))
        kernel = jnp.ones((1, 64, 128))  # [E, K, N]
        group_sizes = jnp.array([128])
        tile_size = (128, 128, 128)

        mock_megablox.return_value = jnp.zeros((128, 128))
        mock_round.side_effect = lambda x, y: x

        gmm_fn(inputs,
               kernel,
               group_sizes,
               tile_size,
               moe_backend=MoEBackend.MEGABLX_GMM,
               dtype=jnp.bfloat16,
               quantized_dtype=False)

        mock_megablox.assert_called_once()

    @mock.patch('tpu_inference.layers.jax.moe.utils.megablox_gmm')
    @mock.patch(
        'tpu_inference.layers.jax.moe.utils.round_up_to_multiple_of_128_within_limit'
    )
    def test_gmm_fn_megablox_quantized(self, mock_round, mock_megablox):
        """Test GMM function calls MegaBlocks correctly."""
        inputs = jnp.ones((128, 64))
        kernel = jnp.ones((1, 64, 128)), jnp.ones((1, 1, 128))
        group_sizes = jnp.array([128])
        tile_size = (128, 128, 128)

        mock_megablox.return_value = jnp.zeros((128, 128))
        mock_round.side_effect = lambda x, y: x

        gmm_fn(inputs,
               kernel,
               group_sizes,
               tile_size,
               moe_backend=MoEBackend.MEGABLX_GMM,
               dtype=jnp.bfloat16,
               quantized_dtype=True)

        mock_megablox.assert_called_once()
