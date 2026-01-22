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

import unittest

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.jax.moe.deepseek_v3_moe import DeepSeekV3Router


class TestDeepSeekV3Router(unittest.TestCase):

    def setUp(self):
        self.cpu_mesh = Mesh(jax.devices('cpu'), axis_names=('data', ))

    def test_get_topk_indices_single_group(self):
        """Test get_topk_indices with single expert group."""
        with jax.set_mesh(self.cpu_mesh):
            router = DeepSeekV3Router(random_init=True,
                                      hidden_size=512,
                                      num_experts=4,
                                      num_experts_per_tok=2,
                                      n_groups=1,
                                      topk_groups=1,
                                      norm_topk_prob=True,
                                      routed_scaling_factor=1.0,
                                      dtype=jnp.bfloat16,
                                      rngs=nnx.Rngs(42))
            router.bias_E = jnp.zeros((4, ))

            scores = jnp.array([[0.1, 0.3, 0.2, 0.4]])  # shape: (1, 4)
            indices = router.get_topk_indices(scores)

            # Should return indices of top 2 experts
            expected_indices = jnp.array([[3,
                                           1]])  # experts with scores 0.4, 0.3
            self.assertTrue(jnp.array_equal(indices, expected_indices))

    def test_get_topk_indices_2_groups(self):
        """Test get_topk_indices with 2 expert groups."""
        with jax.set_mesh(self.cpu_mesh):
            router = DeepSeekV3Router(random_init=True,
                                      hidden_size=512,
                                      num_experts=4,
                                      num_experts_per_tok=2,
                                      n_groups=2,
                                      topk_groups=1,
                                      norm_topk_prob=True,
                                      routed_scaling_factor=1.0,
                                      dtype=jnp.bfloat16,
                                      rngs=nnx.Rngs(42))
            router.bias_E = jnp.zeros((4, ))

            # 4 experts, 2 groups, 2 experts per group
            scores = jnp.array([[[0.1, 0.3, 0.2, 0.4]]])  # shape: (1, 1, 4)
            indices = router.get_topk_indices(scores)

            # Should return indices of top 2 experts
            expected_indices = jnp.array([[[3, 2]]])
            self.assertTrue(jnp.array_equal(indices, expected_indices))

    def test_router_e2e(self):
        with jax.set_mesh(self.cpu_mesh):
            router = DeepSeekV3Router(random_init=True,
                                      hidden_size=512,
                                      num_experts=8,
                                      num_experts_per_tok=2,
                                      n_groups=2,
                                      topk_groups=1,
                                      norm_topk_prob=True,
                                      routed_scaling_factor=1.0,
                                      dtype=jnp.bfloat16,
                                      rngs=nnx.Rngs(42))
            x = jnp.ones((2, 512))
            weights, indices = router(x)
            self.assertEqual(weights.shape, (2, 2))
            self.assertEqual(indices.shape, (2, 2))

    def test_token_replicated_expert_parallel_fwd(self):
        """
        Validates the MoE forward pass against a simple, dense equivalent.
        This specifically tests the is_batch_sharded_by_expert=False path.
        """
        # --- 1. Get the ACTUAL output from the complex distributed MoE layer ---
        # The __call__ method will trigger the shard_map, which requires the mesh context.
        with self.mesh:
            actual_output = self.moe(self.x)

        # --- 2. Calculate the EXPECTED output using a simple, sequential process ---
        # This serves as the "ground truth".

        # Get router decisions (router params are replicated, so this is fine)
        router_weights, selected_experts = self.moe.router(self.x)

        # Gather the full, unsharded weights from all devices ---
        # .value on a sharded param gives the *local* shard.
        # jax.device_get() retrieves the *full* GlobalDeviceArray to the host.
        gating_kernel_full = jax.device_get(self.moe.kernel_gating_EDF.value)
        up_proj_kernel_full = jax.device_get(self.moe.kernel_up_proj_EDF.value)
        down_proj_kernel_full = jax.device_get(
            self.moe.kernel_down_proj_EFD.value)

        # Check that we really got the full weights
        self.assertEqual(gating_kernel_full.shape,
                         (self.E, self.D, self.moe_intermediate_size))

        # Flatten inputs for easier iteration
        flat_x = self.x.reshape(self.B * self.S, self.D)
        flat_weights = router_weights.reshape(self.B * self.S, self.K)
        flat_experts = selected_experts.reshape(self.B * self.S, self.K)

        expected_output = jnp.zeros_like(flat_x)

        # Manually apply each expert to each token sequentially
        for i in range(self.B * self.S):  # For each token
            token_input = flat_x[i]
            combined_expert_output = jnp.zeros(self.D, dtype=jnp.bfloat16)

            for k in range(self.K):  # For each chosen expert for that token
                expert_idx = flat_experts[i, k]
                weight = flat_weights[i, k]

                # Get kernels from the *full* gathered arrays ---
                gating_kernel = gating_kernel_full[expert_idx]
                up_proj_kernel = up_proj_kernel_full[expert_idx]
                down_proj_kernel = down_proj_kernel_full[expert_idx]

                # Perform the expert computation (dense matmuls)
                gating_proj = jnp.dot(token_input, gating_kernel)
                up_proj = jnp.dot(token_input, up_proj_kernel)

                # Note: Assuming 'silu' activation as specified in MoE init
                fused = nnx.silu(gating_proj) * up_proj

                expert_output = jnp.dot(fused, down_proj_kernel)

                # Apply router weight after computation (matches implementation)
                combined_expert_output += weight * expert_output

            expected_output = expected_output.at[i].set(combined_expert_output)

        expected_output = expected_output.reshape(self.B * self.S, self.D)

        # --- 3. Compare the results ---
        self.assertTrue(
            jnp.allclose(actual_output, expected_output, atol=1e-2, rtol=1e-2),
            f"The output of the distributed MoE does not match the dense equivalent.\n"
            f"Actual:\n{actual_output}\n"
            f"Expected:\n{expected_output}")
        print(
            "\n✅ Test Passed: Distributed MoE output matches the dense ground truth."
        )
