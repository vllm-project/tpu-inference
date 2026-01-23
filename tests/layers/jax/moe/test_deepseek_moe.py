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
import numpy as np
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from tpu_inference.layers.jax.moe.deepseek_v3_moe import DeepSeekV3Router
from tpu_inference.layers.jax.moe.moe import MoE, Router
from tpu_inference.layers.jax.moe.utils import MoEBackend


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


class TestMoE(unittest.TestCase):

    def setUp(self):
        """Set up a multi-device mesh for testing."""
        devices = jax.devices()
        self.device_count = len(devices)
        if self.device_count < 8:
            self.skipTest("This test requires at least 8 simulated devices.")

        # This mesh will have a 'model' axis for expert parallelism
        mesh_shape = (self.device_count, 1)
        device_mesh_array = np.array(devices).reshape(mesh_shape)

        # Define the axis names
        axis_names = ('model', 'data')

        # Create the 2D mesh
        self.mesh = Mesh(device_mesh_array, axis_names=axis_names)

        # --- Model Configuration ---
        self.B, self.S, self.D = 2, 8, 32  # Batch, Sequence, Hidden Dim
        self.E, self.K = 8, 2  # Num Experts (Total), Experts per Token
        self.moe_intermediate_size = 64
        self.dtype = jnp.bfloat16
        self.key = jax.random.PRNGKey(42)

        # Input data
        self.x = jax.random.normal(self.key, (self.B * self.S, self.D),
                                   dtype=self.dtype)

    def _create_moe(self, backend: MoEBackend):
        """Helper to instantiate the MoE layer within the mesh context."""
        with self.mesh:
            router = Router(
                dtype=self.dtype,
                hidden_size=self.D,
                num_experts=self.E,
                num_experts_per_tok=self.K,
                router_act="softmax",  # Standard softmax routing
                rngs=nnx.Rngs(self.key),
                activation_ffw_td=PartitionSpec('data', 'model'),
                ed_sharding=PartitionSpec(None, 'model'),
                moe_backend=backend)

            moe = MoE(
                dtype=self.dtype,
                num_local_experts=self.E,
                hidden_size=self.D,
                intermediate_size_moe=self.moe_intermediate_size,
                hidden_act="silu",
                rngs=nnx.Rngs(self.key),
                router=router,
                mesh=self.mesh,

                # Sharding Specs
                activation_ffw_td=PartitionSpec('data', None),
                activation_ffw_ted=PartitionSpec('data', None),
                edf_sharding=PartitionSpec('model', None,
                                           None),  # Expert par on axis 0
                efd_sharding=PartitionSpec('model', None, None),
                apply_expert_weight_before_computation=False,
                moe_backend=backend,
                num_experts_per_tok=self.K,
            )
        return moe

    def _compute_ground_truth(self, moe_instance, x_input):
        """
        Computes the expected MoE output using a simple, sequential Python loop.
        This serves as the 'Gold Standard' to verify distributed logic.
        """
        # 1. Get Router Decisions
        # We run the router (which is replicated/simple) to get weights & indices
        weights, indices = moe_instance.router(x_input)

        # 2. Fetch Full Weights from Devices
        # The MoE weights are sharded. We fetch them to CPU/Host as full arrays.
        gating_kernel_full = jax.device_get(
            moe_instance.kernel_gating_EDF.value)
        up_proj_kernel_full = jax.device_get(
            moe_instance.kernel_up_proj_EDF.value)
        down_proj_kernel_full = jax.device_get(
            moe_instance.kernel_down_proj_EFD.value)

        # 3. Flatten for iteration
        flat_x = x_input.reshape(-1, self.D)
        flat_weights = weights.reshape(-1, self.K)
        flat_indices = indices.reshape(-1, self.K)

        expected_output = np.zeros_like(flat_x)

        # 4. Sequential computation per token per selected expert
        for t in range(flat_x.shape[0]):
            token_val = flat_x[t]
            token_accum = np.zeros_like(token_val)

            for k in range(self.K):
                expert_idx = flat_indices[t, k]
                router_weight = flat_weights[t, k]

                # Extract specific expert weights
                w_gating = gating_kernel_full[expert_idx]
                w_up = up_proj_kernel_full[expert_idx]
                w_down = down_proj_kernel_full[expert_idx]

                # Forward pass: Up(x) * SiLU(Gate(x))
                gate_out = np.dot(token_val, w_gating)
                up_out = np.dot(token_val, w_up)

                # SiLU activation
                # silu(x) = x * sigmoid(x)
                silu_out = gate_out * (1 / (1 + np.exp(-gate_out)))

                expert_out = silu_out * up_out

                # Down projection
                down_out = np.dot(expert_out, w_down)

                # Weighted sum
                token_accum += down_out * router_weight

            expected_output[t] = token_accum

        return jnp.array(expected_output, dtype=self.dtype)

    def test_dense_backend_correctness(self):
        """Verifies the DENSE_MAT backend against the sequential ground truth."""
        moe = self._create_moe(MoEBackend.DENSE_MAT)

        with self.mesh:
            actual_output = moe(self.x)

        expected_output = self._compute_ground_truth(moe, self.x)

        # Dense matmul should be very numerically close
        self.assertTrue(
            jnp.allclose(actual_output, expected_output, atol=1e-2, rtol=1e-2),
            "Dense backend output does not match ground truth.")

    def test_sparse_distributed_backend_correctness_ragged_dot(self):
        """
        Verifies the RAGGED_DOT (Sparse) backend with expert parallelism
        against the sequential ground truth.
        """
        # 1. Instantiate MoE with RAGGED_DOT backend
        # TODO: add MoEBackend.FUSED_MOE, MoEBackend.VLLM_MOE
        for backend in [MoEBackend.RAGGED_DOT, MoEBackend.MEGABLX_GMM]:
            moe = self._create_moe(backend)

            # 2. Run Forward Pass (Distributed)
            # The __call__ invokes shard_map internally for RAGGED_DOT
            with self.mesh:
                actual_output = moe(self.x)

            # 3. Compute Ground Truth using the exact same weights
            # (Weights are pulled from the initialized model)
            expected_output = self._compute_ground_truth(moe, self.x)

            # 4. Compare
            # Due to permutation re-ordering and potential float precision diffs
            # in different kernels, we allow a small tolerance.
            diff = jnp.mean(jnp.abs(actual_output - expected_output))

            self.assertTrue(
                jnp.allclose(actual_output,
                             expected_output,
                             atol=5e-2,
                             rtol=5e-2),
                f"Sparse distributed output mismatch for backebd tyoe {backend}. Mean diff: {diff}"
            )

    def test_backend_consistency(self):
        """
        Ensures that if we initialize two models with identical seeds/weights,
        Dense and Sparse backends yield the same result.
        """
        # This test requires careful weight synchronization or fixed seeds.
        # Since nnx.Rngs(key) is deterministic, we just re-use the key.

        # 1. Run Dense
        moe_dense = self._create_moe(MoEBackend.DENSE_MAT)
        with self.mesh:
            out_dense = moe_dense(self.x)

        # 2. Run Sparse
        # We must re-init with same key to get same weights
        moe_sparse = self._create_moe(MoEBackend.MEGABLX_GMM)
        with self.mesh:
            out_sparse = moe_sparse(self.x)

        self.assertTrue(
            jnp.allclose(out_dense, out_sparse, atol=5e-2, rtol=5e-2),
            "Dense and Sparse backends produced different results for identical initialization."
        )
