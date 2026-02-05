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
from jax.sharding import Mesh

from tpu_inference.layers.jax.moe.moe import JaxMoE, Router
from tpu_inference.layers.jax.moe.utils import (MoEBackend,
                                                get_expert_parallelism)

EXPERT_AXIS_NAME = "model"


class TestMoE(unittest.TestCase):

    def setUp(self):
        """Set up a multi-device mesh for testing."""
        devices = jax.devices()
        self.device_count = len(devices)
        if self.device_count < 2:
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
        self.E, self.K = self.device_count, 2  # Num Experts (Total), Experts per Token
        self.moe_intermediate_size = 64
        self.dtype = jnp.bfloat16
        self.key = jax.random.PRNGKey(42)

        # Input data
        with jax.set_mesh(self.mesh):
            self.x = jax.random.normal(self.key, (self.B * self.S, self.D),
                                       dtype=self.dtype)

    def _create_moe(self, backend: MoEBackend):
        """Helper to instantiate the MoE layer within the mesh context."""
        with jax.set_mesh(self.mesh):
            router = Router(
                dtype=self.dtype,
                hidden_size=self.D,
                num_experts=self.E,
                num_experts_per_tok=self.K,
                router_act="softmax",  # Standard softmax routing
                rngs=nnx.Rngs(self.key),
                activation_ffw_td=('data', 'model'),
                ed_sharding=(None, 'model'),
                moe_backend=backend)
            num_expert_parallelism = get_expert_parallelism(
                EXPERT_AXIS_NAME, self.mesh)
            assert num_expert_parallelism == self.device_count
            use_ep = num_expert_parallelism > 1
            assert use_ep

            moe = JaxMoE(
                dtype=self.dtype,
                num_local_experts=self.E,
                hidden_size=self.D,
                intermediate_size_moe=self.moe_intermediate_size,
                hidden_act="silu",
                rngs=nnx.Rngs(self.key),
                router=router,
                mesh=self.mesh,

                # Sharding Specs
                activation_ffw_td=('data', None),
                activation_ffw_ted=('data', None),
                edf_sharding=('model', None, None),  # Expert par on axis 0
                efd_sharding=('model', None, None),
                apply_expert_weight_before_computation=False,
                moe_backend=backend,
                num_experts_per_tok=self.K,
                expert_axis_name='model',
                num_expert_parallelism=num_expert_parallelism)
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

    @unittest.skip(
        "Skipping dense backend correctness test to allow lib versions upgrade."
    )
    def test_dense_backend_correctness(self):
        """Verifies the DENSE_MAT backend against the sequential ground truth."""
        moe = self._create_moe(MoEBackend.DENSE_MAT)

        with jax.set_mesh(self.mesh):
            actual_output = moe(self.x)

        expected_output = self._compute_ground_truth(moe, self.x)

        # Dense matmul should be very numerically close
        self.assertTrue(
            jnp.allclose(actual_output, expected_output, atol=1e-2, rtol=1e-2),
            "Dense backend output does not match ground truth.")

    @unittest.skip(
        "Skipping dense backend correctness test to allow lib versions upgrade."
    )
    def test_sparse_distributed_backend_correctness_ragged_dot(self):
        """
        Verifies the Sparse backends with expert parallelism
        against the sequential ground truth.
        """
        # TODO: add MoEBackend.FUSED_MOE, MoEBackend.GMM_TP/GMM_EP
        backend = MoEBackend.MEGABLX_GMM
        moe = self._create_moe(backend)

        # 2. Run Forward Pass (Distributed)
        with jax.set_mesh(self.mesh):
            actual_output = moe(self.x)

        # Compute Ground Truth using the exact same weights
        expected_output = self._compute_ground_truth(moe, self.x)

        diff = jnp.mean(jnp.abs(actual_output - expected_output))

        self.assertTrue(
            jnp.allclose(actual_output, expected_output, atol=5e-2, rtol=5e-2),
            f"Sparse distributed output mismatch for backebd tyoe {backend}. Mean diff: {diff}"
        )

    @unittest.skip(
        "Skipping dense backend correctness test to allow lib versions upgrade."
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
        with jax.set_mesh(self.mesh):
            # self.x = jax.random.normal(self.key, (self.B * self.S, self.D),
            #                        dtype=self.dtype)
            out_dense = moe_dense(self.x)

        # 2. Run Sparse
        # We must re-init with same key to get same weights
        moe_sparse = self._create_moe(MoEBackend.MEGABLX_GMM)
        with jax.set_mesh(self.mesh):
            out_sparse = moe_sparse(self.x)

        self.assertTrue(
            jnp.allclose(out_dense, out_sparse, atol=5e-2, rtol=5e-2),
            "Dense and Sparse backends produced different results for identical initialization."
        )
