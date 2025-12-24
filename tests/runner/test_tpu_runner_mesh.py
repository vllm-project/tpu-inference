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
"""Unit tests for TPUModelRunner mesh initialization."""
import os
from unittest.mock import Mock, patch

import pytest

from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestTPUModelRunnerMeshInit:
    """Test suite for TPUModelRunner._init_mesh and related methods."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig with sharding configuration."""
        config = Mock()
        config.sharding_config = Mock()
        config.sharding_config.model_dp_size = 4
        config.sharding_config.attn_dp_size = 2
        config.sharding_config.expert_size = 1
        config.sharding_config.tp_size = 8
        config.sharding_config.device_indexes = None
        config.sharding_config.total_dp_size = 4
        return config

    @pytest.fixture
    def mock_devices(self):
        """Create mock JAX devices."""
        devices = [Mock(id=i) for i in range(64)]
        return devices

    @pytest.fixture
    def runner_instance(self, mock_vllm_config, mock_devices):
        """Create a minimal TPUModelRunner-like object for testing."""
        # Create a minimal object that has the necessary attributes
        runner = Mock(spec=TPUModelRunner)
        runner.vllm_config = mock_vllm_config
        runner.devices = mock_devices
        runner.mesh = None

        # Bind the actual methods to test (methods don't take sharding_strategy param)
        runner._init_mesh = lambda: TPUModelRunner._init_mesh(runner)
        runner._create_new_model_mesh = lambda: TPUModelRunner._create_new_model_mesh(
            runner)
        runner._create_2d_mesh = lambda: TPUModelRunner._create_2d_mesh(runner)
        runner._create_single_slice_mesh = lambda: TPUModelRunner._create_single_slice_mesh(
            runner)
        runner._create_multi_slice_mesh = lambda ns: TPUModelRunner._create_multi_slice_mesh(
            runner, ns)

        return runner

    def test_init_mesh_2d_model_without_device_order(self, runner_instance,
                                                     mock_vllm_config):
        """Test 2d mesh creation without enforced device order."""
        with patch.dict(os.environ, {'NEW_MODEL_DESIGN': ''}), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh') as mock_make_mesh, \
             patch('tpu_inference.runner.tpu_runner.logger'):

            mock_mesh = Mock()
            mock_make_mesh.return_value = mock_mesh

            runner_instance._init_mesh()

            mock_make_mesh.assert_called_once()
            call_args = mock_make_mesh.call_args

            # Verify mesh_shape
            assert call_args[0][0] == (4, 8)  # (model_dp_size, tp_size)
            # Verify axis_names
            assert call_args[0][1] == ("data", "model")
            # Verify devices
            assert call_args[1]['devices'] == runner_instance.devices

            assert runner_instance.mesh == mock_mesh

    def test_init_mesh_2d_model_with_device_order(self, runner_instance,
                                                  mock_vllm_config):
        """Test 2d mesh creation with enforced device order."""
        mock_vllm_config.sharding_config.device_indexes = [0, 1, 2, 3]

        with patch.dict(os.environ, {'NEW_MODEL_DESIGN': ''}), \
             patch('jax.make_mesh') as mock_jax_mesh, \
             patch('tpu_inference.runner.tpu_runner.logger'):

            mock_mesh = Mock()
            mock_jax_mesh.return_value = mock_mesh

            runner_instance._init_mesh()

            mock_jax_mesh.assert_called_once()
            call_args = mock_jax_mesh.call_args

            # Verify mesh_shape
            assert call_args[0][0] == (4, 8)
            # Verify axis_names
            assert call_args[0][1] == ("data", "model")
            # Verify devices
            assert call_args[1]['devices'] == runner_instance.devices

            assert runner_instance.mesh == mock_mesh

    def test_init_mesh_new_model_single_slice(self, runner_instance,
                                              mock_vllm_config):
        """Test new model mesh creation with single slice."""
        with patch.dict(os.environ, {'NEW_MODEL_DESIGN': '1', 'NUM_SLICES': '1'}), \
             patch('tpu_inference.runner.tpu_runner.mesh_utils') as mock_mesh_utils, \
             patch('jax.sharding.Mesh') as mock_jax_mesh, \
             patch('tpu_inference.runner.tpu_runner.logger'):

            mock_devices_array = Mock()
            mock_mesh_utils.create_device_mesh.return_value = mock_devices_array
            mock_mesh = Mock()
            mock_jax_mesh.return_value = mock_mesh

            runner_instance._init_mesh()

            # Verify create_device_mesh was called
            mock_mesh_utils.create_device_mesh.assert_called_once()
            call_args = mock_mesh_utils.create_device_mesh.call_args

            # Verify mesh_shape: (model_dp_size, attn_dp_size, expert_size, tp_size)
            assert call_args[0][0] == (4, 2, 1, 8)
            assert call_args[0][1] == runner_instance.devices
            assert call_args[1]['allow_split_physical_axes'] is True

            # Verify Mesh was created with correct axis names
            mock_jax_mesh.assert_called_once_with(
                mock_devices_array, ("data", "attn_dp", "expert", "model"))

            assert runner_instance.mesh == mock_mesh

    def test_init_mesh_new_model_multi_slice(self, runner_instance,
                                             mock_vllm_config):
        """Test new model mesh creation with multiple slices."""
        num_slices = 2
        with patch.dict(os.environ, {'NEW_MODEL_DESIGN': '1', 'NUM_SLICES': str(num_slices)}), \
             patch('tpu_inference.runner.tpu_runner.mesh_utils') as mock_mesh_utils, \
             patch('jax.sharding.Mesh') as mock_jax_mesh, \
             patch('tpu_inference.runner.tpu_runner.logger'):

            mock_devices_array = Mock()
            mock_mesh_utils.create_hybrid_device_mesh.return_value = mock_devices_array
            mock_mesh = Mock()
            mock_jax_mesh.return_value = mock_mesh

            runner_instance._init_mesh()

            # Verify create_hybrid_device_mesh was called
            mock_mesh_utils.create_hybrid_device_mesh.assert_called_once()
            call_args = mock_mesh_utils.create_hybrid_device_mesh.call_args

            # Verify intra_node_shape: (dp_inner, attn_dp_size, expert_size, tp_size)
            # dp_inner = model_dp_size // num_slices = 4 // 2 = 2
            assert call_args[1]['mesh_shape'] == (2, 2, 1, 8)
            # Verify outer_node_shape: (num_slices, 1, 1, 1)
            assert call_args[1]['dcn_mesh_shape'] == (2, 1, 1, 1)
            assert call_args[1]['devices'] == runner_instance.devices
            assert call_args[1]['allow_split_physical_axes'] is True

            # Verify Mesh was created with correct axis names
            mock_jax_mesh.assert_called_once_with(
                mock_devices_array, ("data", "attn_dp", "expert", "model"))

            assert runner_instance.mesh == mock_mesh

    @pytest.mark.parametrize("num_slices,expected_dp_inner", [
        (1, 4),
        (2, 2),
        (4, 1),
    ])
    def test_multi_slice_mesh_dp_inner_calculation(self, runner_instance,
                                                   mock_vllm_config,
                                                   num_slices,
                                                   expected_dp_inner):
        """Test dp_inner calculation for various num_slices values."""
        with patch('tpu_inference.runner.tpu_runner.mesh_utils'
                   ) as mock_mesh_utils:
            mock_mesh_utils.create_hybrid_device_mesh.return_value = Mock()

            runner_instance._create_multi_slice_mesh(num_slices)

            call_args = mock_mesh_utils.create_hybrid_device_mesh.call_args
            intra_node_shape = call_args[1]['mesh_shape']

            # First dimension of intra_node_shape should be dp_inner
            assert intra_node_shape[0] == expected_dp_inner
