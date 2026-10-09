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

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
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
        config.sharding_config.attn_dp_expert_size = 1
        config.sharding_config.expert_size = 1
        config.sharding_config.tp_size = 8
        config.sharding_config.device_indexes = None
        config.sharding_config.total_dp_size = 4
        config.sharding_config.decode_cp_size = 1
        config.sharding_config.prefill_cp_size = 1
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
             patch('jax.sharding.Mesh') as mock_jax_mesh, \
             patch('tpu_inference.runner.tpu_runner.logger'):

            mock_mesh = Mock()
            mock_jax_mesh.return_value = mock_mesh

            runner_instance._init_mesh()

            mock_jax_mesh.assert_called_once()
            call_args = mock_jax_mesh.call_args

            # Verify mesh_shape
            assert call_args[0][0].shape == (4, 8)
            # Verify axis_names
            assert call_args[0][1] == ("data", "model")
            # Verify devices
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

            # Verify mesh_shape: (model_dp_size, attn_dp_size, attn_dp_expert_size, expert_size, tp_size, dcp_size, pcp_size)
            assert call_args[0][0] == (4, 2, 1, 1, 8, 1, 1)
            assert call_args[0][1] == runner_instance.devices
            assert call_args[1]['allow_split_physical_axes'] is True

            # Verify Mesh was created with correct axis names
            mock_jax_mesh.assert_called_once_with(mock_devices_array,
                                                  MESH_AXIS_NAMES)

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

            # Verify intra_node_shape: (dp_inner, attn_dp_size, attn_dp_expert_size, expert_size, tp_size, dcp_size, pcp_size)
            # dp_inner = model_dp_size // num_slices = 4 // 2 = 2
            assert call_args[1]['mesh_shape'] == (2, 2, 1, 1, 8, 1, 1)
            # Verify outer_node_shape: (num_slices, 1, 1, 1, 1, 1, 1)
            assert call_args[1]['dcn_mesh_shape'] == (2, 1, 1, 1, 1, 1, 1)
            assert call_args[1]['devices'] == runner_instance.devices
            assert call_args[1]['allow_split_physical_axes'] is True

            # Verify Mesh was created with correct axis names
            mock_jax_mesh.assert_called_once_with(mock_devices_array,
                                                  MESH_AXIS_NAMES)

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


def _v7x_devices(num_z):
    """Fake v7x devices of a 2x2xZ slice, in jax.devices() order."""
    from types import SimpleNamespace
    devices = []
    for z in range(num_z):
        for y in range(2):
            for x in range(2):
                for core in range(2):
                    devices.append(
                        SimpleNamespace(id=len(devices),
                                        coords=[x, y, z],
                                        core_on_chip=core))
    return devices


def _hops(a, b):
    return sum(abs(p - q) for p, q in zip(a.coords, b.coords))


class TestAttnDpRingDeviceMesh:
    """attn_dp_ring_device_mesh (TPU_MESH_ATTN_DP_RING)."""

    @pytest.mark.parametrize("num_z", [1, 2, 4])
    def test_layout(self, num_z):
        from tpu_inference.utils import attn_dp_ring_device_mesh
        devices = _v7x_devices(num_z)
        dp = 2 * num_z
        mesh_shape = (1, dp, 1, 1, 4, 1, 1)
        grid = attn_dp_ring_device_mesh(mesh_shape, MESH_AXIS_NAMES, devices)
        assert grid.shape == mesh_shape
        grid = grid.reshape(dp, 4)
        assert sorted(d.id for d in grid.flat) == list(range(len(devices)))
        for r in range(dp):
            row = grid[r]
            # Model indices 2k, 2k+1 are the two cores of one chip, and the
            # two chips of the model group are linked.
            for k in (0, 2):
                assert row[k].coords == row[k + 1].coords
                assert {row[k].core_on_chip, row[k + 1].core_on_chip} == {0, 1}
            assert _hops(row[0], row[2]) == 1
            # A model group stays on one host (one z layer).
            assert len({d.coords[2] for d in row}) == 1
        for m in range(4):
            col = grid[:, m]
            # Every step of each attn_dp ring, wrap included, is one link.
            for r in range(dp):
                assert _hops(col[r], col[(r + 1) % dp]) == 1
            assert len({d.core_on_chip for d in col}) == 1

    def test_production_v7x_32(self):
        from tpu_inference.utils import attn_dp_ring_device_mesh
        grid = attn_dp_ring_device_mesh((1, 8, 1, 1, 4, 1, 1), MESH_AXIS_NAMES,
                                        _v7x_devices(4))
        col = [tuple(d.coords) for d in grid.reshape(8, 4)[:, 0]]
        assert col == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (1, 0, 3),
                       (1, 0, 2), (1, 0, 1), (1, 0, 0)]

    @pytest.mark.parametrize(
        "mesh_shape,num_z",
        [
            ((1, 4, 1, 1, 8, 1, 1), 4),  # model=8
            ((2, 4, 1, 1, 4, 1, 1), 4),  # a data axis in use
            ((1, 2, 1, 1, 4, 1, 1), 2),  # attn_dp=2 on a 2x2x2 slice
            ((1, 3, 1, 1, 4, 1, 1), 2),  # odd attn_dp
        ])
    def test_unsupported_mesh(self, mesh_shape, num_z):
        from tpu_inference.utils import attn_dp_ring_device_mesh
        with pytest.raises(ValueError):
            attn_dp_ring_device_mesh(mesh_shape, MESH_AXIS_NAMES,
                                     _v7x_devices(num_z))

    def test_one_host_of_a_larger_slice(self):
        # Under PP a stage gets one host's devices, here the z=2 layer.
        from tpu_inference.utils import attn_dp_ring_device_mesh
        grid = attn_dp_ring_device_mesh((1, 2, 1, 1, 4, 1, 1), MESH_AXIS_NAMES,
                                        _v7x_devices(4)[16:24])
        assert [tuple(d.coords)
                for d in grid.reshape(2, 4)[:, 0]] == [(0, 0, 2), (1, 0, 2)]

    def test_unsupported_devices(self):
        from tpu_inference.utils import attn_dp_ring_device_mesh

        # A 4x2x1 grid of chips (not 2x2xZ).
        devices = _v7x_devices(1) + _v7x_devices(1)
        for d in devices[8:]:
            d.coords = [d.coords[0] + 2, d.coords[1], 0]
        with pytest.raises(ValueError, match="2x2x2 slice"):
            attn_dp_ring_device_mesh((1, 4, 1, 1, 4, 1, 1), MESH_AXIS_NAMES,
                                     devices)
        # No topology information at all.
        with pytest.raises(ValueError, match="coords"):
            attn_dp_ring_device_mesh((1, 2, 1, 1, 4, 1, 1), MESH_AXIS_NAMES,
                                     [Mock(spec=["id"])] * 8)

    def _runner(self, attn_dp, tp, devices):
        config = Mock()
        sc = config.sharding_config
        sc.model_dp_size, sc.attn_dp_size, sc.attn_dp_expert_size = 1, attn_dp, 1
        sc.expert_size, sc.tp_size = 1, tp
        sc.decode_cp_size, sc.prefill_cp_size = 1, 1
        runner = Mock(spec=TPUModelRunner)
        runner.vllm_config = config
        runner.devices = devices
        return runner

    def test_runner_uses_ring_layout(self):
        runner = self._runner(8, 4, _v7x_devices(4))
        with patch.dict(os.environ, {'TPU_MESH_ATTN_DP_RING': '1'}), \
             patch('tpu_inference.runner.tpu_runner.mesh_utils') as mesh_utils, \
             patch('tpu_inference.runner.tpu_runner.logger'):
            arr = TPUModelRunner._create_single_slice_mesh(runner)
        mesh_utils.create_device_mesh.assert_not_called()
        assert arr.shape == (1, 8, 1, 1, 4, 1, 1)
        assert tuple(arr.reshape(8, 4)[1, 0].coords) == (0, 0, 1)

    def test_runner_falls_back(self):
        runner = self._runner(4, 8, _v7x_devices(4))
        with patch.dict(os.environ, {'TPU_MESH_ATTN_DP_RING': '1'}), \
             patch('tpu_inference.runner.tpu_runner.mesh_utils') as mesh_utils, \
             patch('tpu_inference.runner.tpu_runner.logger') as logger:
            mesh_utils.create_device_mesh.return_value = "default"
            assert TPUModelRunner._create_single_slice_mesh(
                runner) == "default"
        mesh_utils.create_device_mesh.assert_called_once()
        assert "TPU_MESH_ATTN_DP_RING" in logger.warning.call_args[0][0]
