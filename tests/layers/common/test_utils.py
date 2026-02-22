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

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.layers.common.utils import (
    general_device_put,
    get_desired_quant_dtype_for_fp8_moe_weights_from_hf_config)


class UtilsTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        devices = jax.local_devices()
        # Create a mesh with available devices.
        # If running on CPU with 1 device, mesh will be (1, 1).
        self.num_devices = len(devices)
        mesh_shape = (self.num_devices, 1)
        self.mesh = Mesh(np.array(devices).reshape(mesh_shape), ('x', 'y'))
        self.sharding = NamedSharding(self.mesh, P('x', 'y'))

    def test_general_device_put_single_tensor(self):
        tensor = jnp.ones((8, 8))
        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            result = general_device_put(tensor, self.sharding)

        self.assertIsInstance(result, jax.Array)
        self.assertEqual(result.sharding, self.sharding)
        self.assertAllClose(result, tensor)

    def test_general_device_put_pytree(self):
        t1 = jnp.ones((8, 8))
        t2 = jnp.zeros((16, 4))
        tree = {'a': t1, 'b': [t2, t1]}

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            result = general_device_put(tree, self.sharding)

        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['a'], jax.Array)
        self.assertIsInstance(result['b'], list)
        self.assertEqual(result['a'].sharding, self.sharding)
        self.assertEqual(result['b'][0].sharding, self.sharding)
        self.assertAllClose(result['a'], t1)

    def test_general_device_put_multihost_single_tensor(self):
        tensor = jnp.ones((8, 8))

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
            with mock.patch('jax.make_array_from_single_device_arrays'
                            ) as mock_make_array:
                mock_make_array.return_value = tensor

                result = general_device_put(tensor, self.sharding)

                mock_make_array.assert_called_once()
                args, _ = mock_make_array.call_args
                self.assertEqual(args[0], tensor.shape)
                self.assertEqual(args[1], self.sharding)
                # Check that x_split contains device_put result
                # Since we didn't mock device_put, it runs on the tensor slice.
                # We can check the length of x_split
                self.assertEqual(len(args[2]), self.num_devices)
                self.assertAllClose(result, tensor)

    def test_general_device_put_multihost_pytree(self):
        t1 = jnp.ones((8, 8))
        tree = [t1, t1]

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
            with mock.patch('jax.make_array_from_single_device_arrays'
                            ) as mock_make_array:
                mock_make_array.return_value = t1

                result = general_device_put(tree, self.sharding)

                self.assertEqual(mock_make_array.call_count, 2)
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)

    def test_get_desired_quant_dtype_success(self):
        """Tests that the correct dtype is returned when config matches 'e4m3'."""
        mock_config = mock.Mock()
        mock_config.model_config.hf_config.quantization_config = {
            "format": "e4m3"
        }

        result = get_desired_quant_dtype_for_fp8_moe_weights_from_hf_config(
            mock_config)

        self.assertEqual(result, jnp.float8_e4m3fn)

    def test_get_desired_quant_dtype_mismatch_format(self):
        """Tests that None is returned if the format is not 'e4m3'."""
        mock_config = mock.Mock()
        mock_config.model_config.hf_config.quantization_config = {
            "format": "int8"
        }

        result = get_desired_quant_dtype_for_fp8_moe_weights_from_hf_config(
            mock_config)

        self.assertIsNone(result)

    def test_get_desired_quant_dtype_missing_config(self):
        """Tests that None is returned if the config structure is missing."""
        mock_config = mock.Mock()
        # Simulate missing quantization_config by raising an AttributeError
        # when accessing hf_config or underlying attributes.
        type(mock_config.model_config).hf_config = mock.PropertyMock(
            side_effect=AttributeError)

        result = get_desired_quant_dtype_for_fp8_moe_weights_from_hf_config(
            mock_config)

        self.assertIsNone(result)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
