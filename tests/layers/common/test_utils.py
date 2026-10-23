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
from tpu_inference.layers.common.utils import (general_device_put,
                                               truncate_sharded_tensor)


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

    def test_general_device_put_multihost_passes_dtype(self):
        # A host that owns no shard never calls the callback, so the dtype
        # must be passed explicitly.
        tensor = jnp.ones((8, 8), dtype=jnp.bfloat16)

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
            with mock.patch('jax.make_array_from_callback') as mock_make_array:
                mock_make_array.return_value = tensor

                general_device_put(tensor, self.sharding)

                _, kwargs = mock_make_array.call_args
                self.assertEqual(kwargs['dtype'], jnp.bfloat16)

    def test_general_device_put_multihost_single_tensor(self):
        tensor = jnp.ones((8, 8))

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
            with mock.patch('jax.make_array_from_callback') as mock_make_array:
                mock_make_array.return_value = tensor

                result = general_device_put(tensor, self.sharding)

                mock_make_array.assert_called_once()
                args, _ = mock_make_array.call_args
                self.assertEqual(args[0], tensor.shape)
                self.assertEqual(args[1], self.sharding)
                # Check that x_split contains device_put result
                # Since we didn't mock device_put, it runs on the tensor slice.
                # We can check the length of x_split
                self.assertAllClose(result, tensor)

    def test_general_device_put_multihost_pytree(self):
        t1 = jnp.ones((8, 8))
        tree = [t1, t1]

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', 'ray'):
            with mock.patch('jax.make_array_from_callback') as mock_make_array:
                mock_make_array.return_value = t1

                result = general_device_put(tree, self.sharding)

                self.assertEqual(mock_make_array.call_count, 2)
                self.assertIsInstance(result, list)
                self.assertEqual(len(result), 2)

    def test_general_device_put_multihost_non_ray_sharding(self):
        tensor = jnp.ones((8, 8))
        mock_sharding = mock.MagicMock(spec=NamedSharding)
        mock_sharding.is_fully_addressable = False
        mock_sharding.has_addressable_devices = True

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            with mock.patch('jax.make_array_from_callback') as mock_make_array:
                mock_make_array.return_value = tensor
                result = general_device_put(tensor, mock_sharding)

                mock_make_array.assert_called_once()
                args, kwargs = mock_make_array.call_args
                self.assertEqual(args[0], tensor.shape)
                self.assertEqual(args[1], mock_sharding)
                self.assertEqual(kwargs['dtype'], tensor.dtype)
                self.assertAllClose(result, tensor)

    def test_general_device_put_multihost_prng_key(self):
        key = jax.random.key(0)
        mock_sharding = mock.MagicMock(spec=NamedSharding)
        mock_sharding.is_fully_addressable = False
        mock_sharding.has_addressable_devices = True

        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            with mock.patch('jax.make_array_from_callback') as mock_make_array:
                mock_make_array.side_effect = (
                    lambda shape, sharding, cb, **kwargs: cb(()))
                result = general_device_put(key, mock_sharding)

                mock_make_array.assert_called_once()
                self.assertEqual(result.shape, ())

        # Also verify sharded 1D PRNGKeyArray slicing and single np.asarray caching.
        keys = jax.random.split(key, 4)
        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            with mock.patch('jax.make_array_from_callback') as mock_make_array:
                mock_make_array.side_effect = (
                    lambda shape, sharding, cb, **kwargs: cb((slice(0, 2), )))
                result_sharded = general_device_put(keys, mock_sharding)
                self.assertEqual(result_sharded.shape, (2, ))

    def test_general_device_put_multihost_caches_host_numpy_view(self):
        cpu_dev = jax.local_devices(backend='cpu')[0]
        tensor = jax.device_put(
            jnp.arange(64, dtype=jnp.float32).reshape(8, 8), cpu_dev)
        mock_sharding = mock.MagicMock(spec=NamedSharding)
        mock_sharding.is_fully_addressable = False
        mock_sharding.has_addressable_devices = True

        shard_indices = [
            (slice(0, 2), slice(None)),
            (slice(2, 4), slice(None)),
            (slice(4, 6), slice(None)),
            (slice(6, 8), slice(None)),
        ]
        real_asarray = np.asarray
        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            with mock.patch(
                    'tpu_inference.layers.common.utils.np.asarray',
                    wraps=real_asarray) as mock_asarray:
                with mock.patch(
                        'jax.make_array_from_callback') as mock_make_array:
                    mock_make_array.side_effect = (
                        lambda shape, sharding, cb, **kwargs: [
                            cb(idx) for idx in shard_indices
                        ])
                    shards = general_device_put(tensor, mock_sharding)

                    self.assertEqual(mock_asarray.call_count, 1)
                    self.assertEqual(len(shards), 4)
                    for i, shard in enumerate(shards):
                        self.assertAllClose(shard, tensor[shard_indices[i]])

    def test_general_device_put_multihost_skips_numpy_copy_on_tpu_tensor(self):
        tensor = jnp.arange(64, dtype=jnp.float32).reshape(8, 8)
        mock_tpu_dev = mock.MagicMock()
        mock_tpu_dev.platform = 'tpu'
        mock_sharding = mock.MagicMock(spec=NamedSharding)
        mock_sharding.is_fully_addressable = False
        mock_sharding.has_addressable_devices = True

        shard_indices = [
            (slice(0, 4), slice(None)),
            (slice(4, 8), slice(None)),
        ]
        with mock.patch.object(envs, 'TPU_MULTIHOST_BACKEND', ''):
            with mock.patch.object(
                    jax.Array, 'devices', return_value={mock_tpu_dev}):
                with mock.patch(
                        'tpu_inference.layers.common.utils.np.asarray'
                ) as mock_asarray:
                    with mock.patch(
                            'jax.make_array_from_callback') as mock_make_array:
                        mock_make_array.side_effect = (
                            lambda shape, sharding, cb, **kwargs: [
                                cb(idx) for idx in shard_indices
                            ])
                        shards = general_device_put(tensor, mock_sharding)

                        mock_asarray.assert_not_called()
                        self.assertEqual(len(shards), 2)
                        for i, shard in enumerate(shards):
                            self.assertAllClose(shard, tensor[shard_indices[i]])

    def test_truncate_sharded_tensor(self):
        # Shape: (2, 20), n_shards=4, so each shard has size 5 on the last dim
        t = jnp.arange(40).reshape(2, 20)
        n_shards = 4
        truncate_size = 3

        result = truncate_sharded_tensor(t, truncate_size, n_shards)

        # Expected: reshape to (2, 4, 5), truncate to (2, 4, 3), reshape to (2, 12)
        expected = t.reshape(2, 4, 5)[:, :, :truncate_size].reshape(2, 12)

        self.assertEqual(result.shape, (2, 12))
        self.assertAllClose(result, expected)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
