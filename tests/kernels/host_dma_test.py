# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.kernels.dma.host_dma import d2h_dma, h2d_dma

DATA_LOCATION = Literal["device", "host"]


@jtu.with_config(jax_numpy_dtype_promotion='strict')
class HostHbmDmaTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.if_cloud_tpu_at_least(2025, 8, 14):
            return self.skipTest(
                "libtpu version does not support DMA host-hbm")

    def tearDown(self):
        super().tearDown()
        jax.clear_caches()

    def create_mesh(self, axis_shapes, axis_names):
        """Creates a JAX device mesh with the default device order."""
        try:
            num_required_devices = np.prod(axis_shapes)
            devices = np.array(jax.devices())
            if len(devices) < num_required_devices:
                self.skipTest("Not enough devices to create mesh of shape"
                              f" {axis_shapes}. Have {len(devices)}, need"
                              f" {num_required_devices}.")
            device_array = devices[:num_required_devices].reshape(axis_shapes)
            return jax.sharding.Mesh(device_array, axis_names)
        except RuntimeError:
            self.skip(
                "Cannot create mesh. This test must be run on a TPU node.")
            return None

    def create_sharded_array(self, model_axis_size: int,
                             init_location: DATA_LOCATION):
        """Creates a sharded JAX array for testing.

        Args:
            model_axis_size: The size of the model parallelism axis.
            init_location: Where to initialize the array, either "device" or "host".

        Returns:
            A tuple containing the created sharded array, the device sharding spec,
            and the host sharding spec.
        """
        axis_shapes = (1, model_axis_size)
        axis_names = ("data", "model")
        mesh = self.create_mesh(axis_shapes, axis_names)
        if mesh is None:
            return None

        partition_spec = PartitionSpec(None, None, "model")
        device_sharding = NamedSharding(mesh,
                                        partition_spec,
                                        memory_kind="device")
        host_sharding = NamedSharding(mesh,
                                      partition_spec,
                                      memory_kind="pinned_host")

        data_shape = (2, 16, model_axis_size, 2, 128)
        dtype = jnp.bfloat16

        data = jax.device_put(
            jax.random.uniform(jax.random.key(0),
                               shape=data_shape,
                               dtype=dtype),
            device_sharding if init_location == "device" else host_sharding,
        )
        jax.block_until_ready(data)
        return data, device_sharding, host_sharding

    @parameterized.named_parameters([
        dict(testcase_name=f"_model_axis_size_{s}", model_axis_size=s)
        for s in [1, 2, 4, 8]
    ])
    def test_d2h_dma(self, model_axis_size: int):
        """Tests the d2h DMA transfer for various model parallelism sizes."""
        # 1. Create original data on the device
        res = self.create_sharded_array(model_axis_size, "device")
        if res is None:
            return
        original_device_data, device_sharding, host_sharding = res

        # 2. Test Device-to-Host (d2h) DMA
        host_data = d2h_dma(original_device_data, device_sharding,
                            host_sharding)
        jax.block_until_ready(host_data)
        assert host_data.sharding.memory_kind == "pinned_host"

        # 3. Verification
        assert host_data.sharding == host_sharding
        self.assertArraysEqual(original_device_data, host_data)

    @parameterized.named_parameters([
        dict(testcase_name=f"_model_axis_size_{s}", model_axis_size=s)
        for s in [1, 2, 4, 8]
    ])
    def test_h2d_dma(self, model_axis_size: int):
        """Tests the h2d DMA transfer for various model parallelism sizes."""
        # 1. Create original data on the host
        res = self.create_sharded_array(model_axis_size, "host")
        if res is None:
            return
        original_host_data, device_sharding, host_sharding = res

        # 2. Test Host-to-Device (h2d) DMA
        device_data = h2d_dma(original_host_data, host_sharding,
                              device_sharding)
        jax.block_until_ready(device_data)
        assert device_data.sharding.memory_kind == "device"

        # 3. Verification
        assert device_data.sharding == device_sharding
        self.assertArraysEqual(original_host_data, device_data)

    @parameterized.named_parameters([
        dict(testcase_name=f"_model_axis_size_{s}", model_axis_size=s)
        for s in [1, 2, 4, 8]
    ])
    def test_d2h_h2d_dma_roundtrip(self, model_axis_size: int):
        """
        Tests the d2h -> h2d DMA roundtrip for various model parallelism sizes.

        This test verifies that:
        1. Data can be correctly transferred from sharded device memory to sharded
        host memory using `d2h_dma`.
        2. Data can be correctly transferred back from sharded host memory to
        sharded device memory using `h2d_dma`.
        3. The data remains identical after the full roundtrip.
        """
        # 1. Setup: Create sharded array based on the model axis size
        res = self.create_sharded_array(model_axis_size, "device")
        if res is None:
            return
        original_device_data, device_sharding, host_sharding = res

        # 2. Test Device-to-Host (d2h) DMA
        host_data = d2h_dma(original_device_data, device_sharding,
                            host_sharding)
        jax.block_until_ready(host_data)
        assert host_data.sharding.memory_kind == "pinned_host"

        # 3. Verification for d2h
        assert host_data.sharding == host_sharding
        self.assertArraysEqual(original_device_data, host_data)

        # 4. Test Host-to-Device (h2d) DMA
        reloaded_device_data = h2d_dma(host_data, host_sharding,
                                       device_sharding)
        jax.block_until_ready(reloaded_device_data)
        assert reloaded_device_data.sharding.memory_kind == "device"

        # 5. Verification for h2d
        assert reloaded_device_data.sharding == device_sharding
        self.assertArraysEqual(host_data, reloaded_device_data)

        # 6. Final roundtrip verification
        self.assertArraysEqual(original_device_data, reloaded_device_data)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
