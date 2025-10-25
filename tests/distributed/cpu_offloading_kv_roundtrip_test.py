# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized  # Added import
from jax._src import compilation_cache as cc
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tpu_inference.distributed.tpu_connector_local import LoadSpec, SaveSpec
from tpu_inference.distributed.tpu_connector_local import \
    TPUConnector as CPUOffloadingConnector
from tpu_inference.distributed.tpu_connector_local import (
    TPUConnectorMetadata, TPUReqMeta)
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner


class MockTPUModelRunner(TPUModelRunner):
    """A mock TPUModelRunner for testing purposes."""

    def __init__(self, kv_caches: List[jax.Array], mesh: Mesh):
        self.kv_caches = kv_caches
        self.mesh = mesh
        self.model_config = None
        self.sampler = None

    def get_kv_cache_layout(self):
        return "NHD"


class MockVllmConfig:

    def __init__(self):
        self.model_config = self.Model()
        self.cache_config = self.Cache()
        self.kv_transfer_config = self.KVTransfer()

    class Model:
        model = "test-model"

    class Cache:
        block_size = 64

    class KVTransfer:
        kv_ip = "localhost"
        kv_port = 9999


# TODO(jcgu): add into CI tests
class TestCpuOffloadingKVRoundTrip(jtu.JaxTestCase):
    """Test the roundtrip of KV cache in CPU offloading."""

    def setUp(self):
        super().setUp()
        self.vllm_config = MockVllmConfig()
        self.support_pallas_swap_op = jtu.if_cloud_tpu_at_least(2025, 8, 14)

    def tearDown(self):
        super().tearDown()
        # Reset the cache after each test.
        # This can also be achieved by running with JAX_TEST_WITH_PERSISTENT_COMPILATION_CACHE=True
        cc.reset_cache()

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
            self.skipTest(
                "Cannot create mesh. This test must be run on a TPU node.")
            return None

    @parameterized.named_parameters(*[
        dict(
            testcase_name=f"_tp_{s}_swap_op_type_{t}",
            model_axis_size=s,
            swap_op_type=t,
        ) for s in [1, 2, 4, 8] for t in ["pallas", "jax"]
    ])
    def test_tpu_connector_d2h_h2d_roundtrip(self, model_axis_size: int,
                                             swap_op_type: str):
        """
        Tests the full d2h -> save -> load -> h2d KV Cache round trip via TPUConnectorWorker.

        This test simulates the behavior of the TPUConnectorWorker to verify that
        KV cache data can be correctly offloaded from TPU to CPU and then reloaded
        back to the TPU without data corruption.

        Steps:
        1.  Setup:
            - Create a device mesh and sharding configurations.
            - Instantiate a TPUConnector with a worker role.
            - Create mock source and destination KV caches on the TPU.
            - Register a mock TPUModelRunner with the worker.

        2.  Save to CPU (d2h):
            - Simulate a save operation by directly calling the worker's internal
            `_save_blocks_to_cpu` method. This mimics the asynchronous save
            process.
            - This function performs the device-to-host (d2h) swap and stores
            the data in the LocalCPUBackend.

        3.  Verify CPU Cache Content:
            - Generate the expected cache keys for the saved data.
            - Retrieve the data from the LocalCPUBackend using these keys.
            - Compare the retrieved CPU data with the original TPU data to ensure
            the save operation was successful.

        4.  Load from CPU (h2d):
            - Construct the necessary metadata (`TPUConnectorMetadata`) to trigger
            a load operation.
            - Bind this metadata to the connector.
            - Call the worker's `start_load_kv` method. This performs the
            host-to-device (h2d) swap, moving data from the CPU backend to the
            destination TPU cache.

        5.  Verify TPU Reloaded Content:
            - Compare the data in the destination KV cache on the TPU with the
            original source data.
            - A successful match confirms the integrity of the entire
            offload-reload cycle.
        """
        # 0. verify TPU stack
        if swap_op_type == "pallas" and not self.support_pallas_swap_op:
            self.skipTest(f"libtpu version does not support {swap_op_type}")
            return None

        # 1. Setup
        os.environ['TPU_OFFLOADING_SWAP_OP_TYPE'] = swap_op_type
        mesh = self.create_mesh((1, model_axis_size), ("data", "model"))
        if mesh is None:
            return None

        # Instantiate the connector and get the worker
        connector = CPUOffloadingConnector(self.vllm_config,
                                           KVConnectorRole.WORKER)
        assert connector
        worker = connector.connector_worker
        assert worker is not None

        # Define cache properties
        num_layers = 8
        num_blocks = 16
        block_size = self.vllm_config.cache_config.block_size
        num_heads = 8
        head_size = 128
        cache_shape = (num_blocks, block_size, num_heads, 2, head_size)
        cache_dtype = jnp.bfloat16

        # Define shardings
        partition_spec = PartitionSpec(None, None, "model")
        device_sharding = NamedSharding(mesh, partition_spec)

        # Create mock KV caches on TPU
        @functools.partial(jax.jit, out_shardings=device_sharding)
        def create_on_device(key):
            return jax.random.uniform(key,
                                      shape=cache_shape,
                                      dtype=cache_dtype)

        # mock runner's kv cache for swap-out
        source_kv_cache = [
            create_on_device(jax.random.key(0)) for i in range(num_layers)
        ]

        # mock runner's kv cache for swap-in
        dest_kv_cache = [
            jax.device_put(jnp.zeros(cache_shape, dtype=cache_dtype),
                           device_sharding) for i in range(num_layers)
        ]
        jax.block_until_ready(source_kv_cache)
        jax.block_until_ready(dest_kv_cache)

        # Register a mock runner with the worker
        mock_runner = MockTPUModelRunner(kv_caches=source_kv_cache, mesh=mesh)
        worker.register_runner(mock_runner)

        # 2. Save to CPU (d2h)
        req_id = f"req_{model_axis_size}_{swap_op_type}"

        target_block_ids = [2, 4, 6]
        num_target_blocks = len(target_block_ids)
        target_token_ids = list(range(num_target_blocks * block_size))
        save_spec = SaveSpec(skip_leading_tokens=0, is_final_save=True)

        # Manually set the source cache for the save operation
        worker.runner.kv_caches = source_kv_cache
        worker._save_blocks_to_cpu(req_id, target_block_ids, target_token_ids,
                                   save_spec)

        # 3. Verify CPU Cache Content
        keys_generator = worker.token_processor.process_tokens(
            target_token_ids)
        retrieved_chunks = []
        for _, _, key in keys_generator:
            cached_value = worker.cpu_backend.get(key)
            assert cached_value is not None, f"Key {key} not found in CPU cache!"
            assert len(
                cached_value
            ) == num_layers, f"cache_value layer: {len(cached_value)} != {num_layers}"
            # NOTE(jcgu): comment out this assertion since we've reverted back to using SingleDeviceSharding
            # assert cached_value[0].sharding.memory_kind == "pinned_host"
            retrieved_chunks.append(cached_value[0])  # Get first layer

        # Assemble on CPU and compare with original
        assembled_flat_kv_on_cpu = jnp.concatenate(retrieved_chunks, axis=0)
        original_flat_kv_on_tpu = source_kv_cache[0][
            target_block_ids, ...].reshape(-1, *source_kv_cache[0].shape[2:])
        original_flat_kv_on_cpu = jax.device_get(original_flat_kv_on_tpu)
        jax.block_until_ready(assembled_flat_kv_on_cpu)
        jax.block_until_ready(original_flat_kv_on_cpu)
        np.testing.assert_array_equal(np.array(assembled_flat_kv_on_cpu),
                                      np.array(original_flat_kv_on_cpu))

        # 4. Load from CPU (h2d)
        # Set the destination cache for the load operation
        worker.runner.kv_caches = dest_kv_cache

        # Prepare metadata to trigger the load
        load_spec = LoadSpec(num_matched_tokens=len(target_token_ids),
                             can_load=True,
                             is_full_prefix_hit=False)
        req_meta = TPUReqMeta(req_id=req_id,
                              token_ids=target_token_ids,
                              local_block_ids=target_block_ids,
                              load_spec=load_spec)
        connector_metadata = TPUConnectorMetadata(requests_meta=[req_meta])

        # Bind metadata and execute load
        connector.bind_connector_metadata(connector_metadata)
        worker.start_load_kv(fwd_ctx=None)
        jax.block_until_ready(worker.runner.kv_caches)

        # 5. Verify TPU Reloaded Content
        for i in range(num_layers):
            self.assertArraysEqual(
                source_kv_cache[i][target_block_ids, ...],
                worker.runner.kv_caches[i][target_block_ids, ...])
