import functools
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tpu_inference.distributed.tpu_connector_local import (
    LoadSpec, SaveSpec, TPUConnector, TPUConnectorMetadata, TPUReqMeta)
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner


def create_default_mesh(axis_shapes, axis_names):
    """Creates a JAX device mesh with the default device order."""
    try:
        devices = jax.devices()
        print(f"Found {len(devices)} devices. Using default device order.")
        device_array = np.asarray(devices).reshape(axis_shapes)
        return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
        print("No TPU devices found. This script must be run on a TPU node.")
        return None


class MockTPUModelRunner(TPUModelRunner):
    """A mock TPUModelRunner for testing purposes."""

    def __init__(self, kv_caches: List[jax.Array], mesh: Mesh):
        self.kv_caches = kv_caches
        self.mesh = mesh
        self.model_config = None
        self.sampler = None

    def get_kv_cache_layout(self):
        return "NHD"


def test_tpu_connector_d2h_h2d_roundtrip():
    """
    Tests the full d2h -> save -> load -> h2d round trip via TPUConnector.

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
    # 1. Setup
    axis_shapes = (1, 8)
    axis_names = ("data", "model")
    mesh = create_default_mesh(axis_shapes, axis_names)
    if mesh is None:
        print("Skipping test: No TPU devices found.")
        return

    # Mock VllmConfig for the connector
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

    vllm_config = MockVllmConfig()

    # Instantiate the connector and get the worker
    connector = TPUConnector(vllm_config, KVConnectorRole.WORKER)
    worker = connector.connector_worker
    assert worker is not None

    # Define cache properties
    num_layers = 8
    num_blocks = 16
    block_size = vllm_config.cache_config.block_size
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
        return jax.random.uniform(key, shape=cache_shape, dtype=cache_dtype)

    source_kv_cache = [
        create_on_device(jax.random.PRNGKey(0)) for i in range(num_layers)
    ]
    dest_kv_cache = [
        jax.device_put(jnp.zeros(cache_shape, dtype=cache_dtype),
                       device_sharding) for i in range(num_layers)
    ]
    jax.block_until_ready(source_kv_cache)
    jax.block_until_ready(dest_kv_cache)
    # Register a mock runner with the worker
    mock_runner = MockTPUModelRunner(kv_caches=dest_kv_cache, mesh=mesh)
    worker.register_runner(mock_runner)

    # 2. Save to CPU (d2h)
    print("\n--- Simulating Save to CPU (d2h) ---")
    req_id = "test_req_1"

    target_block_ids = [2, 4, 6]
    num_target_blocks = len(target_block_ids)
    target_token_ids = list(range(num_target_blocks * block_size))
    save_spec = SaveSpec(skip_leading_tokens=0, is_final_save=True)

    # Manually set the source cache for the save operation
    worker.runner.kv_caches = source_kv_cache
    worker._save_blocks_to_cpu(req_id, target_block_ids, target_token_ids,
                               save_spec)
    # 3. Verify CPU Cache Content
    print("\n--- Verifying CPU Cache Content ---")
    keys_generator = worker.token_processor.process_tokens(target_token_ids)
    retrieved_chunks = []
    for _, _, key in keys_generator:
        cached_value = worker.cpu_backend.get(key)
        assert cached_value is not None, f"Key {key} not found in CPU cache!"
        assert len(
            cached_value
        ) == num_layers, f"cache_value layer: {len(cached_value)} != {num_layers}"
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
    print("✅ SUCCESS: Data in CPU cache matches original TPU data.")

    # 4. Load from CPU (h2d)
    print("\n--- Simulating Load from CPU (h2d) ---")
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
    print("\n--- Verifying Reloaded TPU Content ---")
    # Compare the destination cache with the original source cache
    np.testing.assert_array_equal(
        np.array(source_kv_cache[0][target_block_ids,
                                    ...].addressable_shards[0].data),
        np.array(dest_kv_cache[0][target_block_ids,
                                  ...].addressable_shards[0].data))

    print(
        "\n✅ SUCCESS: Data matches after d2h -> save -> load -> h2d round trip."
    )


if __name__ == "__main__":
    test_tpu_connector_d2h_h2d_roundtrip()
