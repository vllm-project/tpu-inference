# SPDX-License-Identifier: Apache-2.0

import functools
import os
import random
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import compilation_cache as cc
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tpu_inference.distributed.local_cpu_backend import LocalCPUBackend
from tpu_inference.distributed.tpu_connector_local import LoadSpec, SaveSpec
from tpu_inference.distributed.tpu_connector_local import \
    TPUConnector as CPUOffloadingConnector
from tpu_inference.distributed.tpu_connector_local import (
    TPUConnectorMetadata, TPUReqMeta)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner

logger = init_logger(__name__)


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

    def __init__(self, block_size=64):
        self.model_config = self.Model()
        self.cache_config = self.Cache(block_size)
        self.kv_transfer_config = self.KVTransfer()

    class Model:
        model = "test-model"

    class Cache:

        def __init__(self, block_size):
            self.block_size = block_size

    class KVTransfer:
        kv_ip = "localhost"
        kv_port = 9999


class TestCpuOffloadingDeltaLoad(jtu.JaxTestCase):
    """Test the delta load functionality of the TPUConnectorWorker."""

    def setUp(self):
        super().setUp()
        # Clean the singleton backend instance before each test
        LocalCPUBackend._instance = None
        LocalCPUBackend._initialized = False
        self.vllm_config = MockVllmConfig(block_size=64)

    def tearDown(self):
        super().tearDown()
        cc.reset_cache()

    def create_mesh(self, axis_shapes, axis_names):
        """Creates a JAX device mesh with the default device order."""
        try:
            num_required_devices = np.prod(axis_shapes)
            devices = np.array(jax.devices())
            if len(devices) < num_required_devices:
                self.skipTest(
                    f"Not enough devices to create mesh of shape {axis_shapes}."
                )
            device_array = devices[:num_required_devices].reshape(axis_shapes)
            return jax.sharding.Mesh(device_array, axis_names)
        except RuntimeError:
            self.skipTest("Cannot create mesh. Must be run on a TPU node.")
            return None

    @parameterized.named_parameters(
        dict(
            testcase_name="_full_load",
            num_matched_blocks=2,
        ),
        dict(
            testcase_name="_delta_load_non_aligned_start",
            num_computed_tokens=100,
            num_matched_blocks=5,
        ),
        dict(
            testcase_name="_delta_load_block_aligned_start",
            num_computed_blocks=2,
            num_matched_blocks=6,
        ),
        dict(
            testcase_name="_delta_load_single_block",
            num_computed_tokens=20,
            num_matched_blocks=1,
        ),
        dict(
            testcase_name="_delta_load_partial_block_only",
            num_computed_tokens=20,
            num_matched_blocks=1,
        ),
        dict(
            testcase_name="_no_load_match_equals_computed",
            num_computed_blocks=3,
            num_matched_blocks=3,
        ),
        dict(
            testcase_name="_no_load_match_less_than_computed",
            num_computed_blocks=4,
            num_matched_blocks=2,
        ),
        dict(
            testcase_name="_full_prefix_hit",
            num_matched_blocks=8,
            is_full_prefix_hit=True,
        ),
    )
    def test_tpu_connector_load(
        self,
        num_matched_blocks: int,
        num_computed_tokens: int = 0,
        num_computed_blocks: int = 0,
        is_full_prefix_hit: bool = False,
    ):
        """
        Tests that the TPUConnectorWorker correctly loads only the delta of
        the KV cache when a prefix is already computed by vLLM.

        This test simulates a scenario where vLLM has already computed a certain
        number of tokens (prefix) and the TPUConnectorWorker needs to load
        only the remaining "delta" of the KV cache from the CPU backend.

        Steps:
        1.  Setup:
            - Create a device mesh and sharding configurations.
            - Instantiate a TPUConnector with a worker role.
            - Create mock source (ground truth) and destination KV caches on the TPU.
            - Register a mock TPUModelRunner with the worker.

        2.  Populate CPU Cache:
            - Simulate a save operation to the CPU backend for the "matched" prefix.
            - This represents the KV cache state on the CPU that corresponds to
            the tokens already processed by vLLM.

        3.  Prepare and Execute Delta Load:
            - Calculate the number of tokens to load (the delta).
            - Construct the necessary metadata (`TPUConnectorMetadata`) and `LoadSpec`
            to trigger a delta load operation, skipping the already computed tokens.
            - Bind this metadata to the connector and call the worker's `start_load_kv`
            method to perform the host-to-device (h2d) load for the delta tokens.

        4.  Verification:
            - If no tokens were expected to be loaded, assert that the destination
            KV cache remains zero.
            - Otherwise, extract the expected delta data from the source KV cache
            and the actually loaded data from the destination KV cache.
            - Compare these two sets of data to ensure the loaded delta is correct.
            - Assert that the parts of the destination cache that should not have
            been touched remain zero.
        """
        block_size = self.vllm_config.cache_config.block_size
        num_matched_tokens = num_matched_blocks * block_size

        # If num_computed_blocks is provided, it takes precedence for block-aligned cases.
        if num_computed_blocks > 0:
            num_computed_tokens = num_computed_blocks * block_size

        logger.info(
            f"Starting test_tpu_connector_load with num_computed_tokens={num_computed_tokens}, num_matched_tokens={num_matched_tokens}, is_full_prefix_hit={is_full_prefix_hit}"
        )
        # 1. Setup
        os.environ["TPU_OFFLOADING_SWAP_OP_TYPE"] = "jax"
        logger.info("TPU_OFFLOADING_SWAP_OP_TYPE set to 'jax'")
        mesh = self.create_mesh((1, 8), ("data", "model"))
        if mesh is None:
            logger.info("Mesh creation skipped, returning from test.")
            return

        logger.info("Mesh created successfully.")
        connector = CPUOffloadingConnector(self.vllm_config,
                                           KVConnectorRole.WORKER)
        worker = connector.connector_worker
        assert worker is not None
        logger.info("TPUConnector and worker initialized.")

        # Define cache properties
        num_layers = 2
        num_blocks = 16
        block_size = self.vllm_config.cache_config.block_size
        num_heads = 8
        head_size = 128
        cache_shape = (num_blocks, block_size, num_heads, 2, head_size)
        cache_dtype = jnp.bfloat16
        partition_spec = PartitionSpec(None, None, "model")
        device_sharding = NamedSharding(mesh, partition_spec)
        logger.info(
            f"Cache properties defined: num_layers={num_layers}, num_blocks={num_blocks}, block_size={block_size}, cache_shape={cache_shape}"
        )

        @functools.partial(jax.jit, out_shardings=device_sharding)
        def create_on_device(key):
            return jax.random.uniform(key,
                                      shape=cache_shape,
                                      dtype=cache_dtype)

        # Ground truth cache on TPU
        source_kv_cache = [
            create_on_device(jax.random.key(i)) for i in range(num_layers)
        ]
        # Destination cache on TPU, should be modified by the load operation
        dest_kv_cache = [
            jax.device_put(jnp.zeros(cache_shape, dtype=cache_dtype),
                           device_sharding) for _ in range(num_layers)
        ]
        jax.block_until_ready(source_kv_cache)
        jax.block_until_ready(dest_kv_cache)
        logger.info(
            "Source and destination KV caches created and blocked until ready."
        )

        mock_runner = MockTPUModelRunner(kv_caches=source_kv_cache, mesh=mesh)
        worker.register_runner(mock_runner)
        logger.info("MockTPUModelRunner registered with worker.")

        # 2. Populate CPU Cache
        # Save the part of the source cache that represents the "matched" prefix
        if num_matched_tokens > 0:
            logger.info(
                f"Populating CPU cache with {num_matched_tokens} matched tokens."
            )
            tokens_to_save = list(range(num_matched_tokens))
            num_blocks_to_save = (num_matched_tokens + block_size -
                                  1) // block_size
            blocks_to_save = list(range(num_blocks_to_save))
            save_spec = SaveSpec(skip_leading_tokens=0)
            worker._save_blocks_to_cpu("save_req", blocks_to_save,
                                       tokens_to_save, save_spec)
            logger.info(
                f"Simulated save operation to CPU for {num_matched_tokens} tokens."
            )
        else:
            logger.info("No matched tokens, skipping CPU cache population.")

        # 3. Prepare and Execute Delta Load
        worker.runner.kv_caches = dest_kv_cache
        num_tokens_to_load = max(0, num_matched_tokens - num_computed_tokens)
        # `num_tokens_to_load` cannot be negative. If `num_computed_tokens`
        # is greater than or equal to `num_matched_tokens`, it means all
        # relevant tokens are already on the TPU, and no new tokens need
        # to be loaded from the CPU backend. In such cases, the value should
        # be clamped to 0.
        logger.info(
            f"Calculated num_tokens_to_load: {num_tokens_to_load} (num_matched_tokens={num_matched_tokens} - num_computed_tokens={num_computed_tokens})"
        )
        if is_full_prefix_hit:
            num_tokens_to_load -= 1
            logger.info(
                f"is_full_prefix_hit is True, adjusted num_tokens_to_load: {num_tokens_to_load}"
            )

        # The scheduler allocates blocks for all matched tokens, regardless of
        # how many we actually need to load.
        if num_matched_tokens > 0:
            num_prefix_blocks = (num_matched_tokens + block_size -
                                 1) // block_size
            # We simulate non contiguous blocks for a request's KV cache by sampling
            # a random set of available blocks. This ensures the load logic
            # can correctly handle fragmented cache allocations.
            if num_prefix_blocks > num_blocks:
                self.skipTest(
                    f"Not enough blocks to allocate for prefix of size {num_matched_tokens}"
                )
            available_blocks = list(range(num_blocks))
            local_block_ids = sorted(
                random.sample(available_blocks, num_prefix_blocks))
        else:
            num_prefix_blocks = 0
            local_block_ids = []

        self.assertLen(local_block_ids, num_prefix_blocks)
        logger.info(f"Allocated local_block_ids: {local_block_ids}")

        # The call to start_load_kv should happen regardless of whether there are
        # tokens to load, as the connector should handle the no-op case.
        load_spec = LoadSpec(
            num_matched_tokens=num_matched_tokens,
            can_load=True,
            is_full_prefix_hit=is_full_prefix_hit,
            skip_leading_tokens=num_computed_tokens,
        )
        logger.info(f"LoadSpec created: {load_spec}")
        # The worker needs the full token list to generate keys correctly
        full_token_ids = list(range(num_matched_tokens))
        req_meta = TPUReqMeta(
            req_id="load_req",
            token_ids=full_token_ids,
            local_block_ids=local_block_ids,
            load_spec=load_spec,
        )
        connector_metadata = TPUConnectorMetadata(requests_meta=[req_meta])
        connector.bind_connector_metadata(connector_metadata)
        logger.info("Connector metadata bound, calling start_load_kv.")
        worker.start_load_kv(fwd_ctx=None)
        jax.block_until_ready(worker.runner.kv_caches)
        logger.info("start_load_kv completed and blocked until ready.")

        # 4. Verification
        logger.info("Starting verification phase.")

        if num_tokens_to_load <= 0:
            logger.info(
                "num_tokens_to_load is 0 or less, asserting nothing was loaded."
            )
            # Assert that the entire destination cache remains untouched (all zeros).
            for i in range(num_layers):
                self.assertArraysEqual(
                    dest_kv_cache[i],
                    jnp.zeros(cache_shape, dtype=cache_dtype),
                )
            logger.info("Assertion passed: Destination KV cache is all zeros.")
            return

        # Helper to flatten and extract a token range from a cache given a block map
        def get_token_slice(kv_cache, start_token, num_tokens, block_map):
            if num_tokens <= 0:
                return jnp.empty((0, *kv_cache.shape[2:]),
                                 dtype=kv_cache.dtype)
            start_block_logical = start_token // block_size
            start_offset = start_token % block_size
            end_token = start_token + num_tokens
            end_block_logical = (end_token + block_size - 1) // block_size

            if end_block_logical > len(block_map):
                raise ValueError(
                    f"Not enough blocks in block_map to satisfy token range. "
                    f"Need {end_block_logical} blocks, but map has {len(block_map)}."
                )

            physical_blocks_to_gather = [
                block_map[i]
                for i in range(start_block_logical, end_block_logical)
            ]

            flat_cache = kv_cache[physical_blocks_to_gather,
                                  ...].reshape(-1, *kv_cache.shape[2:])
            return flat_cache[start_offset:start_offset + num_tokens, ...]

        # Define the block maps for source and destination
        source_block_map = list(range(num_blocks))
        dest_block_map = local_block_ids

        # Get the ground truth data from the source cache
        expected_data_from_source_tpu = [
            get_token_slice(
                source_kv_cache[i],
                start_token=num_computed_tokens,
                num_tokens=num_tokens_to_load,
                block_map=source_block_map,
            ) for i in range(num_layers)
        ]
        logger.info(
            f"Extracted expected data from source cache. Shape of first layer: {expected_data_from_source_tpu[0].shape}"
        )

        # Get the data that was actually loaded into the destination cache
        loaded_data_on_dest_tpu = [
            get_token_slice(
                worker.runner.kv_caches[i],
                start_token=num_computed_tokens,
                num_tokens=num_tokens_to_load,
                block_map=dest_block_map,
            ) for i in range(num_layers)
        ]
        logger.info(
            f"Extracted loaded data from destination cache. Shape of first layer: {loaded_data_on_dest_tpu[0].shape}"
        )

        # Assert that the loaded delta is correct. This works for no-load cases too.
        for i in range(num_layers):
            self.assertArraysEqual(np.array(expected_data_from_source_tpu[i]),
                                   np.array(loaded_data_on_dest_tpu[i]))
        logger.info("Assertion passed: Loaded delta matches expected data.")

        # Assert that blocks not in local_block_ids are still zero
        untouched_blocks = sorted(
            list(set(range(num_blocks)) - set(local_block_ids)))
        logger.info(
            f"Asserting that {len(untouched_blocks)} untouched blocks are still zero."
        )
        if untouched_blocks:
            for i in range(num_layers):
                zero_slice = worker.runner.kv_caches[i][untouched_blocks, ...]
                self.assertTrue(jnp.all(zero_slice == 0))
                expected_zeros = jnp.zeros(
                    (len(untouched_blocks), *cache_shape[1:]),
                    dtype=cache_dtype)
                self.assertArraysEqual(np.array(zero_slice),
                                       np.array(expected_zeros))
        logger.info("Assertion passed: Untouched blocks are zero.")
        logger.info(
            "Test test_tpu_connector_delta_load completed successfully.")
