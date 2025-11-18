# SPDX-License-Identifier: Apache-2.0

import functools
import os
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import compilation_cache as cc
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tpu_inference.distributed.offload.cpu_backend import LocalCPUBackend
from tpu_inference.distributed.offload.tpu_offload_connector import \
    TPUOffloadConnector as CPUOffloadingConnector
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_jax_runner import TPUModelRunner

logger = init_logger(__name__)

_DEFAULT_BLOCK_SIZE = 64


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

    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
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


class TestHostOffloadingPrecompile(jtu.JaxTestCase):
    """Test the host offloading precompilation and related functionalities."""

    def setUp(self):
        super().setUp()
        self.vllm_config = MockVllmConfig(block_size=_DEFAULT_BLOCK_SIZE)
        self.num_layers = 2
        self.num_blocks = 128  # Increased for larger tests
        self.block_size = self.vllm_config.cache_config.block_size
        self.num_heads = 8
        self.head_size = 128
        self.mesh = self.create_mesh((1, 8), ("data", "model"))
        if self.mesh is None:
            self.skipTest("Cannot create mesh. Must be run on a TPU node.")
            return

        # Define cache properties
        self.cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_heads,
            2,
            self.head_size,
        )
        self.cache_dtype = jnp.bfloat16
        partition_spec = PartitionSpec(None, None, "model")
        self.device_sharding = NamedSharding(self.mesh, partition_spec)

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
            return None

    def _create_connector(self, swap_op_type: str = "jax"):
        # Clean the singleton backend instance before each test
        LocalCPUBackend._instance = None
        LocalCPUBackend._initialized = False

        os.environ["TPU_OFFLOAD_SWAP_OP_TYPE"] = swap_op_type
        connector = CPUOffloadingConnector(self.vllm_config,
                                           KVConnectorRole.WORKER)
        worker = connector.connector_worker
        assert worker is not None

        @functools.partial(jax.jit, out_shardings=self.device_sharding)
        def create_on_device(key):
            return jax.random.uniform(key,
                                      shape=self.cache_shape,
                                      dtype=self.cache_dtype)

        source_kv_cache = [
            create_on_device(jax.random.key(i)) for i in range(self.num_layers)
        ]
        jax.block_until_ready(source_kv_cache)

        mock_runner = MockTPUModelRunner(kv_caches=source_kv_cache,
                                         mesh=self.mesh)
        worker.register_runner(mock_runner)
        return connector

    @parameterized.named_parameters(
        dict(testcase_name="_zero_blocks", num_blocks=0, expected_buckets=[]),
        dict(testcase_name="_one_block", num_blocks=1, expected_buckets=[1]),
        dict(testcase_name="_five_blocks",
             num_blocks=5,
             expected_buckets=[4, 1]),
        dict(testcase_name="_sixteen_blocks",
             num_blocks=16,
             expected_buckets=[16]),
        dict(testcase_name="_seventeen_blocks",
             num_blocks=17,
             expected_buckets=[16, 1]),
        dict(testcase_name="_twenty_three_blocks",
             num_blocks=23,
             expected_buckets=[16, 4, 2, 1]),
        dict(testcase_name="_thirty_two_blocks",
             num_blocks=32,
             expected_buckets=[16, 16]),
        dict(testcase_name="_large_number_blocks",
             num_blocks=100,
             expected_buckets=[16, 16, 16, 16, 16, 16, 4]),
    )
    def test_decompose_into_buckets(self, num_blocks: int,
                                    expected_buckets: List[int]):
        """
        Tests the _decompose_into_buckets function for correct greedy decomposition.
        """
        os.environ["TPU_OFFLOAD_SKIP_JAX_PRECOMPILE"] = "0"
        connector = self._create_connector()
        worker = connector.connector_worker
        self.assertEqual(worker._decompose_into_buckets(num_blocks),
                         expected_buckets)
        logger.info(
            f"Decomposition for {num_blocks} blocks: {worker._decompose_into_buckets(num_blocks)} matched expected: {expected_buckets}"
        )

    @parameterized.named_parameters(
        dict(testcase_name="_jax", swap_op_type="jax"),
        dict(testcase_name="_pallas", swap_op_type="pallas"),
    )
    def test_precompile_run_success(self, swap_op_type: str):
        """
        Tests that _precompile_kv_swap_operations runs without errors and
        modifies the cache content.
        """
        # Unset skip flag to allow precompilation to run
        os.environ["TPU_OFFLOAD_SKIP_JAX_PRECOMPILE"] = "0"
        connector = self._create_connector(swap_op_type=swap_op_type)
        worker = connector.connector_worker

        # Keep a copy of the original cache content on the host
        original_cache_host = [
            np.array(cache) for cache in worker.runner.kv_caches
        ]

        worker._precompile_kv_swap_operations()

        # Fetch the new cache content to the host
        new_cache_host = [np.array(cache) for cache in worker.runner.kv_caches]
        self.assertTrue(
            all(
                np.array_equal(orig, new)
                for orig, new in zip(original_cache_host, new_cache_host)),
            "Cache content should not have changed after precompilation.",
        )
