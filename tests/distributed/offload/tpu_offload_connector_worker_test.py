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

from tpu_inference.distributed.offload.tpu_offload_connector import (LoadSpec,
                                                                     SaveSpec)
from tpu_inference.distributed.offload.tpu_offload_connector import \
    TPUOffloadConnector as CPUOffloadingConnector
from tpu_inference.distributed.offload.tpu_offload_connector import (
    TPUOffloadConnectorMetadata, TPUReqMeta)
from tpu_inference.logger import init_logger
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

_DEFAULT_BLOCK_SIZE = 64


class MockTPUModelRunner(TPUModelRunner):
    """A mock TPUModelRunner for testing purposes."""

    def __init__(self, kv_caches: List[jax.Array], mesh: Mesh):
        self.kv_caches = kv_caches
        self.mesh = mesh
        self.model_config = None
        self.sampler = None
        self.devices = jax.devices()

    def get_kv_cache_layout(self):
        return "NHD"


class MockVllmConfig:

    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
        self.model_config = self.Model()
        self.cache_config = self.Cache(block_size)
        self.kv_transfer_config = self.KVTransferConfig()

    class Model:
        model = "test-model"

    class Cache:

        def __init__(self, block_size):
            self.block_size = block_size

    class KVTransferConfig:
        ip = "ip"
        port = 1234


class TestCpuOffloadingSave(jtu.JaxTestCase):
    """Test the save functionality of the TPUOffloadConnectorWorker."""

    def setUp(self):
        super().setUp()
        self.vllm_config = MockVllmConfig(block_size=_DEFAULT_BLOCK_SIZE)
        self.num_layers = 2
        self.num_blocks = 24
        self.num_cpu_chunks = 24
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

    def _create_connector(self,
                          swap_op_type: str = "jax",
                          use_precompiled_swap_ops: bool = False):
        os.environ["TPU_OFFLOAD_SWAP_OP_TYPE"] = swap_op_type
        os.environ[
            "TPU_OFFLOAD_SKIP_JAX_PRECOMPILE"] = "0" if use_precompiled_swap_ops else "1"
        os.environ["TPU_OFFLOAD_NUM_CPU_CHUNKS"] = str(self.num_cpu_chunks)

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
        connector = self._create_connector(use_precompiled_swap_ops="0")
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
        connector = self._create_connector(swap_op_type,
                                           use_precompiled_swap_ops="0")

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

    @parameterized.named_parameters(
        dict(
            testcase_name="_regular_multi_block_save",
            num_blocks_to_save=5,
        ),
        dict(
            testcase_name="_regular_multi_block_save_with_compile_jax",
            num_blocks_to_save=5,
            use_precompiled_swap_ops=True,
        ),
        dict(
            testcase_name="_regular_multi_block_save_with_compile_pallas",
            num_blocks_to_save=5,
            use_precompiled_swap_ops=True,
            swap_op_type="pallas",
        ),
        dict(
            testcase_name="_final_save",
            num_blocks_to_save=1,
            is_final_save=True,
            skip_save=False,
        ),
        dict(
            testcase_name="_final_skip_save",
            num_blocks_to_save=0,
            is_final_save=True,
            skip_save=True,
        ),
    )
    def test_tpu_connector_save(
        self,
        num_blocks_to_save: int,
        is_final_save: bool = False,
        skip_save: bool = False,
        use_precompiled_swap_ops: bool = False,
        swap_op_type: str = "jax",
    ):
        if num_blocks_to_save > self.num_blocks or num_blocks_to_save > self.num_cpu_chunks:
            self.skipTest(
                f"num_blocks_to_save {num_blocks_to_save} exceeds ModelRunner / OffloadConnectorWorker's capacity"
            )

        # Prepare and Execute Save
        all_block_ids = list(range(self.num_blocks))
        all_chunk_ids = list(range(self.num_cpu_chunks))
        src_block_ids = random.sample(all_block_ids, num_blocks_to_save)
        dst_chunk_ids = random.sample(all_chunk_ids, num_blocks_to_save)
        num_tokens_to_save = num_blocks_to_save * self.block_size
        num_total_tokens = num_tokens_to_save
        save_spec = SaveSpec(
            num_skip_leading_tokens=0,
            num_total_tokens=num_total_tokens,
            is_final_save=is_final_save,
            skip_save=skip_save,
            src_blocks=src_block_ids,
            dst_chunks=dst_chunk_ids,
        )

        logger.info(f"Starting test_tpu_connector_save with: "
                    f"num_blocks_to_save={num_blocks_to_save}, "
                    f"is_final_save={is_final_save}, "
                    f"skip_save={skip_save}, "
                    f"use_precompiled_swap_ops={use_precompiled_swap_ops}, "
                    f"swap_op_type={swap_op_type};"
                    f"Swapspec: {save_spec}")

        total_token_ids = list(range(num_total_tokens))

        req_id = "save_req"
        req_meta = TPUReqMeta(
            req_id=req_id,
            token_ids=total_token_ids,
            local_block_ids=src_block_ids,
            save_spec=save_spec,
        )

        connector_metadata = TPUOffloadConnectorMetadata(
            requests_meta=[req_meta])

        connector = self._create_connector(swap_op_type,
                                           use_precompiled_swap_ops)
        worker = connector.connector_worker
        connector.bind_connector_metadata(connector_metadata)
        logger.info(
            "Connector metadata bound, calling worker.wait_for_save().")
        worker.wait_for_save()
        logger.info("worker.wait_for_save() completed.")

        # Verification
        logger.info("Starting verification phase.")
        cpu_backend = worker.cpu_backend
        kv_caches = worker.runner.kv_caches

        if skip_save or num_tokens_to_save == 0:
            logger.info(" no blocks to save")
            assert cpu_backend.num_saved_cpu_chunks == 0
            self.assertEmpty(worker.finished_save_reqs)
            self.assertEmpty(worker.offload_stats.data["finished_save_chunks"])
            return

        # verify the saved chunks
        assert req_id in worker.offload_stats.data["finished_save_chunks"]
        assert dst_chunk_ids == worker.offload_stats.data[
            "finished_save_chunks"][req_id]

        for tpu_block_id, cpu_chunk_id in zip(src_block_ids, dst_chunk_ids):
            cpu_kv_chunk = cpu_backend.get(cpu_chunk_id)
            for layer_idx in range(self.num_layers):
                tpu_kv_block = kv_caches[layer_idx][tpu_block_id]
                self.assertArraysEqual(np.array(tpu_kv_block),
                                       np.array(cpu_kv_chunk[layer_idx]))

        logger.info("Saved data verification completed.")

        if is_final_save:
            finished_saves, _ = worker.get_finished()
            logger.info(
                f"is_final_save is True. Finished requests: {finished_saves}")
            self.assertIn(req_id, finished_saves)

    @parameterized.named_parameters(
        dict(
            testcase_name="_single_block_",
            num_blocks_to_operate=1,
        ),
        dict(
            testcase_name="_multi_blocks_compile_jax",
            num_blocks_to_operate=5,
            use_precompiled_swap_ops=True,
            swap_op_type="jax",
        ),
        dict(
            testcase_name="_multi_blocks_compile_pallas",
            num_blocks_to_operate=5,
            use_precompiled_swap_ops=True,
            swap_op_type="pallas",
        ),
    )
    def test_tpu_connector_load(
        self,
        num_blocks_to_operate: int,
        use_precompiled_swap_ops: bool = False,
        swap_op_type: str = "jax",
    ):
        """
        This test simulates a scenario where some amount of blocks get
        offloaded to cpu cache, and then get loaded into tpu kv cache.
        Both swap-out and swap-in are tested.

        Steps:
        1. Setup:
        2. Simulate a save operation
        3. Load the data
        4. Verification
        """
        if num_blocks_to_operate > self.num_blocks or num_blocks_to_operate > self.num_cpu_chunks:
            self.skipTest(
                f"num_blocks_to_save {num_blocks_to_operate} exceeds ModelRunner / OffloadConnectorWorker's capacity"
            )
        # 1. Setup
        connector = self._create_connector(swap_op_type,
                                           use_precompiled_swap_ops)
        worker = connector.connector_worker
        # Ground truth cache on TPU
        src_kv_cache = worker.runner.kv_caches
        # Destination cache on TPU, should be modified by the load operation
        dst_kv_cache = [
            jax.device_put(jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                           self.device_sharding)
            for _ in range(self.num_layers)
        ]
        jax.block_until_ready(dst_kv_cache)

        # Prepare
        all_block_ids = list(range(self.num_blocks))
        all_chunk_ids = list(range(self.num_cpu_chunks))
        src_block_ids = random.sample(all_block_ids, num_blocks_to_operate)
        dst_chunk_ids = random.sample(all_chunk_ids, num_blocks_to_operate)
        num_tokens_to_save = num_blocks_to_operate * self.block_size
        num_total_tokens = num_tokens_to_save
        save_spec = SaveSpec(
            num_skip_leading_tokens=0,
            num_total_tokens=num_tokens_to_save,
            is_final_save=False,
            skip_save=False,
            src_blocks=src_block_ids,
            dst_chunks=dst_chunk_ids,
        )
        total_token_ids = list(range(num_total_tokens))
        req_id = "save_req"
        req_meta = TPUReqMeta(
            req_id=req_id,
            token_ids=total_token_ids,
            local_block_ids=src_block_ids,
            save_spec=save_spec,
        )
        connector_metadata = TPUOffloadConnectorMetadata(
            requests_meta=[req_meta])
        connector.bind_connector_metadata(connector_metadata)
        logger.info(
            "Connector metadata bound, calling worker.wait_for_save().")
        worker.wait_for_save()
        logger.info("worker.wait_for_save() completed.")

        # 3. Prepare and Execute Delta Load
        new_req_id = "load_req"
        worker.runner.kv_caches = dst_kv_cache
        load_spec = LoadSpec(
            num_matched_tokens=num_tokens_to_save,
            dst_blocks=src_block_ids,
            src_chunks=dst_chunk_ids,
            can_load=True,
            num_skip_leading_tokens=0,
        )
        req_meta = TPUReqMeta(
            req_id="load_req",
            token_ids=total_token_ids,
            local_block_ids=src_block_ids,
            load_spec=load_spec,
        )
        connector_metadata = TPUOffloadConnectorMetadata(
            requests_meta=[req_meta])
        connector.bind_connector_metadata(connector_metadata)
        logger.info("Connector metadata bound, calling start_load_kv.")
        worker.start_load_kv(fwd_ctx=None)
        jax.block_until_ready(worker.runner.kv_caches)
        logger.info("start_load_kv completed and blocked until ready.")

        # verify the data
        # we will donate the original kv_cache ref
        dst_kv_cache = worker.runner.kv_caches
        for src_block_id in src_block_ids:
            for layer_idx in range(self.num_layers):
                self.assertArraysEqual(
                    np.array(src_kv_cache[layer_idx][src_block_id]),
                    np.array(dst_kv_cache[layer_idx][src_block_id]))

        # verify the saved chunks
        assert new_req_id in worker.offload_stats.data["finished_load_chunks"]
        assert dst_chunk_ids == worker.offload_stats.data[
            "finished_load_chunks"][new_req_id]
