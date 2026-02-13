# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import random
import time
from typing import List
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import compilation_cache as cc
from jax._src import test_util as jtu
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tpu_inference.logger import init_logger
from tpu_inference.offload.tpu_offload_connector import LoadSpec, SaveSpec
from tpu_inference.offload.tpu_offload_connector import \
    TPUOffloadConnector as CPUOffloadingConnector
from tpu_inference.offload.tpu_offload_connector import (
    TPUOffloadConnectorMetadata, TPUReqMeta)
from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

_DEFAULT_BLOCK_SIZE = 128


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


class TestTPUOffloadConnectorWorker(jtu.JaxTestCase):
    """Test the save functionality of the TPUOffloadConnectorWorker."""

    def setUp(self):
        super().setUp()
        self.vllm_config = MockVllmConfig(block_size=_DEFAULT_BLOCK_SIZE)
        self.num_layers = 64
        self.num_blocks = 128
        self.num_cpu_chunks = 128
        self.block_size = self.vllm_config.cache_config.block_size
        num_devices = len(list(jax.devices()))
        self.num_heads = num_devices
        self.head_size = 128
        self.mesh = self.create_mesh((1, num_devices), ("data", "model"))
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
        # Destroy references explicitly
        if hasattr(self, 'connector'):
            del self.connector

        # Force JAX to release memory
        cc.reset_cache()
        jax.clear_caches()

        # Force Python GC
        gc.collect()

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
            return jax.sharding.Mesh(
                device_array, axis_names, None)
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

    # @parameterized.named_parameters(
    #     dict(testcase_name="_zero_blocks", num_blocks=0, expected_buckets=[]),
    #     dict(testcase_name="_one_block", num_blocks=1, expected_buckets=[1]),
    #     dict(testcase_name="_five_blocks",
    #          num_blocks=5,
    #          expected_buckets=[4, 1]),
    #     dict(testcase_name="_sixteen_blocks",
    #          num_blocks=16,
    #          expected_buckets=[16]),
    #     dict(testcase_name="_seventeen_blocks",
    #          num_blocks=17,
    #          expected_buckets=[16, 1]),
    #     dict(testcase_name="_twenty_three_blocks",
    #          num_blocks=23,
    #          expected_buckets=[16, 4, 2, 1]),
    #     dict(testcase_name="_thirty_two_blocks",
    #          num_blocks=32,
    #          expected_buckets=[16, 16]),
    #     dict(testcase_name="_large_number_blocks",
    #          num_blocks=100,
    #          expected_buckets=[16, 16, 16, 16, 16, 16, 4]),
    # )
    # @mock.patch(
    #     "tpu_inference.offload.tpu_offload_connector.BLOCK_SIZE_BUCKETS",
    #     [1, 2, 4, 8, 16],
    # )
    # def test_decompose_into_buckets(self, num_blocks: int,
    #                                 expected_buckets: List[int]):
    #     """
    #     Tests the _decompose_into_buckets function for correct greedy decomposition.
    #     fix BLOCK_SIZE_BUCKETS = [1, 2, 4, 8, 16]
    #     """
    #     connector = self._create_connector()
    #     worker = connector.connector_worker
    #     chunks_list = worker._decompose_into_buckets(num_blocks)
    #     self.assertEqual(chunks_list, expected_buckets)
    #     logger.info(
    #         f"Decomposition for {num_blocks} blocks into {expected_buckets}.")

    # @parameterized.named_parameters(
    #     dict(testcase_name="_jax", swap_op_type="jax"),
    #     dict(testcase_name="_pallas", swap_op_type="pallas"),
    #     dict(testcase_name="_jax_copy", swap_op_type="jax_copy"),
    # )
    # def test_precompile_run_success(self, swap_op_type: str):
    #     """
    #     Tests that _precompile_kv_swap_operations runs without errors and
    #     modifies the cache content.
    #     """
    #     connector = self._create_connector(swap_op_type,
    #                                        use_precompiled_swap_ops=True)

    #     worker = connector.connector_worker

    #     # Keep a copy of the original cache content on the host
    #     original_cache_host = [
    #         np.array(cache) for cache in worker.runner.kv_caches
    #     ]

    #     worker._precompile_kv_swap_operations()

    #     # Fetch the new cache content to the host
    #     new_cache_host = [np.array(cache) for cache in worker.runner.kv_caches]
    #     self.assertTrue(
    #         all(
    #             np.array_equal(orig, new)
    #             for orig, new in zip(original_cache_host, new_cache_host)),
    #         "Cache content should not have changed after precompilation.",
    #     )

    @parameterized.named_parameters(
        # dict(
        #     testcase_name="_single_block",
        #     num_blocks_to_save=1,
        #     num_requests=1,
        # ),
        # dict(
        #     testcase_name="_multi_requests_single_block",
        #     num_blocks_to_save=1,
        #     num_requests=6,
        # ),
        # dict(
        #     testcase_name="_multi_blocks",
        #     num_blocks_to_save=5,
        #     num_requests=1,
        # ),
        # dict(
        #     testcase_name="_multi_requests_multi_blocks",
        #     num_blocks_to_save=5,
        #     num_requests=6,
        # ),
        dict(
            testcase_name="_multi_requests_multi_blocks",
            num_blocks_to_save=16,
            num_requests=6,
        ),
        # dict(
        #     testcase_name="_multi_blocks_with_compile_jax",
        #     num_blocks_to_save=5,
        #     num_requests=1,
        #     use_precompiled_swap_ops=True,
        # ),
        # dict(
        #     testcase_name="_multi_requests_single_block_with_compile_jax",
        #     num_blocks_to_save=1,
        #     num_requests=6,
        #     use_precompiled_swap_ops=True,
        # ),
        # dict(
        #     testcase_name="_multi_requests_multi_blocks_with_compile_jax",
        #     num_blocks_to_save=5,
        #     num_requests=6,
        #     use_precompiled_swap_ops=True,
        # ),
        # dict(
        #     testcase_name="_multi_blocks_with_compile_jax_copy",
        #     num_blocks_to_save=5,
        #     num_requests=1,
        #     use_precompiled_swap_ops=True,
        #     swap_op_type="jax_copy",
        # ),
        # dict(
        #     testcase_name="_multi_requests_multi_blocks_with_compile_jax_copy",
        #     num_blocks_to_save=5,
        #     num_requests=6,
        #     use_precompiled_swap_ops=True,
        #     swap_op_type="jax_copy",
        # ),
        # dict(
        #     testcase_name="_multi_blocks_with_compile_pallas",
        #     num_blocks_to_save=5,
        #     num_requests=1,
        #     use_precompiled_swap_ops=True,
        #     swap_op_type="pallas",
        # ),
        # dict(
        #     testcase_name="_multi_requests_multi_blocks_with_compile_pallas",
        #     num_blocks_to_save=5,
        #     num_requests=6,
        #     use_precompiled_swap_ops=True,
        #     swap_op_type="pallas",
        # ),
        # dict(
        #     testcase_name="_final_save",
        #     num_blocks_to_save=1,
        #     num_requests=1,
        #     is_final_save=True,
        #     skip_save=False,
        # ),
        # dict(
        #     testcase_name="_final_skip_save",
        #     num_blocks_to_save=0,
        #     num_requests=1,
        #     is_final_save=True,
        #     skip_save=True,
        # ),
    )
    def test_tpu_connector_save(
        self,
        num_blocks_to_save: int,
        num_requests: int = 1,
        is_final_save: bool = False,
        skip_save: bool = False,
        use_precompiled_swap_ops: bool = False,
        swap_op_type: str = "jax",
    ):
        total_num_blocks_to_save = num_blocks_to_save * num_requests
        if total_num_blocks_to_save > self.num_blocks or total_num_blocks_to_save > self.num_cpu_chunks:
            self.skipTest(
                f"num_blocks_to_save {total_num_blocks_to_save} exceeds ModelRunner / OffloadConnectorWorker's capacity"
            )

        exp_name = f"{num_blocks_to_save}-{num_requests}-{swap_op_type}"
        profile_dir = f"/mnt/disks/jcgu/code/ullm/offload_tests/microbenchmark_traces/{exp_name}"
        options = jax.profiler.ProfileOptions()
        # default: https://docs.jax.dev/en/latest/profiling.html#general-options
        options.python_tracer_level = 1
        options.host_tracer_level = os.getenv("HOST_TRACER_LEVEL", 2)
        jax.profiler.start_trace(profile_dir, profiler_options=options)
        # Prepare and Execute Save
        all_block_ids = list(range(self.num_blocks))
        all_chunk_ids = list(range(self.num_cpu_chunks))
        src_block_ids = random.sample(all_block_ids, total_num_blocks_to_save)
        dst_chunk_ids = random.sample(all_chunk_ids, total_num_blocks_to_save)

        src_block_ids_split = np.array_split(src_block_ids, num_requests)
        dst_chunk_ids_split = np.array_split(dst_chunk_ids, num_requests)

        requests_meta = []
        for i in range(num_requests):
            req_id = f"save_req_{i}"
            src_blocks = src_block_ids_split[i].tolist()
            dst_chunks = dst_chunk_ids_split[i].tolist()

            num_tokens_to_save_per_req = len(src_blocks) * self.block_size

            save_spec = SaveSpec(
                num_skip_leading_tokens=0,
                num_total_tokens=num_tokens_to_save_per_req,
                is_final_save=is_final_save,
                skip_save=skip_save,
                src_blocks=src_blocks,
                dst_chunks=dst_chunks,
            )

            total_token_ids = list(range(num_tokens_to_save_per_req))

            req_meta = TPUReqMeta(
                req_id=req_id,
                token_ids=total_token_ids,
                local_block_ids=src_blocks,
                save_spec=save_spec,
            )
            requests_meta.append(req_meta)

        logger.info(f"Starting test_tpu_connector_save with: "
                    f"num_blocks_to_save={num_blocks_to_save}, "
                    f"num_requests={num_requests}, "
                    f"is_final_save={is_final_save}, "
                    f"skip_save={skip_save}, "
                    f"use_precompiled_swap_ops={use_precompiled_swap_ops}, "
                    f"swap_op_type={swap_op_type};")

        connector_metadata = TPUOffloadConnectorMetadata(
            requests_meta=requests_meta)

        connector = self._create_connector(swap_op_type,
                                           use_precompiled_swap_ops)
        worker = connector.connector_worker
        connector.bind_connector_metadata(connector_metadata)
        logger.info(
            "Connector metadata bound, calling worker.start_save_kv().")
        worker.start_save_kv()
        logger.info("Waiting for all save operations to complete...")
        while worker._pending_save_futures:
            worker._process_completed_saves()
            time.sleep(0.01)

        jax.profiler.stop_trace()
        logger.info("complete profiling")
        # Verification
        logger.info("Starting verification phase.")
        cpu_backend = worker.cpu_backend
        kv_caches = worker.runner.kv_caches

        if skip_save or total_num_blocks_to_save == 0:
            logger.info(" no blocks to save")
            assert cpu_backend.num_saved_cpu_chunks == 0
            # self.assertEmpty(worker.finished_save_reqs)
            self.assertEmpty(worker.offload_stats.data["finished_save_chunks"])
            return

        # verify the saved chunks
        all_req_ids = {f"save_req_{i}" for i in range(num_requests)}
        self.assertSetEqual(
            all_req_ids,
            set(worker.offload_stats.data["finished_save_chunks"].keys()))

        for i in range(num_requests):
            req_id = f"save_req_{i}"
            src_blocks = src_block_ids_split[i].tolist()
            dst_chunks = dst_chunk_ids_split[i].tolist()
            self.assertListEqual(
                dst_chunks,
                worker.offload_stats.data["finished_save_chunks"][req_id])

            for tpu_block_id, cpu_chunk_id in zip(src_blocks, dst_chunks):
                cpu_kv_chunk = cpu_backend.get(cpu_chunk_id)
                for layer_idx in range(self.num_layers):
                    tpu_kv_block = kv_caches[layer_idx][tpu_block_id]
                    if swap_op_type != "jax_copy":
                        # assert cpu_kv_chunk[
                        #     layer_idx].sharding.memory_kind == 'pinned_host'
                        single_layer_cpu = cpu_kv_chunk[layer_idx]
                        if len(single_layer_cpu.shape) == 5:
                            single_layer_cpu = jnp.squeeze(single_layer_cpu, axis=0)
                        self.assertArraysEqual(
                            np.array(tpu_kv_block),
                            np.array(single_layer_cpu))
                    else:
                        cpu_shards_for_layer = cpu_kv_chunk[layer_idx]
                        tpu_shards_for_layer = tpu_kv_block.addressable_shards
                        self.assertEqual(
                            len(cpu_shards_for_layer),
                            len(tpu_shards_for_layer),
                        )
                        for cpu_shard, tpu_shard in zip(
                                cpu_shards_for_layer, tpu_shards_for_layer):
                            # cpu_shard is using SingleDeviceSharding
                            self.assertArraysEqual(np.array(cpu_shard),
                                                   np.array(tpu_shard.data))
        logger.info("Saved data verification completed.")

    # @parameterized.named_parameters(
    #     dict(
    #         testcase_name="_single_block",
    #         num_blocks_to_operate=1,
    #         num_requests=1,
    #     ),
    #     dict(
    #         testcase_name="_multi_requests_single_block",
    #         num_blocks_to_operate=1,
    #         num_requests=4,
    #     ),
    #     dict(
    #         testcase_name="_multi_blocks_compile_jax",
    #         num_blocks_to_operate=5,
    #         num_requests=1,
    #         use_precompiled_swap_ops=True,
    #         swap_op_type="jax",
    #     ),
    #     dict(
    #         testcase_name="_multi_requests_single_block_compile_jax",
    #         num_blocks_to_operate=1,
    #         num_requests=6,
    #         use_precompiled_swap_ops=True,
    #         swap_op_type="jax",
    #     ),
    #     dict(
    #         testcase_name="_multi_requests_multi_blocks_compile_jax",
    #         num_blocks_to_operate=5,
    #         num_requests=6,
    #         use_precompiled_swap_ops=True,
    #         swap_op_type="jax",
    #     ),
    #     dict(
    #         testcase_name="_multi_requests_multi_blocks_compile_jax_copy",
    #         num_blocks_to_operate=5,
    #         num_requests=6,
    #         use_precompiled_swap_ops=True,
    #         swap_op_type="jax_copy",
    #     ),
    #     dict(
    #         testcase_name="_multi_requests_multi_blocks_compile_pallas",
    #         num_blocks_to_operate=5,
    #         num_requests=6,
    #         use_precompiled_swap_ops=True,
    #         swap_op_type="pallas",
    #     ),
    # )
    # def test_tpu_connector_load(
    #     self,
    #     num_blocks_to_operate: int,
    #     num_requests: int = 1,
    #     use_precompiled_swap_ops: bool = False,
    #     swap_op_type: str = "jax",
    # ):
    #     """
    #     This test simulates a scenario where some amount of blocks get
    #     offloaded to cpu cache, and then get loaded into tpu kv cache.
    #     Both swap-out and swap-in are tested.

    #     Steps:
    #     1. Setup:
    #     2. Simulate a save operation
    #     3. Load the data
    #     4. Verification
    #     """
    #     total_num_blocks_to_operate = num_blocks_to_operate * num_requests
    #     if total_num_blocks_to_operate > self.num_blocks or total_num_blocks_to_operate > self.num_cpu_chunks:
    #         self.skipTest(
    #             f"num_blocks_to_save {total_num_blocks_to_operate} exceeds ModelRunner / OffloadConnectorWorker's capacity"
    #         )
    #     # 1. Setup
    #     connector = self._create_connector(swap_op_type,
    #                                        use_precompiled_swap_ops)
    #     worker = connector.connector_worker
    #     # Ground truth cache on TPU
    #     src_kv_cache = worker.runner.kv_caches
    #     # Destination cache on TPU, should be modified by the load operation
    #     dst_kv_cache = [
    #         jax.device_put(jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
    #                        self.device_sharding)
    #         for _ in range(self.num_layers)
    #     ]
    #     jax.block_until_ready(dst_kv_cache)

    #     # 2. Simulate a save operation
    #     all_block_ids = list(range(self.num_blocks))
    #     all_chunk_ids = list(range(self.num_cpu_chunks))
    #     src_block_ids = random.sample(all_block_ids,
    #                                   total_num_blocks_to_operate)
    #     dst_chunk_ids = random.sample(all_chunk_ids,
    #                                   total_num_blocks_to_operate)

    #     src_block_ids_split = np.array_split(src_block_ids, num_requests)
    #     dst_chunk_ids_split = np.array_split(dst_chunk_ids, num_requests)

    #     save_requests_meta = []
    #     for i in range(num_requests):
    #         req_id = f"save_req_{i}"
    #         src_blocks = src_block_ids_split[i].tolist()
    #         dst_chunks = dst_chunk_ids_split[i].tolist()
    #         num_tokens_to_save_per_req = len(src_blocks) * self.block_size

    #         save_spec = SaveSpec(
    #             num_skip_leading_tokens=0,
    #             num_total_tokens=num_tokens_to_save_per_req,
    #             is_final_save=False,
    #             skip_save=False,
    #             src_blocks=src_blocks,
    #             dst_chunks=dst_chunks,
    #         )
    #         total_token_ids = list(range(num_tokens_to_save_per_req))
    #         req_meta = TPUReqMeta(
    #             req_id=req_id,
    #             token_ids=total_token_ids,
    #             local_block_ids=src_blocks,
    #             save_spec=save_spec,
    #         )
    #         save_requests_meta.append(req_meta)

    #     connector_metadata = TPUOffloadConnectorMetadata(
    #         requests_meta=save_requests_meta)
    #     connector.bind_connector_metadata(connector_metadata)
    #     logger.info(
    #         "Connector metadata bound, calling worker.start_save_kv().")
    #     worker.start_save_kv()
    #     logger.info("Waiting for all save operations to complete...")
    #     while worker._pending_save_futures:
    #         worker._process_completed_saves()
    #         time.sleep(0.01)

    #     logger.info("worker save completed.")

    #     # 3. Prepare and Execute Delta Load
    #     worker.runner.kv_caches = dst_kv_cache

    #     load_requests_meta = []
    #     for i in range(num_requests):
    #         req_id = f"load_req_{i}"
    #         src_blocks = src_block_ids_split[i].tolist()
    #         dst_chunks = dst_chunk_ids_split[i].tolist()
    #         num_tokens_to_load_per_req = len(src_blocks) * self.block_size

    #         load_spec = LoadSpec(
    #             num_matched_tokens=num_tokens_to_load_per_req,
    #             dst_blocks=src_blocks,
    #             src_chunks=dst_chunks,
    #             can_load=True,
    #             num_skip_leading_tokens=0,
    #         )
    #         total_token_ids = list(range(num_tokens_to_load_per_req))
    #         req_meta = TPUReqMeta(
    #             req_id=req_id,
    #             token_ids=total_token_ids,
    #             local_block_ids=src_blocks,
    #             load_spec=load_spec,
    #         )
    #         load_requests_meta.append(req_meta)

    #     connector_metadata = TPUOffloadConnectorMetadata(
    #         requests_meta=load_requests_meta)
    #     connector.bind_connector_metadata(connector_metadata)
    #     logger.info("Connector metadata bound, calling start_load_kv.")
    #     worker.start_load_kv(fwd_ctx=None)
    #     jax.block_until_ready(worker.runner.kv_caches)
    #     logger.info("start_load_kv completed and blocked until ready.")

    #     # 4. Verification
    #     # verify the data
    #     dst_kv_cache = worker.runner.kv_caches
    #     for i in range(num_requests):
    #         src_blocks = src_block_ids_split[i].tolist()
    #         for src_block_id in src_blocks:
    #             for layer_idx in range(self.num_layers):
    #                 self.assertArraysEqual(
    #                     np.array(src_kv_cache[layer_idx][src_block_id]),
    #                     np.array(dst_kv_cache[layer_idx][src_block_id]))

    #     # verify the loaded chunks
    #     all_load_req_ids = {f"load_req_{i}" for i in range(num_requests)}
    #     self.assertSetEqual(
    #         all_load_req_ids,
    #         set(worker.offload_stats.data["finished_load_chunks"].keys()))

    #     for i in range(num_requests):
    #         req_id = f"load_req_{i}"
    #         dst_chunks = dst_chunk_ids_split[i].tolist()
    #         self.assertListEqual(
    #             dst_chunks,
    #             worker.offload_stats.data["finished_load_chunks"][req_id])
