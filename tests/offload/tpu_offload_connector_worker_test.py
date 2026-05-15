# Copyright 2026 Google LLC
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec
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
            return jax.sharding.Mesh(device_array, axis_names, None)
        except RuntimeError:
            return None

    def _create_connector(self, use_precompiled_swap_ops: bool = False):
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
    @mock.patch(
        "tpu_inference.offload.tpu_offload_connector.BLOCK_SIZE_BUCKETS",
        [1, 2, 4, 8, 16],
    )
    def test_decompose_into_buckets(self, num_blocks: int,
                                    expected_buckets: List[int]):
        """
        Tests the _decompose_into_buckets function for correct greedy decomposition.
        fix BLOCK_SIZE_BUCKETS = [1, 2, 4, 8, 16]
        """
        connector = self._create_connector()
        worker = connector.connector_worker
        block_ids = [i for i in range(num_blocks)]
        chunks_list = worker._decompose_into_buckets(block_ids)
        chunk_size_list = [len(x) for x in chunks_list]
        self.assertEqual(chunk_size_list, expected_buckets)
        logger.info(
            f"Decomposition for {num_blocks} blocks into {expected_buckets}.")

    def test_precompile_run_success(self):
        """
        Tests that _precompile_kv_swap_operations runs without errors and
        modifies the cache content.
        """
        connector = self._create_connector(use_precompiled_swap_ops=True)

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
        # Pinned Host Cases
        dict(
            testcase_name="_single_block_pinned",
            num_blocks_to_save=1,
            num_requests=1,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_requests_single_block_pinned",
            num_blocks_to_save=1,
            num_requests=6,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_blocks_pinned",
            num_blocks_to_save=5,
            num_requests=1,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_requests_multi_blocks_pinned",
            num_blocks_to_save=16,
            num_requests=6,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_blocks_with_compile_jax_pinned",
            num_blocks_to_save=5,
            num_requests=1,
            use_precompiled_swap_ops=True,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name=
            "_multi_requests_single_block_with_compile_jax_pinned",
            num_blocks_to_save=1,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name=
            "_multi_requests_multi_blocks_with_compile_jax_pinned",
            num_blocks_to_save=5,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_final_save_pinned",
            num_blocks_to_save=1,
            num_requests=1,
            is_final_save=True,
            skip_save=False,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_final_skip_save_pinned",
            num_blocks_to_save=0,
            num_requests=1,
            is_final_save=True,
            skip_save=True,
            use_unpinned_host=False,
        ),

        # Unpinned Host Cases
        dict(
            testcase_name="_single_block_unpinned",
            num_blocks_to_save=1,
            num_requests=1,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_requests_single_block_unpinned",
            num_blocks_to_save=1,
            num_requests=6,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_blocks_unpinned",
            num_blocks_to_save=5,
            num_requests=1,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_requests_multi_blocks_unpinned",
            num_blocks_to_save=16,
            num_requests=6,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_blocks_with_compile_jax_unpinned",
            num_blocks_to_save=5,
            num_requests=1,
            use_precompiled_swap_ops=True,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name=
            "_multi_requests_single_block_with_compile_jax_unpinned",
            num_blocks_to_save=1,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name=
            "_multi_requests_multi_blocks_with_compile_jax_unpinned",
            num_blocks_to_save=5,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_final_save_unpinned",
            num_blocks_to_save=1,
            num_requests=1,
            is_final_save=True,
            skip_save=False,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_final_skip_save_unpinned",
            num_blocks_to_save=0,
            num_requests=1,
            is_final_save=True,
            skip_save=True,
            use_unpinned_host=True,
        ),
    )
    def test_tpu_connector_save(
        self,
        num_blocks_to_save: int,
        num_requests: int = 1,
        is_final_save: bool = False,
        skip_save: bool = False,
        use_precompiled_swap_ops: bool = False,
        use_unpinned_host: bool = False,
    ):
        total_num_blocks_to_save = num_blocks_to_save * num_requests
        if total_num_blocks_to_save > self.num_blocks or total_num_blocks_to_save > self.num_cpu_chunks:
            self.skipTest(
                f"num_blocks_to_save {total_num_blocks_to_save} exceeds ModelRunner / OffloadConnectorWorker's capacity"
            )

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
                    f"use_precompiled_swap_ops={use_precompiled_swap_ops};")

        connector_metadata = TPUOffloadConnectorMetadata(
            requests_meta=requests_meta)

        with mock.patch.dict(os.environ, {
                "TPU_OFFLOAD_USE_UNPINNED_HOST":
                "1" if use_unpinned_host else "0"
        }):
            connector = self._create_connector(use_precompiled_swap_ops)
            worker = connector.connector_worker
            connector.bind_connector_metadata(connector_metadata)
            logger.info(
                "Connector metadata bound, calling worker.start_save_kv().")

            worker.start_save_kv()
            logger.info("Waiting for all save operations to complete...")
            while worker._pending_save_futures:
                worker._process_completed_saves()
                time.sleep(0.01)

            # Verification
            logger.info("Starting verification phase.")
            cpu_backend = worker.cpu_backend
            kv_caches = worker.runner.kv_caches

            if skip_save or total_num_blocks_to_save == 0:
                logger.info(" no blocks to save")
                assert cpu_backend.num_saved_cpu_chunks == 0
                # self.assertEmpty(worker.finished_save_reqs)
                self.assertEmpty(
                    worker.offload_stats.data["finished_save_chunks"])
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
                    np_cpu_kv_chunk = np.array(cpu_kv_chunk)
                    if len(np_cpu_kv_chunk.shape) == 6:
                        np_cpu_kv_chunk = np.squeeze(np_cpu_kv_chunk, axis=0)
                    for layer_idx in range(self.num_layers):
                        tpu_kv_block = kv_caches[layer_idx][tpu_block_id]
                        self.assertArraysEqual(np.array(tpu_kv_block),
                                               np_cpu_kv_chunk[layer_idx])
            logger.info("Saved data verification completed.")

    @parameterized.named_parameters(
        # Pinned Host Cases
        dict(
            testcase_name="_single_block_pinned",
            num_blocks_to_operate=1,
            num_requests=1,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_requests_single_block_pinned",
            num_blocks_to_operate=1,
            num_requests=4,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_blocks_compile_jax_pinned",
            num_blocks_to_operate=5,
            num_requests=1,
            use_precompiled_swap_ops=True,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_requests_single_block_compile_jax_pinned",
            num_blocks_to_operate=1,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=False,
        ),
        dict(
            testcase_name="_multi_requests_multi_blocks_compile_jax_pinned",
            num_blocks_to_operate=16,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=False,
        ),

        # Unpinned Host Cases
        dict(
            testcase_name="_single_block_unpinned",
            num_blocks_to_operate=1,
            num_requests=1,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_requests_single_block_unpinned",
            num_blocks_to_operate=1,
            num_requests=4,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_blocks_compile_jax_unpinned",
            num_blocks_to_operate=5,
            num_requests=1,
            use_precompiled_swap_ops=True,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_requests_single_block_compile_jax_unpinned",
            num_blocks_to_operate=1,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=True,
        ),
        dict(
            testcase_name="_multi_requests_multi_blocks_compile_jax_unpinned",
            num_blocks_to_operate=16,
            num_requests=6,
            use_precompiled_swap_ops=True,
            use_unpinned_host=True,
        ),
    )
    def test_tpu_connector_load(
        self,
        num_blocks_to_operate: int,
        num_requests: int = 1,
        use_precompiled_swap_ops: bool = False,
        use_unpinned_host: bool = False,
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
        total_num_blocks_to_operate = num_blocks_to_operate * num_requests
        if total_num_blocks_to_operate > self.num_blocks or total_num_blocks_to_operate > self.num_cpu_chunks:
            self.skipTest(
                f"num_blocks_to_save {total_num_blocks_to_operate} exceeds ModelRunner / OffloadConnectorWorker's capacity"
            )

        with mock.patch.dict(os.environ, {
                "TPU_OFFLOAD_USE_UNPINNED_HOST":
                "1" if use_unpinned_host else "0"
        }):
            # 1. Setup
            connector = self._create_connector(use_precompiled_swap_ops)
            worker = connector.connector_worker
            # Ground truth cache. We copy it to host early because the save
            # operation will donate via stack_kv_cache_cross_layers.
            src_kv_cache_baseline = [
                np.array(cache) for cache in worker.runner.kv_caches
            ]
            # Destination cache on TPU, should be modified by the load operation
            dst_kv_cache = [
                jax.device_put(
                    jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                    self.device_sharding) for _ in range(self.num_layers)
            ]
            jax.block_until_ready(dst_kv_cache)

            # 2. Simulate a save operation
            all_block_ids = list(range(self.num_blocks))
            all_chunk_ids = list(range(self.num_cpu_chunks))
            src_block_ids = random.sample(all_block_ids,
                                          total_num_blocks_to_operate)
            dst_chunk_ids = random.sample(all_chunk_ids,
                                          total_num_blocks_to_operate)

            src_block_ids_split = np.array_split(src_block_ids, num_requests)
            dst_chunk_ids_split = np.array_split(dst_chunk_ids, num_requests)

            save_requests_meta = []
            for i in range(num_requests):
                req_id = f"save_req_{i}"
                src_blocks = src_block_ids_split[i].tolist()
                dst_chunks = dst_chunk_ids_split[i].tolist()
                num_tokens_to_save_per_req = len(src_blocks) * self.block_size

                save_spec = SaveSpec(
                    num_skip_leading_tokens=0,
                    num_total_tokens=num_tokens_to_save_per_req,
                    is_final_save=False,
                    skip_save=False,
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
                save_requests_meta.append(req_meta)

            connector_metadata = TPUOffloadConnectorMetadata(
                requests_meta=save_requests_meta)
            connector.bind_connector_metadata(connector_metadata)
            logger.info(
                "Connector metadata bound, calling worker.start_save_kv().")
            worker.start_save_kv()
            logger.info("Waiting for all save operations to complete...")
            while worker._pending_save_futures:
                worker._process_completed_saves()
                time.sleep(0.01)

            logger.info("worker save completed.")
            # 3. Prepare and Execute Delta Load
            worker.runner.kv_caches = dst_kv_cache

            load_requests_meta = []
            for i in range(num_requests):
                req_id = f"load_req_{i}"
                src_blocks = src_block_ids_split[i].tolist()
                dst_chunks = dst_chunk_ids_split[i].tolist()
                num_tokens_to_load_per_req = len(src_blocks) * self.block_size

                load_spec = LoadSpec(
                    num_matched_tokens=num_tokens_to_load_per_req,
                    dst_blocks=src_blocks,
                    src_chunks=dst_chunks,
                    can_load=True,
                    num_skip_leading_tokens=0,
                )
                total_token_ids = list(range(num_tokens_to_load_per_req))
                req_meta = TPUReqMeta(
                    req_id=req_id,
                    token_ids=total_token_ids,
                    local_block_ids=src_blocks,
                    load_spec=load_spec,
                )
                load_requests_meta.append(req_meta)

            connector_metadata = TPUOffloadConnectorMetadata(
                requests_meta=load_requests_meta)
            connector.bind_connector_metadata(connector_metadata)
            logger.info("Connector metadata bound, calling start_load_kv.")
            worker.start_load_kv(fwd_ctx=None)
            jax.block_until_ready(worker.runner.kv_caches)
            logger.info("start_load_kv completed and blocked until ready.")

            # 4. Verification
            # verify the data
            dst_kv_cache = worker.runner.kv_caches
            for i in range(num_requests):
                src_blocks = src_block_ids_split[i].tolist()
                for src_block_id in src_blocks:
                    for layer_idx in range(self.num_layers):
                        self.assertArraysEqual(
                            src_kv_cache_baseline[layer_idx][src_block_id],
                            np.array(dst_kv_cache[layer_idx][src_block_id]))

            # verify the loaded chunks
            all_load_req_ids = {f"load_req_{i}" for i in range(num_requests)}
            self.assertSetEqual(
                all_load_req_ids,
                set(worker.offload_stats.data["finished_load_chunks"].keys()))

            for i in range(num_requests):
                req_id = f"load_req_{i}"
                dst_chunks = dst_chunk_ids_split[i].tolist()
                self.assertListEqual(
                    dst_chunks,
                    worker.offload_stats.data["finished_load_chunks"][req_id])

    def test_tpu_connector_async_save_integrity(self):
        """
        Verify that the worker ensures data integrity using the dependency graph.
        This test mimics the scenario of two adjacent model iterations where
        a save in iteration N must complete its read before iteration N+1
        modifies the same memory.
        """
        # 1. Setup
        connector = self._create_connector()
        worker = connector.connector_worker
        block_to_save = 0
        dst_chunk = 7

        # 2. Define Kernel for modification (Mimics Step N+1 compute)
        @jax.jit
        def _modify_kernel(caches, block_id, val):
            return jax.tree.map(lambda x: x.at[block_id].set(val), caches)

        # stall kernel
        @jax.jit
        def _stall_kernel(a, b, num_loops=20):
            # Heavy matmul loop to keep the TPU busy for a significant duration.
            def body_fun(i, val):
                return jnp.matmul(val, b)

            return jax.lax.fori_loop(0, num_loops, body_fun, a)

        # 3. Warmup to prevent JIT stalls during the timed race
        logger.info("Warming up integrity test kernels...")
        dummy_val = jnp.full(self.cache_shape[1:],
                             1.23,
                             dtype=self.cache_dtype)
        worker.runner.kv_caches = _modify_kernel(worker.runner.kv_caches, 10,
                                                 dummy_val)

        # Warm up stall kernel
        size = 4096  # Balanced size and loops for a solid stall
        key = jax.random.key(42)
        a = jax.random.normal(key, (size, size), dtype=jnp.float32)
        b = jax.random.normal(key, (size, size), dtype=jnp.float32)
        # Scale matrices to prevent overflow during repeated multiplication
        a = a / jnp.sqrt(size)
        b = b / jnp.sqrt(size)
        _stall_kernel(a, b).block_until_ready()

        worker._precompile_kv_swap_operations()
        jax.block_until_ready(worker.runner.kv_caches)

        # 4. Baseline Capture
        # Capture the pre-corruption ground truth on the host for all layers.
        src_kv_cache_baseline = [np.array(c) for c in worker.runner.kv_caches]

        # 5. START THE RACE
        # Stall the TPU with a heavy compute operation.
        # This ensures the following save and corruption are queued together
        # while the TPU is busy, creating a deep command queue.
        logger.info("Dispatching stall compute...")
        _stall_kernel(a, b)

        # A. Step N Save: Dispatch Async Save for Block 0.
        # Due to the removal of block_until_ready, Python moves to 'B' immediately.
        save_spec = SaveSpec(
            num_skip_leading_tokens=0,
            num_total_tokens=self.block_size,
            is_final_save=False,
            skip_save=False,
            src_blocks=[block_to_save],
            dst_chunks=[dst_chunk],
        )
        req_meta = TPUReqMeta(
            req_id="async_integrity_test",
            token_ids=list(range(self.block_size)),
            local_block_ids=[block_to_save],
            save_spec=save_spec,
        )
        connector.bind_connector_metadata(
            TPUOffloadConnectorMetadata(requests_meta=[req_meta]))

        logger.info("Dispatching Async Save...")
        save_start_time = time.time()
        worker.start_save_kv()
        logger.info(
            f"Async Save dispatched in {time.time() - save_start_time:.6f}s")

        # B. Step N+1 Compute: IMMEDIATELY CORRUPT Block 0
        # If the dependency graph/barrier is broken, this might overwrite
        # Block 0 before the DMA engine finished reading it for the save.
        corruption_val = jnp.full(self.cache_shape[1:],
                                  6.66,
                                  dtype=self.cache_dtype)
        logger.info(
            f"RACE: Immediately dispatching corruption with value {6.66}...")
        corruption_start_time = time.time()
        worker.runner.kv_caches = _modify_kernel(worker.runner.kv_caches,
                                                 block_to_save, corruption_val)
        logger.info(
            f"Corruption dispatched in {time.time() - corruption_start_time:.6f}s"
        )

        # 6. Wait for Hardware and Poll completion
        logger.info(
            "Waiting for hardware to finish queued operations and polling for completion..."
        )
        jax.block_until_ready(worker.runner.kv_caches)
        poll_count = 0
        while worker._pending_save_futures:
            worker._process_completed_saves()
            poll_count += 1
            time.sleep(0.01)
        logger.info(f"Completion detected after {poll_count} polls.")

        # 7. Verification
        logger.info(
            f"Starting integrity verification for chunk {dst_chunk}...")
        cpu_val = np.array(worker.cpu_backend.get(dst_chunk))
        if len(cpu_val.shape) == 6:
            cpu_val = np.squeeze(cpu_val, axis=0)

        # Verify all layers for the saved block. Success proves that
        # hardware-level sequencing is enforced across the entire KV stack.
        for layer in range(self.num_layers):
            self.assertArraysEqual(src_kv_cache_baseline[layer][block_to_save],
                                   cpu_val[layer])

        # Sanity check: verify corruption actually happened on device.
        current_device_val = np.array(
            worker.runner.kv_caches[0][block_to_save])
        self.assertArraysEqual(current_device_val, np.array(corruption_val))

    def test_host_memory_kind_override(self):
        """Test that TPU_OFFLOAD_USE_UNPINNED_HOST overrides memory kind."""
        with mock.patch.dict(os.environ,
                             {"TPU_OFFLOAD_USE_UNPINNED_HOST": "1"}):
            connector = self._create_connector()
            worker = connector.connector_worker
            self.assertEqual(worker.host_sharding.memory_kind, "unpinned_host")
            self.assertEqual(worker.flatten_host_sharding.memory_kind,
                             "unpinned_host")
            self.assertEqual(worker.expanded_host_sharding.memory_kind,
                             "unpinned_host")

    def test_host_memory_kind_default(self):
        """Test that TPU_OFFLOAD_USE_UNPINNED_HOST defaults to false."""
        with mock.patch.dict(os.environ):
            if "TPU_OFFLOAD_USE_UNPINNED_HOST" in os.environ:
                del os.environ["TPU_OFFLOAD_USE_UNPINNED_HOST"]
            connector = self._create_connector()
            worker = connector.connector_worker
            self.assertEqual(worker.host_sharding.memory_kind, "pinned_host")
