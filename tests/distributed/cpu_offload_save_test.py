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
from tpu_inference.distributed.tpu_connector_local import SaveSpec
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


class TestCpuOffloadingSave(jtu.JaxTestCase):
    """Test the save functionality of the TPUConnectorWorker."""

    def setUp(self):
        super().setUp()
        # Clean the singleton backend instance before each test
        LocalCPUBackend._instance = None
        LocalCPUBackend._initialized = False
        self.vllm_config = MockVllmConfig(block_size=64)

        os.environ["TPU_OFFLOADING_SWAP_OP_TYPE"] = "jax"
        self.mesh = self.create_mesh((1, 8), ("data", "model"))
        if self.mesh is None:
            self.skipTest("Cannot create mesh. Must be run on a TPU node.")
            return

        self.connector = CPUOffloadingConnector(self.vllm_config,
                                                KVConnectorRole.WORKER)
        self.worker = self.connector.connector_worker
        assert self.worker is not None

        # Define cache properties
        self.num_layers = 2
        self.num_blocks = 16
        self.block_size = self.vllm_config.cache_config.block_size
        self.num_heads = 8
        self.head_size = 128
        cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_heads,
            2,
            self.head_size,
        )
        cache_dtype = jnp.bfloat16
        partition_spec = PartitionSpec(None, None, "model")
        device_sharding = NamedSharding(self.mesh, partition_spec)

        @functools.partial(jax.jit, out_shardings=device_sharding)
        def create_on_device(key):
            return jax.random.uniform(key,
                                      shape=cache_shape,
                                      dtype=cache_dtype)

        self.source_kv_cache = [
            create_on_device(jax.random.key(i)) for i in range(self.num_layers)
        ]
        jax.block_until_ready(self.source_kv_cache)

        mock_runner = MockTPUModelRunner(kv_caches=self.source_kv_cache,
                                         mesh=self.mesh)
        self.worker.register_runner(mock_runner)

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

    def _verify_saved_data(
        self,
        worker,
        req_id,
        source_kv_cache,
        token_ids,
        local_block_ids,
        skip_leading_tokens,
        num_tokens_to_save,
        num_layers,
        block_size,
    ):
        cpu_backend = LocalCPUBackend._instance
        token_processor = worker.token_processor

        # Create a map from token index to its corresponding chunk key info
        token_to_chunk_map = {}
        all_keys_generator = token_processor.process_tokens(token_ids)
        for abs_start_idx, abs_end_idx, key in all_keys_generator:
            for i in range(abs_start_idx, abs_end_idx):
                token_to_chunk_map[i] = {
                    "key": key,
                    "start_idx": abs_start_idx,
                }

        # Cache fetched chunks to avoid getting the same chunk multiple times
        fetched_chunks = {}

        total_tokens = skip_leading_tokens + num_tokens_to_save
        for token_idx in range(skip_leading_tokens, total_tokens):
            if token_idx not in token_to_chunk_map:
                self.fail(f"Token index {token_idx} not found in any chunk.")

            chunk_info = token_to_chunk_map[token_idx]
            chunk_key = chunk_info["key"]

            # Fetch the chunk from backend if not already fetched
            if chunk_key not in fetched_chunks:
                chunk_data = cpu_backend.get(chunk_key)
                self.assertIsNotNone(
                    chunk_data,
                    f"Key {chunk_key} for token {token_idx} not found in backend",
                )
                fetched_chunks[chunk_key] = chunk_data

            saved_chunk_data = fetched_chunks[chunk_key]

            # Get the original data from the source TPU cache
            logical_block_idx = token_idx // block_size
            block_offset = token_idx % block_size
            physical_block_id = local_block_ids[logical_block_idx]

            # Get the saved data for the specific token from the chunk
            offset_in_chunk = token_idx - chunk_info["start_idx"]

            for layer_idx in range(num_layers):
                original_token_data = source_kv_cache[layer_idx][
                    physical_block_id, block_offset, ...]
                saved_token_data = saved_chunk_data[layer_idx][offset_in_chunk,
                                                               ...]
                self.assertArraysEqual(np.array(saved_token_data),
                                       np.array(original_token_data))

    @parameterized.named_parameters(
        dict(
            testcase_name="_full_save_single_step",
            num_blocks_to_save=2,
            skip_leading_blocks=0,
        ),
        dict(
            testcase_name="_save_partial_block_only",
            num_blocks_to_save=1,
            skip_leading_blocks=0,
        ),
        dict(
            testcase_name="_no_save_required",
            num_blocks_to_save=0,
            skip_leading_blocks=2,
        ),
        dict(
            testcase_name="_final_save_with_data",
            num_blocks_to_save=2,
            skip_leading_blocks=0,
            is_final_save=True,
        ),
        dict(
            testcase_name="_final_save_without_data",
            num_blocks_to_save=0,
            skip_leading_blocks=0,
            is_final_save=True,
            skip_save=True,
        ),
    )
    def test_tpu_connector_save(
        self,
        num_blocks_to_save: int,
        skip_leading_blocks: int = 0,
        is_final_save: bool = False,
        skip_save: bool = False,
    ):
        num_tokens_to_save = num_blocks_to_save * self.block_size
        skip_leading_tokens = skip_leading_blocks * self.block_size

        logger.info(
            f"Starting test_tpu_connector_save with num_blocks_to_save={num_blocks_to_save}, "
            f"skip_leading_blocks={skip_leading_blocks}, is_final_save={is_final_save}, "
            f"skip_save={skip_save}. Derived: num_tokens_to_save={num_tokens_to_save}, skip_leading_tokens={skip_leading_tokens}"
        )

        # Prepare and Execute Save
        total_tokens = skip_leading_tokens + num_tokens_to_save
        token_ids = list(range(total_tokens))
        num_blocks_for_tokens = (total_tokens + self.block_size -
                                 1) // self.block_size
        if num_blocks_for_tokens > self.num_blocks:
            self.skipTest("Not enough blocks to run test")
        available_blocks = list(range(self.num_blocks))
        local_block_ids = sorted(
            random.sample(available_blocks, num_blocks_for_tokens))
        logger.info(
            f"Prepared for save: total_tokens={total_tokens}, token_ids={token_ids}, "
            f"num_blocks_for_tokens={num_blocks_for_tokens}, local_block_ids={local_block_ids}"
        )

        req_id = "save_req"
        save_spec = SaveSpec(
            skip_leading_tokens=skip_leading_tokens,
            is_final_save=is_final_save,
            skip_save=skip_save,
        )
        req_meta = TPUReqMeta(
            req_id=req_id,
            token_ids=token_ids,
            local_block_ids=local_block_ids,
            save_spec=save_spec,
        )
        connector_metadata = TPUConnectorMetadata(requests_meta=[req_meta])
        self.connector.bind_connector_metadata(connector_metadata)
        logger.info(
            "Connector metadata bound, calling worker.wait_for_save().")
        self.worker.wait_for_save()
        logger.info("worker.wait_for_save() completed.")

        # Verification
        logger.info("Starting verification phase.")
        cpu_backend = self.worker.cpu_backend
        saved_keys = cpu_backend.cache.keys()

        if num_tokens_to_save == 0 or skip_save:
            logger.info(
                f"num_tokens_to_save is 0 or skip_save is True. Asserting no keys saved. "
                f"Saved keys: {saved_keys}")
            self.assertEmpty(saved_keys)
            if is_final_save:
                finished_saves, _ = self.worker.get_finished()
                logger.info(
                    f"is_final_save is True. Finished requests: {finished_saves}"
                )
                self.assertIn(req_id, finished_saves)
            logger.info("Verification completed for no-save scenario.")
            return

        # Verify that the correct number of chunks were saved
        token_processor = self.worker.token_processor
        all_keys_generator = token_processor.process_tokens(token_ids)
        expected_num_keys = 0
        for start_idx, _, _ in all_keys_generator:
            # The logic in _save_blocks_to_cpu filters keys based on the start
            # of the chunk.
            if start_idx >= skip_leading_tokens:
                expected_num_keys += 1
        logger.info(
            f"Expected number of saved keys: {expected_num_keys}, Actual saved keys: {len(saved_keys)}"
        )
        self.assertLen(saved_keys, expected_num_keys)
        self._verify_saved_data(
            self.worker,
            req_id,
            self.source_kv_cache,
            token_ids,
            local_block_ids,
            skip_leading_tokens,
            num_tokens_to_save,
            self.num_layers,
            self.block_size,
        )
        logger.info("Saved data verification completed.")

        if is_final_save:
            finished_saves, _ = self.worker.get_finished()
            logger.info(
                f"is_final_save is True. Finished requests: {finished_saves}")
            self.assertIn(req_id, finished_saves)
        logger.info("Test test_tpu_connector_save completed successfully.")

    @parameterized.named_parameters(
        dict(
            testcase_name="_both_steps_block_aligned",
            num_blocks_step1=2,
            num_blocks_step2=1,
        ),
        dict(
            testcase_name="_zero_token_step2",
            num_blocks_step1=2,
            num_blocks_step2=0,
        ),
        dict(
            testcase_name="_multiple_full_blocks_each_step",
            num_blocks_step1=2,
            num_blocks_step2=2,
        ),
    )
    def test_tpu_connector_multi_step_save(
        self,
        num_blocks_step1: int,
        num_blocks_step2: int,
    ):
        """
        Tests that the TPUConnectorWorker correctly saves the KV cache in multiple
        steps, respecting the save watermark (skip_leading_tokens).
        """
        num_tokens_step1 = num_blocks_step1 * self.block_size
        num_tokens_step2 = num_blocks_step2 * self.block_size
        logger.info(
            f"Starting test_tpu_connector_multi_step_save with "
            f"num_tokens_step1={num_tokens_step1}, num_tokens_step2={num_tokens_step2}"
        )

        # --- Step 1: Initial Save ---
        logger.info("--- Multi-step save: Step 1 ---")
        skip_leading_tokens_step1 = 0

        total_tokens_step1 = num_tokens_step1
        token_ids_step1 = list(range(total_tokens_step1))
        logger.info(
            f"Step 1: num_tokens_step1={num_tokens_step1}, total_tokens_step1={total_tokens_step1}, num_blocks_step1={num_blocks_step1}"
        )
        if num_blocks_step1 > self.num_blocks:
            self.skipTest("Not enough blocks for step 1")

        available_blocks = list(range(self.num_blocks))
        local_block_ids_step1 = sorted(
            random.sample(available_blocks, num_blocks_step1))
        logger.info(f"Step 1: local_block_ids_step1={local_block_ids_step1}")

        req_id = "multi_step_save_req"
        save_spec_step1 = SaveSpec(
            skip_leading_tokens=skip_leading_tokens_step1)
        req_meta_step1 = TPUReqMeta(
            req_id=req_id,
            token_ids=token_ids_step1,
            local_block_ids=local_block_ids_step1,
            save_spec=save_spec_step1,
        )
        logger.info(
            f"Step 1: req_meta_step1.token_ids={req_meta_step1.token_ids}, req_meta_step1.local_block_ids={req_meta_step1.local_block_ids}, req_meta_step1.save_spec.skip_leading_tokens={req_meta_step1.save_spec.skip_leading_tokens}"
        )
        connector_metadata_step1 = TPUConnectorMetadata(
            requests_meta=[req_meta_step1])
        self.connector.bind_connector_metadata(connector_metadata_step1)
        self.worker.wait_for_save()

        # Verification for Step 1
        logger.info("--- Verifying Step 1 ---")
        self._verify_saved_data(
            self.worker,
            req_id,
            self.source_kv_cache,
            token_ids_step1,
            local_block_ids_step1,
            skip_leading_tokens_step1,
            num_tokens_step1,
            self.num_layers,
            self.block_size,
        )

        # --- Step 2: Incremental Save ---
        logger.info("--- Multi-step save: Step 2 ---")
        skip_leading_tokens_step2 = num_tokens_step1
        total_tokens_step2 = skip_leading_tokens_step2 + num_tokens_step2
        token_ids_step2 = list(range(total_tokens_step2))
        num_blocks_step2_total = total_tokens_step2 // self.block_size
        logger.info(
            f"Step 2: num_tokens_step2={num_tokens_step2}, skip_leading_tokens_step2={skip_leading_tokens_step2}, total_tokens_step2={total_tokens_step2}, num_blocks_step2_total={num_blocks_step2_total}"
        )
        if num_blocks_step2_total > self.num_blocks:
            self.skipTest("Not enough blocks for step 2")

        num_additional_blocks = num_blocks_step2_total - num_blocks_step1
        logger.info(f"Step 2: num_additional_blocks={num_additional_blocks}")
        remaining_blocks = sorted(
            list(set(available_blocks) - set(local_block_ids_step1)))
        if num_additional_blocks > len(remaining_blocks):
            self.skipTest("Not enough remaining blocks for step 2")

        additional_blocks = random.sample(remaining_blocks,
                                          num_additional_blocks)
        local_block_ids_step2 = local_block_ids_step1 + additional_blocks
        logger.info(f"Step 2: local_block_ids_step2={local_block_ids_step2}")

        save_spec_step2 = SaveSpec(
            skip_leading_tokens=skip_leading_tokens_step2)
        req_meta_step2 = TPUReqMeta(
            req_id=req_id,
            token_ids=token_ids_step2,
            local_block_ids=local_block_ids_step2,
            save_spec=save_spec_step2,
        )
        logger.info(
            f"Step 2: req_meta_step2.token_ids={req_meta_step2.token_ids}, req_meta_step2.local_block_ids={req_meta_step2.local_block_ids}, req_meta_step2.save_spec.skip_leading_tokens={req_meta_step2.save_spec.skip_leading_tokens}"
        )
        connector_metadata_step2 = TPUConnectorMetadata(
            requests_meta=[req_meta_step2])

        # Manually reset worker state to simulate a new scheduler step
        self.worker._processed_save_for_step = False
        self.connector.bind_connector_metadata(connector_metadata_step2)
        self.worker.wait_for_save()

        # Verification for Step 2 (only the new data)
        logger.info("--- Verifying Step 2 (new data) ---")
        self._verify_saved_data(
            self.worker,
            req_id,
            self.source_kv_cache,
            token_ids_step2,
            local_block_ids_step2,
            skip_leading_tokens_step2,
            num_tokens_step2,
            self.num_layers,
            self.block_size,
        )

        # Verification for Step 1 data (to ensure it is not corrupted)
        logger.info("--- Verifying Step 1 data after Step 2 ---")
        self._verify_saved_data(
            self.worker,
            req_id,
            self.source_kv_cache,
            token_ids_step2,
            local_block_ids_step2,
            skip_leading_tokens_step1,
            num_tokens_step1,
            self.num_layers,
            self.block_size,
        )
        logger.info(
            "Test test_tpu_connector_multi_step_save completed successfully.")
