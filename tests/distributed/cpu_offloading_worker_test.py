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


class TestCpuOffloadingSave(jtu.JaxTestCase):
    """Test the save functionality of the TPUConnectorWorker."""

    def setUp(self):
        super().setUp()
        self.vllm_config = MockVllmConfig(block_size=_DEFAULT_BLOCK_SIZE)
        self.num_layers = 2
        self.num_blocks = 24
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

        os.environ["TPU_OFFLOADING_SWAP_OP_TYPE"] = swap_op_type
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

    def _verify_saved_data(
        self,
        worker,
        source_kv_cache,
        process_token_ids,
        local_block_ids,
        skip_leading_tokens,
        num_layers,
        block_size,
    ):
        cpu_backend = LocalCPUBackend._instance
        token_processor = worker.token_processor

        # Create a map from token index to its corresponding chunk key info
        token_to_chunk_map = {}
        all_keys_generator = token_processor.process_tokens(process_token_ids)
        for abs_start_idx, abs_end_idx, key in all_keys_generator:
            for i in range(abs_start_idx, abs_end_idx):
                token_to_chunk_map[i] = {
                    "key": key,
                    "start_idx": abs_start_idx,
                }

        # Cache fetched chunks to avoid getting the same chunk multiple times
        fetched_chunks = {}

        num_processed_tokens = len(process_token_ids)
        for token_idx in range(skip_leading_tokens, num_processed_tokens):
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

            # logger.info(f"token_idx: {token_idx}, logical_block_idx: {logical_block_idx}, block_offset: {block_offset}, physical_block_id: {physical_block_id}, offset_in_chunk: {offset_in_chunk}")
            assert offset_in_chunk == block_offset, f"{offset_in_chunk} != {block_offset}"
            for layer_idx in range(num_layers):
                original_token_data = source_kv_cache[layer_idx][
                    physical_block_id, block_offset, ...]
                saved_chunk = jax.device_put(saved_chunk_data[layer_idx],
                                             jax.devices("cpu")[0])
                saved_token_data = saved_chunk[offset_in_chunk, ...]
                self.assertArraysEqual(np.array(saved_token_data),
                                       np.array(original_token_data))

    @parameterized.named_parameters(
        dict(
            testcase_name="_prefill_no_skip_save_2_drop",
            num_skip_leading_tokens=0,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE * 2,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_blocks_to_save=2,
        ),
        dict(
            testcase_name="_prefill_no_skip_save_2_drop_pallas",
            num_skip_leading_tokens=0,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE * 2,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_blocks_to_save=2,
            swap_op_type="pallas",
        ),
        dict(
            testcase_name="_prefill_no_skip_save_2_pad",
            num_skip_leading_tokens=0,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_blocks_to_save=3,
        ),
        dict(
            testcase_name="_prefill_no_skip_save_2_pad_pallas",
            num_skip_leading_tokens=0,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_blocks_to_save=3,
            swap_op_type="pallas",
        ),
        dict(
            testcase_name="_prefill_skip_2_save_2_drop",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE * 2,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 4 + 10,
            num_blocks_to_save=2,
        ),
        dict(
            testcase_name="_prefill_skip_2_save_2_pad",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 4 + 10,
            num_blocks_to_save=3,
        ),
        dict(
            testcase_name="_decode_skip_3_save_1",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 3,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 4,
            num_blocks_to_save=1,
        ),
        dict(
            testcase_name="_no_save",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_tokens_to_save=0,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_blocks_to_save=0,
            is_final_save=False,
            skip_save=False,
        ),
        dict(
            testcase_name="_final_save_save_1_drop",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_tokens_to_save=_DEFAULT_BLOCK_SIZE,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 3 + 10,
            num_blocks_to_save=1,
            is_final_save=True,
            skip_save=False,
        ),
        dict(
            testcase_name="_final_save_save_1_pad",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_tokens_to_save=10,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2 + 10,
            num_blocks_to_save=1,
            is_final_save=True,
            skip_save=False,
        ),
        dict(
            testcase_name="_final_save_without_data",
            num_skip_leading_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_tokens_to_save=0,
            num_total_tokens=_DEFAULT_BLOCK_SIZE * 2,
            num_blocks_to_save=0,
            is_final_save=True,
            skip_save=True,
        ),
    )
    def test_tpu_connector_save(
        self,
        num_skip_leading_tokens: int,
        num_tokens_to_save: int,
        num_total_tokens: int,
        num_blocks_to_save: int,
        is_final_save: bool = False,
        skip_save: bool = False,
        swap_op_type: str = "jax",
    ):

        # Prepare and Execute Save
        total_token_ids = list(range(num_total_tokens))
        num_blocks_for_tokens = (num_total_tokens + self.block_size -
                                 1) // self.block_size

        if num_blocks_for_tokens > self.num_blocks:
            self.skipTest(
                f"Not enough blocks to run test, blocks for tokens {num_blocks_for_tokens} > {self.num_blocks}"
            )
        if num_blocks_for_tokens < num_blocks_to_save:
            self.skipTest(
                f"Not enough blocks to save, blocks for tokens {num_blocks_for_tokens} < {num_blocks_to_save}"
            )
        if num_skip_leading_tokens % self.block_size != 0:
            self.skipTest(
                "num_skip_leading_tokens must be a multiple of block_size")
        if num_total_tokens < (num_skip_leading_tokens + num_tokens_to_save):
            self.skipTest(
                f"num_total_tokens {num_total_tokens} must be no less than num_skip_leading_tokens + num_tokens_to_save"
            )
        if (num_blocks_to_save -
            (num_tokens_to_save // self.block_size)) not in [0, 1]:
            self.skipTest(
                f"num_blocks_to_save {num_blocks_to_save} does not match with the given num_tokens_to_save {num_tokens_to_save}"
            )

        total_blocks = list(range(self.num_blocks))
        local_block_ids = sorted(
            random.sample(total_blocks, num_blocks_for_tokens))
        num_skip_blocks = num_skip_leading_tokens // self.block_size
        src_blocks_to_save = local_block_ids[num_skip_blocks:(
            num_skip_blocks + num_blocks_to_save)]

        logger.info(
            f"Starting test_tpu_connector_save with: "
            f"num_blocks_to_save={num_blocks_to_save}, skip_leading_tokens={num_skip_leading_tokens}, num_tokens_to_save={num_tokens_to_save}, "
            f"num_total_tokens={num_total_tokens}, is_final_save={is_final_save}, skip_save={skip_save}, swap_op_type={swap_op_type}. \n"
            f" Prepared for save: total_token_ids={total_token_ids}, num_blocks_for_tokens={num_blocks_for_tokens}, "
            f"blocks_to_save={src_blocks_to_save}, local_block_ids={local_block_ids}."
        )

        connector = self._create_connector(swap_op_type)
        worker = connector.connector_worker

        req_id = "save_req"
        num_process_tokens = num_skip_leading_tokens + num_tokens_to_save
        save_spec = SaveSpec(
            num_skip_leading_tokens=num_skip_leading_tokens,
            num_total_tokens=num_process_tokens,
            is_final_save=is_final_save,
            skip_save=skip_save,
            src_blocks=src_blocks_to_save,
        )
        req_meta = TPUReqMeta(
            req_id=req_id,
            token_ids=total_token_ids,
            local_block_ids=local_block_ids,
            save_spec=save_spec,
        )

        connector_metadata = TPUConnectorMetadata(requests_meta=[req_meta])
        connector.bind_connector_metadata(connector_metadata)
        logger.info(
            "Connector metadata bound, calling worker.wait_for_save().")
        worker.wait_for_save()
        logger.info("worker.wait_for_save() completed.")

        # Verification
        logger.info("Starting verification phase.")
        cpu_backend = worker.cpu_backend
        saved_keys = cpu_backend.cache.keys()

        if num_tokens_to_save == 0 or skip_save:
            logger.info(
                f"num_tokens_to_save is 0 or skip_save is True. Asserting no keys saved. "
                f"Saved keys: {saved_keys}")
            self.assertEmpty(saved_keys)
            if is_final_save:
                finished_saves, _ = worker.get_finished()
                logger.info(
                    f"is_final_save is True. Finished requests: {finished_saves}"
                )
                self.assertIn(req_id, finished_saves)
            logger.info("Verification completed for no-save scenario.")
            return

        # Verify that the correct number of chunks were saved
        processed_token_ids = total_token_ids[:num_process_tokens]
        token_processor = worker.token_processor
        all_keys_generator = token_processor.process_tokens(
            processed_token_ids)
        expected_num_keys = 0
        for start_idx, _, _ in all_keys_generator:
            # The logic in _save_blocks_to_cpu filters keys based on the start
            # of the chunk.
            if start_idx >= num_skip_leading_tokens:
                expected_num_keys += 1
        logger.info(
            f"Expected number of saved keys: {expected_num_keys}, Actual saved keys: {len(saved_keys)}"
        )
        self.assertLen(saved_keys, expected_num_keys)
        self._verify_saved_data(
            worker,
            worker.runner.kv_caches,
            processed_token_ids,
            local_block_ids,
            num_skip_leading_tokens,
            self.num_layers,
            self.block_size,
        )
        logger.info("Saved data verification completed.")

        if is_final_save:
            finished_saves, _ = worker.get_finished()
            logger.info(
                f"is_final_save is True. Finished requests: {finished_saves}")
            self.assertIn(req_id, finished_saves)
        logger.info("Test test_tpu_connector_save completed successfully.")

    @parameterized.named_parameters(
        dict(
            testcase_name="_2_steps",
            num_blocks_step1=2,
            num_blocks_step2=1,
        ),
        dict(
            testcase_name="_zero_token_step2",
            num_blocks_step1=2,
            num_blocks_step2=0,
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

        connector = self._create_connector()
        worker = connector.connector_worker
        available_blocks = list(range(self.num_blocks))

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

        local_block_ids_step1 = sorted(
            random.sample(available_blocks, num_blocks_step1))
        num_skip_blocks_step1 = skip_leading_tokens_step1 // self.block_size
        src_blocks_to_save_step1 = local_block_ids_step1[
            num_skip_blocks_step1:(num_skip_blocks_step1 + num_blocks_step1)]

        logger.info(
            f"Step 1: local_block_ids_step1={local_block_ids_step1}, src_blocks_to_save_step1={src_blocks_to_save_step1}"
        )

        req_id = "multi_step_save_req"
        save_spec_step1 = SaveSpec(
            num_skip_leading_tokens=skip_leading_tokens_step1,
            num_total_tokens=total_tokens_step1,
            is_final_save=False,
            skip_save=False,
            src_blocks=src_blocks_to_save_step1,
        )

        req_meta_step1 = TPUReqMeta(
            req_id=req_id,
            token_ids=token_ids_step1,
            local_block_ids=local_block_ids_step1,
            save_spec=save_spec_step1,
        )
        logger.info(
            f"Step 1: req_meta_step1.token_ids={req_meta_step1.token_ids}, req_meta_step1.local_block_ids={req_meta_step1.local_block_ids}, req_meta_step1.save_spec.skip_leading_tokens={req_meta_step1.save_spec.num_skip_leading_tokens}"
        )
        connector_metadata_step1 = TPUConnectorMetadata(
            requests_meta=[req_meta_step1])
        connector.bind_connector_metadata(connector_metadata_step1)
        worker.wait_for_save()

        # Verification for Step 1
        logger.info("--- Verifying Step 1 ---")
        self._verify_saved_data(
            worker,
            worker.runner.kv_caches,
            token_ids_step1,
            local_block_ids_step1,
            skip_leading_tokens_step1,
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
        src_blocks_to_save_step2 = additional_blocks
        logger.info(
            f"Step 2: local_block_ids_step2={local_block_ids_step2}, src_blocks_to_save_step2={src_blocks_to_save_step2}"
        )

        save_spec_step2 = SaveSpec(
            num_skip_leading_tokens=skip_leading_tokens_step2,
            num_total_tokens=total_tokens_step2,
            is_final_save=False,
            skip_save=False,
            src_blocks=src_blocks_to_save_step2,
        )
        req_meta_step2 = TPUReqMeta(
            req_id=req_id,
            token_ids=token_ids_step2,
            local_block_ids=local_block_ids_step2,
            save_spec=save_spec_step2,
        )
        logger.info(
            f"Step 2: req_meta_step2.token_ids={req_meta_step2.token_ids}, req_meta_step2.local_block_ids={req_meta_step2.local_block_ids}, req_meta_step2.save_spec.skip_leading_tokens={req_meta_step2.save_spec.num_skip_leading_tokens}"
        )
        connector_metadata_step2 = TPUConnectorMetadata(
            requests_meta=[req_meta_step2])

        # Manually reset worker state to simulate a new scheduler step
        worker._processed_save_for_step = False
        connector.bind_connector_metadata(connector_metadata_step2)
        worker.wait_for_save()

        # Verification for Step 2 (only the new data)
        logger.info("--- Verifying Step 2 (new data) ---")
        self._verify_saved_data(
            worker,
            worker.runner.kv_caches,
            token_ids_step2,
            local_block_ids_step2,
            skip_leading_tokens_step2,
            self.num_layers,
            self.block_size,
        )

        # Verification for Step 1 data (to ensure it is not corrupted)
        logger.info("--- Verifying Step 1 data after Step 2 ---")
        self._verify_saved_data(
            worker,
            worker.runner.kv_caches,
            token_ids_step1,
            local_block_ids_step1,
            skip_leading_tokens_step1,
            self.num_layers,
            self.block_size,
        )
        logger.info(
            "Test test_tpu_connector_multi_step_save completed successfully.")

    @parameterized.named_parameters(
        dict(
            testcase_name="_full_load_jax",
            swap_op_type="jax",
            num_matched_blocks=4,
            num_computed_blocks=0,
        ),
        dict(
            testcase_name="_delta_load_jax",
            swap_op_type="jax",
            num_matched_blocks=4,
            num_computed_blocks=1,
        ),
        dict(
            testcase_name="_delta_load_pallas",
            swap_op_type="pallas",
            num_matched_blocks=4,
            num_computed_blocks=1,
        ),
        dict(
            testcase_name="_no_load_jax",
            swap_op_type="jax",
            num_matched_blocks=1,
            num_computed_blocks=1,
        ),
    )
    def test_tpu_connector_load(
        self,
        swap_op_type: str,
        num_matched_blocks: int,
        num_computed_blocks: int = 0,
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
        num_matched_tokens = num_matched_blocks * self.block_size
        num_computed_tokens = num_computed_blocks * self.block_size
        if num_matched_blocks > self.num_blocks:
            self.skipTest(
                f"num_matched_blocks {num_matched_blocks} > vllm_config.num_blocks {self.num_blocks}"
            )
        if num_computed_blocks > num_matched_blocks:
            self.skipTest(
                f"num_computed_blocks {num_computed_blocks} > num_matched_blocks {num_matched_blocks}"
            )

        logger.info(
            f"Starting test_tpu_connector_load with num_computed_tokens={num_computed_tokens}, num_matched_tokens={num_matched_tokens}, swap_op_type={swap_op_type}."
        )
        # 1. Setup
        connector = self._create_connector(swap_op_type)
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

        req_id = "save_req"
        matched_token_ids = list(range(num_matched_tokens))
        total_blocks = list(range(self.num_blocks))
        local_block_ids = sorted(
            random.sample(total_blocks, num_matched_blocks))

        # 2. Populate CPU Cache
        # Save the part of the source cache that represents the "matched" prefix
        if num_matched_tokens > 0:
            logger.info(
                f"Populating CPU cache with {num_matched_tokens} matched tokens."
            )

            src_blocks_to_save = local_block_ids[
                num_computed_blocks:num_matched_blocks]
            save_spec = SaveSpec(
                num_skip_leading_tokens=num_computed_tokens,
                num_total_tokens=num_matched_tokens,
                is_final_save=False,
                skip_save=False,
                src_blocks=src_blocks_to_save,
            )
            req_meta = TPUReqMeta(
                req_id=req_id,
                token_ids=matched_token_ids,
                local_block_ids=local_block_ids,
                save_spec=save_spec,
            )
            connector_metadata = TPUConnectorMetadata(requests_meta=[req_meta])
            connector.bind_connector_metadata(connector_metadata)
            worker.wait_for_save()
            logger.info(
                f"Simulated save operation to CPU for {num_matched_tokens} tokens."
            )
        else:
            logger.info("No matched tokens, skipping CPU cache population.")

        # 3. Prepare and Execute Delta Load
        worker.runner.kv_caches = dst_kv_cache
        num_tokens_to_load = max(0, num_matched_tokens - num_computed_tokens)
        # `num_tokens_to_load` cannot be negative. If `num_computed_tokens`
        # is greater than or equal to `num_matched_tokens`, it means all
        # relevant tokens are already on the TPU, and no new tokens need
        # to be loaded from the CPU backend. In such cases, the value should
        # be clamped to 0.
        logger.info(
            f"Calculated num_tokens_to_load: {num_tokens_to_load} (num_matched_tokens={num_matched_tokens} - num_computed_tokens={num_computed_tokens})"
        )
        if num_tokens_to_load > 0:
            dst_blocks = local_block_ids[
                num_computed_blocks:num_matched_blocks]
            load_spec = LoadSpec(
                num_matched_tokens=num_matched_tokens,
                dst_blocks=dst_blocks,
                is_full_prefix_hit=False,
                can_load=True,
                num_skip_leading_tokens=num_computed_tokens,
            )

            logger.info(f"LoadSpec created: {load_spec}")
            # The worker needs the full token list to generate keys correctly
            req_meta = TPUReqMeta(
                req_id="load_req",
                token_ids=matched_token_ids,
                local_block_ids=local_block_ids,
                load_spec=load_spec,
            )
            connector_metadata = TPUConnectorMetadata(requests_meta=[req_meta])
            connector.bind_connector_metadata(connector_metadata)
            logger.info("Connector metadata bound, calling start_load_kv.")
            worker.start_load_kv(fwd_ctx=None)
            jax.block_until_ready(worker.runner.kv_caches)
            logger.info("start_load_kv completed and blocked until ready.")
            # we will donate the original kv_cache ref
            dst_kv_cache = worker.runner.kv_caches

        # worker.runner.kv_caches = src_kv_cache

        # 4. Verification
        logger.info("Starting verification phase.")

        if num_tokens_to_load <= 0:
            logger.info(
                "num_tokens_to_load is 0 or less, asserting nothing was loaded."
            )
            # Assert that the entire destination cache remains untouched (all zeros).
            for i in range(self.num_layers):
                self.assertArraysEqual(
                    dst_kv_cache[i],
                    jnp.zeros(self.cache_shape, dtype=self.cache_dtype),
                )
            logger.info("Assertion passed: Destination KV cache is all zeros.")
            return

        # Helper to flatten and extract a token range from a cache given a block map
        def get_token_slice(kv_cache, start_token, num_tokens, block_map):
            if num_tokens <= 0:
                return jnp.empty((0, *kv_cache.shape[2:]),
                                 dtype=kv_cache.dtype)
            start_block_logical = start_token // self.block_size
            start_offset = start_token % self.block_size
            end_token = start_token + num_tokens
            end_block_logical = (end_token + self.block_size -
                                 1) // self.block_size

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

        # Get the ground truth data from the source cache
        expected_data_from_source_tpu = [
            get_token_slice(
                src_kv_cache[i],
                start_token=num_computed_tokens,
                num_tokens=num_tokens_to_load,
                block_map=local_block_ids,
            ) for i in range(self.num_layers)
        ]
        logger.info(
            f"Extracted expected data from source cache. Shape of first layer: {expected_data_from_source_tpu[0].shape}"
        )

        # Get the data that was actually loaded into the destination cache
        loaded_data_on_dest_tpu = [
            get_token_slice(
                dst_kv_cache[i],
                start_token=num_computed_tokens,
                num_tokens=num_tokens_to_load,
                block_map=local_block_ids,
            ) for i in range(self.num_layers)
        ]
        logger.info(
            f"Extracted loaded data from destination cache. Shape of first layer: {loaded_data_on_dest_tpu[0].shape}"
        )

        # Assert that the loaded delta is correct. This works for no-load cases too.
        for i in range(self.num_layers):
            self.assertArraysEqual(np.array(expected_data_from_source_tpu[i]),
                                   np.array(loaded_data_on_dest_tpu[i]))
        logger.info("Assertion passed: Loaded delta matches expected data.")

        # Assert that blocks not in local_block_ids are still zero
        untouched_blocks = sorted(
            list(set(range(self.num_blocks)) - set(local_block_ids)))
        logger.info(
            f"Asserting that {len(untouched_blocks)} untouched blocks are still zero."
        )
        if untouched_blocks:
            for i in range(self.num_layers):
                zero_slice = worker.runner.kv_caches[i][untouched_blocks, ...]
                self.assertTrue(jnp.all(zero_slice == 0))
                expected_zeros = jnp.zeros(
                    (len(untouched_blocks), *self.cache_shape[1:]),
                    dtype=self.cache_dtype)
                self.assertArraysEqual(np.array(zero_slice),
                                       np.array(expected_zeros))
        logger.info("Assertion passed: Untouched blocks are zero.")
        logger.info(
            "Test test_tpu_connector_delta_load completed successfully.")
