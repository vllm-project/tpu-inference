# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request

from tpu_inference.distributed.local_cpu_backend import LocalCPUBackend
from tpu_inference.distributed.tpu_connector_local import (
    ReqId, RequestTracker, TPUConnectorScheduler, _StagingBufferManager)
from tpu_inference.logger import init_logger

from .cpu_offloading_worker_test import MockVllmConfig

logger = init_logger(__name__)

_DEFAULT_BLOCK_SIZE = 4


def create_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    num_computed_tokens: int = 0,
) -> Request:
    """Creates a mock vLLM request object."""
    req = MagicMock(spec=Request)
    req.request_id = request_id
    req.req_id = request_id
    req.prompt_token_ids = prompt_token_ids
    req.all_token_ids = prompt_token_ids
    req.num_computed_tokens = num_computed_tokens
    req.block_size = block_size
    req.block_ids = [[]]  # Mock structure
    return req


@pytest.fixture
def clean_backend_instance():
    """
    Provides a clean instance of the LocalCPUBackend for each test.
    """
    LocalCPUBackend._instance = None
    LocalCPUBackend._initialized = False
    yield
    LocalCPUBackend._instance = None
    LocalCPUBackend._initialized = False


@pytest.fixture
def scheduler_factory():
    """Provides a factory function for TPUConnectorScheduler instances."""

    def _scheduler(
        block_size: int = _DEFAULT_BLOCK_SIZE,
        offload_decode_save: int = 0,
        offload_partial_block_save_behavior: str = "drop",
        offload_partial_block_dynamic_pad_lower_limit: int = 0,
        offload_staging_buffer_tokens: int = -1,
    ):
        # update config
        vllm_config = MockVllmConfig(block_size=block_size)
        os.environ["TPU_OFFLOAD_DECODE_SAVE"] = str(offload_decode_save)
        os.environ[
            "TPU_OFFLOAD_PARTIAL_BLOCK_SAVE_BEHAVIOR"] = offload_partial_block_save_behavior
        os.environ["TPU_OFFLOAD_PARTIAL_BLOCK_DYNAMIC_PAD_LOWER_LIMIT"] = str(
            offload_partial_block_dynamic_pad_lower_limit)
        logger.info(f"-------jcgu----------: {offload_staging_buffer_tokens}")
        if offload_staging_buffer_tokens >= 0:
            os.environ["TPU_OFFLOAD_STAGING_BUFFER_TOKENS"] = str(
                offload_staging_buffer_tokens)
        return TPUConnectorScheduler(vllm_config)

    return _scheduler


class TestStagingBufferManager:

    def test_initialization(self):
        manager = _StagingBufferManager(num_blocks=100)
        assert manager.num_blocks == 100
        assert manager.get_num_free_staging_blocks() == 100
        assert manager.get_num_used_staging_blocks() == 0

    def test_allocate_simple(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id1: ReqId = "req1"
        req_id2: ReqId = "req2"

        allocated1 = manager.allocate(req_id1, 10, "load")
        assert allocated1 == 10
        assert manager.get_num_free_staging_blocks() == 90
        assert manager.get_num_used_staging_blocks() == 10
        assert manager._num_blocks_for_load == 10
        assert manager._num_blocks_for_save == 0

        allocated2 = manager.allocate(req_id2, 20, "save")
        assert allocated2 == 20
        assert manager.get_num_free_staging_blocks() == 70
        assert manager.get_num_used_staging_blocks() == 30
        assert manager._num_blocks_for_load == 10
        assert manager._num_blocks_for_save == 20

    def test_allocate_insufficient_capacity(self):
        manager = _StagingBufferManager(num_blocks=10)
        req_id: ReqId = "req1"
        allocated = manager.allocate(req_id, 20, "load")
        assert allocated == 0
        assert manager.get_num_free_staging_blocks() == 10
        assert manager.get_num_used_staging_blocks() == 0

    def test_allocate_existing_load_request(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "load")
        with pytest.raises(ValueError):
            # multiple concurrent loads from a single request is not allowed.
            manager.allocate(req_id, 5, "load")

    def test_allocate_existing_save_request(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "save")
        assert manager._blocks_for_save[req_id] == 10
        manager.allocate(req_id, 5, "save")
        assert manager._blocks_for_save[req_id] == 15
        assert manager.get_num_free_staging_blocks() == 85
        assert manager.get_num_used_staging_blocks() == 15

    def test_allocate_negative_blocks(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        allocated = manager.allocate(req_id, -5, "load")
        assert allocated == -5
        assert manager.get_num_free_staging_blocks() == 100

    def test_free_full(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "load")
        freed = manager.free(req_id, "load")
        assert freed == 10
        assert manager.get_num_free_staging_blocks() == 100
        assert manager.get_num_used_staging_blocks() == 0
        assert req_id not in manager._blocks_for_load

    def test_free_partial(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "save")
        freed = manager.free(req_id, "save", num_finished_blocks=4)
        assert freed == 4
        assert manager.get_num_free_staging_blocks() == 94
        assert manager.get_num_used_staging_blocks() == 6
        assert manager._blocks_for_save[req_id] == 6

    def test_free_more_than_allocated(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        manager.allocate(req_id, 10, "load")
        manager.free(req_id, "load", num_finished_blocks=15)
        assert req_id not in manager._blocks_for_load

    def test_free_non_existent_request(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id: ReqId = "req1"
        freed = manager.free(req_id, "load")
        assert freed == 0

    def test_get_usage(self):
        manager = _StagingBufferManager(num_blocks=100)
        req_id1: ReqId = "req1"
        req_id2: ReqId = "req2"
        manager.allocate(req_id1, 10, "load")
        manager.allocate(req_id2, 20, "save")

        usage_str = manager.get_usage()
        expected_str = "Staging Buffer: total=100, free=70, used_for_load=10, used_for_save=20;"
        assert usage_str == expected_str

        usage_str_details = manager.get_usage(with_details=True)
        assert "save_details:{req2:20,}" in usage_str_details
        assert "load_details:{req1:10,}" in usage_str_details

    def test_complex_scenario(self):
        manager = _StagingBufferManager(num_blocks=50)
        req1, req2, req3 = "req1", "req2", "req3"

        # req1 loads 10, req2 saves 15
        assert manager.allocate(req1, 10, "load") == 10
        assert manager.allocate(req2, 15, "save") == 15
        assert manager.get_num_free_staging_blocks() == 25
        assert manager.get_num_used_staging_blocks() == 25

        # req3 tries to load 30, fails
        assert manager.allocate(req3, 30, "load") == 0
        assert manager.get_num_free_staging_blocks() == 25

        # req1 finishes loading
        assert manager.free(req1, "load") == 10
        assert manager.get_num_free_staging_blocks() == 35

        # req3 can now load 20
        assert manager.allocate(req3, 20, "load") == 20
        assert manager.get_num_free_staging_blocks() == 15
        assert manager.get_num_used_staging_blocks(
        ) == 35  # 15 for save (req2) + 20 for load (req3)

        # req2 saves another 5
        assert manager.allocate(req2, 5, "save") == 5
        assert manager.get_num_free_staging_blocks() == 10
        assert manager._blocks_for_save[req2] == 20

        # req2 frees 8 blocks
        assert manager.free(req2, "save", 8) == 8
        assert manager.get_num_free_staging_blocks() == 18
        assert manager._blocks_for_save[req2] == 12

        # req2 and req3 finish
        assert manager.free(req2, "save") == 12
        assert manager.free(req3, "load") == 20
        assert manager.get_num_free_staging_blocks() == 50
        assert manager.get_num_used_staging_blocks() == 0


class TestTPUConnectorScheduler:

    def test_get_num_new_matched_tokens_no_hit(self, scheduler_factory,
                                               clean_backend_instance):
        """
        Tests that get_num_new_matched_tokens returns 0 when there is no
        matching prefix in the CPU cache.
        """
        scheduler = scheduler_factory()
        assert len(scheduler.cpu_backend.cache) == 0
        request = create_request("req1",
                                 list(range(scheduler.block_size * 2)),
                                 block_size=scheduler.block_size)
        num_matched, _ = scheduler.get_num_new_matched_tokens(request, 0)
        assert num_matched == 0
        assert request.request_id not in scheduler.load_specs

    @pytest.mark.parametrize(
        "num_computed_blocks, num_matched_blocks, num_prompt_blocks",
        [(0, 3, 4), (1, 3, 4), (3, 3, 4)],
    )
    def test_get_num_new_matched_tokens_partial_hit(self, scheduler_factory,
                                                    clean_backend_instance,
                                                    num_computed_blocks,
                                                    num_matched_blocks,
                                                    num_prompt_blocks):
        """
        Tests that get_num_new_matched_tokens correctly identifies a partial
        prefix hit and creates a LoadSpec.
        """

        scheduler = scheduler_factory()
        assert len(scheduler.cpu_backend.cache) == 0
        num_computed_tokens = num_computed_blocks * scheduler.block_size
        num_matched_tokens = num_matched_blocks * scheduler.block_size
        num_prompt_tokens = num_prompt_blocks * scheduler.block_size

        prompt_tokens = list(range(num_prompt_tokens))
        request = create_request("req1",
                                 prompt_tokens,
                                 block_size=scheduler.block_size)

        # Simulate a cache hit for the first 3 block
        keys_gen = scheduler.token_processor.process_tokens(prompt_tokens)
        keys = list(keys_gen)
        for i in range(num_matched_blocks):
            start, end, key = keys[i]
            scheduler.cpu_backend.add(key, "dummy_data")

        num_tokens_to_load, _ = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

        assert num_tokens_to_load == num_matched_tokens - num_computed_tokens
        if num_tokens_to_load > 0:
            assert request.request_id in scheduler.load_specs
            load_spec = scheduler.load_specs[request.request_id]
            assert load_spec.num_matched_tokens == num_matched_tokens
            assert load_spec.num_skip_leading_tokens == num_computed_tokens
            assert not load_spec.can_load

    @pytest.mark.parametrize(
        "num_computed_blocks, num_prompt_blocks",
        [(0, 4), (3, 4), (4, 4)],
    )
    def test_get_num_new_matched_tokens_full_hit(self, scheduler_factory,
                                                 clean_backend_instance,
                                                 num_computed_blocks,
                                                 num_prompt_blocks):
        """
        Tests the special case of a full prefix hit, where N-1 tokens are
        reported to the vLLM scheduler.
        """
        scheduler = scheduler_factory()
        assert len(scheduler.cpu_backend.cache) == 0

        num_computed_tokens = num_computed_blocks * scheduler.block_size
        num_prompt_tokens = num_prompt_blocks * scheduler.block_size
        num_matched_tokens = num_prompt_tokens

        prompt_tokens = list(range(num_prompt_tokens))
        request = create_request("req1",
                                 prompt_tokens,
                                 block_size=scheduler.block_size)

        # Simulate a cache hit for the entire prompt
        keys_gen = scheduler.token_processor.process_tokens(prompt_tokens)
        keys = list(keys_gen)
        for i in range(num_prompt_blocks):
            start, end, key = keys[i]
            scheduler.cpu_backend.add(key, "dummy_data")

        num_tokens_to_load, _ = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

        # Should report N-1 to scheduler, but LoadSpec should have the full N
        assert num_tokens_to_load == max(
            0, num_matched_tokens - num_computed_tokens - 1)
        if num_matched_tokens > num_computed_tokens:
            assert request.request_id in scheduler.load_specs
            load_spec = scheduler.load_specs[request.request_id]
            assert load_spec.num_matched_tokens == num_matched_tokens
            assert load_spec.num_skip_leading_tokens == num_computed_tokens
            assert not load_spec.can_load

    @pytest.mark.parametrize(
        "num_computed_blocks, num_prompt_blocks, num_staging_blocks",
        [(0, 4, 0), (0, 4, 2), (2, 4, 1)],
    )
    def test_get_num_new_matched_tokens_hit_with_limited_staging_buffer(
            self, scheduler_factory, clean_backend_instance,
            num_computed_blocks, num_prompt_blocks, num_staging_blocks):
        """
        Tests the special case of a full prefix hit, where N-1 tokens are
        reported to the vLLM scheduler.
        """
        num_staging_tokens = num_staging_blocks * _DEFAULT_BLOCK_SIZE
        scheduler = scheduler_factory(
            offload_staging_buffer_tokens=num_staging_tokens)
        assert len(scheduler.cpu_backend.cache) == 0

        num_computed_tokens = num_computed_blocks * scheduler.block_size
        num_prompt_tokens = num_prompt_blocks * scheduler.block_size
        num_matched_tokens = num_prompt_tokens

        prompt_tokens = list(range(num_prompt_tokens))
        request = create_request("req1",
                                 prompt_tokens,
                                 block_size=scheduler.block_size)

        # Simulate a cache hit for the entire prompt
        keys_gen = scheduler.token_processor.process_tokens(prompt_tokens)
        keys = list(keys_gen)
        for i in range(num_prompt_blocks):
            start, end, key = keys[i]
            scheduler.cpu_backend.add(key, "dummy_data")

        num_tokens_to_load, _ = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

        gt_num_tokens_to_load = min(num_matched_tokens - num_computed_tokens,
                                    num_staging_tokens)
        gt_num_matched_tokens = gt_num_tokens_to_load + num_computed_tokens

        # Should report N-1 to scheduler, but LoadSpec should have the full N
        if gt_num_matched_tokens == num_prompt_tokens:
            assert num_tokens_to_load == gt_num_tokens_to_load - 1

        if gt_num_matched_tokens > num_computed_tokens:
            assert request.request_id in scheduler.load_specs
            load_spec = scheduler.load_specs[request.request_id]
            assert load_spec.num_matched_tokens == gt_num_matched_tokens
            assert load_spec.num_skip_leading_tokens == num_computed_tokens
            assert len(load_spec.dst_blocks
                       ) == gt_num_tokens_to_load // scheduler.block_size
            assert not load_spec.can_load

    @pytest.mark.parametrize(
        "num_skip_leading_tokens, num_matched_tokens, save_behavior, dynamic_pad_lower_limit",
        [(0, _DEFAULT_BLOCK_SIZE * 4, "drop", 0),
         (0, _DEFAULT_BLOCK_SIZE * 4, "pad", 0),
         (0, _DEFAULT_BLOCK_SIZE * 4, "dynamic", 1),
         (0, _DEFAULT_BLOCK_SIZE * 4, "dynamic", _DEFAULT_BLOCK_SIZE - 1),
         (_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE * 4 + 2, "drop", 0),
         (_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE * 4 + 2, "pad", 0),
         (_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE * 4 + 1, "dynamic", 1),
         (_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE * 4 + 2, "dynamic", 1),
         (_DEFAULT_BLOCK_SIZE, _DEFAULT_BLOCK_SIZE * 5 - 1, "dynamic",
          _DEFAULT_BLOCK_SIZE - 1)],
    )
    def test_update_state_after_alloc(self, scheduler_factory,
                                      clean_backend_instance,
                                      num_skip_leading_tokens,
                                      num_matched_tokens, save_behavior,
                                      dynamic_pad_lower_limit):
        """
        Tests that update_state_after_alloc correctly updates the LoadSpec
        when blocks are allocated for a request with a cache hit.
        """
        scheduler = scheduler_factory(
            offload_partial_block_save_behavior=save_behavior,
            offload_partial_block_dynamic_pad_lower_limit=
            dynamic_pad_lower_limit)
        assert len(scheduler.cpu_backend.cache) == 0

        # The ground truth of loading decisions
        num_partial_block_tokens = num_matched_tokens % scheduler.block_size
        if save_behavior == "drop" or \
            (save_behavior == "dynamic"
             and num_partial_block_tokens < dynamic_pad_lower_limit):
            # the last partial blocks needs to be dropped
            num_matched_tokens -= num_partial_block_tokens

        num_external_tokens = num_matched_tokens - num_skip_leading_tokens
        num_blocks_to_skip = num_skip_leading_tokens // scheduler.block_size
        num_blocks_to_load = (num_external_tokens + scheduler.block_size -
                              1) // scheduler.block_size

        prompt_tokens = list(range(num_matched_tokens))
        request = create_request("req1",
                                 prompt_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=num_skip_leading_tokens)

        # Setup a pending load operation
        scheduler.load_specs[request.request_id] = MagicMock(
            num_matched_tokens=num_matched_tokens,
            num_skip_leading_tokens=num_skip_leading_tokens,
            dst_blocks=[-1] * num_blocks_to_load,
            can_load=False)

        # Mock allocated blocks
        allocated_blocks = MagicMock(spec=KVCacheBlocks)
        num_blocks = (num_matched_tokens + scheduler.block_size -
                      1) // scheduler.block_size
        allocated_block_ids = [i for i in range(num_blocks)]
        allocated_blocks.get_block_ids.return_value = [allocated_block_ids]

        scheduler.update_state_after_alloc(request, allocated_blocks,
                                           num_external_tokens)

        load_spec = scheduler.load_specs[request.request_id]
        assert load_spec.can_load
        assert len(load_spec.dst_blocks) == num_blocks_to_load
        assert load_spec.dst_blocks == allocated_block_ids[num_blocks_to_skip:(
            num_blocks_to_load + num_blocks_to_skip)]

    @pytest.mark.parametrize(
        "save_behavior, dynamic_pad_lower_limit, prompt_len, num_computed_tokens",
        [("drop", 0, _DEFAULT_BLOCK_SIZE * 4 + 2, 0),
         ("pad", 0, _DEFAULT_BLOCK_SIZE * 4 + 2, _DEFAULT_BLOCK_SIZE),
         ("dynamic", 1, _DEFAULT_BLOCK_SIZE * 4 + 2, 0),
         ("dynamic", _DEFAULT_BLOCK_SIZE - 1, _DEFAULT_BLOCK_SIZE * 4 + 2,
          _DEFAULT_BLOCK_SIZE)])
    def test_build_connector_meta_new_request(self, scheduler_factory,
                                              clean_backend_instance,
                                              save_behavior,
                                              dynamic_pad_lower_limit,
                                              prompt_len, num_computed_tokens):
        """
        Tests metadata generation for a new request (prefill) that has no
        cache hit and generates enough tokens to trigger a save.

        NOTE(jcgu):
        1. we will not cover load + save for new_request here, since load
           is determined by `get_num_new_matched_tokens()`

        """
        scheduler = scheduler_factory(
            offload_partial_block_save_behavior=save_behavior,
            offload_partial_block_dynamic_pad_lower_limit=
            dynamic_pad_lower_limit,
            offload_staging_buffer_tokens=2 * prompt_len)
        assert len(scheduler.cpu_backend.cache) == 0

        prompt_tokens = list(range(prompt_len))
        request = create_request("req1",
                                 prompt_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=num_computed_tokens)
        num_blocks = (prompt_len + scheduler.block_size -
                      1) // scheduler.block_size
        request.block_ids = [[i for i in range(num_blocks)]]
        new_scheduled_tokens = prompt_len - num_computed_tokens

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"req1": new_scheduled_tokens},
            total_num_scheduled_tokens=new_scheduled_tokens,
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        metadata = scheduler.build_connector_meta(scheduler_output)

        # ground_truth
        num_tokens_in_partial_block = prompt_len % scheduler.block_size
        num_processed_tokens = prompt_len
        if save_behavior == "drop" or (save_behavior == "dynamic"
                                       and num_tokens_in_partial_block
                                       < dynamic_pad_lower_limit):
            num_processed_tokens = (
                prompt_len // scheduler.block_size) * scheduler.block_size
        num_skip_blocks = num_computed_tokens // scheduler.block_size
        num_blocks_to_save = (num_processed_tokens + scheduler.block_size -
                              1) // scheduler.block_size - num_skip_blocks

        assert len(metadata.requests_meta) == 1
        req_meta = metadata.requests_meta[0]
        assert req_meta.req_id == "req1"
        assert req_meta.load_spec is None
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.num_total_tokens == num_processed_tokens
        assert req_meta.save_spec.num_skip_leading_tokens == num_computed_tokens
        assert len(req_meta.save_spec.src_blocks) == num_blocks_to_save
        assert req_meta.save_spec.src_blocks == request.block_ids[0][
            num_skip_blocks:(num_skip_blocks + num_blocks_to_save)]
        assert not req_meta.save_spec.is_final_save

        tracker = scheduler._request_trackers["req1"]
        assert tracker.save_watermark == num_processed_tokens
        assert tracker.block_ids == request.block_ids[0]

    @pytest.mark.parametrize(
        "save_behavior, dynamic_pad_lower_limit, prompt_len, num_computed_tokens, num_staging_blocks",
        [
            ("drop", 0, _DEFAULT_BLOCK_SIZE * 4 + 2, 0, 0),
            ("drop", 0, _DEFAULT_BLOCK_SIZE * 4 + 2, 0, 2),
        ])
    def test_build_connector_meta_new_request_with_limited_staging_buffer(
            self, scheduler_factory, clean_backend_instance, save_behavior,
            dynamic_pad_lower_limit, prompt_len, num_computed_tokens,
            num_staging_blocks):
        """
        get a new request, but limited staging buffer.
        """
        num_staging_buffer_tokens = num_staging_blocks * _DEFAULT_BLOCK_SIZE

        scheduler = scheduler_factory(
            offload_partial_block_save_behavior=save_behavior,
            offload_partial_block_dynamic_pad_lower_limit=
            dynamic_pad_lower_limit,
            offload_staging_buffer_tokens=num_staging_buffer_tokens)
        assert len(scheduler.cpu_backend.cache) == 0

        prompt_tokens = list(range(prompt_len))
        request = create_request("req1",
                                 prompt_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=num_computed_tokens)
        num_blocks = (prompt_len + scheduler.block_size -
                      1) // scheduler.block_size
        request.block_ids = [[i for i in range(num_blocks)]]
        new_scheduled_tokens = prompt_len - num_computed_tokens

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={"req1": new_scheduled_tokens},
            total_num_scheduled_tokens=new_scheduled_tokens,
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        metadata = scheduler.build_connector_meta(scheduler_output)

        # ground_truth
        num_tokens_in_partial_block = prompt_len % scheduler.block_size
        num_processed_tokens = prompt_len
        if save_behavior == "drop" or (save_behavior == "dynamic"
                                       and num_tokens_in_partial_block
                                       < dynamic_pad_lower_limit):
            num_processed_tokens = (
                prompt_len // scheduler.block_size) * scheduler.block_size
        num_skip_blocks = num_computed_tokens // scheduler.block_size
        num_blocks_to_save = (num_processed_tokens + scheduler.block_size -
                              1) // scheduler.block_size - num_skip_blocks

        # throttled by the limited staging buffer
        if num_blocks_to_save > num_staging_blocks:
            num_blocks_to_save = num_staging_blocks
            num_processed_tokens = num_staging_buffer_tokens + num_computed_tokens

        if num_blocks_to_save > 0:
            assert len(metadata.requests_meta) == 1
            req_meta = metadata.requests_meta[0]
            assert req_meta.req_id == "req1"
            assert req_meta.load_spec is None
            assert req_meta.save_spec is not None
            assert req_meta.save_spec.num_total_tokens == num_processed_tokens
            assert req_meta.save_spec.num_skip_leading_tokens == num_computed_tokens
            assert len(req_meta.save_spec.src_blocks) == num_blocks_to_save
            assert req_meta.save_spec.src_blocks == request.block_ids[0][
                num_skip_blocks:(num_skip_blocks + num_blocks_to_save)]
            assert not req_meta.save_spec.is_final_save

            tracker = scheduler._request_trackers["req1"]
            assert tracker.save_watermark == num_processed_tokens
            assert tracker.block_ids == request.block_ids[0]

    @pytest.mark.parametrize(
        "save_behavior, dynamic_pad_lower_limit, decode_save, prompt_len",
        [("drop", 0, 0, _DEFAULT_BLOCK_SIZE * 4 + 2),
         ("drop", 0, 1, _DEFAULT_BLOCK_SIZE * 4 + 3),
         ("pad", 0, 0, _DEFAULT_BLOCK_SIZE * 4 + 2),
         ("pad", 0, 1, _DEFAULT_BLOCK_SIZE * 4 + 3),
         ("dynamic", 1, 0, _DEFAULT_BLOCK_SIZE * 4 + 2),
         ("dynamic", _DEFAULT_BLOCK_SIZE - 1, 1, _DEFAULT_BLOCK_SIZE * 4 + 3)])
    def test_build_connector_meta_cached_request_with_one_decode(
            self, scheduler_factory, clean_backend_instance, save_behavior,
            dynamic_pad_lower_limit, decode_save, prompt_len):
        """
        Tests metadata generation for a running request (chunked prefill)
        that gets more tokens scheduled, styled as a single unit test.
        """
        scheduler = scheduler_factory(
            offload_decode_save=decode_save,
            offload_partial_block_save_behavior=save_behavior,
            offload_partial_block_dynamic_pad_lower_limit=
            dynamic_pad_lower_limit)
        assert len(scheduler.cpu_backend.cache) == 0

        gen_len = 1  # single decode step
        num_total_tokens = prompt_len + gen_len
        request_tokens = list(range(num_total_tokens))
        num_prompt_blocks = (prompt_len + scheduler.block_size -
                             1) // scheduler.block_size
        num_total_blocks = (num_total_tokens + scheduler.block_size -
                            1) // scheduler.block_size
        request = create_request("req1",
                                 request_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=prompt_len)
        request.block_ids = [[i for i in range(num_total_blocks)]]

        # Arrange: Set up the scheduler's state to simulate a request that has
        # already been partially processed.
        initial_tokens = request_tokens[:prompt_len]
        initial_block_ids = [i for i in range(num_prompt_blocks)]

        initial_save_watermark = prompt_len
        num_tokens_in_partial_block = prompt_len % scheduler.block_size
        if save_behavior == "drop" or (save_behavior == "dynamic"
                                       and num_tokens_in_partial_block
                                       < dynamic_pad_lower_limit):
            initial_save_watermark = (
                prompt_len // scheduler.block_size) * scheduler.block_size

        tracker = RequestTracker(
            req_id="req1",
            prompt_len=prompt_len,
            token_ids=initial_tokens,
            block_ids=initial_block_ids,
            save_watermark=initial_save_watermark,
            is_decode_phase=False,
        )
        scheduler._request_trackers["req1"] = tracker
        scheduler._unfinished_requests["req1"] = request

        # Act: Simulate a decode step
        new_blocks_ids = [
            i for i in range(num_prompt_blocks, num_total_blocks)
        ]
        logger.info(f"new_blocks_ids: {new_blocks_ids}")
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = ["req1"]
        cached_req_data.new_block_ids = (new_blocks_ids, )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={"req1": gen_len},
            total_num_scheduled_tokens=gen_len,
            finished_req_ids=set(),
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        metadata = scheduler.build_connector_meta(scheduler_output)

        # The tracker should be updated with the new tokens and blocks.
        updated_tracker = scheduler._request_trackers["req1"]
        assert updated_tracker.token_ids == request_tokens
        assert updated_tracker.block_ids == request.block_ids[0]
        assert updated_tracker.is_decode_phase

        # ground-truth of save
        if not decode_save:
            assert updated_tracker.save_watermark == initial_save_watermark
        else:
            # NOTE(jcgu): currently still mimic the internal logic,
            # find a better way to
            next_save_boundary = (
                initial_save_watermark // scheduler.block_size +
                1) * scheduler.block_size
            if num_total_tokens < next_save_boundary:
                # nothing to save
                assert updated_tracker.save_watermark == initial_save_watermark
            else:
                # Assert: Verify the generated metadata and the updated tracker state.
                assert len(metadata.requests_meta) == 1
                req_meta = metadata.requests_meta[0]
                assert req_meta.req_id == "req1"
                assert req_meta.load_spec is None

                # a block (maybe part of its tokens has been saved) should be saved.
                assert req_meta.save_spec.num_total_tokens == num_total_tokens
                assert req_meta.save_spec.num_skip_leading_tokens == num_total_tokens - scheduler.block_size
                assert req_meta.save_spec.src_blocks == [
                    request.block_ids[0][-1]
                ]
                assert not req_meta.save_spec.is_final_save

                assert updated_tracker.save_watermark == num_total_tokens

    @pytest.mark.parametrize(
        "save_behavior, dynamic_pad_lower_limit, decode_save, prompt_len, gen_len",
        [("drop", 0, 0, _DEFAULT_BLOCK_SIZE * 4 + 2, 3),
         ("pad", 0, 1, _DEFAULT_BLOCK_SIZE * 4 + 2, 1),
         ("pad", 0, 1, _DEFAULT_BLOCK_SIZE * 4 + 2, 4),
         ("dynamic", 1, 1, _DEFAULT_BLOCK_SIZE * 4 + 2, 1),
         ("dynamic", _DEFAULT_BLOCK_SIZE - 1, 1, _DEFAULT_BLOCK_SIZE * 4 + 2,
          4)])
    def test_build_connector_meta_finished_request(
            self, scheduler_factory, clean_backend_instance, save_behavior,
            dynamic_pad_lower_limit, decode_save, prompt_len, gen_len):
        """
        Tests metadata generation for a finishing request.
        """
        scheduler = scheduler_factory(
            offload_decode_save=decode_save,
            offload_partial_block_save_behavior=save_behavior,
            offload_partial_block_dynamic_pad_lower_limit=
            dynamic_pad_lower_limit)
        assert len(scheduler.cpu_backend.cache) == 0

        num_total_tokens = prompt_len + gen_len
        request_tokens = list(range(num_total_tokens))
        num_total_blocks = (num_total_tokens + scheduler.block_size -
                            1) // scheduler.block_size
        request = create_request("req1",
                                 request_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=prompt_len)
        request.block_ids = [[i for i in range(num_total_blocks)]]

        # Arrange: Set up the scheduler's state to simulate a request that has
        # already been processed.
        num_tokens_in_partial_block = prompt_len % scheduler.block_size

        adjusted_prompt_len = prompt_len
        if save_behavior == "drop" or (save_behavior == "dynamic"
                                       and num_tokens_in_partial_block
                                       < dynamic_pad_lower_limit):
            adjusted_prompt_len = (prompt_len //
                                   scheduler.block_size) * scheduler.block_size

        latest_save_watermark = adjusted_prompt_len
        if decode_save:
            num_full_block_tokens = num_total_tokens // scheduler.block_size * scheduler.block_size
            latest_save_watermark = max(num_full_block_tokens,
                                        adjusted_prompt_len)
            logger.info(
                f"latest_save_watermark: {latest_save_watermark}, {num_full_block_tokens}, {adjusted_prompt_len}"
            )

        tracker = RequestTracker(
            req_id="req1",
            prompt_len=prompt_len,
            token_ids=request_tokens,
            block_ids=request.block_ids[0],
            save_watermark=latest_save_watermark,
            is_decode_phase=True,
        )
        scheduler._request_trackers["req1"] = tracker
        scheduler._unfinished_requests["req1"] = request

        finished_req_ids = set()
        finished_req_ids.add("req1")
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=finished_req_ids,
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )

        metadata = scheduler.build_connector_meta(scheduler_output)
        req_meta = metadata.requests_meta[0]
        assert req_meta.load_spec is None

        # ground-truth of save
        if not decode_save:
            assert req_meta.save_spec.is_final_save
            assert req_meta.save_spec.skip_save
            assert req_meta.save_spec.src_blocks == []
        else:
            # since it's a finished request, tokens are saved until the last full block (thanks to decode_save's next_block_boundary)
            num_tokens_in_last_partial_block = num_total_tokens % scheduler.block_size
            if save_behavior == "drop" or (save_behavior == "dynamic"
                                           and num_tokens_in_last_partial_block
                                           < dynamic_pad_lower_limit):
                # if drop, then no blocks to save
                assert req_meta.save_spec.is_final_save
                assert req_meta.save_spec.skip_save
                assert req_meta.save_spec.src_blocks == []
            else:
                # otherwise, save
                num_skip_leading_blocks = tracker.save_watermark // scheduler.block_size
                num_skip_leading_tokens = num_skip_leading_blocks * scheduler.block_size
                assert req_meta.save_spec.num_total_tokens == num_total_tokens
                assert req_meta.save_spec.num_skip_leading_tokens == num_skip_leading_tokens
                assert req_meta.save_spec.src_blocks == request.block_ids[0][
                    num_skip_leading_blocks:]
                assert req_meta.save_spec.is_final_save
                assert not req_meta.save_spec.skip_save
