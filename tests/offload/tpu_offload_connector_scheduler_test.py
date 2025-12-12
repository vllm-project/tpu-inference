# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pytest
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.request import Request

from tpu_inference.offload.tpu_offload_connector import (
    RequestTracker, TPUOffloadConnectorScheduler)

_DEFAULT_BLOCK_SIZE = 16


class MockVllmConfig:

    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
        self.model_config = self.Model()
        self.cache_config = self.Cache(block_size)

    class Model:
        model = "test-model"

    class Cache:

        def __init__(self, block_size):
            self.block_size = block_size


def create_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int,
    num_computed_tokens: int = 0,
    generated_token_ids: list[int] = [],
) -> Request:
    """Creates a mock vLLM request object."""
    req = MagicMock(spec=Request)
    req.request_id = request_id
    req.req_id = request_id  # for NewRequestData
    req.prompt_token_ids = prompt_token_ids
    req.all_token_ids = prompt_token_ids + generated_token_ids
    req.num_computed_tokens = num_computed_tokens + len(generated_token_ids)
    req.block_size = block_size
    req.block_ids = [[]]
    # Mock the block_hashes property to return a list of mock hashes
    req.block_hashes = [
        f"hash_{i}".encode()
        for i in range(len(req.all_token_ids) // block_size)
    ]
    return req


@pytest.fixture
def scheduler_factory():
    """Provides a factory function for Scheduler instances."""

    def _scheduler(
        block_size: int = _DEFAULT_BLOCK_SIZE,
        offload_decode_save: int = 0,
        offload_num_staging_blocks: int = -1,
        offload_num_cpu_chunks: int = -1,
    ):
        # update config
        vllm_config = MockVllmConfig(block_size=block_size)
        os.environ["TPU_OFFLOAD_DECODE_SAVE"] = str(offload_decode_save)
        if offload_num_staging_blocks >= 0:
            os.environ["TPU_OFFLOAD_NUM_STAGING_BLOCKS"] = str(
                offload_num_staging_blocks)
        if offload_num_cpu_chunks > 0:
            os.environ["TPU_OFFLOAD_NUM_CPU_CHUNKS"] = str(
                offload_num_cpu_chunks)

        return TPUOffloadConnectorScheduler(vllm_config)

    return _scheduler


class TestTPUOffloadConnectorScheduler:

    def test_get_num_new_matched_tokens_no_hit(self, scheduler_factory):
        """
        Tests that get_num_new_matched_tokens returns 0 for a cache miss.
        """
        scheduler = scheduler_factory()
        request = create_request("req1", [1] * 32, scheduler.block_size)

        num_matched, _ = scheduler.get_num_new_matched_tokens(request, 0)
        assert num_matched == 0
        assert "req1" not in scheduler.load_specs

    @pytest.mark.parametrize(
        "num_computed_blocks, num_matched_blocks, num_prompt_blocks, num_staging_blocks",
        [(0, 2, 4, 10), (1, 2, 4, 10), (0, 4, 4, 10), (1, 4, 4, 10),
         (1, 4, 4, 1), (1, 4, 4, 0)])
    def test_get_num_new_matched_tokens_hit(self, scheduler_factory,
                                            num_computed_blocks,
                                            num_matched_blocks,
                                            num_prompt_blocks,
                                            num_staging_blocks):
        """
        Tests correct identification of a prefix hit (partial and full).
        test cases:
        1. no-skip + load 2 blocks + no staging buffer limit
        2. skip 1 block + load 1 block + no staging buffer limit
        3. no-skip + full-hit + no staging buffer limit
        4. skip 1 block + full-hit + no staging buffer limit
        5. skip 1 block + full-hit + only 1 staging block
        6. skip 1 block + full-hit + no staging block
        """
        scheduler = scheduler_factory(
            offload_num_staging_blocks=num_staging_blocks)
        prompt_len = scheduler.block_size * num_prompt_blocks
        num_computed_tokens = scheduler.block_size * num_computed_blocks
        num_blocks_to_load = num_matched_blocks - num_computed_blocks
        # consider the case of limited staging blocks
        num_blocks_to_load = min(num_blocks_to_load, num_staging_blocks)
        num_matched_blocks = num_blocks_to_load + num_computed_blocks
        num_matched_tokens = num_matched_blocks * scheduler.block_size

        request = create_request("req1", list(range(prompt_len)),
                                 scheduler.block_size)

        # init offload_manager state
        matched_block_hashes = request.block_hashes[:num_matched_blocks]
        allocated_chunks, _ = scheduler.offload_manager.allocate_for_save(
            matched_block_hashes)
        scheduler.offload_manager.complete_save(matched_block_hashes)

        # call fn
        num_external_matched_tokens, _ = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

        # check external_matched_tokens
        if num_matched_blocks == num_prompt_blocks:
            assert num_external_matched_tokens == num_blocks_to_load * scheduler.block_size - 1
        else:
            assert num_external_matched_tokens == num_blocks_to_load * scheduler.block_size

        # check scheduler internal states
        # cache_hits
        assert "req1" in scheduler._external_cache_hits
        assert scheduler._external_cache_hits["req1"] == num_matched_tokens
        if num_blocks_to_load > 0:
            # load_spec
            assert "req1" in scheduler._pre_load_specs
            load_spec = scheduler._pre_load_specs["req1"]
            assert load_spec.num_matched_tokens == num_matched_tokens
            assert not load_spec.can_load
            assert len(load_spec.src_chunks) == num_blocks_to_load
            assert load_spec.num_skip_leading_tokens == num_computed_tokens
            assert len(load_spec.dst_blocks) == num_blocks_to_load
            # staging_buffer
            assert "req1" in scheduler.staging_buffer_manager._blocks_for_load
            assert scheduler.staging_buffer_manager._blocks_for_load[
                "req1"] == num_blocks_to_load
            assert scheduler.staging_buffer_manager.get_num_free_staging_blocks(
            ) == num_staging_blocks - num_blocks_to_load
        else:
            assert "req1" not in scheduler._pre_load_specs
            assert "req1" not in scheduler.staging_buffer_manager._blocks_for_load

    def test_update_state_after_alloc(self, scheduler_factory):
        """
        Tests that a LoadSpec is correctly updated after block allocation.
        """
        scheduler = scheduler_factory()
        req_id = "req1"
        num_prompt_blocks = 4
        num_matched_blocks = 3
        num_computed_blocks = 2
        num_blocks_to_load = num_matched_blocks - num_computed_blocks
        num_prompt_tokens = num_prompt_blocks * scheduler.block_size
        num_matched_tokens = num_matched_blocks * scheduler.block_size
        num_tokens_to_load = scheduler.block_size * num_blocks_to_load

        request = create_request(req_id, [0] * num_prompt_tokens,
                                 scheduler.block_size)

        # init offload_manager state
        matched_block_hashes = request.block_hashes[:num_matched_blocks]
        allocated_chunks, _ = scheduler.offload_manager.allocate_for_save(
            matched_block_hashes)
        scheduler.offload_manager.complete_save(matched_block_hashes)

        # Setup a pending load
        scheduler._pre_load_specs[req_id] = MagicMock(
            num_matched_tokens=num_matched_tokens,
            num_skip_leading_tokens=num_computed_blocks * scheduler.block_size,
            dst_blocks=[-1] * num_blocks_to_load,
            src_chunks=[i for i in range(num_blocks_to_load)],
            can_load=False)

        # Mock allocated blocks
        allocated_blocks = MagicMock(spec=KVCacheBlocks)
        allocated_block_ids = [i for i in range(num_prompt_blocks)]
        allocated_blocks.get_block_ids.return_value = [allocated_block_ids]

        scheduler.update_state_after_alloc(request, allocated_blocks,
                                           num_tokens_to_load)

        load_spec = scheduler.load_specs[req_id]
        assert load_spec.can_load
        assert load_spec.dst_blocks == allocated_block_ids[
            num_computed_blocks:num_matched_blocks]
        assert req_id in scheduler._reqs_being_loaded
        assert len(scheduler._reqs_being_loaded[req_id]) == num_blocks_to_load

    @pytest.mark.parametrize(
        "num_computed_tokens, num_matched_tokens, num_prompt_tokens, num_staging_tokens",
        [(0, 0, 64, 160),
         (0, 32, 64, 160), (16, 32, 64, 160), (0, 64, 64, 160),
         (16, 64, 64, 160), (0, 32, 64, 48), (0, 32, 64, 16)])
    def test_build_connector_meta_new_prefill(self, scheduler_factory,
                                              num_computed_tokens,
                                              num_matched_tokens,
                                              num_prompt_tokens,
                                              num_staging_tokens):
        """
        Tests metadata generation for a new request (prefill) with no cache hit.
        1. no hit + save 4 blocks
        2. partial hit (no-skip + load 2 blocks) + save 2 blocks
        3. partial hit (skip 1 block + load 1 blocks) + save 2 blocks
        4. full hit (no-skip + load 4 blocks) + no-save
        5. full hit (skip 1 block + load 3 blocks) + no-save
        6. partial hit (no-skip + load 2 blocks) + save 2 blocks + 3 staging blocks limit
        7. partial hit (no-skip + load 2 blocks) + save 2 blocks + 1 staging blocks limit
        """
        num_staging_blocks = num_staging_tokens // _DEFAULT_BLOCK_SIZE
        scheduler = scheduler_factory(
            offload_num_staging_blocks=num_staging_blocks,
            offload_num_cpu_chunks=100)

        # calculate the groundtruth
        num_computed_blocks = num_computed_tokens // scheduler.block_size
        num_matched_blocks = num_matched_tokens // scheduler.block_size
        num_prompt_blocks = cdiv(num_prompt_tokens, scheduler.block_size)

        num_blocks_to_load = num_matched_blocks - num_computed_blocks
        # adjustment based on staging_block limitation
        if num_blocks_to_load > num_staging_blocks:
            num_blocks_to_load = num_staging_blocks
            num_matched_blocks = num_blocks_to_load + num_computed_blocks
            num_matched_tokens = num_matched_blocks * scheduler.block_size

        remaining_staging_blocks = num_staging_blocks - num_blocks_to_load
        num_blocks_to_save = num_prompt_blocks - num_matched_blocks
        if num_blocks_to_save > remaining_staging_blocks:
            num_blocks_to_save = remaining_staging_blocks
            # reconfig staging_buffer limit for save
            scheduler.staging_buffer_manager._num_free_blocks = remaining_staging_blocks
        num_tokens_in_cache = (num_matched_blocks +
                               num_blocks_to_save) * scheduler.block_size

        req_id = "req1"
        request = create_request(req_id,
                                 list(range(num_prompt_tokens)),
                                 scheduler.block_size,
                                 num_computed_tokens=num_computed_tokens)
        request.block_ids = [[i for i in range(num_prompt_blocks)]]

        # init offload_manager state
        if num_matched_blocks > 0:
            matched_block_hashes = request.block_hashes[:num_matched_blocks]
            allocated_chunks, _ = scheduler.offload_manager.allocate_for_save(
                matched_block_hashes)
            scheduler.offload_manager.complete_save(matched_block_hashes)
            # allocated_chunk_ids = [chunk.chunk_id for chunk in allocated_chunks]
            # load_src_chunk_ids = allocated_chunk_ids[num_computed_blocks:]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={
                "req1": num_prompt_tokens - num_computed_tokens
            },
            total_num_scheduled_tokens=num_prompt_tokens - num_computed_tokens,
            finished_req_ids=set(),
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            free_encoder_mm_hashes=[],
        )

        # Mock that the scheduler has seen this request
        scheduler._unfinished_requests["req1"] = request
        scheduler._external_cache_hits["req1"] = num_matched_tokens
        if num_blocks_to_load > 0:
            scheduler.load_specs[req_id] = MagicMock(
                num_matched_tokens=num_matched_tokens,
                num_skip_leading_tokens=num_computed_tokens,
                dst_blocks=[-1] * num_blocks_to_load,
                src_chunks=[i for i in range(num_blocks_to_load)],
                can_load=True)

        metadata = scheduler.build_connector_meta(scheduler_output)

        if num_blocks_to_load + num_blocks_to_save == 0:
            # no load or store
            assert len(metadata.requests_meta) == 0
        else:
            req_meta = metadata.requests_meta[0]
            assert req_meta.req_id == "req1"
            if num_blocks_to_load == 0:
                assert req_meta.load_spec is None
            else:
                # load
                assert req_meta.load_spec is not None
                # NOTE(jcgu): no need to check details, since they are
                # generated by other functions.
            if num_blocks_to_save == 0:
                assert req_meta.save_spec is None
            else:
                # save
                assert req_meta.save_spec is not None
                assert req_meta.save_spec.num_total_tokens == num_tokens_in_cache
                assert req_meta.save_spec.num_skip_leading_tokens == num_matched_blocks * scheduler.block_size
                assert req_meta.save_spec.src_blocks == request.block_ids[0][
                    num_matched_blocks:num_matched_blocks + num_blocks_to_save]
                assert len(req_meta.save_spec.dst_chunks) == num_blocks_to_save
                assert not req_meta.save_spec.is_final_save
                assert "req1" in scheduler.staging_buffer_manager._blocks_for_save
                assert scheduler.staging_buffer_manager._blocks_for_save[
                    "req1"] == num_blocks_to_save
                assert "req1" in scheduler._reqs_being_saved
                assert len(
                    scheduler._reqs_being_saved["req1"]) == num_blocks_to_save

        assert "req1" in scheduler._request_trackers
        tracker = scheduler._request_trackers["req1"]
        # after creating SaveSpec, we also update tracker.save_watermark
        assert tracker.save_watermark == num_tokens_in_cache

    @pytest.mark.parametrize("prompt_len, seq_len, decode_save", [(63, 64, 1),
                                                                  (18, 64, 1),
                                                                  (18, 64, 0)])
    def test_build_connector_meta_decode_with_save(self, scheduler_factory,
                                                   prompt_len, seq_len,
                                                   decode_save):
        """
        Tests metadata generation for a decode step that triggers a save.
        1. the first decode (hit block boundary) + decode_save (save one block)
        2. th N-th decode (hit block bounary) + decode_save (save one block)
        2. th N-th decode (hit block bounary) + not decode_save (no save)
        """

        scheduler = scheduler_factory(offload_decode_save=decode_save,
                                      offload_num_staging_blocks=10,
                                      offload_num_cpu_chunks=10)

        prompt_tokens = list(range(prompt_len))
        generated_tokens = list(range(prompt_len, seq_len))
        req_id = "req1"
        request = create_request(req_id,
                                 prompt_token_ids=prompt_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=seq_len,
                                 generated_token_ids=generated_tokens)
        num_blocks = cdiv(seq_len, scheduler.block_size)
        request.block_ids = [i for i in range(num_blocks)]

        if decode_save == 1:
            # the last token in seq hasn't been computed (kv) yet
            num_saved_tokens = (
                (seq_len - 1) // scheduler.block_size) * scheduler.block_size
        else:
            num_saved_tokens = (prompt_len //
                                scheduler.block_size) * scheduler.block_size

        # Setup initial state
        # request tracker only tracks the computed tokens
        tracker = RequestTracker(req_id="req1",
                                 prompt_len=prompt_len,
                                 token_ids=request.all_token_ids[:-1],
                                 block_ids=request.block_ids,
                                 save_watermark=num_saved_tokens)

        scheduler._request_trackers["req1"] = tracker
        scheduler._unfinished_requests["req1"] = request

        # Simulate a decode step
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = ["req1"]
        cached_req_data.new_block_ids = ([], )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_req_data,
            num_scheduled_tokens={"req1": 1},
            total_num_scheduled_tokens=1,
            finished_req_ids=set(),
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            free_encoder_mm_hashes=[],
        )

        metadata = scheduler.build_connector_meta(scheduler_output)

        if seq_len % scheduler.block_size != 0 or decode_save != 1:
            # no save when there is no new full computed block
            assert len(metadata.requests_meta) == 0
        else:
            req_meta = metadata.requests_meta[0]
            # save spec
            assert req_meta.req_id == "req1"
            assert req_meta.load_spec is None
            assert req_meta.save_spec is not None
            assert req_meta.save_spec.num_total_tokens == seq_len
            assert req_meta.save_spec.num_skip_leading_tokens == num_saved_tokens
            assert req_meta.save_spec.src_blocks == [num_blocks - 1]
            assert len(req_meta.save_spec.dst_chunks) == 1
            assert not req_meta.save_spec.is_final_save
            # staging buffer
            assert "req1" in scheduler.staging_buffer_manager._blocks_for_save
            assert scheduler.staging_buffer_manager._blocks_for_save[
                "req1"] == 1
            # chunk_id for save
            assert "req1" in scheduler._reqs_being_saved
            assert len(scheduler._reqs_being_saved["req1"]) == 1

            assert tracker.save_watermark == seq_len

    def test_build_connector_meta_finished_request(self, scheduler_factory):
        """
        Tests metadata generation for a finished request.
        When using request's default block hash (fully-computed blocks only),
        a finished request either saves the last full block in their last
        decode step, or given up the last partial block; when it's treated as a
        finished request, there is no blocks to save.

        """

        scheduler = scheduler_factory(offload_decode_save=1)
        prompt_len = scheduler.block_size + 4
        final_seq_len = scheduler.block_size * 2 + 3
        prompt_tokens = list(range(prompt_len))
        generated_tokens = list(range(prompt_len, final_seq_len))
        req_id = "req1"
        request = create_request(req_id,
                                 prompt_token_ids=prompt_tokens,
                                 block_size=scheduler.block_size,
                                 num_computed_tokens=final_seq_len,
                                 generated_token_ids=generated_tokens)
        num_blocks = cdiv(final_seq_len, scheduler.block_size)
        request.block_ids = [i for i in range(num_blocks)]

        num_saved_tokens = (final_seq_len //
                            scheduler.block_size) * scheduler.block_size

        # Setup initial state
        tracker = RequestTracker(req_id="req1",
                                 prompt_len=prompt_len,
                                 token_ids=request.all_token_ids[:-1],
                                 block_ids=request.block_ids,
                                 save_watermark=num_saved_tokens)
        scheduler._request_trackers["req1"] = tracker
        scheduler._unfinished_requests["req1"] = request

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids={"req1"},
            scheduled_encoder_inputs={},
            scheduled_spec_decode_tokens={},
            num_common_prefix_blocks=0,
            free_encoder_mm_hashes=[],
        )

        metadata = scheduler.build_connector_meta(scheduler_output)

        assert req_id not in scheduler._unfinished_requests
        assert req_id not in scheduler._request_trackers
        assert len(metadata.requests_meta) == 1
        req_meta = metadata.requests_meta[0]
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.is_final_save
        assert req_meta.save_spec.skip_save
        assert req_meta.save_spec.src_blocks == []
        assert req_meta.save_spec.dst_chunks == []
