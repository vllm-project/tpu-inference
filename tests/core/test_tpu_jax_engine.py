# SPDX-License-Identifier: Apache-2.0
"""Tests for JaxEngine._schedule."""
from unittest.mock import Mock

import unittest
import torch
from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request

from tpu_commons.core.tpu_jax_engine import JaxEngine

EOS_TOKEN_ID = 50256


class JaxEngineTest(unittest.TestCase):
    def create_test_jax_engine(
        self,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 1024,
        block_size: int = 16,
        num_blocks: int = 1000,
    ) -> JaxEngine:
        """Creates a JaxEngine instance for testing."""
        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_num_batched_tokens,
        )
        model_config = ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            task="auto",
        )
        cache_config = CacheConfig(
            block_size=block_size,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
        )
        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
        )

        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(['layer'],
                                 FullAttentionSpec(block_size, 1, 1, torch.float32,
                                                   False))
            ],
        )
        cache_config.num_gpu_blocks = num_blocks

        kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=vllm_config.scheduler_config.max_model_len,
            enable_caching=False,
        )

        mock_executor = Mock()
        mock_executor.driver_worker.model_runner = Mock()

        engine = JaxEngine(vllm_config, kv_cache_manager, mock_executor)
        return engine


    def create_requests(self,
                        num_requests: int,
                        num_tokens: int = 10,
                        max_tokens: int = 16) -> list[Request]:
        """Creates a list of requests for testing."""
        sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
        requests = []
        for i in range(num_requests):
            request = Request(
                request_id=f"{i}",
                prompt_token_ids=[i] * num_tokens,
                sampling_params=sampling_params,
                pooling_params=None,
                multi_modal_inputs=None,
                multi_modal_placeholders=None,
                multi_modal_hashes=None,
                eos_token_id=EOS_TOKEN_ID,
            )
            requests.append(request)
        return requests


    def test_jax_engine_schedule_new_requests(self):
        """Tests scheduling of new requests that fit within the token limit."""
        engine = self.create_test_jax_engine(max_num_batched_tokens=100)
        requests = self.create_requests(num_requests=2, num_tokens=40)

        for req in requests:
            engine.add_request(req)

        assert len(engine._new_requests) == 2
        assert not engine._requests

        output = engine._schedule()

        assert len(output.scheduled_new_reqs) == 2
        assert not output.scheduled_cached_reqs
        assert output.total_num_scheduled_tokens == 80

        assert output.num_scheduled_tokens[requests[0].request_id] == 40
        assert output.num_scheduled_tokens[requests[1].request_id] == 40

        assert not engine._new_requests
        assert len(engine._requests) == 2
        assert engine._requests[0].request_id == "0"
        assert engine._requests[1].request_id == "1"


    def test_jax_engine_schedule_new_requests_with_limit(self):
        """Tests scheduling new requests when they exceed the token limit."""
        engine = self.create_test_jax_engine(max_num_batched_tokens=50)
        requests = self.create_requests(num_requests=2, num_tokens=40)

        for req in requests:
            engine.add_request(req)

        output = engine._schedule()

        # The first request is fully scheduled, the second is partially scheduled.
        assert len(output.scheduled_new_reqs) == 2
        assert not output.scheduled_cached_reqs
        assert output.total_num_scheduled_tokens == 50

        assert output.num_scheduled_tokens[requests[0].request_id] == 40
        assert output.num_scheduled_tokens[requests[1].request_id] == 10

        assert not engine._new_requests
        assert len(engine._requests) == 2


    def test_jax_engine_schedule_running_requests_chunked(self):
        """Tests scheduling of a running request (chunked prefill)."""
        engine = self.create_test_jax_engine(max_num_batched_tokens=50)
        requests = self.create_requests(num_requests=1, num_tokens=80)
        req = requests[0]

        engine.add_request(req)

        # First schedule call
        output1 = engine._schedule()

        assert len(output1.scheduled_new_reqs) == 1
        assert not output1.scheduled_cached_reqs
        assert output1.total_num_scheduled_tokens == 50
        assert output1.num_scheduled_tokens[req.request_id] == 50

        # Simulate model execution by updating computed tokens
        req.num_computed_tokens += 50
        assert len(engine._requests) == 1
        assert not engine._new_requests

        # Second schedule call
        output2 = engine._schedule()

        assert not output2.scheduled_new_reqs
        assert len(output2.scheduled_cached_reqs) == 1
        assert output2.total_num_scheduled_tokens == 30  # 80 - 50
        assert output2.num_scheduled_tokens[req.request_id] == 30

        cached_req_data = output2.scheduled_cached_reqs[0]
        assert cached_req_data.req_id == req.request_id
        assert len(cached_req_data.new_token_ids) == 30


    def test_jax_engine_schedule_mixed_requests(self):
        """Tests scheduling a mix of running and new requests."""
        engine = self.create_test_jax_engine(max_num_batched_tokens=100)

        # Add and partially schedule a long request
        running_req = self.create_requests(num_requests=1, num_tokens=80)[0]
        engine.add_request(running_req)

        # Temporarily reduce capacity to simulate chunking
        engine._max_num_tokens = 50
        _ = engine._schedule()
        running_req.num_computed_tokens += 50
        engine._max_num_tokens = 100  # Restore capacity

        # Add a new request
        new_req = self.create_requests(num_requests=1, num_tokens=40)[0]
        new_req.request_id = "new_req_1"
        engine.add_request(new_req)

        # Second schedule call
        output2 = engine._schedule()

        # The running request's next chunk should be scheduled first
        assert len(output2.scheduled_cached_reqs) == 1
        assert output2.scheduled_cached_reqs[0].req_id == running_req.request_id
        assert output2.num_scheduled_tokens[running_req.request_id] == 30

        # Then the new request should be scheduled
        assert len(output2.scheduled_new_reqs) == 1
        assert output2.scheduled_new_reqs[0].req_id == new_req.request_id
        assert output2.num_scheduled_tokens[new_req.request_id] == 40

        assert output2.total_num_scheduled_tokens == 70  # 30 + 40
