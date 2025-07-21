# SPDX-License-Identifier: Apache-2.0
"""Tests for JaxEngine."""
import unittest
from unittest.mock import Mock

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.request import Request, RequestStatus

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
                                 FullAttentionSpec(block_size, 1, 1,
                                                   torch.float32, False))
            ],
        )
        cache_config.num_gpu_blocks = num_blocks

        mock_executor = Mock()
        mock_executor.driver_worker.model_runner = Mock()

        engine = JaxEngine(vllm_config, kv_cache_config, mock_executor)
        return engine

    def create_requests(self,
                        num_requests: int,
                        num_tokens: int = 10,
                        max_tokens: int = 16) -> list[Request]:
        """Creates a list of requests for testing."""
        sampling_params = SamplingParams(ignore_eos=False,
                                         max_tokens=max_tokens)
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
            engine.add_request(req, 40)

        assert len(engine._new_requests) == 2
        assert not engine._requests

        output = engine._schedule_prefill()

        assert len(output.scheduled_new_reqs) == 2
        assert output.scheduled_cached_reqs.num_reqs == 0
        assert output.total_num_scheduled_tokens == 80

        assert output.num_scheduled_tokens[requests[0].request_id] == 40
        assert output.num_scheduled_tokens[requests[1].request_id] == 40

        assert not engine._new_requests
        assert len(engine._requests) == 2

    def test_jax_engine_schedule_new_requests_with_limit(self):
        """Tests scheduling new requests when they exceed the token limit."""
        engine = self.create_test_jax_engine(max_num_batched_tokens=50)
        requests = self.create_requests(num_requests=2, num_tokens=40)

        for req in requests:
            engine.add_request(req, 40)

        output = engine._schedule_prefill()

        # The first request is fully scheduled, the second is partially scheduled.
        assert len(output.scheduled_new_reqs) == 2
        assert output.scheduled_cached_reqs.num_reqs == 0
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

        engine.add_request(req, 80)

        # First schedule call
        output1 = engine._schedule_prefill()

        assert len(output1.scheduled_new_reqs) == 1
        assert output1.scheduled_cached_reqs.num_reqs == 0
        assert output1.total_num_scheduled_tokens == 50
        assert output1.num_scheduled_tokens[req.request_id] == 50

        # Simulate model execution by updating computed tokens
        req.num_computed_tokens += 50
        assert len(engine._requests) == 1
        assert not engine._new_requests

        # Second schedule call
        output2 = engine._schedule_prefill()

        assert not output2.scheduled_new_reqs
        assert output2.scheduled_cached_reqs.num_reqs == 1
        assert output2.total_num_scheduled_tokens == 30  # 80 - 50
        assert output2.num_scheduled_tokens[req.request_id] == 30

        assert output2.scheduled_cached_reqs.req_ids[0] == req.request_id
        assert len(output2.scheduled_cached_reqs.new_token_ids[0]) == 30

    def test_jax_engine_schedule_mixed_requests(self):
        """Tests scheduling a mix of running and new requests."""
        engine = self.create_test_jax_engine(max_num_batched_tokens=100)

        # Add and partially schedule a long request
        running_req = self.create_requests(num_requests=1, num_tokens=80)[0]
        engine.add_request(running_req, 80)

        # Temporarily reduce capacity to simulate chunking
        engine._max_num_tokens = 50
        _ = engine._schedule_prefill()
        running_req.num_computed_tokens += 50
        engine._max_num_tokens = 100  # Restore capacity

        # Add a new request
        new_req = self.create_requests(num_requests=1, num_tokens=40)[0]
        new_req.request_id = "new_req_1"
        engine.add_request(new_req, 40)

        # Second schedule call
        output2 = engine._schedule_prefill()

        # The running request's next chunk should be scheduled first
        assert output2.scheduled_cached_reqs.num_reqs == 1
        assert output2.scheduled_cached_reqs.req_ids[
            0] == running_req.request_id
        assert output2.num_scheduled_tokens[running_req.request_id] == 30

        # Then the new request should be scheduled
        assert len(output2.scheduled_new_reqs) == 1
        assert output2.scheduled_new_reqs[0].req_id == new_req.request_id
        assert output2.num_scheduled_tokens[new_req.request_id] == 40

        assert output2.total_num_scheduled_tokens == 70  # 30 + 40

    def test_has_more_capacity_true(self):
        """Tests has_more_capacity when there is room for more requests."""
        engine = self.create_test_jax_engine(max_num_seqs=8,
                                             max_num_batched_tokens=1024)
        engine.model_runner.max_num_reqs = 8
        engine.model_runner.input_batch.num_reqs = 4

        engine._pending_num_prefill_tokens = 512
        engine._requests = [Mock()] * 2
        engine._new_requests = [Mock()] * 2

        self.assertTrue(engine.has_more_capacity())

    def test_has_more_capacity_token_limit_reached(self):
        """Tests has_more_capacity when the token limit is reached."""
        engine = self.create_test_jax_engine(max_num_seqs=8,
                                             max_num_batched_tokens=1024)
        engine.model_runner.max_num_reqs = 8
        engine.model_runner.input_batch.num_reqs = 4

        # Token limit reached
        engine._pending_num_prefill_tokens = 1024
        engine._requests = [Mock()] * 2
        engine._new_requests = [Mock()] * 2
        self.assertFalse(engine.has_more_capacity())

        # Token limit exceeded
        engine._pending_num_prefill_tokens = 1025
        self.assertFalse(engine.has_more_capacity())

    def test_has_more_capacity_seq_limit_reached(self):
        """Tests has_more_capacity when the sequence limit is reached."""
        engine = self.create_test_jax_engine(max_num_seqs=8,
                                             max_num_batched_tokens=1024)
        engine.model_runner.max_num_reqs = 8
        engine.model_runner.input_batch.num_reqs = 4

        engine._pending_num_prefill_tokens = 512
        # Sequence limit reached
        engine._requests = [Mock()] * 4
        engine._new_requests = [Mock()] * 4
        self.assertFalse(engine.has_more_capacity())

        # Sequence limit exceeded
        engine._requests = [Mock()] * 5
        engine._new_requests = [Mock()] * 4
        self.assertFalse(engine.has_more_capacity())

    def test_has_more_capacity_model_runner_limit_reached(self):
        """Tests has_more_capacity when model runner's request limit is reached."""
        engine = self.create_test_jax_engine(max_num_seqs=8,
                                             max_num_batched_tokens=1024)
        engine.model_runner.max_num_reqs = 8
        # Model runner limit reached
        engine.model_runner.input_batch.num_reqs = 8

        engine._pending_num_prefill_tokens = 512
        engine._requests = [Mock()] * 2
        engine._new_requests = [Mock()] * 2
        self.assertFalse(engine.has_more_capacity())

        # Model runner limit exceeded
        engine.model_runner.input_batch.num_reqs = 9
        self.assertFalse(engine.has_more_capacity())

    def test_has_more_capacity_at_limits(self):
        """Tests has_more_capacity at the boundary of the limits."""
        engine = self.create_test_jax_engine(max_num_seqs=8,
                                             max_num_batched_tokens=1024)
        engine.model_runner.max_num_reqs = 8

        # Just under the limits
        engine._pending_num_prefill_tokens = 1023
        engine._requests = [Mock()] * 3
        engine._new_requests = [Mock()] * 4  # Total 7 < 8
        engine.model_runner.input_batch.num_reqs = 7
        self.assertTrue(engine.has_more_capacity())

    def test_prefill(self):
        """Tests the prefill method with a mocked model runner."""
        engine = self.create_test_jax_engine()
        requests = self.create_requests(num_requests=2, num_tokens=10)

        for req in requests:
            engine.add_request(req, req.num_tokens)

        # Mock the model runner's _execute_model method
        mock_runner_output = Mock()
        mock_runner_output.req_ids = [req.request_id for req in requests]
        mock_runner_output.req_id_to_index = {
            req.request_id: i
            for i, req in enumerate(requests)
        }
        mock_runner_output.sampled_token_ids = [[100]] * len(requests)
        engine.model_runner._execute_model.return_value = (None,
                                                           mock_runner_output)

        for req in requests:
            self.assertIn(req.request_id, engine._request_map)

        kv_cache_map, runner_output = engine.prefill()

        # Assert that the model runner's execute method was called
        engine.model_runner._execute_model.assert_called_once()

        # Check the number of scheduled tokens is correctly updated.
        # By default all prompt tokens are scheduled
        self.assertEqual(engine._pending_num_prefill_tokens, 2 * 10 - 2 * 10)

        # Verify results for each request
        for req in requests:
            self.assertNotIn(req.request_id, engine._request_map)
            self.assertIn(req.request_id, engine._completed_requests)
            # We generate the first token in prefill, hence 11 here.
            self.assertEqual(req.num_computed_tokens, 11)
            self.assertEqual(req.num_cached_tokens, 11)
            self.assertEqual(req.status.value, RequestStatus.RUNNING)

        # Check the model runner output
        self.assertEqual(runner_output.req_ids,
                         [req.request_id for req in requests])
        self.assertEqual(runner_output.req_id_to_index, {
            req.request_id: i
            for i, req in enumerate(requests)
        })
        self.assertEqual(runner_output.sampled_token_ids,
                         [[100]] * len(requests))
        self.assertEqual(len(kv_cache_map), 2)

    def test_generate(self):
        """Tests the generate method with a mocked model runner."""
        engine = self.create_test_jax_engine()
        requests = self.create_requests(num_requests=2, num_tokens=10)
        # Add requests to the engine
        for req in requests:
            engine.add_request(req, 1)

        # Mock the model runner's _execute_model method for generation
        mock_runner_output = Mock()
        mock_runner_output.req_ids = [req.request_id for req in requests]
        mock_runner_output.req_id_to_index = {
            req.request_id: i
            for i, req in enumerate(requests)
        }
        mock_runner_output.sampled_token_ids = [[200]] * len(requests)
        engine.model_runner._execute_model.return_value = (None,
                                                           mock_runner_output)

        # Perform generation
        runner_output = engine.generate()

        # Assert that the model runner's execute method was called
        engine.model_runner._execute_model.assert_called_once()

        # Verify results for each request
        for req in requests:
            self.assertIn(req.request_id, engine._request_map)
            self.assertEqual(req.num_computed_tokens, 1)
            self.assertEqual(req.num_cached_tokens, 1)
            self.assertEqual(req.status.value, RequestStatus.RUNNING)

        # Check the model runner output
        self.assertEqual(runner_output.req_ids,
                         [req.request_id for req in requests])
        self.assertEqual(runner_output.req_id_to_index, {
            req.request_id: i
            for i, req in enumerate(requests)
        })
        self.assertEqual(runner_output.sampled_token_ids,
                         [[200]] * len(requests))

        # Test with empty output
        engine.model_runner._execute_model.return_value = (None,
                                                           Mock(req_ids=[]))
        with self.assertRaises(RuntimeError):
            engine.generate()


if __name__ == '__main__':
    unittest.main()
