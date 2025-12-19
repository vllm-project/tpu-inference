# Copyright 2025 Google LLC
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

import unittest
from unittest.mock import MagicMock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.request import RequestStatus

from tpu_inference.distributed import tpu_connector


class MockVllmConfig:

    def __init__(self):
        self.kv_transfer_config = MagicMock()
        self.kv_transfer_config.is_kv_producer = True
        self.cache_config = MagicMock()
        self.cache_config.block_size = 16
        self.parallel_config = MagicMock()


@patch("tpu_inference.distributed.tpu_connector.TPUConnectorWorker")
@patch("tpu_inference.distributed.tpu_connector.TPUConnectorScheduler")
class TestTPUConnector(unittest.TestCase):

    def setUp(self):
        self.vllm_config = MockVllmConfig()

    def test_init_scheduler_role(self, mock_scheduler_cls, mock_worker_cls):
        """
        Tests that TPUConnector initializes the scheduler connector for the
        SCHEDULER role.
        """
        connector = tpu_connector.TPUConnector(self.vllm_config,
                                               KVConnectorRole.SCHEDULER)
        mock_scheduler_cls.assert_called_once_with(self.vllm_config)
        mock_worker_cls.assert_not_called()
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    def test_init_worker_role(self, mock_scheduler_cls, mock_worker_cls):
        """
        Tests that TPUConnector initializes the worker connector for the WORKER
        role.
        """
        connector = tpu_connector.TPUConnector(self.vllm_config,
                                               KVConnectorRole.WORKER)
        mock_worker_cls.assert_called_once_with(self.vllm_config)
        mock_scheduler_cls.assert_not_called()
        self.assertIsNone(connector.connector_scheduler)
        self.assertIsNotNone(connector.connector_worker)

    def test_scheduler_methods_are_called(self, mock_scheduler_cls,
                                          mock_worker_cls):
        """Tests that scheduler-side methods are correctly delegated."""
        mock_scheduler_instance = mock_scheduler_cls.return_value
        connector = tpu_connector.TPUConnector(self.vllm_config,
                                               KVConnectorRole.SCHEDULER)

        mock_request = MagicMock()
        mock_blocks = MagicMock()
        mock_scheduler_output = MagicMock()

        connector.get_num_new_matched_tokens(mock_request, 16)
        mock_scheduler_instance.get_num_new_matched_tokens.assert_called_once_with(
            mock_request, 16)

        connector.update_state_after_alloc(mock_request, mock_blocks, 16)
        mock_scheduler_instance.update_state_after_alloc.assert_called_once_with(
            mock_request, mock_blocks, 16)

        connector.build_connector_meta(mock_scheduler_output)
        mock_scheduler_instance.build_connector_meta.assert_called_once_with()

        connector.request_finished(mock_request, [1, 2])
        mock_scheduler_instance.request_finished.assert_called_once_with(
            mock_request, [1, 2])

    def test_worker_methods_are_called(self, mock_scheduler_cls,
                                       mock_worker_cls):
        """Tests that worker-side methods are correctly delegated."""
        mock_worker_instance = mock_worker_cls.return_value
        connector = tpu_connector.TPUConnector(self.vllm_config,
                                               KVConnectorRole.WORKER)
        connector._connector_metadata = tpu_connector.TPUConnectorMetadata(
        )  # need to set this for start_load_kv

        mock_runner = MagicMock()

        connector.register_runner(mock_runner)
        mock_worker_instance.register_runner.assert_called_once_with(
            mock_runner)

        connector.start_load_kv(None)
        mock_worker_instance.process_send_load.assert_called_once_with(
            connector._connector_metadata)

        connector.get_finished(set())
        mock_worker_instance.get_finished.assert_called_once_with()


class TestTPUConnectorScheduler(unittest.TestCase):

    def setUp(self):
        self.vllm_config = MockVllmConfig()
        self.vllm_config.cache_config.block_size = 16
        self.vllm_config.kv_transfer_config.is_kv_producer = False

        with patch("tpu_inference.distributed.tpu_connector.get_kv_ips",
                   return_value="1.1.1.1"), patch(
                       "tpu_inference.distributed.tpu_connector.get_kv_ports",
                       return_value=12345):
            self.scheduler = tpu_connector.TPUConnectorScheduler(
                self.vllm_config)

    def test_get_num_new_matched_tokens_producer(self):
        """Tests that producer returns 0 tokens to load."""
        self.scheduler.is_producer = True
        mock_request = MagicMock()
        num_tokens, is_async = self.scheduler.get_num_new_matched_tokens(
            mock_request, 16)
        self.assertEqual(num_tokens, 0)
        self.assertFalse(is_async)

    def test_get_num_new_matched_tokens_consumer_needs_loading(self):
        """Tests consumer calculates correct number of tokens to load."""
        mock_request = MagicMock()
        mock_request.prompt_token_ids = [0] * 35  # 2 blocks worth, plus some
        num_computed_tokens = 16  # 1 block
        # rounded_down(35) = 32. 32 - 16 = 16.
        expected_tokens = 16
        num_tokens, is_async = self.scheduler.get_num_new_matched_tokens(
            mock_request, num_computed_tokens)
        self.assertEqual(num_tokens, expected_tokens)
        self.assertTrue(is_async)

    def test_get_num_new_matched_tokens_consumer_no_loading(self):
        """Tests consumer returns 0 if prompt is fully cached."""
        mock_request = MagicMock()
        mock_request.prompt_token_ids = [0] * 31  # less than 2 blocks
        num_computed_tokens = 32  # 2 blocks computed
        expected_tokens = 0
        num_tokens, is_async = self.scheduler.get_num_new_matched_tokens(
            mock_request, num_computed_tokens)
        self.assertEqual(num_tokens, expected_tokens)
        self.assertFalse(is_async)

    def test_update_state_after_alloc_producer(self):
        """Tests that update_state_after_alloc is a no-op for producers."""
        self.scheduler.is_producer = True
        self.scheduler.update_state_after_alloc(MagicMock(), MagicMock(), 16)
        self.assertEqual(len(self.scheduler.reqs_to_load), 0)

    def test_update_state_after_alloc_consumer_with_external_tokens(self):
        """
        Tests consumer state is updated when external tokens are needed.
        """
        mock_request = MagicMock()
        mock_request.request_id = "req1"
        mock_request.kv_transfer_params = {
            "uuid": 123,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 54321
        }
        mock_blocks = MagicMock()
        mock_blocks.get_block_ids.return_value = [[1, 2]]
        num_external_tokens = 32

        self.scheduler.update_state_after_alloc(mock_request, mock_blocks,
                                                num_external_tokens)

        self.assertIn("req1", self.scheduler.reqs_to_load)
        load_meta = self.scheduler.reqs_to_load["req1"]
        self.assertEqual(load_meta.uuid, 123)
        self.assertEqual(load_meta.local_block_ids, [1, 2])
        self.assertEqual(load_meta.remote_block_ids, [10, 11])

    def test_update_state_after_alloc_consumer_no_external_tokens(self):
        """
        Tests consumer state is updated for notification when no external
        tokens are needed.
        """
        mock_request = MagicMock()
        mock_request.request_id = "req1"
        mock_request.kv_transfer_params = {
            "uuid": 123,
            "remote_block_ids": [10, 11],
            "remote_host": "2.2.2.2",
            "remote_port": 54321
        }
        mock_blocks = MagicMock()
        num_external_tokens = 0

        self.scheduler.update_state_after_alloc(mock_request, mock_blocks,
                                                num_external_tokens)

        self.assertIn("req1", self.scheduler.reqs_to_load)
        load_meta = self.scheduler.reqs_to_load["req1"]
        self.assertEqual(load_meta.uuid, 123)
        self.assertIsNone(load_meta.local_block_ids)
        self.assertIsNone(load_meta.remote_block_ids)

    def test_build_connector_meta(self):
        """Tests that metadata is built and state is cleared."""
        self.scheduler.is_producer = True
        self.scheduler.reqs_to_send = {"req1": "meta1"}
        meta = self.scheduler.build_connector_meta()
        self.assertEqual(meta.reqs_to_send, {"req1": "meta1"})
        self.assertEqual(len(self.scheduler.reqs_to_send),
                         0)  # check it was cleared

        self.scheduler.is_producer = False
        self.scheduler.reqs_to_load = {"req2": "meta2"}
        meta = self.scheduler.build_connector_meta()
        self.assertEqual(meta.reqs_to_load, {"req2": "meta2"})
        self.assertEqual(len(self.scheduler.reqs_to_load), 0)

    def test_request_finished_consumer(self):
        """Tests request_finished is a no-op for consumers."""
        self.scheduler.is_producer = False
        delay_free, params = self.scheduler.request_finished(MagicMock(), [])
        self.assertFalse(delay_free)
        self.assertIsNone(params)

    @patch("tpu_inference.distributed.tpu_connector.get_uuid",
           return_value=456)
    def test_request_finished_producer_finished_by_length(self, mock_get_uuid):
        """Tests producer logic when a request finishes normally."""
        self.scheduler.is_producer = True
        mock_request = MagicMock()
        mock_request.request_id = "req-finished"
        mock_request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_request.num_computed_tokens = 32  # 2 blocks
        block_ids = [1, 2]

        delay_free, params = self.scheduler.request_finished(
            mock_request, block_ids)

        self.assertTrue(delay_free)
        self.assertIn("req-finished", self.scheduler.reqs_to_send)
        send_meta = self.scheduler.reqs_to_send["req-finished"]
        self.assertEqual(send_meta.uuid, 456)
        self.assertEqual(send_meta.local_block_ids, [1, 2])

        self.assertIsNotNone(params)
        self.assertEqual(params["uuid"], 456)
        self.assertEqual(params["remote_block_ids"], [1, 2])
        self.assertEqual(params["remote_host"], "1.1.1.1")
        self.assertEqual(params["remote_port"], 12345)

    def test_request_finished_producer_not_finished(self):
        """Tests producer logic when a request is not yet finished."""
        self.scheduler.is_producer = True
        mock_request = MagicMock()
        mock_request.status = RequestStatus.RUNNING  # Not finished
        delay_free, params = self.scheduler.request_finished(
            mock_request, [1, 2])
        self.assertFalse(delay_free)
        self.assertIsNone(params)

    def test_request_finished_producer_prompt_too_short(self):
        """Tests producer logic when prompt is too short to transfer."""
        self.scheduler.is_producer = True
        mock_request = MagicMock()
        mock_request.request_id = "req-short"
        mock_request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        mock_request.num_computed_tokens = 10  # less than a block
        block_ids = [1]

        delay_free, params = self.scheduler.request_finished(
            mock_request, block_ids)

        self.assertFalse(delay_free)
        self.assertEqual(params, {})
        self.assertNotIn("req-short", self.scheduler.reqs_to_send)


class TestTPUConnectorWorker(unittest.TestCase):

    def setUp(self):
        self.vllm_config = MockVllmConfig()
        patchers = {
            "jax":
            patch('tpu_inference.distributed.tpu_connector.jax'),
            "get_host_ip":
            patch('tpu_inference.distributed.tpu_connector.get_host_ip',
                  return_value='127.0.0.1'),
            "get_kv_transfer_port":
            patch(
                'tpu_inference.distributed.tpu_connector.get_kv_transfer_port',
                return_value=10000),
            "get_side_channel_port":
            patch(
                'tpu_inference.distributed.tpu_connector.get_side_channel_port',
                return_value=20000),
            "start_transfer_server":
            patch(
                'tpu_inference.distributed.tpu_connector.start_transfer_server'
            ),
            "zmq":
            patch('tpu_inference.distributed.tpu_connector.zmq'),
            "threading":
            patch('tpu_inference.distributed.tpu_connector.threading'),
            "ThreadPoolExecutor":
            patch(
                'tpu_inference.distributed.tpu_connector.ThreadPoolExecutor'),
            "device_array":
            patch('tpu_inference.distributed.tpu_connector.device_array'),
            "select_from_kv_caches":
            patch(
                'tpu_inference.distributed.tpu_connector.select_from_kv_caches'
            ),
            "scatter_kv_slices":
            patch('tpu_inference.distributed.tpu_connector.scatter_kv_slices'),
            "time":
            patch('tpu_inference.distributed.tpu_connector.time'),
            "make_zmq_path":
            patch('tpu_inference.distributed.tpu_connector.make_zmq_path'),
            "make_zmq_socket":
            patch('tpu_inference.distributed.tpu_connector.make_zmq_socket'),
        }
        self.all_mocks = {k: p.start() for k, p in patchers.items()}
        self.all_mocks["jax"].local_devices.return_value = [MagicMock()]
        for p in patchers.values():
            self.addCleanup(p.stop)

    def test_init_producer(self):
        """Tests worker initialization for the producer role."""
        self.vllm_config.kv_transfer_config.is_kv_producer = True
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)

        self.all_mocks["zmq"].Context.assert_called_once()
        self.all_mocks["threading"].Thread.assert_called_once()
        self.all_mocks["threading"].Event.assert_called()
        self.all_mocks["ThreadPoolExecutor"].assert_not_called()
        self.assertTrue(worker.is_producer)

    def test_init_consumer(self):
        """Tests worker initialization for the consumer role."""
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)

        self.all_mocks["zmq"].Context.assert_called_once()
        self.all_mocks["threading"].Thread.assert_not_called()
        self.all_mocks["ThreadPoolExecutor"].assert_called_once_with(
            max_workers=64)
        self.assertFalse(worker.is_producer)

    def test_register_runner(self):
        """Tests that runner registration correctly sets worker attributes."""
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)

        mock_runner = MagicMock()
        mock_kv_cache_layer = MagicMock()
        mock_kv_cache_layer.shape = [10, 20, 30, 40]
        mock_kv_cache_layer.dtype = 'float32'
        mock_kv_cache_layer.sharding = 'sharding_spec'
        mock_runner.kv_caches = [mock_kv_cache_layer] * 5
        mock_runner.mesh = 'mesh'

        worker.register_runner(mock_runner)

        self.all_mocks["start_transfer_server"].assert_called_once()
        self.assertEqual(worker.runner, mock_runner)
        self.assertEqual(worker.mesh, 'mesh')
        self.assertEqual(worker.num_layers, 5)
        self.assertEqual(worker.shape, [10, 20, 30, 40])
        self.assertEqual(worker.dtype, 'float32')
        self.assertEqual(worker.sharding, 'sharding_spec')

    def test_process_send_load_for_producer(self):
        """Tests process_send_load for the producer role."""
        self.vllm_config.kv_transfer_config.is_kv_producer = True
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)
        worker._prepare_kv_and_wait = MagicMock()

        meta = tpu_connector.TPUConnectorMetadata()
        send_meta = tpu_connector.SendMeta(uuid=1,
                                           local_block_ids=[1],
                                           expiration_time=123)
        meta.reqs_to_send = {"req1": send_meta}

        worker.process_send_load(meta)

        worker._prepare_kv_and_wait.assert_called_once_with("req1", send_meta)

    def test_process_send_load_for_consumer_loading(self):
        """Tests process_send_load for a consumer that needs to load KV."""
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)
        worker._maybe_build_kv_connection = MagicMock(return_value="conn")

        meta = tpu_connector.TPUConnectorMetadata()
        load_meta = tpu_connector.LoadMeta(uuid=1,
                                           local_block_ids=[1],
                                           remote_block_ids=[10],
                                           remote_host="host",
                                           remote_port=123)
        meta.reqs_to_load = {"req1": load_meta}

        worker.process_send_load(meta)

        worker._maybe_build_kv_connection.assert_called_once_with(load_meta)
        self.all_mocks[
            "ThreadPoolExecutor"].return_value.submit.assert_called_once_with(
                worker._pull_kv, "conn", load_meta)

    def test_process_send_load_for_consumer_notifying(self):
        """Tests process_send_load for a consumer that needs to notify."""
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)
        worker._maybe_build_notif_socket = MagicMock(return_value="socket")
        worker._notify_pull_done = MagicMock()

        meta = tpu_connector.TPUConnectorMetadata()
        load_meta = tpu_connector.LoadMeta(uuid=1,
                                           local_block_ids=None,
                                           remote_block_ids=None,
                                           remote_host="host",
                                           remote_port=123)
        meta.reqs_to_load = {"req1": load_meta}

        worker.process_send_load(meta)

        worker._maybe_build_notif_socket.assert_called_once_with(load_meta)
        worker._notify_pull_done.assert_called_once_with("socket", "req1")

    def test_get_finished_recving(self):
        """Tests get_finished for a request that has finished pulling."""
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)
        worker.runner = MagicMock()
        original_kv_caches = worker.runner.kv_caches

        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = ('kv_data', 'indices')
        worker.reqs_pulling = {'req1': mock_future}

        done_sending, done_recving = worker.get_finished()

        self.assertEqual(done_sending, set())
        self.assertEqual(done_recving, {'req1'})
        self.assertNotIn('req1', worker.reqs_pulling)
        self.all_mocks['scatter_kv_slices'].assert_called_once_with(
            original_kv_caches, 'kv_data', 'indices')

    def test_get_finished_sending_expired(self):
        """Tests get_finished for a request that has expired."""
        self.vllm_config.kv_transfer_config.is_kv_producer = True
        worker = tpu_connector.TPUConnectorWorker(self.vllm_config)

        self.all_mocks['time'].perf_counter.return_value = 1000
        worker.reqs_wait_pull = {'req1': ['kv_data', 900]}

        done_sending, done_recving = worker.get_finished()

        self.assertEqual(done_sending, {'req1'})
        self.assertEqual(done_recving, set())
        self.assertNotIn('req1', worker.reqs_wait_pull)


if __name__ == "__main__":
    unittest.main()
