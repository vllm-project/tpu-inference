# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, PropertyMock

from tpu_commons.core import adapters


class TestVllmConfigAdapter(unittest.TestCase):

    def test_config_adapter(self):
        mock_vllm_config = MagicMock()
        type(mock_vllm_config).scheduler_config = PropertyMock(
            return_value="scheduler")
        type(mock_vllm_config).cache_config = PropertyMock(
            return_value="cache")

        adapter = adapters.VllmConfigAdapter(mock_vllm_config)

        self.assertEqual(adapter.scheduler_config, "scheduler")
        self.assertEqual(adapter.cache_config, "cache")


class TestVllmSchedulerAdapter(unittest.TestCase):

    def test_add_request(self):
        mock_scheduler = MagicMock()
        mock_request = MagicMock()
        mock_request.vllm_request = "vllm_request"
        adapter = adapters.VllmSchedulerAdapter(mock_scheduler)
        adapter.add_request(mock_request)
        mock_scheduler.add_request.assert_called_once_with("vllm_request")

    def test_getattr(self):
        mock_scheduler = MagicMock()
        adapter = adapters.VllmSchedulerAdapter(mock_scheduler)
        adapter.schedule()
        mock_scheduler.schedule.assert_called_once()


class TestVllmEngineAdapter(unittest.TestCase):

    def test_engine_adapter(self):
        mock_engine_core = MagicMock()
        mock_engine_core.scheduler = "scheduler"
        type(mock_engine_core).model_executor = PropertyMock(
            return_value="executor")

        adapter = adapters.VllmEngineAdapter(mock_engine_core)

        self.assertIsInstance(adapter.scheduler, adapters.VllmSchedulerAdapter)
        self.assertEqual(adapter.model_executor, "executor")

        adapter.execute_model_with_error_logging("arg1", kwarg1="kwarg1")
        mock_engine_core.execute_model_with_error_logging.assert_called_once_with(
            "arg1", kwarg1="kwarg1")

        adapter.shutdown()
        mock_engine_core.shutdown.assert_called_once()


class TestVllmRequestAdapter(unittest.TestCase):

    def test_request_adapter(self):
        mock_vllm_request = MagicMock()
        type(mock_vllm_request).request_id = PropertyMock(return_value="123")

        # Mock properties that can be written to by setting them as attributes
        # on the mock object.
        mock_vllm_request.num_computed_tokens = 10
        mock_vllm_request.status = "COMPLETED"

        adapter = adapters.VllmRequestAdapter(mock_vllm_request)

        self.assertEqual(adapter.vllm_request, mock_vllm_request)
        self.assertEqual(adapter.request_id, "123")
        self.assertEqual(adapter.num_computed_tokens, 10)
        self.assertEqual(adapter.status, "COMPLETED")

        adapter.num_computed_tokens = 20
        self.assertEqual(mock_vllm_request.num_computed_tokens, 20)

        adapter.status = "RUNNING"
        self.assertEqual(mock_vllm_request.status, "RUNNING")
