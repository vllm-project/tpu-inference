"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
from unittest.mock import ANY, MagicMock, patch

from tpu_commons.adapters.vllm_adapters import VllmSchedulerOutputAdapter
from tpu_commons.backend import TPUBackend
from tpu_commons.di.interfaces import HostInterface


class BackendTest(unittest.TestCase):

    @patch('tpu_commons.backend.get_tpu_worker_cls')
    def test_backend_initialization(self, mock_get_worker_cls):
        """Tests that the TPUBackend correctly initializes a worker."""
        # Arrange
        mock_worker_cls = MagicMock()
        mock_get_worker_cls.return_value = mock_worker_cls
        mock_host = MagicMock(spec=HostInterface)
        worker_kwargs = {'arg1': 'value1', 'arg2': 123}

        # Act
        backend = TPUBackend(host_interface=mock_host, **worker_kwargs)

        # Assert
        mock_get_worker_cls.assert_called_once_with()
        mock_worker_cls.assert_called_once_with(host_interface=mock_host,
                                                arg1='value1',
                                                arg2=123)
        self.assertIsInstance(backend.worker, MagicMock)

    @patch('tpu_commons.backend.get_tpu_worker_cls')
    def test_launch_tpu_batch_delegates_to_worker(self, mock_get_worker_cls):
        """Tests that launch_tpu_batch correctly calls the worker's method."""
        # Arrange
        mock_worker_instance = MagicMock()
        mock_worker_cls = MagicMock(return_value=mock_worker_instance)
        mock_get_worker_cls.return_value = mock_worker_cls
        mock_host = MagicMock(spec=HostInterface)

        backend = TPUBackend(host_interface=mock_host)
        test_batch = "test_batch_data"

        # Act
        backend.launch_tpu_batch(test_batch)

        # Assert
        mock_worker_instance.execute_model.assert_called_once_with(ANY)
        adapter_arg = mock_worker_instance.execute_model.call_args[0][0]
        self.assertIsInstance(adapter_arg, VllmSchedulerOutputAdapter)
        self.assertEqual(adapter_arg.vllm_scheduler_output, test_batch)


if __name__ == '__main__':
    unittest.main()
