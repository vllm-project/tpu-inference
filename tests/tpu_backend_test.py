import unittest
from unittest.mock import Mock, patch

from tpu_commons.backend import TPUBackend


class TPUBackendTest(unittest.TestCase):

    @patch('tpu_commons.backend.TPUWorker')
    def test_tpu_backend_initialization(self, mock_tpu_worker_class):
        """Test that TPUBackend initializes the worker correctly."""
        mock_host_interface = Mock()
        mock_worker_kwargs = {'worker_arg': 'test_value'}

        backend = TPUBackend(host_interface=mock_host_interface,
                             **mock_worker_kwargs)

        # Assert that the TPUWorker was instantiated with the correct arguments
        mock_tpu_worker_class.assert_called_once_with(
            host_interface=mock_host_interface, **mock_worker_kwargs)

        # Assert that the worker attribute is an instance of the mock class
        self.assertEqual(backend.worker, mock_tpu_worker_class.return_value)

    @patch('tpu_commons.backend.VllmSchedulerOutputAdapter')
    @patch('tpu_commons.backend.TPUWorker')
    def test_launch_tpu_batch(self, mock_tpu_worker_class, mock_adapter_class):
        """Test that launch_tpu_batch delegates to the worker correctly."""
        mock_worker_instance = mock_tpu_worker_class.return_value

        backend = TPUBackend()
        mock_batch = Mock()

        backend.launch_tpu_batch(mock_batch)

        # Assert that the adapter was created with the correct input
        mock_adapter_class.assert_called_once_with(mock_batch)

        # Assert that the worker's execute_model method was called with the mock adapter's return value
        mock_worker_instance.execute_model.assert_called_once_with(
            mock_adapter_class.return_value)

    @patch('tpu_commons.backend.VllmLoRARequestAdapter')
    @patch('tpu_commons.backend.TPUWorker')
    def test_add_lora(self, mock_tpu_worker_class, mock_adapter_class):
        """Test that add_lora delegates to the worker correctly."""
        mock_worker_instance = mock_tpu_worker_class.return_value

        backend = TPUBackend()
        mock_lora_request = Mock()

        backend.add_lora(mock_lora_request)

        # Assert that the adapter was created with the correct input
        mock_adapter_class.assert_called_once_with(mock_lora_request)

        # Assert that the worker's add_lora method was called with the mock adapter's return value
        mock_worker_instance.add_lora.assert_called_once_with(
            mock_adapter_class.return_value)
