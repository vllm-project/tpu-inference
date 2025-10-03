# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

from vllm.config import ModelConfig, VllmConfig

from tpu_inference.core.disagg_executor import DisaggExecutor


class DisaggExecutorTest(unittest.TestCase):

    def setUp(self):
        """Set up the test environment by mocking dependencies."""
        # Mock configurations
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.model_config = ModelConfig(
            tokenizer_mode="auto",
            trust_remote_code=False,
            seed=0,
            dtype='bfloat16')
        self.mock_vllm_config.cache_config = MagicMock()
        self.mock_vllm_config.scheduler_config = MagicMock()
        self.mock_vllm_config.load_config = MagicMock()
        self.mock_vllm_config.lora_config = None
        self.mock_vllm_config.parallel_config = MagicMock()
        self.mock_vllm_config.device_config = MagicMock()
        self.mock_vllm_config.speculative_config = None
        self.mock_vllm_config.prompt_adapter_config = None
        self.mock_vllm_config.observability_config = MagicMock()

        # Patch the collective_rpc method to avoid actual RPC calls
        self.patcher = patch(
            "tpu_inference.core.disagg_executor.DisaggExecutor.collective_rpc")
        self.mock_collective_rpc = self.patcher.start()
        self.addCleanup(self.patcher.stop)

        # Create a DisaggExecutor instance with the mock config
        self.executor = DisaggExecutor(vllm_config=self.mock_vllm_config)

    def test_init_with_devices(self):
        """Test init_with_devices."""
        self.executor._init_executor()

        # Check that collective_rpc was called with the expected arguments
        self.mock_collective_rpc.assert_called()
        calls = self.mock_collective_rpc.call_args_list

        # Asserts for init_worker
        self.assertEqual(calls[0][0][0], "init_worker")
        self.assertEqual(calls[1][0][0], "init_device")
        self.assertEqual(calls[2][0][0], "load_model")

    def test_check_health(self):
        """Test check_health."""
        # Call check_health (it should always pass)
        self.executor.check_health()


if __name__ == '__main__':
    unittest.main()
