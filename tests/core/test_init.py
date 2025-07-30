import importlib
import unittest
from unittest.mock import patch


class TestPathwaysInit(unittest.TestCase):

    @patch("os.environ.get")
    def test_VLLM_TPU_USING_PATHWAYS_enabled(self, mock_environ_get):
        """Test when JAX_PLATFORMS contains 'proxy'."""
        # Mock JAX_PLATFORMS to contain "proxy"
        mock_environ_get.return_value = "proxy,cpu"

        # Import vllm.envs to test the VLLM_TPU_USING_PATHWAYS logic
        import vllm.envs as envs

        # Reload the module to ensure fresh import
        importlib.reload(envs)

        # Check that VLLM_TPU_USING_PATHWAYS is True when JAX_PLATFORMS contains "proxy"
        self.assertTrue(envs.VLLM_TPU_USING_PATHWAYS)
        mock_environ_get.assert_called_with("JAX_PLATFORMS", "")

    @patch("os.environ.get")
    def test_VLLM_TPU_USING_PATHWAYS_not_enabled(self, mock_environ_get):
        """Test when JAX_PLATFORMS does not contain 'proxy'."""
        # Mock JAX_PLATFORMS to not contain "proxy"
        mock_environ_get.return_value = "cpu"

        # Import vllm.envs to test the VLLM_TPU_USING_PATHWAYS logic
        import vllm.envs as envs

        # Reload the module to ensure fresh import
        importlib.reload(envs)

        # Check that VLLM_TPU_USING_PATHWAYS is False when JAX_PLATFORMS doesn't contain "proxy"
        self.assertFalse(envs.VLLM_TPU_USING_PATHWAYS)
        mock_environ_get.assert_called_with("JAX_PLATFORMS", "")

    @patch("os.environ.get")
    def test_VLLM_TPU_USING_PATHWAYS_case_insensitive(self, mock_environ_get):
        """Test that JAX_PLATFORMS check is case insensitive."""
        # Mock JAX_PLATFORMS to contain "PROXY" (uppercase)
        mock_environ_get.return_value = "PROXY,CPU"

        # Import vllm.envs to test the VLLM_TPU_USING_PATHWAYS logic
        import vllm.envs as envs

        # Reload the module to ensure fresh import
        importlib.reload(envs)

        # Check that VLLM_TPU_USING_PATHWAYS is True even with uppercase "PROXY"
        self.assertTrue(envs.VLLM_TPU_USING_PATHWAYS)
        mock_environ_get.assert_called_with("JAX_PLATFORMS", "")


if __name__ == "__main__":
    unittest.main()
