import importlib
import unittest
from unittest.mock import patch


class TestPathwaysInit(unittest.TestCase):

    @patch.dict("os.environ", {"JAX_PLATFORMS": "proxy,cpu"})
    def test_VLLM_TPU_USING_PATHWAYS_enabled(self):
        """Test when JAX_PLATFORMS contains 'proxy'."""

        # Import vllm.envs to test the VLLM_TPU_USING_PATHWAYS logic
        import vllm.envs as envs

        # Reload the module to ensure fresh import
        importlib.reload(envs)

        # Check that VLLM_TPU_USING_PATHWAYS is True when JAX_PLATFORMS contains "proxy"
        self.assertTrue(envs.VLLM_TPU_USING_PATHWAYS)

    @patch.dict("os.environ", {"JAX_PLATFORMS": "cpu"})
    def test_VLLM_TPU_USING_PATHWAYS_not_enabled(self):
        """Test when JAX_PLATFORMS does not contain 'proxy'."""

        # Import vllm.envs to test the VLLM_TPU_USING_PATHWAYS logic
        import vllm.envs as envs

        # Reload the module to ensure fresh import
        importlib.reload(envs)

        # Check that VLLM_TPU_USING_PATHWAYS is False when JAX_PLATFORMS doesn't contain "proxy"
        self.assertFalse(envs.VLLM_TPU_USING_PATHWAYS)

    @patch.dict("os.environ", {"JAX_PLATFORMS": "PROXY,CPU"})
    def test_VLLM_TPU_USING_PATHWAYS_case_insensitive(self):
        """Test that JAX_PLATFORMS check is case insensitive."""

        # Import vllm.envs to test the VLLM_TPU_USING_PATHWAYS logic
        import vllm.envs as envs

        # Reload the module to ensure fresh import
        importlib.reload(envs)

        # Check that VLLM_TPU_USING_PATHWAYS is True even with uppercase "PROXY"
        self.assertTrue(envs.VLLM_TPU_USING_PATHWAYS)


if __name__ == "__main__":
    unittest.main()
