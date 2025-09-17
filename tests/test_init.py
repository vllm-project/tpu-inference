import os
import unittest
from unittest.mock import Mock, patch


class PlatformDetectionTest(unittest.TestCase):

    @patch.dict(os.environ, {}, clear=True)
    @patch('jax.lib.xla_bridge.get_backend')
    @patch("tpu_commons.tpu_info.get_tpu_metadata")
    def test_detect_platform_tpu(self, mock_get_tpu_metadata,
                                 mock_get_backend):
        """Test detect_platform returns 'tpu' when JAX backend is TPU."""
        # Create a mock backend object with platform attribute
        mock_backend = Mock()
        mock_backend.platform = "tpu"
        mock_get_backend.return_value = mock_backend

        mock_get_tpu_metadata.return_value = Mock(None)

        # Import the module to test the function
        from tpu_commons import detect_platform

        result = detect_platform()
        self.assertEqual(result, 'tpu')

    @patch.dict(os.environ, {}, clear=True)
    @patch('jax.lib.xla_bridge.get_backend')
    def test_detect_platform_cpu(self, mock_get_backend):
        """Test detect_platform returns 'cpu' when JAX backend is CPU."""
        mock_backend = Mock()
        mock_backend.platform = 'cpu'
        mock_get_backend.return_value = mock_backend

        from tpu_commons import detect_platform

        result = detect_platform()
        self.assertEqual(result, 'cpu')

    @patch.dict(os.environ, {'JAX_PLATFORMS': 'proxy'})
    @patch('jax.lib.xla_bridge.get_backend')
    def test_detect_platform_pathways_lowercase(self, mock_get_backend):
        """Test detect_platform returns 'pathways' when JAX_PLATFORMS contains 'proxy'."""
        from tpu_commons import detect_platform

        result = detect_platform()
        self.assertEqual(result, 'pathways')
        # Should not call get_backend when proxy is detected
        mock_get_backend.assert_not_called()

    @patch.dict(os.environ, {'JAX_PLATFORMS': 'PROXY'})
    @patch('jax.lib.xla_bridge.get_backend')
    def test_detect_platform_pathways_uppercase(self, mock_get_backend):
        """Test detect_platform returns 'pathways' when JAX_PLATFORMS contains 'PROXY' (case insensitive)."""
        from tpu_commons import detect_platform

        result = detect_platform()
        self.assertEqual(result, 'pathways')
        mock_get_backend.assert_not_called()

    @patch.dict(os.environ, {'JAX_PLATFORMS': 'something,proxy,other'})
    @patch('jax.lib.xla_bridge.get_backend')
    def test_detect_platform_pathways_in_list(self, mock_get_backend):
        """Test detect_platform returns 'pathways' when JAX_PLATFORMS contains 'proxy' in a list."""
        from tpu_commons import detect_platform

        result = detect_platform()
        self.assertEqual(result, 'pathways')
        mock_get_backend.assert_not_called()
