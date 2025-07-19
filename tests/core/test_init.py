import importlib
import unittest
from unittest.mock import patch

from tpu_commons import core


class TestPathwaysInit(unittest.TestCase):

    @patch("importlib.util.find_spec")
    def test_pathways_enabled(self, mock_find_spec):
        """Test when pathwaysutils is found."""
        mock_pathwaysutils = unittest.mock.MagicMock()
        mock_find_spec.return_value = True
        with patch.dict("sys.modules", {"pathwaysutils": mock_pathwaysutils}):
            with patch.object(mock_pathwaysutils,
                              "initialize") as mock_initialize:
                importlib.reload(core)
                self.assertTrue(core.PATHWAYS_ENABLED)
                mock_initialize.assert_called_once()

    @patch("importlib.util.find_spec")
    def test_pathways_not_enabled(self, mock_find_spec):
        """Test when pathwaysutils is not found."""
        mock_find_spec.return_value = None
        with patch("builtins.print") as mock_print:
            importlib.reload(core)
            self.assertFalse(core.PATHWAYS_ENABLED)
            mock_print.assert_called_once_with(
                "Running uLLM without Pathways. "
                "Module pathwaysutils is not imported.")


if __name__ == "__main__":
    unittest.main()
