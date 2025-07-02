import unittest

from tpu_commons.core.disagg_utils import _parse_slices


class DisaggUtilsTest(unittest.TestCase):

    def test_parse_slices_valid_cases(self):
        """Tests valid slice strings."""
        # Test with a single slice
        self.assertEqual(_parse_slices("2x2"), (4, ))
        self.assertEqual(_parse_slices("2"), (2, ))

        # Test with multiple slices
        self.assertEqual(_parse_slices("2x2,2x1,3,2x4"), (4, 2, 3, 8))

        # Test with various dimensions
        self.assertEqual(_parse_slices("1x1,10x10,5x3"), (1, 100, 15))

        # Test with an empty string
        self.assertEqual(_parse_slices(""), ())

    def test_parse_slices_with_whitespace(self):
        """Tests valid slice strings with extra whitespace."""
        self.assertEqual(_parse_slices(" 2x2 "), (4, ))
        self.assertEqual(_parse_slices(" 2x2 , 2x1 , 2x4 "), (4, 2, 8))
        # The current implementation allows spaces inside the slice definition
        # because int() can handle them.
        self.assertEqual(_parse_slices("2 x 2"), (4, ))
        self.assertEqual(_parse_slices(" 10 x 10 "), (100, ))

    def test_parse_slices_invalid_cases(self):
        """Tests malformed slice strings that should raise ValueError."""
        invalid_strings = [
            "2*2",  # wrong separator
            "2x",  # incomplete
            "axb",  # not integers
            "2x2x2",  # too many dimensions
            "2x2,3*3",  # partially malformed
            ",2x2",  # leading comma
            "2x2,",  # trailing comma
            "2x2,,2x1",  # empty slice in middle
        ]
        for invalid_str in invalid_strings:
            with self.subTest(invalid_str=invalid_str):
                with self.assertRaises(ValueError):
                    _parse_slices(invalid_str)


if __name__ == '__main__':
    unittest.main()
