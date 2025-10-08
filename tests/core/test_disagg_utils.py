import unittest

from tpu_inference.core.disagg_utils import _parse_slices


class DisaggUtilsTest(unittest.TestCase):

    def test_parse_slices_valid_cases(self):
        """Tests valid slice strings."""
        # Test with a single slice
        self.assertEqual(_parse_slices("2x2"), ((2, 2), ))
        self.assertEqual(_parse_slices("2"), (2, ))

        # Test with multiple slices
        self.assertEqual(_parse_slices("2x2,2x1,3,2x4"),
                         ((2, 2), (2, 1), 3, (2, 4)))

        # Test with various dimensions
        self.assertEqual(_parse_slices("1x1,10x10,5x3"),
                         ((1, 1), (10, 10), (5, 3)))

        # Test with an empty string
        self.assertEqual(_parse_slices(""), ())

    def test_parse_slices_with_whitespace(self):
        """Tests valid slice strings with extra whitespace."""
        self.assertEqual(_parse_slices(" 2x2 "), ((2, 2), ))
        self.assertEqual(_parse_slices(" 2x2 , 2x1 , 2x4 "),
                         ((2, 2), (2, 1), (2, 4)))
        # The current implementation allows spaces inside the slice definition
        self.assertEqual(_parse_slices("2 x 2"), ((2, 2), ))
        self.assertEqual(_parse_slices(" 10 x 10 "), ((10, 10), ))

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
