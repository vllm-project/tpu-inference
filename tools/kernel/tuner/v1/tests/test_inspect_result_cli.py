# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from tools.kernel.tuner.v1.inspect_result_cli import (FilterResult,
                                                      _matches_filter)


class TestMatchesFilter(unittest.TestCase):

    def setUp(self):
        # A sample CaseKeyValue dictionary containing both tuning_key and tunable_params
        self.sample_kv = {
            "tuning_key": {
                "max_num_tokens": 4,
                "q_dtype": "fp8",
                "use_bias": True,
                "threshold": 0.5,
                "empty_val": None
            },
            "tunable_params": {
                "block_sizes": [16, 32],
                "name": "test_case",
                "is_active": False
            }
        }

    # --- Match & No Match Tests ---

    def test_integer_filtering(self):
        self.assertEqual(_matches_filter(self.sample_kv, ["max_num_tokens=4"]),
                         FilterResult.MATCH)
        self.assertEqual(_matches_filter(self.sample_kv, ["max_num_tokens=8"]),
                         FilterResult.NO_MATCH)

    def test_string_filtering(self):
        self.assertEqual(_matches_filter(self.sample_kv, ["q_dtype=fp8"]),
                         FilterResult.MATCH)
        self.assertEqual(_matches_filter(self.sample_kv, ["q_dtype=bf16"]),
                         FilterResult.NO_MATCH)

    def test_boolean_filtering_true(self):
        # The function accepts 'true', '1', and 'yes' as True equivalents
        for val in ["true", "1", "yes", "TRUE"]:
            self.assertEqual(
                _matches_filter(self.sample_kv, [f"use_bias={val}"]),
                FilterResult.MATCH)
        self.assertEqual(_matches_filter(self.sample_kv, ["use_bias=false"]),
                         FilterResult.NO_MATCH)

    def test_boolean_filtering_false(self):
        # The function accepts 'false', '0', and 'no' as False equivalents
        for val in ["false", "0", "no", "FALSE"]:
            self.assertEqual(
                _matches_filter(self.sample_kv, [f"is_active={val}"]),
                FilterResult.MATCH)
        self.assertEqual(_matches_filter(self.sample_kv, ["is_active=true"]),
                         FilterResult.NO_MATCH)

    def test_float_filtering(self):
        self.assertEqual(_matches_filter(self.sample_kv, ["threshold=0.5"]),
                         FilterResult.MATCH)
        self.assertEqual(_matches_filter(self.sample_kv, ["threshold=0.6"]),
                         FilterResult.NO_MATCH)

    def test_list_filtering(self):
        self.assertEqual(
            _matches_filter(self.sample_kv, ["block_sizes=[16, 32]"]),
            FilterResult.MATCH)
        # Should be NO_MATCH if the list doesn't match exactly
        self.assertEqual(_matches_filter(self.sample_kv, ["block_sizes=[16]"]),
                         FilterResult.NO_MATCH)

    def test_none_filtering(self):
        # The function accepts 'none', 'null', or empty string as None equivalents
        for val in ["none", "null", "", "NONE"]:
            self.assertEqual(
                _matches_filter(self.sample_kv, [f"empty_val={val}"]),
                FilterResult.MATCH)
        self.assertEqual(_matches_filter(self.sample_kv, ["empty_val=0"]),
                         FilterResult.NO_MATCH)

    def test_multiple_filters(self):
        # Match if all are true
        self.assertEqual(
            _matches_filter(self.sample_kv,
                            ["max_num_tokens=4", "q_dtype=fp8"]),
            FilterResult.MATCH)
        # No match if even one is false
        self.assertEqual(
            _matches_filter(self.sample_kv,
                            ["max_num_tokens=4", "q_dtype=bf16"]),
            FilterResult.NO_MATCH)

    def test_empty_filter_list(self):
        # Should default to MATCH if no filters are applied
        self.assertEqual(_matches_filter(self.sample_kv, []),
                         FilterResult.MATCH)

    # --- Invalid Filter Tests ---

    def test_missing_equals_sign(self):
        self.assertEqual(_matches_filter(self.sample_kv, ["max_num_tokens4"]),
                         FilterResult.INVALID_FILTER)

    def test_missing_field(self):
        self.assertEqual(
            _matches_filter(self.sample_kv, ["non_existent_field=1"]),
            FilterResult.INVALID_FILTER)

    def test_invalid_integer_coercion(self):
        self.assertEqual(
            _matches_filter(self.sample_kv, ["max_num_tokens=four"]),
            FilterResult.INVALID_FILTER)

    def test_invalid_boolean_coercion(self):
        self.assertEqual(_matches_filter(self.sample_kv, ["use_bias=maybe"]),
                         FilterResult.INVALID_FILTER)

    def test_invalid_float_coercion(self):
        self.assertEqual(_matches_filter(self.sample_kv, ["threshold=half"]),
                         FilterResult.INVALID_FILTER)

    def test_invalid_list_evaluation(self):
        self.assertEqual(
            _matches_filter(self.sample_kv, ["block_sizes=not_a_list"]),
            FilterResult.INVALID_FILTER)


if __name__ == '__main__':
    unittest.main()
