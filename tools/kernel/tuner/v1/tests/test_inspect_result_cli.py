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

import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tools.kernel.tuner.v1.inspect_result_cli import (
    FilterResult, _build_parser, _matches_filter, _might_add_baseline_latency,
    _print_case_latency, _print_min_latency, dump_tuned_params_mapping,
    local_get_baseline_latency_map)


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


class TestMightAddBaselineLatency(unittest.TestCase):

    def test_none_baseline_map(self):
        row = {"tuning_key": {"max_tokens": 4}, "Latency": 80.0}
        formatted = {"case_id": "c1"}
        _might_add_baseline_latency(None, row, formatted)
        self.assertEqual(formatted, {"case_id": "c1"})

    def test_key_not_in_baseline_map(self):
        baseline_map = {json.dumps({"max_tokens": 8}, sort_keys=True): 100.0}
        row = {"tuning_key": {"max_tokens": 4}, "Latency": 80.0}
        formatted = {"case_id": "c1"}
        _might_add_baseline_latency(baseline_map, row, formatted)
        self.assertEqual(formatted["baseline_latency"], "N/A")
        self.assertEqual(formatted["latency_improvement%"], "N/A")

    def test_positive_improvement(self):
        baseline_map = {json.dumps({"max_tokens": 4}, sort_keys=True): 100.0}
        row = {"tuning_key": {"max_tokens": 4}, "Latency": 80.0}
        formatted = {"case_id": "c1"}
        _might_add_baseline_latency(baseline_map, row, formatted)
        self.assertEqual(formatted["baseline_latency"], 100.0)
        self.assertEqual(formatted["latency_improvement%"], "+20.0%")

    def test_negative_improvement_regression(self):
        baseline_map = {json.dumps({"max_tokens": 4}, sort_keys=True): 100.0}
        row = {"tuning_key": {"max_tokens": 4}, "Latency": 125.0}
        formatted = {"case_id": "c1"}
        _might_add_baseline_latency(baseline_map, row, formatted)
        self.assertEqual(formatted["baseline_latency"], 100.0)
        self.assertEqual(formatted["latency_improvement%"], "-25.0%")

    def test_none_latency_in_row(self):
        baseline_map = {json.dumps({"max_tokens": 4}, sort_keys=True): 100.0}
        row = {"tuning_key": {"max_tokens": 4}, "Latency": None}
        formatted = {"case_id": "c1"}
        _might_add_baseline_latency(baseline_map, row, formatted)
        self.assertEqual(formatted["baseline_latency"], 100.0)
        self.assertEqual(formatted["latency_improvement%"], "N/A")

    def test_zero_or_negative_baseline_latency(self):
        baseline_map = {json.dumps({"max_tokens": 4}, sort_keys=True): 0.0}
        row = {"tuning_key": {"max_tokens": 4}, "Latency": 50.0}
        formatted = {"case_id": "c1"}
        _might_add_baseline_latency(baseline_map, row, formatted)
        self.assertEqual(formatted["baseline_latency"], 0.0)
        self.assertEqual(formatted["latency_improvement%"], "N/A")


class TestLocalGetBaselineLatencyMap(unittest.TestCase):

    def test_local_get_baseline_latency_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cases = [
                {
                    "ID":
                    "cs1",
                    "CaseId":
                    "c1",
                    "CaseKeyValue":
                    json.dumps({
                        "tuning_key": {
                            "m": 128
                        },
                        "is_baseline": True
                    }),
                },
                {
                    "ID":
                    "cs1",
                    "CaseId":
                    "c2",
                    "CaseKeyValue":
                    json.dumps({
                        "tuning_key": {
                            "m": 128
                        },
                        "is_baseline": False
                    }),
                },
                {
                    "ID":
                    "cs1",
                    "CaseId":
                    "c3",
                    "CaseKeyValue":
                    json.dumps({
                        "tuning_key": {
                            "m": 256
                        },
                        "is_baseline": True
                    }),
                },
            ]
            results = [
                {
                    "ID": "cs1",
                    "RunId": "r1",
                    "CaseId": "c1",
                    "ProcessedStatus": "SUCCESS",
                    "Latency": 150.0,
                },
                {
                    "ID": "cs1",
                    "RunId": "r1",
                    "CaseId": "c2",
                    "ProcessedStatus": "SUCCESS",
                    "Latency": 100.0,
                },
                {
                    "ID": "cs1",
                    "RunId": "r1",
                    "CaseId": "c3",
                    "ProcessedStatus": "FAILED",
                    "Latency": None,
                },
            ]

            with open(os.path.join(tmpdir, "KernelTuningCases.json"),
                      "w") as f:
                json.dump(cases, f)
            with open(os.path.join(tmpdir, "CaseResults.json"), "w") as f:
                json.dump(results, f)

            baseline_map = local_get_baseline_latency_map(tmpdir, "cs1", "r1")
            expected_key = json.dumps({"m": 128}, sort_keys=True)
            self.assertIn(expected_key, baseline_map)
            self.assertEqual(baseline_map[expected_key], 150.0)

            # c3 is FAILED, so m: 256 shouldn't be in baseline_map
            missing_key = json.dumps({"m": 256}, sort_keys=True)
            self.assertNotIn(missing_key, baseline_map)

            # Different case_set_id/run_id returns empty map
            empty_map = local_get_baseline_latency_map(tmpdir, "cs2", "r1")
            self.assertEqual(empty_map, {})


class TestBaselineDisplayAndParser(unittest.TestCase):

    def test_parser_show_baseline_query_min_latency(self):
        parser = _build_parser()
        args = parser.parse_args([
            "query_min_latency", "--case_set_id", "cs1", "--run_id", "r1",
            "--show-baseline"
        ])
        self.assertTrue(args.show_baseline)

    def test_parser_show_baseline_query_case_latency(self):
        parser = _build_parser()
        args = parser.parse_args([
            "query_case_latency", "--case_set_id", "cs1", "--run_id", "r1",
            "--show-baseline"
        ])
        self.assertTrue(args.show_baseline)

    def test_print_min_latency_with_baseline(self):
        tk = {"m": 128}
        tk_str = json.dumps(tk, sort_keys=True)
        baseline_map = {tk_str: 100.0}
        rows = [{
            "CaseId": "c2",
            "Latency": 80.0,
            "WarmupTime": 10.0,
            "tuning_key": tk,
            "tunable_params": {
                "tile": 16
            },
        }]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            _print_min_latency(rows, baseline_map=baseline_map)
            output = mock_stdout.getvalue()

        self.assertIn("baseline_latency", output)
        self.assertIn("latency_improvement%", output)
        self.assertIn("100.0", output)
        self.assertIn("+20.0%", output)

    def test_print_case_latency_with_baseline(self):
        tk = {"m": 128}
        tk_str = json.dumps(tk, sort_keys=True)
        baseline_map = {tk_str: 100.0}
        rows = [{
            "CaseId": "c2",
            "ProcessedStatus": "SUCCESS",
            "Latency": 80.0,
            "WarmupTime": 10.0,
            "TotalTime": 90.0,
            "tuning_key": tk,
            "tunable_params": {
                "tile": 16
            },
        }]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            _print_case_latency(rows, baseline_map=baseline_map)
            output = mock_stdout.getvalue()

        self.assertIn("baseline_latency", output)
        self.assertIn("latency_improvement%", output)
        self.assertIn("100.0", output)
        self.assertIn("+20.0%", output)


class TestDumpTunedParamsMapping(unittest.TestCase):

    def test_dump_tuned_params_mapping(self):
        sample_results = [{
            'tuning_key': {
                'case': 'batched_decode',
                'max_num_tokens': 4,
            },
            'tunable_params': {
                'decode_batch_size': 8,
                'num_kv_pages_per_block': 3,
            },
            'Latency': 100.5,
            'WarmupTime': 10.0,
            'CaseId': 'case_1',
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'tuned_params.py')
            result_path = dump_tuned_params_mapping(sample_results,
                                                    output_path=output_path)
            self.assertEqual(result_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.assertIn(
                'tuned_params_mapping: dict[TuningKey, TunableParams] = {',
                content)
            self.assertIn('TuningKey(', content)
            self.assertIn("case='batched_decode'", content)
            self.assertIn('max_num_tokens=4', content)
            self.assertIn('TunableParams(', content)
            self.assertIn('decode_batch_size=8', content)
            self.assertIn('num_kv_pages_per_block=3', content)


class TestCheckDuplicateShownTuningKeys(unittest.TestCase):

    def test_duplicate_shown_tuning_keys_debug_print(self):
        rows = [
            {
                "CaseId": "73",
                "Latency": 329.0,
                "WarmupTime": 10.0,
                "tuning_key": {
                    "total_q_tokens": 512,
                    "sliding_window": 1024,
                    "num_seqs": 256,
                    "num_page_indices": 4096,
                },
                "tunable_params": {
                    "batch_size": 8
                },
            },
            {
                "CaseId": "1241",
                "Latency": 443.0,
                "WarmupTime": 10.0,
                "tuning_key": {
                    "total_q_tokens": 512,
                    "sliding_window": 1024,
                    "num_seqs": 818,
                    "num_page_indices": 4908,
                },
                "tunable_params": {
                    "batch_size": 16
                },
            },
        ]
        show_fields = [
            "total_q_tokens", "sliding_window", "batch_size", "latency_us"
        ]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            _print_min_latency(rows, show_fields=show_fields)
            output = mock_stdout.getvalue()

        self.assertIn(
            "Debug: 2 distinct TuningKeys share identical displayed key field(s) (total_q_tokens=512, sliding_window=1024)",
            output)
        self.assertIn(
            "hidden differing fields: num_page_indices=4096, num_seqs=256",
            output)
        self.assertIn(
            "hidden differing fields: num_page_indices=4908, num_seqs=818",
            output)


if __name__ == '__main__':

    unittest.main()
