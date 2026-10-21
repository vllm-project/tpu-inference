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

import json
import os
import sys
import uuid
from datetime import datetime


def fmt(val, is_str=False):
    """Safely formats python values to SQL-compatible strings for file output."""
    if val is None or val == 'NULL':
        return 'NULL'
    if str(val).lower() == 'inf' or str(val) == 'Infinity':
        return "CAST('inf' AS FLOAT64)"
    if is_str:
        # Prevent SQL injection by escaping single quotes
        # 'Hello' becomes ''Hello'' (Standard SQL escaping)
        safe_val = str(val).replace("'", "''")
        return f"'{safe_val}'"
    return str(val)


def parse_and_dump(file_path, record_id):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Extract input_len and output_len from filename
    # Assuming filename format like path/to/1024_8192.json
    base_name = os.path.basename(file_path)
    name_part, _ = os.path.splitext(base_name)
    try:
        input_len_str, output_len_str = name_part.split('_')
        input_len = int(input_len_str)
        output_len = int(output_len_str)
    except ValueError:
        print(
            f"Warning: Could not parse input/output len from filename: {base_name}. Using NULL.",
            file=sys.stderr)
        input_len = 'NULL'
        output_len = 'NULL'

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from line: {line}",
                      file=sys.stderr)
                continue

            rate = data.get('request_rate', 'unknown')
            concurrency = data.get('max_concurrency', 'unknown')
            short_suffix = uuid.uuid4().hex[:5]
            # Parse date from JSON
            # Example: "20260402-195133" -> "2026-04-02T19:51:33Z"
            date_str = data.get('date')
            spanner_timestamp = 'CURRENT_TIMESTAMP()'
            if date_str:
                try:
                    dt = datetime.strptime(date_str, '%Y%m%d-%H%M%S')
                    spanner_timestamp = f"TIMESTAMP '{dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
                except ValueError:
                    print(
                        f"Warning: Could not parse date string: {date_str}. Using CURRENT_TIMESTAMP()",
                        file=sys.stderr)

            # Sanitize the components of the record ID
            unique_record_id = fmt(
                f"{record_id}_{rate}_c{concurrency}_{short_suffix}",
                is_str=True)

            sql = f"""
            INSERT INTO RunRecord (
                RecordId, Status, CreatedTime, LastUpdate,
                Device, Model, RunType, CodeHash, Dataset, CreatedBy, InputLen, OutputLen,
                EndpointType, Backend, Label, TokenizerId, NumPrompts, RequestRate, Burstiness, MaxConcurrency,
                Duration, Completed, Failed, TotalInputTokens, TotalOutputTokens, MaxConcurrentRequests,
                RequestThroughput, RequestGoodput, OutputThroughput, TotalTokenThroughput, MaxOutputTokensPerS, Rtfx,
                MeanTTFT, MedianTTFT, StdTTFT, P90TTFT, P99TTFT,
                MeanTPOT, MedianTPOT, StdTPOT, P90TPOT, P99TPOT,
                MeanITL, MedianITL, StdITL, P90ITL, P99ITL
            ) VALUES (
                {unique_record_id}, 'COMPLETED', {spanner_timestamp}, CURRENT_TIMESTAMP(),
                'GKE', {fmt(data.get('model_id', 'N/A'), True)}, 'GKE_DISAGG', 'N/A', 'random', 'scheduler', {fmt(input_len)}, {fmt(output_len)},
                {fmt(data.get('endpoint_type'), True)}, {fmt(data.get('backend'), True)}, {fmt(data.get('label'), True)}, {fmt(data.get('tokenizer_id'), True)}, 
                {fmt(data.get('num_prompts'))}, {fmt(data.get('request_rate'))}, {fmt(data.get('burstiness'))}, {fmt(data.get('max_concurrency'))},
                {fmt(data.get('duration'))}, {fmt(data.get('completed'))}, {fmt(data.get('failed'))}, {fmt(data.get('total_input_tokens'))}, {fmt(data.get('total_output_tokens'))}, {fmt(data.get('max_concurrent_requests'))},
                {fmt(data.get('request_throughput'))}, {fmt(data.get('request_goodput'))}, {fmt(data.get('output_throughput'))}, {fmt(data.get('total_token_throughput'))}, {fmt(data.get('max_output_tokens_per_s'))}, {fmt(data.get('rtfx'))},
                {fmt(data.get('mean_ttft_ms'))}, {fmt(data.get('median_ttft_ms'))}, {fmt(data.get('std_ttft_ms'))}, {fmt(data.get('p90_ttft_ms'))}, {fmt(data.get('p99_ttft_ms'))},
                {fmt(data.get('mean_tpot_ms'))}, {fmt(data.get('median_tpot_ms'))}, {fmt(data.get('std_tpot_ms'))}, {fmt(data.get('p90_tpot_ms'))}, {fmt(data.get('p99_tpot_ms'))},
                {fmt(data.get('mean_itl_ms'))}, {fmt(data.get('median_itl_ms'))}, {fmt(data.get('std_itl_ms'))}, {fmt(data.get('p90_itl_ms'))}, {fmt(data.get('p99_itl_ms'))}
            );
            """

            # Print as a single line for bash processing
            print(" ".join(sql.split()))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_gke_results.py <file_path> <record_id>",
              file=sys.stderr)
        sys.exit(1)
    parse_and_dump(sys.argv[1], sys.argv[2])
