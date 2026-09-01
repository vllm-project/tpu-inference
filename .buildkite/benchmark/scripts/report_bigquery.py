#!/usr/bin/env python3
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
"""Mirror one benchmark result into BigQuery alongside the Spanner upsert.

report_result.sh calls this immediately after it writes RunRecord, with
the same RECORD_ID, so a row in either sink can be joined to the other.
Dashboards move to BigQuery on their own schedule; until they have, both
writes have to stay truthful.

Metric names are translated from this repo's Spanner column names to the
leaf names `vllm bench serve` prints, because that is the vocabulary the
shared table already speaks -- vllm-torchtpu writes the raw payload. A
query spanning both repositories should not have to know which harness
produced a row.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bq_utils  # noqa: E402

# parse_benchmark_log.py names metrics after their Spanner columns; the
# shared BigQuery table keys them the way `vllm bench serve` prints them.
SPANNER_TO_BENCH_METRIC = {
    "Throughput": "request_throughput",
    "OutputTokenThroughput": "output_throughput",
    "TotalTokenThroughput": "total_token_throughput",
    "MedianTTFT": "median_ttft_ms",
    "P99TTFT": "p99_ttft_ms",
    "MedianTPOT": "median_tpot_ms",
    "P99TPOT": "p99_tpot_ms",
    "MedianITL": "median_itl_ms",
    "P99ITL": "p99_itl_ms",
    "MedianETEL": "median_e2el_ms",
    "P99ETEL": "p99_e2el_ms",
}


def read_result_file(path):
    """Parse the KEY=VALUE result file into (bench metrics, lm_eval metrics).

    Unrecognized keys are dropped rather than guessed at; report_result.sh
    already warns about them when building the Spanner statement.
    """
    bench = {}
    accuracy = {}
    if not path or not os.path.exists(path):
        print(f"Warning: result file not found: {path}", file=sys.stderr)
        return bench, accuracy

    with open(path, "r") as fh:
        for line in fh:
            key, sep, value = line.strip().partition("=")
            if not sep or not key or not value:
                continue
            if key == "AccuracyMetrics":
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as e:
                    print(f"Warning: AccuracyMetrics is not valid JSON: {e}",
                          file=sys.stderr)
                    continue
                if isinstance(parsed, dict):
                    accuracy = parsed
                continue
            bench_key = SPANNER_TO_BENCH_METRIC.get(key)
            if not bench_key:
                continue
            try:
                bench[bench_key] = float(value)
            except ValueError:
                print(f"Warning: non-numeric value for {key}: {value!r}",
                      file=sys.stderr)
    return bench, accuracy


def env_int(name, default=None):
    val = os.getenv(name)
    if val in (None, ""):
        return default
    try:
        return int(val)
    except ValueError:
        return default


def env_str(name, default=None):
    # run_job.sh exports several of these as empty strings when unset, so
    # `or` (empty falls through) is used instead of os.getenv's
    # unset-only default.
    return os.getenv(name) or default


def build_config(status, attempt):
    """The non-metric half of the row: everything a chart might group or
    filter by. Mirrors the shape vllm-torchtpu writes so the shared table
    stays queryable without per-repository special cases."""
    config = {
        "code_hash": env_str("CODE_HASH"),
        "created_by": env_str("USER", "buildkite-agent"),
        "device_type": env_str("DEVICE"),
        "run_by": env_str("BUILDKITE_AGENT_NAME"),
        "job_reference": env_str("JOB_REFERENCE"),
        "backend": env_str("DATASET"),
        "case_name": env_str("TARGET_CASE_NAME"),
        "model_tag": env_str("MODELTAG", "PROD"),
        "status": status,
        "attempt": attempt,
        "engine_flags": {
            "max_num_seqs": env_int("MAX_NUM_SEQS"),
            "max_num_batched_tokens": env_int("MAX_NUM_BATCHED_TOKENS"),
            "tensor_parallel_size": env_int("TENSOR_PARALLEL_SIZE"),
            "max_model_len": env_int("MAX_MODEL_LEN"),
            "extra_envs": env_str("EXTRA_ENVS"),
            "extra_args": env_str("EXTRA_ARGS"),
            "additional_config": env_str("ADDITIONAL_CONFIG"),
        },
        # Defaults match report_result.sh's, so the same run does not land
        # in Spanner and BigQuery describing two different workloads.
        "workload": {
            "dataset": env_str("DATASET"),
            "input_len": env_int("INPUT_LEN"),
            "output_len": env_int("OUTPUT_LEN"),
            "prefix_len": env_int("PREFIX_LEN", 0),
            "num_prompts": env_int("NUM_PROMPTS", 1000),
            "expected_etel": env_int("EXPECTED_ETEL", 3600000),
        },
    }

    # The full case definition, so a row stays interpretable even after
    # the case file changes underneath it.
    raw_case = env_str("CASE_CONFIG_JSON")
    if raw_case:
        try:
            config["case"] = json.loads(raw_case)
        except json.JSONDecodeError as e:
            print(f"Warning: CASE_CONFIG_JSON is not valid JSON: {e}",
                  file=sys.stderr)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Dual-write a benchmark result to BigQuery.")
    parser.add_argument("--record-id",
                        required=True,
                        help="Spanner RecordId; reused as the BigQuery "
                        "record_id so the two rows join.")
    parser.add_argument("--result-file",
                        required=True,
                        help="KEY=VALUE file written by "
                        "parse_benchmark_log.py.")
    parser.add_argument("--status",
                        default="COMPLETED",
                        help="Final Spanner Status for this run.")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="Print the SQL without executing it.")
    parser.add_argument("--bq-project",
                        default=None,
                        help="GCP project used to run BigQuery jobs "
                        "(default: BQ_PROJECT_ID env or "
                        f"{bq_utils.DEFAULT_BQ_PROJECT_ID}).")
    parser.add_argument("--bq-table",
                        default=None,
                        help="Fully-qualified BigQuery table "
                        "(default: BQ_TABLE env or "
                        f"{bq_utils.DEFAULT_BQ_TABLE}).")
    args = parser.parse_args()

    bench, accuracy = read_result_file(args.result_file)

    model = env_str("MODEL_NAME") or env_str("MODEL")
    if not model:
        print("Error: neither MODEL_NAME nor MODEL is set.", file=sys.stderr)
        return 1

    attempt = env_int("BUILDKITE_RETRY_COUNT") or 0

    # Spanner reports a retried step by updating its row in place. Clear
    # the previous BigQuery row first so the retry replaces it rather
    # than double-counting in any chart that averages over runs.
    if attempt > 0:
        delete_sql = bq_utils.build_delete_sql(args.record_id,
                                               bq_table=args.bq_table)
        if not bq_utils.run_query(delete_sql,
                                  project=args.bq_project,
                                  label=f"retry cleanup attempt {attempt}",
                                  record_id=args.record_id,
                                  skip=args.dry_run):
            return 1

    metrics = []
    if bench:
        metrics.append(("vllm_bench", bench))
    if accuracy:
        metrics.append(("lm_eval", accuracy))
    if not metrics:
        # A failed run still belongs in the table -- config.status is how
        # a dashboard tells a regression from an outage -- but the row
        # needs at least one array element to stay shaped like the rest.
        metrics.append(("vllm_bench", {}))

    sql = bq_utils.build_insert_sql(record_id=args.record_id,
                                    created_time="CURRENT_TIMESTAMP()",
                                    run_type=env_str("RUN_TYPE", "DAILY"),
                                    model_id=model,
                                    metrics=metrics,
                                    config=build_config(args.status, attempt),
                                    bq_table=args.bq_table)

    ok = bq_utils.run_query(sql,
                            project=args.bq_project,
                            label=env_str("TARGET_CASE_NAME", args.record_id),
                            record_id=args.record_id,
                            skip=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
