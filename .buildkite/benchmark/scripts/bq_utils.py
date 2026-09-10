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
"""Shared helpers for inserting benchmark rows into BigQuery.

This is a port of vllm-torchtpu's scripts/vllm/benchmarking/bq_utils.py.
Both repositories write to the same table, so the two copies are meant to
stay in sync -- keep changes here mirrored there. Two deliberate
differences from that copy: sql_escape escapes backslashes and uses
GoogleSQL's \\' quote escape rather than the doubled '' form, which
BigQuery rejects (see sql_escape -- that copy has the same bug, latent
only because its payloads happen to contain no apostrophes), and
build_delete_sql exists because reporting here is an upsert keyed on a
stable RecordId.

Table schema: record_id, created_time, repository, run_type, model_id,
metrics ARRAY<STRUCT<type STRING, values JSON>>, config JSON. Each
metrics element's type names the tool that produced its payload (e.g.
'vllm_bench', 'lm_eval'); values keeps that tool's flat leaf keys
verbatim, so generic names (completed, failed, duration) can't clash with
other harnesses writing to the shared table and provenance stays explicit.
"""

import json
import os
import subprocess
import sys

DEFAULT_BQ_PROJECT_ID = "cloud-ullm-inference-ci-cd"
DEFAULT_BQ_TABLE = ("cloud-ullm-inference-ci-cd"
                    ".llm_benchmark_analytics.benchmark_runs")
DEFAULT_REPOSITORY = "tpu-inference"


def sql_escape(val: str) -> str:
    """Escape a value for a single-quoted GoogleSQL string literal.

    GoogleSQL escapes a quote as \\' -- it does not accept the doubled ''
    form, and parses 'it''s' as two adjacent literals rather than one
    string. Backslashes are escaped first so the added ones survive.
    """
    return val.replace("\\", "\\\\").replace("'", "\\'")


def resolve_table(bq_table=None) -> str:
    """`bq_table` falls back to the BQ_TABLE env var, then the default.
    Pipelines may export BQ_TABLE as an empty string, so `or` (empty falls
    through) is used instead of os.getenv's unset-only default. Backticks
    around an override are accepted and normalized."""
    return (bq_table or os.getenv("BQ_TABLE") or DEFAULT_BQ_TABLE).strip("`")


def build_insert_sql(record_id: str,
                     created_time: str,
                     run_type: str,
                     model_id: str,
                     metrics: list,
                     config: dict,
                     bq_table=None) -> str:
    """Render one INSERT statement for the benchmark_runs table.

    `metrics` is a list of (type, payload-dict) pairs, one array element
    per producing tool. `created_time` is a SQL expression, e.g.
    CURRENT_TIMESTAMP().
    """
    table = resolve_table(bq_table)
    repository = os.getenv("REPOSITORY") or DEFAULT_REPOSITORY
    config_json = sql_escape(json.dumps(config))
    metrics_elems = ", ".join(
        f"STRUCT('{metrics_type}', "
        f"PARSE_JSON('{sql_escape(json.dumps(payload))}', "
        "wide_number_mode=>'round'))" for metrics_type, payload in metrics)
    sql = f"""
    INSERT INTO `{table}` (
        record_id, created_time, repository, run_type, model_id, metrics, config
    ) VALUES (
        '{sql_escape(record_id)}', {created_time},
        '{sql_escape(repository)}', '{sql_escape(run_type)}',
        '{sql_escape(model_id)}',
        [{metrics_elems}],
        PARSE_JSON('{config_json}', wide_number_mode=>'round')
    );
    """
    # Single line so the statement survives shell/CLI round-trips intact.
    return " ".join(sql.split())


def build_delete_sql(record_id: str, bq_table=None) -> str:
    """Render the DELETE that clears a record_id before re-inserting it.

    Spanner reporting is an upsert keyed on RecordId, so a retried step
    replaces its row rather than adding one. BigQuery has no such
    constraint, so a retry is preceded by this DELETE to keep the two
    sinks telling the same story.
    """
    table = resolve_table(bq_table)
    repository = os.getenv("REPOSITORY") or DEFAULT_REPOSITORY
    return (f"DELETE FROM `{table}` WHERE record_id = "
            f"'{sql_escape(record_id)}' AND repository = "
            f"'{sql_escape(repository)}';")


def run_query(sql: str,
              project=None,
              label: str = "",
              record_id: str = "",
              skip: bool = False) -> bool:
    """Execute one statement via the bq CLI. Prints the statement either
    way; with skip=True it is printed but not executed."""
    print(f"SQL for BigQuery ({label}):")
    print(sql)

    if skip:
        print(f"=== Skipping BigQuery upload ===. record_id: {record_id}")
        return True

    project = project or os.getenv("BQ_PROJECT_ID") or DEFAULT_BQ_PROJECT_ID
    cmd = [
        "bq", "query", "--use_legacy_sql=false", f"--project_id={project}", sql
    ]
    print(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
    except FileNotFoundError:
        print(
            f"Failed to run BigQuery statement for {label}: 'bq' CLI not found.",
            file=sys.stderr)
        return False
    if proc.returncode != 0:
        print(f"Failed to run BigQuery statement for {label}!",
              file=sys.stderr)
        print(f"Stdout:\n{proc.stdout}", file=sys.stderr)
        print(f"Stderr:\n{proc.stderr}", file=sys.stderr)
        return False
    print(f"BigQuery statement succeeded for {label}. record_id: {record_id}")
    return True
