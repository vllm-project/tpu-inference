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

import ast
import dataclasses
import html
import json
import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from absl import app, flags

from tools.kernel.tuner.v1.autotune.kernel_autotune_config import \
    kernel_autotune_mapping

logger = logging.getLogger(__name__)

_GCP_PROJECT_ID = flags.DEFINE_string(
    'gcp_project_id', 'cloud-tpu-inference-test',
    'The GCP project ID to use for Spanner. Only used when --run_locally is false.'
)
_SPANNER_INSTANCE_ID = flags.DEFINE_string(
    'spanner_instance_id', 'vllm-bm-inst',
    'The Spanner instance ID to use. Only used when --run_locally is false.')
_SPANNER_DATABASE_ID = flags.DEFINE_string(
    'spanner_database_id', 'tune-gmm',
    'The Spanner database ID to use. Only used when --run_locally is false.')
_AUTOTUNE_ID = flags.DEFINE_string(
    'autotune_id', '',
    'The autotune ID to use for this run, for example, "KERNEL_AUTOTUNE_2026-06-23-07-10".'
)
_PROCESS_STEP = flags.DEFINE_string(
    'process_step', 'PATCH_KERNEL_AUTOTUNE_RESULT',
    'The process step to run. Options: EVALUATE_AND_CREATE_PR, PATCH_KERNEL_AUTOTUNE_RESULT'
)
_METRIC_IMPROVEMENT_THRESHOLD = flags.DEFINE_float(
    'metric_improvement_threshold', 0.004,
    'The metric improvement threshold to use for this run.')

REPORT_OUTPUT_PATH_PREFIX = "/tmp/kernel_tuning/kernel_autotune_report"


class KernelAutoTuneResultProcessor:

    def __init__(self):
        self.autotune_id = self._get_flag_value(_AUTOTUNE_ID)
        self.gcp_project_id = self._get_flag_value(_GCP_PROJECT_ID)
        self.spanner_instance_id = self._get_flag_value(_SPANNER_INSTANCE_ID)
        self.spanner_database_id = self._get_flag_value(_SPANNER_DATABASE_ID)
        process_step = self._get_flag_value(_PROCESS_STEP)
        assert process_step in [
            'EVALUATE_AND_CREATE_PR', 'PATCH_KERNEL_AUTOTUNE_RESULT'
        ], f"Invalid process step: {process_step}. Must be one of ['EVALUATE_AND_CREATE_PR', 'PATCH_KERNEL_AUTOTUNE_RESULT']"
        self.process_step = process_step
        self._spanner_db = None

    def _get_flag_value(self, flag_def):
        try:
            return flag_def.value
        except Exception:  # pylint: disable=broad-exception-caught
            return flag_def.default

    def _get_spanner_db(self, project, instance_id, database_id):
        if self._spanner_db is None:
            from google.cloud import \
                spanner as gspanner  # pylint: disable=import-outside-toplevel
            client = gspanner.Client(project=project,
                                     disable_builtin_metrics=True)
            self._spanner_db = client.instance(instance_id).database(
                database_id)
        return self._spanner_db

    def get_best_results(self, case_set_id) -> list[dict]:
        """
        Get the best results for each case in a kernel auto-tuning run.

        Args:
            case_set_id (str): The ID of the case set.

        Returns:
            list[dict]: A list of dictionaries containing the best results for each case.
        """
        from google.cloud import \
            spanner as gspanner  # pylint: disable=import-outside-toplevel
        query = """
            SELECT cr.CaseId, cr.Latency, cr.WarmupTime, ktc.CaseKeyValue
            FROM CaseResults cr
            JOIN KernelTuningCases ktc ON cr.ID = ktc.ID AND cr.CaseId = ktc.CaseId
            WHERE cr.ID = @id AND cr.RunId = @rid AND cr.ProcessedStatus = 'SUCCESS'
            ORDER BY cr.CaseId
        """
        db = self._get_spanner_db(self.gcp_project_id,
                                  self.spanner_instance_id,
                                  self.spanner_database_id)
        key_best = {}
        with db.snapshot() as snap:
            for case_id, lat, warmup, kv_str in snap.execute_sql(
                    query,
                    params={
                        'id': case_set_id,
                        'rid': '0'
                    },
                    param_types={
                        'id': gspanner.param_types.STRING,
                        'rid': gspanner.param_types.STRING
                    }):
                try:
                    kv = json.loads(kv_str)
                except (json.JSONDecodeError, TypeError):
                    continue
                tk_str = json.dumps(kv.get('tuning_key', {}), sort_keys=True)
                if tk_str not in key_best or lat < key_best[tk_str]['Latency']:
                    key_best[tk_str] = {
                        'tuning_key': kv.get('tuning_key'),
                        'tunable_params': kv.get('tunable_params'),
                        'Latency': lat,
                        'WarmupTime': warmup,
                        'CaseId': case_id,
                    }
        result = sorted(
            key_best.values(),
            key=lambda x: json.dumps(x['tuning_key'], sort_keys=True))
        return result

    def update_best_results(self, best_tunable_params: list[dict],
                            kernel_tuner_name: str):
        """
        Update the best results for each case in a kernel auto-tuning run.

        Args:
            best_tunable_params (list[dict]): A list of dictionaries containing the best tunable parameters for each case.
            kernel_tuner_name (str): The name of the kernel tuner.
        """
        if kernel_tuner_name not in kernel_autotune_mapping:
            raise ValueError(
                f"Kernel tuner name '{kernel_tuner_name}' not found in kernel_autotune_mapping."
            )
        file_path = kernel_autotune_mapping[kernel_tuner_name]
        # Dynamically import the tuned params module from file_path.
        import importlib.util
        spec = importlib.util.spec_from_file_location("tuned_params_module",
                                                      file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec for '{file_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the dataclasses from the module and construct new entries.
        TuningKey = getattr(module, "TuningKey")
        TunableParams = getattr(module, "TunableParams")
        tuning_key_to_params = {}
        sorted_best_tunable_params = sorted(
            best_tunable_params,
            key=lambda item: json.dumps(item.get('tuning_key', {}),
                                        sort_keys=True),
        )
        for item in sorted_best_tunable_params:
            tuning_key = TuningKey(**item['tuning_key'])
            tunable_params = TunableParams(**item['tunable_params'])
            tuning_key_to_params[tuning_key] = tunable_params

        # Before updating the file, log how many existing entries will be
        # updated and how many new entries will be added.
        existing_keys = set(module.tuned_params_mapping.keys())
        new_keys = set(tuning_key_to_params.keys())
        may_updated_keys = existing_keys.intersection(new_keys)
        # Check whether the overlapping keys have different values.
        existing_params_updated = 0
        for key in may_updated_keys:
            if module.tuned_params_mapping[key] != tuning_key_to_params[key]:
                logger.info(
                    f"Updating existing key: {key} with new value: {tuning_key_to_params[key]}"
                )
                existing_params_updated += 1
        logger.info(
            f"Updating {existing_params_updated} existing keys with new values."
        )
        new_keys_to_add = new_keys - existing_keys
        logger.info(f"Adding {len(new_keys_to_add)} new keys to the mapping.")
        # Update the in-memory mapping first.
        module.tuned_params_mapping.update(tuning_key_to_params)
        # Persist only tuned_params_mapping back to the original file.
        self._write_tuned_params_mapping(Path(file_path), module)

    def _write_tuned_params_mapping(self, file_path: Path, module):
        source = file_path.read_text(encoding='utf-8')
        tree = ast.parse(source)

        mapping_node = None
        for node in tree.body:
            target = None
            if isinstance(node, ast.Assign) and node.targets and isinstance(
                    node.targets[0], ast.Name):
                target = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name):
                target = node.target.id

            if target == 'tuned_params_mapping':
                mapping_node = node
                break

        if mapping_node is None:
            raise ValueError(
                f"Could not find 'tuned_params_mapping' assignment in '{file_path}'"
            )

        if mapping_node.end_lineno is None:
            raise ValueError(
                f"Could not determine source range for 'tuned_params_mapping' in '{file_path}'"
            )

        lines = source.splitlines(keepends=True)
        start_idx = mapping_node.lineno - 1
        end_idx = mapping_node.end_lineno

        mapping_text = self._build_mapping_text(module)
        lines[start_idx:end_idx] = [mapping_text]
        new_source = ''.join(lines)
        self._validate_generated_source(new_source, file_path)
        file_path.write_text(new_source, encoding='utf-8')

    def _validate_generated_source(self, source: str, file_path: Path) -> None:
        try:
            ast.parse(source, filename=str(file_path))
            compile(source, str(file_path), 'exec')
        except SyntaxError as exc:
            raise ValueError(
                f"Generated source for 'tuned_params_mapping' in '{file_path}' "
                f"is invalid: {exc}") from exc

    def _build_mapping_text(self, module) -> str:
        key_cls = module.TuningKey
        params_cls = module.TunableParams
        key_fields = [field.name for field in dataclasses.fields(key_cls)]
        params_fields = [
            field.name for field in dataclasses.fields(params_cls)
        ]

        chunks = ["tuned_params_mapping: dict[TuningKey, TunableParams] = {\n"]
        sorted_items = sorted(
            module.tuned_params_mapping.items(),
            key=lambda item: json.dumps(dataclasses.asdict(item[0]),
                                        sort_keys=True),
        )
        for key, params in sorted_items:
            key_values = dataclasses.asdict(key)
            param_values = dataclasses.asdict(params)

            chunks.append("    TuningKey(\n")
            for field_name in key_fields:
                if field_name in key_values:
                    chunks.append(
                        f"        {field_name}={self._py_repr(key_values[field_name])},\n"
                    )
            chunks.append("    ):\n")

            chunks.append("    TunableParams(\n")
            for field_name in params_fields:
                if field_name in param_values:
                    chunks.append(
                        f"        {field_name}={self._py_repr(param_values[field_name])},\n"
                    )
            chunks.append("    ),\n")

        chunks.append("}\n")
        return ''.join(chunks)

    def _py_repr(self, value):
        return repr(value)

    def _evaluate_metric_result(self, baseline, tuned, metric, threshold,
                                lower_is_better_metrics):
        delta_pct = None
        verdict = 'NO_DATA'

        if baseline is None or tuned is None:
            verdict = 'MISSING'
        elif math.isclose(baseline, 0.0, abs_tol=1e-12):
            verdict = 'BASELINE_ZERO'
        else:
            if metric in lower_is_better_metrics:
                delta_pct = (baseline - tuned) / baseline
            else:
                delta_pct = (tuned - baseline) / baseline

            improved = delta_pct > threshold
            regressed = delta_pct < -threshold
            if regressed:
                verdict = 'REGRESSION'
            elif improved:
                verdict = 'IMPROVED'
            else:
                verdict = 'NEUTRAL'

        return delta_pct, verdict

    def _should_create_pr(self, monitor_improved, has_regression,
                          hard_blocker):
        return monitor_improved and not has_regression and not hard_blocker

    def patch_tuned_results(self):
        for kernel_tuner_name in kernel_autotune_mapping.keys():
            case_set_id = f"{kernel_tuner_name}_{self.autotune_id}"
            logger.info(f"Processing results for case set ID: {case_set_id}")
            best_results = self.get_best_results(case_set_id)
            self.update_best_results(best_results, kernel_tuner_name)

    def _github_request(self,
                        method: str,
                        path: str,
                        token: str,
                        payload: dict | None = None) -> dict:
        url = f"https://api.github.com{path}"
        body = None
        headers = {
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {token}',
            'X-GitHub-Api-Version': '2022-11-28',
            'User-Agent': 'tpu-inference-kernel-autotune-bot',
        }
        if payload is not None:
            body = json.dumps(payload).encode('utf-8')
            headers['Content-Type'] = 'application/json'

        req = urllib.request.Request(url,
                                     data=body,
                                     headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode('utf-8')
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(
                f"GitHub API request failed: method={method}, path={path}, "
                f"status={exc.code}, body={error_body}") from exc

    def _create_or_update_pr(self, pr_body: str) -> str | None:
        token = os.environ.get('GITHUB_CI_BOT_TOKEN', '').strip()
        if not token:
            logger.warning(
                "GITHUB_CI_BOT_TOKEN is not set; skipping PR creation.")
            return None

        owner = os.environ.get('KERNEL_AUTOTUNE_PR_OWNER', 'vllm-project')
        repo = os.environ.get('KERNEL_AUTOTUNE_PR_REPO', 'tpu-inference')
        head_branch = (
            os.environ.get('KERNEL_AUTOTUNE_PR_HEAD_BRANCH', '').strip()
            or f"kernel_autotune.update_tuned_params_{self.autotune_id}")
        base_branch = (os.environ.get(
            'KERNEL_AUTOTUNE_PR_BASE_BRANCH', '').strip() or os.environ.get(
                'BUILDKITE_PULL_REQUEST_BASE_BRANCH', '').strip() or 'main')
        pr_title = (os.environ.get('KERNEL_AUTOTUNE_PR_TITLE', '').strip()
                    or f"Kernel autotune update for {self.autotune_id}")

        encoded_head = urllib.parse.quote(f"{owner}:{head_branch}", safe='')
        encoded_base = urllib.parse.quote(base_branch, safe='')
        existing = self._github_request(
            'GET',
            f"/repos/{owner}/{repo}/pulls?state=open&head={encoded_head}&base={encoded_base}",
            token,
        )
        if existing:
            pr_number = existing[0]['number']
            updated = self._github_request(
                'PATCH',
                f"/repos/{owner}/{repo}/pulls/{pr_number}",
                token,
                payload={
                    'title': pr_title,
                    'body': pr_body,
                    'base': base_branch,
                },
            )
            logger.info("Updated existing PR #%s", pr_number)
            return updated.get('html_url')

        created = self._github_request(
            'POST',
            f"/repos/{owner}/{repo}/pulls",
            token,
            payload={
                'title': pr_title,
                'head': head_branch,
                'base': base_branch,
                'body': pr_body,
                'maintainer_can_modify': True,
            },
        )
        logger.info("Created new PR for branch %s", head_branch)
        return created.get('html_url')

    def evaluate_benchmark_metrics_and_create_pr_body(self):
        logger.info("Comparing benchmark metrics")

        monitor_metrics = {
            'Throughput',
            'MedianITL',
        }
        all_metrics = [
            'Throughput',
            'MedianITL',
            'MedianTPOT',
            'MedianTTFT',
            'MedianETEL',
            'P99ITL',
            'P99TPOT',
            'P99TTFT',
            'P99ETEL',
            'OutputTokenThroughput',
            'TotalTokenThroughput',
        ]
        lower_is_better_metrics = {
            'MedianITL',
            'MedianTPOT',
            'MedianTTFT',
            'MedianETEL',
            'P99ITL',
            'P99TPOT',
            'P99TTFT',
            'P99ETEL',
        }
        threshold = _METRIC_IMPROVEMENT_THRESHOLD.value

        from google.cloud import \
            spanner as gspanner  # pylint: disable=import-outside-toplevel

        postfix_match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2})$',
                                  self.autotune_id)
        autotune_postfix = postfix_match.group(
            1) if postfix_match else self.autotune_id
        autotune_like = f"%{autotune_postfix}%"

        metric_columns_sql = ', '.join(all_metrics)
        query = f"""
            SELECT
                rr.Config,
                rr.Status,
                {metric_columns_sql}
            FROM RunRecord rr
            WHERE rr.ExtraEnvs LIKE @autotune_like
              AND rr.ExtraEnvs LIKE @stage_like
        """

        db = self._get_spanner_db(self.gcp_project_id,
                                  self.spanner_instance_id,
                                  self.spanner_database_id)

        warnings: list[str] = []

        def _normalize_config(config_value) -> str:
            if isinstance(config_value, (dict, list)):
                return json.dumps(config_value, sort_keys=True)
            if isinstance(config_value, str):
                try:
                    parsed = json.loads(config_value)
                    return json.dumps(parsed, sort_keys=True)
                except json.JSONDecodeError:
                    return config_value
            return repr(config_value)

        def _to_float(value):
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _fetch_stage_records(stage: str):
            stage_like = f"%{stage}%"
            grouped = {}
            with db.snapshot() as snap:
                rows = snap.execute_sql(query,
                                        params={
                                            'autotune_like': autotune_like,
                                            'stage_like': stage_like,
                                        },
                                        param_types={
                                            'autotune_like':
                                            gspanner.param_types.STRING,
                                            'stage_like':
                                            gspanner.param_types.STRING,
                                        })
                for row in rows:
                    config_key = _normalize_config(row[0])
                    record = {
                        'status': row[1],
                        'metrics': {
                            metric: _to_float(row[idx + 2])
                            for idx, metric in enumerate(all_metrics)
                        },
                    }
                    grouped.setdefault(config_key, []).append(record)
            return grouped

        def _pick_record(records, stage_label: str, config_key: str):
            if len(records) > 1:
                warnings.append(
                    f"Found {len(records)} rows for stage={stage_label}, config={config_key}. "
                    "Using first COMPLETED row when available.")
            completed = [r for r in records if r['status'] == 'COMPLETED']
            if completed:
                return completed[0]
            return records[0]

        pre_stage = 'PRE_KERNEL_AUTOTUNE_CASES_COLLECTION'
        post_stage = 'POST_KERNEL_AUTOTUNE_BM_RERUN'
        pre_by_config = _fetch_stage_records(pre_stage)
        post_by_config = _fetch_stage_records(post_stage)

        all_config_keys = sorted(set(pre_by_config) | set(post_by_config))

        rows = []
        status_by_config = {}
        monitor_improved = False
        has_regression = False
        hard_blocker = False

        for config_key in all_config_keys:
            pre_records = pre_by_config.get(config_key, [])
            post_records = post_by_config.get(config_key, [])

            if not pre_records:
                warnings.append(
                    f"No baseline row found for config={config_key}; skipping comparison."
                )
                hard_blocker = True
                continue
            if not post_records:
                warnings.append(
                    f"No tuned row found for config={config_key}; skipping comparison."
                )
                hard_blocker = True
                continue

            pre = _pick_record(pre_records, pre_stage, config_key)
            post = _pick_record(post_records, post_stage, config_key)

            if pre['status'] != 'COMPLETED' or post['status'] != 'COMPLETED':
                warnings.append(
                    f"Config={config_key} skipped because status is not COMPLETED "
                    f"(pre={pre['status']}, post={post['status']}).")
                hard_blocker = True
                continue

            status_by_config[config_key] = {
                'baseline_status': pre['status'],
                'tuned_status': post['status'],
            }

            for metric in all_metrics:
                baseline = pre['metrics'][metric]
                tuned = post['metrics'][metric]
                delta_pct, verdict = self._evaluate_metric_result(
                    baseline,
                    tuned,
                    metric,
                    threshold,
                    lower_is_better_metrics,
                )
                improved = verdict == 'IMPROVED'
                regressed = verdict == 'REGRESSION'

                if metric in monitor_metrics and improved:
                    monitor_improved = True

                if regressed:
                    has_regression = True

                rows.append({
                    'config': config_key,
                    'metric': metric,
                    'baseline': baseline,
                    'tuned': tuned,
                    'delta_pct': delta_pct,
                    'monitor': metric in monitor_metrics,
                    'verdict': verdict,
                })

        should_create_pr = self._should_create_pr(monitor_improved,
                                                  has_regression, hard_blocker)

        def _fmt_float(value):
            if value is None:
                return 'N/A'
            return f"{value:.6g}"

        def _fmt_pct(value):
            if value is None:
                return 'N/A'
            sign = '+' if value >= 0 else ''
            return f"{sign}{value * 100:.3f}%"

        def _verdict_cell(verdict, monitor=False):
            styles = {
                'IMPROVED': 'color:#0B6E4F;font-weight:600;',
                'REGRESSION': 'color:#B42318;font-weight:700;',
                'NEUTRAL': 'color:#344054;',
                'MISSING': 'color:#667085;',
                'BASELINE_ZERO': 'color:#667085;',
                'NO_DATA': 'color:#667085;',
            }
            text = verdict
            if monitor and verdict in {'IMPROVED', 'REGRESSION'}:
                text = f"MONITOR_{verdict}"
            return f"<td style=\"{styles.get(verdict, '')}\">{html.escape(text)}</td>"

        chunks = [
            '<h2>Kernel Auto-Tuning Benchmark Evaluation</h2>',
            f'<p><b>Autotune ID:</b> {html.escape(self.autotune_id)}</p>',
            f'<p><b>Autotune postfix filter:</b> {html.escape(autotune_postfix)}</p>',
            f"<p><b>Decision:</b> {'CREATE_PR' if should_create_pr else 'DO_NOT_CREATE_PR'}</p>",
        ]

        if warnings:
            chunks.append('<h3>Warnings</h3>')
            chunks.append('<ul>')
            for warning in warnings:
                chunks.append(f'<li>{html.escape(warning)}</li>')
            chunks.append('</ul>')

        if not rows:
            chunks.append('<p>No comparable benchmark rows were found.</p>')
        else:
            rows_by_config = {}
            for row in rows:
                rows_by_config.setdefault(row['config'], []).append(row)

            chunks.append('<h3>Per-Config Metric Comparison</h3>')
            for config_key in sorted(rows_by_config):
                chunks.append('<details>')
                chunks.append('<summary><b>Config</b></summary>')
                chunks.append(f"<pre>{html.escape(config_key)}</pre>")
                chunks.append(
                    '<table border="1" cellspacing="0" cellpadding="4">'
                    '<thead><tr>'
                    '<th>Metric</th><th>Baseline</th><th>Tuned</th><th>Delta</th><th>Verdict</th>'
                    '</tr></thead><tbody>')
                status = status_by_config.get(config_key, {
                    'baseline_status': 'UNKNOWN',
                    'tuned_status': 'UNKNOWN',
                })
                chunks.append('<tr>')
                chunks.append('<td><b>RunStatus</b></td>')
                chunks.append(
                    f"<td>{html.escape(str(status['baseline_status']))}</td>")
                chunks.append(
                    f"<td>{html.escape(str(status['tuned_status']))}</td>")
                chunks.append('<td>N/A</td>')
                chunks.append('<td>N/A</td>')
                chunks.append('</tr>')
                for row in rows_by_config[config_key]:
                    metric_label = html.escape(row['metric'])
                    if row['monitor']:
                        metric_label = f'<b>{metric_label}</b>'
                    chunks.append('<tr>')
                    chunks.append(f'<td>{metric_label}</td>')
                    chunks.append(f"<td>{_fmt_float(row['baseline'])}</td>")
                    chunks.append(f"<td>{_fmt_float(row['tuned'])}</td>")
                    chunks.append(f"<td>{_fmt_pct(row['delta_pct'])}</td>")
                    chunks.append(
                        _verdict_cell(row['verdict'], monitor=row['monitor']))
                    chunks.append('</tr>')
                chunks.append('</tbody></table>')
                chunks.append('</details>')

        pr_body = '\n'.join(chunks)
        output_path = f"{REPORT_OUTPUT_PATH_PREFIX}_{self.autotune_id}.html"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(pr_body)

        if should_create_pr:
            try:
                pr_url = self._create_or_update_pr(pr_body)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception("Failed to create/update PR")
                return f"NO_PR_CREATED\nPR creation failed: {exc}\n{pr_body}"

            if pr_url:
                return f"PR_CREATED:{pr_url}\n{pr_body}"
            return f"NO_PR_CREATED\nPR creation skipped (missing GITHUB_CI_BOT_TOKEN).\n{pr_body}"

        return f"NO_PR_CREATED\nBenchmark gates not met.\n{pr_body}"

    def execute(self):
        if self.process_step == 'PATCH_KERNEL_AUTOTUNE_RESULT':
            self.patch_tuned_results()
        elif self.process_step == 'EVALUATE_AND_CREATE_PR':
            return self.evaluate_benchmark_metrics_and_create_pr_body()


if __name__ == "__main__":

    def _main(_):
        result = KernelAutoTuneResultProcessor().execute()
        if result is not None:
            print(result)

    app.run(_main)  # pylint: disable=no-value-for-parameter
