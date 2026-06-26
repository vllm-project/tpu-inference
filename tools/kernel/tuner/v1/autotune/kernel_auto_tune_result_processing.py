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
import json
import logging
from pathlib import Path

from absl import app, flags

kernel_auto_tune_mapping = {
    'mla_kernel_tuner':
    '/workspace/tpu_inference/tpu_inference/kernels/mla/v2/tuned_params.py',
}

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
    'The auto tune ID to use for this run, for example, "KERNEL_AUTOTUNE_2026-06-23-07-10".'
)
_PROCESS_STEP = flags.DEFINE_string(
    'process_step', 'PATCH_KERNEL_AUTOTUNE_RESULT',
    'The process step to run. Options: COMPARE_BM_METRICS, PATCH_KERNEL_AUTOTUNE_RESULT'
)


class KernelAutoTuneResultProcessor:

    def __init__(self):
        self.auto_tune_id = _AUTOTUNE_ID.value
        self.gcp_project_id = _GCP_PROJECT_ID.value
        self.spanner_instance_id = _SPANNER_INSTANCE_ID.value
        self.spanner_database_id = _SPANNER_DATABASE_ID.value
        assert _PROCESS_STEP.value in [
            'COMPARE_BM_METRICS', 'PATCH_KERNEL_AUTOTUNE_RESULT'
        ], f"Invalid process step: {_PROCESS_STEP.value}. Must be one of ['COMPARE_BM_METRICS', 'PATCH_KERNEL_AUTOTUNE_RESULT']"
        self.process_step = _PROCESS_STEP.value

    def _get_spanner_db(self, project, instance_id, database_id):
        from google.cloud import \
            spanner as gspanner  # pylint: disable=import-outside-toplevel
        client = gspanner.Client(project=project, disable_builtin_metrics=True)
        return client.instance(instance_id).database(database_id)

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
        if kernel_tuner_name not in kernel_auto_tune_mapping:
            raise ValueError(
                f"Kernel tuner name '{kernel_tuner_name}' not found in kernel_auto_tune_mapping."
            )
        file_path = kernel_auto_tune_mapping[kernel_tuner_name]
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

        # Before updating the file, log how many existing entries will be updated and how many new entries will be added.
        existing_keys = set(module.tuned_params_mapping.keys())
        new_keys = set(tuning_key_to_params.keys())
        may_updated_keys = existing_keys.intersection(new_keys)
        # check whether the may_updated_keys have different values in the new mapping
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
        file_path.write_text(''.join(lines), encoding='utf-8')

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

    def patch_tuned_results(self):
        for kernel_tuner_name in kernel_auto_tune_mapping.keys():
            case_set_id = f"{kernel_tuner_name}_{self.auto_tune_id}"
            logger.info(f"Processing results for case set ID: {case_set_id}")
            best_results = self.get_best_results(case_set_id)
            self.update_best_results(best_results, kernel_tuner_name)

    def compare_benchmark_metrics(self):
        logger.info("Comparing benchmark metrics")
        query = """
            SELECT OutputTokenThroughput  
            FROM RunRecord rr where rr.status = 'COMPLETED' and rr.run_type like 'KERNEL_AUTOTUNE'
        """
        db = self._get_spanner_db(self.gcp_project_id,
                                  self.spanner_instance_id,
                                  self.spanner_database_id)
        with db.snapshot() as snap:
            for r in snap.execute_sql(query):
                print(r)
                return
        return "Comparison of benchmark metrics completed."

    def execute(self):
        if self.process_step == 'PATCH_KERNEL_AUTOTUNE_RESULT':
            self.patch_tuned_results()
        elif self.process_step == 'COMPARE_BM_METRICS':
            return self.compare_benchmark_metrics()


if __name__ == "__main__":
    app.run(lambda _: KernelAutoTuneResultProcessor().execute())  # pylint: disable=no-value-for-parameter
