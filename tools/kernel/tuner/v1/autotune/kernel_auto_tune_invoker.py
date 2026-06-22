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

import os

import yaml
from absl import app, flags

from tools.kernel.tuner.v1.storage_management.spanner_database_manager import \
    SpannerStorageManager
import logging
logger = logging.getLogger(__name__)

# invoke the kernel tuning pipeline for each kernel
# when in auto tune mode, the kernel tuner will read the cases from spanner instead


class LiteralString(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(LiteralString, _literal_representer)

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
_CASE_SET_ID = flags.DEFINE_string('case_set_id', '', 'The case set ID to use for this run.')

OUTPUT_PATH = "/tmp/kernel_tuning/generated_pipeline.yml"


class KernelAutoTuneInvoker:

    def __init__(self, case_set_id: str):
        self.case_set_id = case_set_id
        # the case_set_id will be used to query
        self.storage_manager = SpannerStorageManager(
            gcp_project_id=_GCP_PROJECT_ID.value,
            spanner_instance_id=_SPANNER_INSTANCE_ID.value,
            spanner_database_id=_SPANNER_DATABASE_ID.value)

    def _build_generate_tuning_cases_step(self,
                                          kernel_tuner_name: str,
                                          case_set_id: str,
                                          tpu_version: str,
                                          tpu_cores: int,
                                          case_set_desc: str,
                                          max_execution_minutes: int = 20,
                                          job_priority: int = -10,
                                          parent_step_key: str = None) -> dict:
        step_key = f'{tpu_version}_{kernel_tuner_name}_autotune_generate_case_set'
        parent_step_key = parent_step_key
        return {
            "label":
            step_key,
            "key":
            step_key,
            "depends_on":
            parent_step_key,
            "agents": {
                "queue": 'cpu'
            },
            "env": {
                "USE_PREBUILT_IMAGE": "1",
                "TPU_VERSION": tpu_version
            },
            "commands": [
                LiteralString(f'rm -f {OUTPUT_PATH}'),
                LiteralString(
                    '.buildkite/scripts/run_in_docker.sh bash -c \''
                    'pip install --upgrade google-cloud-spanner google-api-core google-auth absl-py && '
                    'python -m tools.kernel.tuner.v1.kernel_tuner_runner '
                    f'--kernel_tuner_name={kernel_tuner_name} '
                    f'  --case_set_id={case_set_id} --run_id=0 '
                    f'  --tpu_version={tpu_version} '
                    f'  --tpu_cores={tpu_cores} '
                    f'  --case_set_desc=\"{case_set_desc}\" '
                    f'  --autotune_model=True '
                    f'  --max_execution_minutes={max_execution_minutes} '
                    f'  --job_priority={job_priority} '),
                LiteralString(
                    f'if [ -f {OUTPUT_PATH} ]; then '
                    f'  buildkite-agent artifact upload {OUTPUT_PATH} && '
                    f'  echo \"Upload generated pipeline YAML to Buildkite artifacts with priority {job_priority}\" && '
                    f'  {{ '
                    f'      echo \"priority: {job_priority}\"; '
                    f'      cat {OUTPUT_PATH}; '
                    f'  }} | buildkite-agent pipeline upload; '
                    f'  else '
                    f'      echo \"File {OUTPUT_PATH} does not exist. Exiting successfully.\"; '
                    f'fi')
            ]
        }

    def generate_kernel_tuning_cases(self):
        pipeline = {"steps": []}
        autotune_cases = self.storage_manager.read_auto_tune_cases(
            case_set_id=self.case_set_id)
        tpus = set(case['TPU'] for case in autotune_cases)
        assert all(
            'tpu6e' in tpu or 'tpu7x' in tpu for tpu in tpus
        ), f'Unsupported TPU versions found in auto-tune cases: {tpus}. Supported versions are tpu6e and tpu7x.'
        tpus = set('tpu6e' if 'tpu6e' in tpu else 'tpu7x' for tpu in tpus)

        generated_cases = set()
        for row in autotune_cases:
            kernel_tuner_name = row['KernelTunerName']
            tpu = 'tpu6e' if 'tpu6e' in row['TPU'] else 'tpu7x'
            if (kernel_tuner_name, tpu) in generated_cases:
                continue
            generated_cases.add((kernel_tuner_name, tpu))
            supported_core_num = 1 if tpu == 'tpu6e' else 2
            pipeline['steps'].append(
                self._build_generate_tuning_cases_step(
                    kernel_tuner_name=kernel_tuner_name,
                    case_set_id=f'{kernel_tuner_name}_{self.case_set_id}',
                    tpu_version=tpu,
                    #(TODO): Only support kernel without communication for now and we don't have TPU 7x with 1 core queue
                    tpu_cores=supported_core_num,
                    case_set_desc=f'{kernel_tuner_name}_autotune',
                    parent_step_key=os.environ.get('BUILDKITE_STEP_KEY', None)
                ))
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            logger.info(f"Writing generated pipeline \n {yaml.dump(pipeline, default_flow_style=False, sort_keys=False)} \n to {OUTPUT_PATH}")
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    app.run(lambda _: KernelAutoTuneInvoker(
        case_set_id=_CASE_SET_ID.value).generate_kernel_tuning_cases())
