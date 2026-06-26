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

import logging
import os

import yaml
from absl import app, flags

from tools.kernel.tuner.v1.storage_management.spanner_database_manager import \
    SpannerStorageManager

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
_AUTO_TUNE_ID = flags.DEFINE_string('auto_tune_id', '',
                                    'The auto tune ID to use for this run.')

OUTPUT_PATH = "/tmp/kernel_tuning/generated_pipeline.yml"


class BootstrapKernelTuners:

    def __init__(self, auto_tune_id: str):
        self.auto_tune_id = auto_tune_id
        self.storage_manager = SpannerStorageManager(
            gcp_project_id=_GCP_PROJECT_ID.value,
            spanner_instance_id=_SPANNER_INSTANCE_ID.value,
            spanner_database_id=_SPANNER_DATABASE_ID.value)

    def _build_trigger_kernel_tuner_pipeline(self,
                                          kernel_tuner_name: str,
                                          case_set_id: str,
                                          tpu_version: str,
                                          tpu_cores: int,
                                          case_set_desc: str,
                                          max_execution_minutes: int = 20,
                                          job_priority: int = -10) -> dict:

        return {
            "label": f"Tune Collected Cases For ({kernel_tuner_name}, {tpu_version}-{tpu_cores})",
            "key": f"tune_collected_cases_{kernel_tuner_name}_{tpu_version}_{tpu_cores}",
            "trigger": "tpu-inference-kernel-tuning",
            "build": {
                "branch": os.environ.get('BUILDKITE_BRANCH', 'patrickji.kernel_autotune_pipeline'),
                "env": {
                    "KERNEL_TUNING_AUTOTUNE_MODE": True,
                    "KERNEL_TUNING_KERNEL_TUNER_NAME": kernel_tuner_name,
                    "KERNEL_TUNING_CASE_SET_ID": case_set_id,
                    "KERNEL_TUNING_RUN_ID": '000',
                    "KERNEL_TUNING_TPU_VERSION": tpu_version,
                    "KERNEL_TUNING_TPU_CORES": str(tpu_cores),
                    "KERNEL_TUNING_CASE_SET_DESC": case_set_desc,
                    "KERNEL_TUNING_MAX_EXECUTION_MINUTES": str(max_execution_minutes),
                    "KERNEL_TUNING_JOB_PRIORITY": str(job_priority),
                }
            }
        }

    def generate_kernel_tuning_cases(self):
        pipeline = {"steps": []}
        autotune_cases = self.storage_manager.read_auto_tune_cases(
            case_set_id=self.auto_tune_id)
        tpus = set(case['TPU'] for case in autotune_cases)
        assert all(
            'tpu6e' in tpu or 'tpu7x' in tpu for tpu in tpus
        ), f'Unsupported TPU versions found in auto-tune cases: {tpus}. Supported versions are tpu6e and tpu7x.'
        tpus = set('tpu6e' if 'tpu6e' in tpu else 'tpu7x' for tpu in tpus)

        generated_cases = set()
        tuning_group_keys = []
        for row in autotune_cases:
            kernel_tuner_name = row['KernelTunerName']
            tpu = 'tpu6e' if 'tpu6e' in row['TPU'] else 'tpu7x'
            if (kernel_tuner_name, tpu) in generated_cases:
                continue
            tuning_group_keys.append(
                f'{kernel_tuner_name}_{tpu}_tuning_group')
            generated_cases.add((kernel_tuner_name, tpu))
            supported_core_num = 1 if tpu == 'tpu6e' else 2
            pipeline['steps'].append(
                self._build_trigger_kernel_tuner_pipeline(
                    kernel_tuner_name=kernel_tuner_name,
                    case_set_id=f'{kernel_tuner_name}_{self.auto_tune_id}',
                    tpu_version=tpu,
                    #(TODO): Only support kernel without communication for now and we don't have TPU 7x with 1 core queue
                    tpu_cores=supported_core_num,
                    case_set_desc=f'{kernel_tuner_name}_autotune'))
        print(
            yaml.dump(pipeline,
                    sort_keys=False,
                    default_flow_style=False,
                    width=float('inf')))
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            logger.info(
                f"Writing generated pipeline \n {yaml.dump(pipeline, default_flow_style=False, sort_keys=False)} \n to {OUTPUT_PATH}"
            )
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    app.run(lambda _: BootstrapKernelTuners(auto_tune_id=_AUTO_TUNE_ID.value).
            generate_kernel_tuning_cases())
