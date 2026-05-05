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

import datetime
import logging
import os

from absl import app, flags

from tools.kernel.tuner.v1.example_kernel_tuner import ExampleKernelTuner
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner
from tools.kernel.tuner.v1.storage_management.local_db_manager import \
    LocalDbManager
from tools.kernel.tuner.v1.storage_management.spanner_database_manager import \
    SpannerStorageManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEBUG = flags.DEFINE_bool(
    'debug', False, 'If true, prints results after each case iteration.')
_RUN_LOCALLY = flags.DEFINE_bool(
    'run_locally', False,
    'If true, uses local storage instead of cloud storage.')
_KERNEL_TUNER_NAME = flags.DEFINE_string('kernel_tuner_name',
                                         'example_kernel_tuner',
                                         'Name of the kernel tuner to run.')
_CASE_SET_ID = flags.DEFINE_string('case_set_id', '',
                                   'The case set ID to use for this run.')
_RUN_ID = flags.DEFINE_string(
    'run_id', '0',
    'The run ID to use for this run. If not specified, a timestamp-based ID will be generated.'
)
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '',
                                     'Description of the case set.')
_GENERATE_BUILDKITE_PIPELINE = flags.DEFINE_bool(
    'generate_buildkite_pipeline', False,
    'If true, generates Buildkite pipeline YAML instead of running tuning jobs.'
)
_BEGIN_CASE_ID = flags.DEFINE_integer(
    'begin_case_id', None,
    'The begin case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.'
)
_END_CASE_ID = flags.DEFINE_integer(
    'end_case_id', None,
    'The end case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.'
)
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
_WORKER_ID = flags.DEFINE_string('worker_id',
                                 os.getenv('HOST_NAME',
                                           'unknown'), 'The worker id')
_TPU_VERSION = flags.DEFINE_string(
    'tpu_version', '',
    'The TPU version to use for tuning. Supported values are "tpu6e" and "tpu7x".'
)

# Note: For simplicity, we are directly referencing the kernel tuner class
# here. In the future, we can consider a more flexible plugin-based system
# if we have more kernel tuners. For example, we can define an interface for
# kernel tuners and dynamically load kernel tuner classes based on the
# --kernel_tuner_name flag. This would allow us to add new kernel tuners
# without modifying this runner code. For now, after we implement more kernel
# tuners, we can simply add them to the KERNEL_TUNER_REGISTRY dictionary below.
KERNEL_TUNER_REGISTRY = {
    'example_kernel_tuner': ExampleKernelTuner,
    'rpa_v3_kernel_tuner': RpaV3KernelTuner,
}


def main(argv):
    del argv  # Unused.

    # env validation
    if _KERNEL_TUNER_NAME.value not in KERNEL_TUNER_REGISTRY:
        raise ValueError(
            f'Kernel tuner {_KERNEL_TUNER_NAME.value} is not registered. Available tuners: {list(KERNEL_TUNER_REGISTRY.keys())}'
        )

    case_set_id = _CASE_SET_ID.value
    run_id = _RUN_ID.value
    case_set_desc = _CASE_SET_DESC.value
    if not case_set_id:
        # If case_set_id is not provided, generate one using the current timestamp but in the format of YYYYMMDDHHMMSS to ensure it is sortable and easily readable.
        case_set_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        run_id = '0'
    logger.info(
        f'Using case_set_id: {case_set_id}, run_id: {run_id}, case_set_desc: {case_set_desc} for this tuning run.'
    )

    # Initialize storage manager
    if _RUN_LOCALLY.value:
        storage_manager = LocalDbManager()
    else:
        storage_manager = SpannerStorageManager()

    # Initialize kernel tuner
    kernel_tuner_cls = KERNEL_TUNER_REGISTRY.get(_KERNEL_TUNER_NAME.value)

    kernel_tuner = kernel_tuner_cls(storage_manager)

    if _RUN_LOCALLY.value:
        logger.info(
            'Running in locally mode. Skipping Buildkite pipeline generation and running tuning jobs directly.'
        )
        buckets = kernel_tuner._generate_tuning_jobs(case_set_id,
                                                     desc=case_set_desc)
        for bucket in buckets:
            begin_case_id, end_case_id = bucket
            kernel_tuner.measure_latency(case_set_id, run_id, begin_case_id,
                                         end_case_id)
    else:
        logger.info(
            'Running in cloud mode. Generating Buildkite pipeline YAML or running tuning jobs directly.'
        )
        if _GENERATE_BUILDKITE_PIPELINE.value:
            logger.info(
                'Generating Buildkite pipeline YAML. No tuning jobs will be run.'
            )
            tpu_version = _TPU_VERSION.value
            assert tpu_version in [
                'tpu6e', 'tpu7x'
            ], f'Unsupported TPU version: {tpu_version}. Supported versions are "tpu6e" and "tpu7x".'
            tpu_queue_multi = 'tpu_v6e_8_queue' if tpu_version == 'tpu6e' else 'tpu_v7x_8_queue'

            kernel_tuner.generate_buildkite_pipeline(
                case_set_id=case_set_id,
                run_id=run_id,
                desc=case_set_desc,
                tpu_version=tpu_version,
                tpu_queue_multi=tpu_queue_multi)
        else:
            begin_case_id = _BEGIN_CASE_ID.value
            end_case_id = _END_CASE_ID.value
            logger.debug(
                'Running tuning jobs directly. Skipping Buildkite pipeline generation. Bucket [%d, %d)',
                begin_case_id, end_case_id)
            kernel_tuner.measure_latency(case_set_id,
                                         run_id=run_id,
                                         begin_case_id=begin_case_id,
                                         end_case_id=end_case_id)


if __name__ == '__main__':
    app.run(main)
