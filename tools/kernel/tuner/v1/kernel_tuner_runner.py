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

from absl import app, flags

from tools.kernel.tuner.v1.batched_rpa_kernel_tuner import \
    BatchedRpaKernelTuner
from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.example_kernel_tuner import ExampleKernelTuner
from tools.kernel.tuner.v1.mla_kernel_tuner import MlaKernelTuner
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner
from tools.kernel.tuner.v1.utils import get_tpu_queue_by_version_and_cores

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEBUG = flags.DEFINE_bool(
    'debug', False, 'If true, prints results after each case iteration.')
_RUN_LOCALLY = flags.DEFINE_bool(
    'run_locally', False,
    'If true, uses local storage instead of cloud storage.')
_AUTOTUNE_MODE = flags.DEFINE_bool(
    'autotune_mode', False,
    'If true, runs the kernel tuner in autotune mode, which reads tuning cases from Spanner and generates Buildkite pipeline YAML for tuning jobs. '
)
_KERNEL_TUNER_NAME = flags.DEFINE_string('kernel_tuner_name',
                                         'example_kernel_tuner',
                                         'Name of the kernel tuner to run.')
_CASE_SET_ID = flags.DEFINE_string('case_set_id', '',
                                   'The case set ID to use for this run.')
_RUN_ID = flags.DEFINE_string(
    'run_id', '',
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

_TPU_CORES = flags.DEFINE_integer(
    'tpu_cores', 0,
    'The number of TPU cores to use for tuning. Default is 0. TPU tpu6e has 1 core per chip, TPU tpu7x has 2 cores per chip.'
)

_TPU_QUEUE_MULTI = flags.DEFINE_string(
    'tpu_queue_multi', '',
    'The TPU queue to use for tuning. This will be automatically determined based on the TPU version and cores if not specified. Supported values are "tpu_v6e_queue", "tpu_v6e_8_queue", "tpu_v7x_2_queue", "tpu_v7x_8_queue", and "tpu_v7x_16_queue".'
)

_JOB_PRIORITY = flags.DEFINE_integer(
    'job_priority', -10,
    'The priority to use for kernel tuning jobs. Higher priority jobs will be scheduled before lower priority ones. Default is -10, which is lower than typical user jobs to avoid impacting them.'
)

_MAX_EXECUTION_MINUTES = flags.DEFINE_integer(
    'max_execution_minutes', 20,
    'Only used when the kernel tuning job is scheduled through Buildkite. The maximum execution time in minutes for each kernel tuning job. If the job exceeds this time, it will save the job progresss, generate a new job to be scheduled by Buildkite and exit.'
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
    'mla_kernel_tuner': MlaKernelTuner,
    'batched_rpa_kernel_tuner': BatchedRpaKernelTuner,
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
    assert case_set_id, 'case_set_id is required. Please specify it through --case_set_id flag.'
    assert run_id, 'run_id is required. Please specify it through --run_id flag.'
    logger.info(
        f'Using case_set_id: {case_set_id}, run_id: {run_id}, case_set_desc: {case_set_desc} for this tuning run.'
    )

    tpu_version = _TPU_VERSION.value
    tpu_cores = _TPU_CORES.value
    tpu_queue_multi = _TPU_QUEUE_MULTI.value

    tpu_queue_multi = get_tpu_queue_by_version_and_cores(
        tpu_version, tpu_cores, tpu_queue_multi)

    run_config = RunConfig(case_set_id=case_set_id,
                           run_id=run_id,
                           case_set_desc=case_set_desc,
                           tpu_version=tpu_version,
                           tpu_cores=tpu_cores,
                           tpu_queue_multi=tpu_queue_multi,
                           run_locally=_RUN_LOCALLY.value,
                           job_priority=_JOB_PRIORITY.value,
                           max_execution_minutes=_MAX_EXECUTION_MINUTES.value,
                           gcp_project_id=_GCP_PROJECT_ID.value,
                           spanner_instance_id=_SPANNER_INSTANCE_ID.value,
                           spanner_database_id=_SPANNER_DATABASE_ID.value,
                           worker_id=_WORKER_ID.value,
                           autotune_mode=_AUTOTUNE_MODE.value,
                           debug=_DEBUG.value)
    kernel_tuner_cls = KERNEL_TUNER_REGISTRY.get(_KERNEL_TUNER_NAME.value)
    kernel_tuner = kernel_tuner_cls(run_config=run_config)

    if kernel_tuner.run_config.run_locally:
        logger.info(
            'Running in locally mode. Skipping Buildkite pipeline generation and running tuning jobs directly.'
        )
        buckets = kernel_tuner._generate_tuning_jobs()
        for bucket in buckets:
            begin_case_id, end_case_id = bucket
            kernel_tuner.measure_latency(begin_case_id, end_case_id)
    else:
        logger.info(
            'Running in cloud mode. Generating Buildkite pipeline YAML or running tuning jobs directly.'
        )
        if _GENERATE_BUILDKITE_PIPELINE.value:
            logger.info(
                'Generating Buildkite pipeline YAML. No tuning jobs will be run.'
            )

            kernel_tuner.generate_buildkite_pipeline()
        else:
            begin_case_id = _BEGIN_CASE_ID.value
            end_case_id = _END_CASE_ID.value
            logger.debug(
                'Running tuning jobs directly. Skipping Buildkite pipeline generation. Bucket [%d, %d)',
                begin_case_id, end_case_id)
            kernel_tuner.measure_latency(begin_case_id=begin_case_id,
                                         end_case_id=end_case_id)


if __name__ == '__main__':
    app.run(main)
