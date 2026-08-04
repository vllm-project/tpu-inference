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

import importlib
import json
import logging
import os
import subprocess
import sys

from absl import app, flags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
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

_WORKER_PROCESS = flags.DEFINE_bool(
    '_worker_process', False,
    'Internal flag used by the runner to invoke a worker subprocess for kernel tuning execution.'
)


def _load_kernel_tuner_class(kernel_tuner_name: str):
    module_name = (f'tools.kernel.tuner.v1.{kernel_tuner_name}'
                   if not kernel_tuner_name.endswith('_kernel_tuner') else
                   f'tools.kernel.tuner.v1.{kernel_tuner_name}')
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f'Kernel tuner module for {kernel_tuner_name} not found: {e}')
    class_name = ''.join(part.title() for part in kernel_tuner_name.split('_'))
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise ValueError(
            f'Kernel tuner class {class_name} not found in module {module_name}: {e}'
        )


def _format_flag_value(name: str, value):
    if isinstance(value, bool):
        return f'--{name}={str(value).lower()}'
    return f'--{name}={value}'


def _get_worker_log_path(run_config: RunConfig, begin_case_id: int,
                         end_case_id: int) -> str:
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(
        log_dir,
        f'kernel_tuner_worker_{run_config.run_id}_{begin_case_id}_{end_case_id}.log',
    )


def _stream_subprocess_output(stream, log_file_path: str) -> str:
    output_chunks = []
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        for line in stream:
            if line is None:
                continue
            output_chunks.append(line)
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
    return ''.join(output_chunks)


def _invoke_worker_process(kernel_tuner_name: str, run_config: RunConfig,
                           begin_case_id: int, end_case_id: int):
    worker_script = os.path.abspath(__file__)
    command = [
        sys.executable,
        worker_script,
        _format_flag_value('kernel_tuner_name', kernel_tuner_name),
        _format_flag_value('case_set_id', run_config.case_set_id),
        _format_flag_value('run_id', run_config.run_id),
        _format_flag_value('case_set_desc', run_config.case_set_desc),
        _format_flag_value('tpu_version', run_config.tpu_version),
        _format_flag_value('tpu_cores', run_config.tpu_cores),
        _format_flag_value('tpu_queue_multi', run_config.tpu_queue_multi),
        _format_flag_value('run_locally', run_config.run_locally),
        _format_flag_value('max_execution_minutes',
                           run_config.max_execution_minutes),
        _format_flag_value('job_priority', run_config.job_priority),
        _format_flag_value('begin_case_id', begin_case_id),
        _format_flag_value('end_case_id', end_case_id),
        _format_flag_value('gcp_project_id', run_config.gcp_project_id),
        _format_flag_value('spanner_instance_id',
                           run_config.spanner_instance_id),
        _format_flag_value('spanner_database_id',
                           run_config.spanner_database_id),
        _format_flag_value('worker_id', run_config.worker_id),
        _format_flag_value('autotune_mode', run_config.autotune_mode),
        _format_flag_value('debug', run_config.debug),
        _format_flag_value('_worker_process', True),
    ]

    log_file_path = _get_worker_log_path(run_config, begin_case_id,
                                         end_case_id)
    logger.info(
        'Starting kernel tuner worker subprocess for bucket [%d, %d). Logs will be streamed to %s',
        begin_case_id, end_case_id, log_file_path)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    if proc.stdout is None:
        raise RuntimeError('Worker subprocess stdout pipe is not available.')

    stdout_text = _stream_subprocess_output(proc.stdout, log_file_path)
    proc.wait()

    if proc.returncode != 0:
        logger.error(
            'Kernel tuner worker subprocess failed with return code %d',
            proc.returncode)
        raise RuntimeError(
            f'Worker subprocess failed with return code {proc.returncode}')

    try:
        text_output = None
        for line in stdout_text.splitlines():
            if 'next_begin_case_id' in line:
                text_output = line
                break
        logger.info('Worker subprocess output: %s', text_output)
        result = json.loads(text_output)
    except json.JSONDecodeError:
        logger.info(
            'Failed to find next_begin_case_id in worker subprocess output, set it to 0'
        )
        result = {'next_begin_case_id': 0}

    if not isinstance(result, dict):
        raise RuntimeError(
            f'Worker subprocess returned invalid result: {result}')

    return result.get('next_begin_case_id')


def _create_kernel_tuner(kernel_tuner_name: str, run_config: RunConfig):
    kernel_tuner_cls = _load_kernel_tuner_class(kernel_tuner_name)
    return kernel_tuner_cls(run_config=run_config)


def _worker_main():
    case_set_id = _CASE_SET_ID.value
    run_id = _RUN_ID.value
    case_set_desc = _CASE_SET_DESC.value
    assert case_set_id, 'case_set_id is required. Please specify it through --case_set_id flag.'
    assert run_id, 'run_id is required. Please specify it through --run_id flag.'

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
    kernel_tuner = _create_kernel_tuner(_KERNEL_TUNER_NAME.value, run_config)
    begin_case_id = _BEGIN_CASE_ID.value
    end_case_id = _END_CASE_ID.value
    next_begin_case_id = kernel_tuner.measure_latency(
        begin_case_id=begin_case_id, end_case_id=end_case_id)
    sys.stdout.write(
        json.dumps({
            'next_begin_case_id': next_begin_case_id,
        }) + '\n')
    sys.stdout.flush()
    sys.exit(0)


def main(argv):
    del argv  # Unused.

    if _WORKER_PROCESS.value:
        _worker_main()
        return

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

    if _RUN_LOCALLY.value:
        if _BEGIN_CASE_ID.value is None:
            kernel_tuner = _create_kernel_tuner(_KERNEL_TUNER_NAME.value,
                                                run_config)
            buckets = kernel_tuner._generate_tuning_jobs()
            del kernel_tuner  # Free up memory before running tuning jobs.
        else:
            buckets = [(_BEGIN_CASE_ID.value, _END_CASE_ID.value)]
        for bucket in buckets:
            begin_case_id, end_case_id = bucket
            current_begin_case_id = begin_case_id
            next_begin_case_id = begin_case_id  # Initialize to a value less than begin_case_id
            while next_begin_case_id < end_case_id:
                next_begin_case_id = _invoke_worker_process(
                    _KERNEL_TUNER_NAME.value, run_config, begin_case_id,
                    end_case_id)
                if next_begin_case_id < end_case_id:
                    logger.info(
                        'Local bucket [%d, %d) partially processed. Retrying from case %d.',
                        begin_case_id, end_case_id, next_begin_case_id)
                else:
                    logger.info(
                        'Local bucket [%d, %d) fully processed in worker subprocess.',
                        begin_case_id, end_case_id)
    else:
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
            next_begin_case_id = begin_case_id
            while next_begin_case_id < end_case_id:
                next_begin_case_id = _invoke_worker_process(
                    _KERNEL_TUNER_NAME.value, run_config, begin_case_id,
                    end_case_id)
                if next_begin_case_id < end_case_id:
                    logger.info(
                        'Bucket [%d, %d) not fully processed. Retrying from case %d.',
                        current_begin_case_id, end_case_id, next_begin_case_id)
                else:
                    logger.info(
                        'Bucket [%d, %d) was fully processed in worker subprocess.',
                        begin_case_id, end_case_id)


if __name__ == '__main__':
    app.run(main)
