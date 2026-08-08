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

import gc
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time

from absl import app
from absl import logging as absl_logging

from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.kernel_tuner_factory import (
    create_kernel_tuner, create_run_config_from_flags, create_storage_manager,
    run_config_to_json)
from tools.kernel.tuner.v1.kernel_tuner_flags import (
    BEGIN_CASE_ID, END_CASE_ID, GENERATE_BUILDKITE_PIPELINE, KERNEL_TUNER_NAME,
    RUN_LOCALLY)
from tools.kernel.tuner.v1.storage_management.storage_manager import \
    StorageManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Maximum number of consecutive retries when a worker subprocess makes no
# forward progress (returns the same or earlier next_begin_case_id).
_MAX_WORKER_RETRIES = 3

# Maximum wall-clock time (in seconds) a single worker subprocess is allowed
# to run before being forcibly killed.  Default: 2 hours.
_WORKER_TIMEOUT_SECONDS = 2 * 60 * 60


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
    """Spawns a worker subprocess and returns its next_begin_case_id.

    RunConfig is passed via a JSON temp file instead of mirroring every
    field as CLI flags — this keeps runner and worker in sync automatically
    when new fields are added to RunConfig.

    The worker writes its result to a JSON file (``result_path``) rather
    than embedding it in stdout, so log output can never interfere with
    result parsing.
    """
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'kernel_tuner_worker.py')

    # --- Serialize RunConfig to a temp file ---
    run_config_fd, run_config_path = tempfile.mkstemp(prefix='run_config_',
                                                      suffix='.json')
    try:
        with os.fdopen(run_config_fd, 'w') as f:
            f.write(run_config_to_json(run_config))
    except Exception:
        os.close(run_config_fd)
        raise

    # --- Result file the worker will write to ---
    result_fd, result_path = tempfile.mkstemp(prefix='worker_result_',
                                              suffix='.json')
    os.close(result_fd)  # Worker will open it for writing.

    command = [
        sys.executable,
        '-m',
        'tools.kernel.tuner.v1.kernel_tuner_worker',
        f'--kernel_tuner_name={kernel_tuner_name}',
        f'--begin_case_id={begin_case_id}',
        f'--end_case_id={end_case_id}',
        f'--run_config_path={run_config_path}',
        f'--result_path={result_path}',
    ]

    log_file_path = _get_worker_log_path(run_config, begin_case_id,
                                         end_case_id)
    logger.info(
        f'Starting kernel tuner worker subprocess for bucket [{begin_case_id}, {end_case_id})'
    )
    logger.info(f'Worker Logs: {log_file_path}')
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    cwd = os.getcwd()
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{cwd}:{pythonpath}' if pythonpath else cwd
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,  # Own process group for clean killpg.
    )

    if proc.stdout is None:
        raise RuntimeError('Worker subprocess stdout pipe is not available.')

    # Stream stdout in a daemon thread so the main thread can enforce a
    # wall-clock timeout via proc.wait(timeout=...).
    stream_thread = threading.Thread(
        target=_stream_subprocess_output,
        args=(proc.stdout, log_file_path),
        daemon=True,
    )
    stream_thread.start()

    def _kill_worker_process_group():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    def _cleanup_artifacts():
        for path in (run_config_path, result_path):
            try:
                os.remove(path)
            except OSError:
                pass

    try:
        proc.wait(timeout=_WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_worker_process_group()
        stream_thread.join(timeout=5)
        _cleanup_artifacts()
        raise RuntimeError(f'Worker subprocess timed out after '
                           f'{_WORKER_TIMEOUT_SECONDS} seconds for bucket '
                           f'[{begin_case_id}, {end_case_id}).')
    except BaseException as e:
        _kill_worker_process_group()
        stream_thread.join(timeout=5)
        _cleanup_artifacts()
        raise Exception(
            f'Worker subprocess for bucket [{begin_case_id}, {end_case_id}) interrupted or encountered error ({e}). '
        )

    stream_thread.join(timeout=5)

    # --- Clean up the RunConfig temp file ---
    try:
        os.remove(run_config_path)
    except OSError:
        pass

    if proc.returncode != 0:
        logger.error(
            'Kernel tuner worker subprocess failed with return code %d',
            proc.returncode)
        _cleanup_artifacts()
        raise RuntimeError(
            f'Worker subprocess failed with return code {proc.returncode}')

    # --- Read result from file ---
    try:
        with open(result_path, 'r') as f:
            result = json.load(f)
        logger.debug('Worker subprocess result: %s', result)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        raise RuntimeError(
            f'Worker subprocess returned invalid or missing result file '
            f'{result_path}: {e}')
    finally:
        _cleanup_artifacts()

    return result.get('next_begin_case_id')


def _run_bucket(run_config: RunConfig, begin_case_id: int, end_case_id: int):
    """Runs a bucket of tuning cases, retrying on partial completion.

    If the worker subprocess makes no forward progress (returns the same
    or an earlier ``next_begin_case_id``), it is retried up to
    ``_MAX_WORKER_RETRIES`` times before raising.
    """
    next_begin_case_id = begin_case_id
    retries_without_progress = 0
    while next_begin_case_id < end_case_id:
        prev_begin_case_id = next_begin_case_id
        try:
            next_begin_case_id = _invoke_worker_process(
                KERNEL_TUNER_NAME.value, run_config, next_begin_case_id,
                end_case_id)
        except Exception:
            logger.critical('Worker process raise unhandled exception')
            next_begin_case_id = prev_begin_case_id
        if next_begin_case_id <= prev_begin_case_id:
            retries_without_progress += 1
            logger.warning(
                'Worker made no forward progress for bucket [%d, %d). '
                'Retry %d / %d.', begin_case_id, end_case_id,
                retries_without_progress, _MAX_WORKER_RETRIES)
            if retries_without_progress >= _MAX_WORKER_RETRIES:
                logger.critical(
                    f'Worker stuck at case {next_begin_case_id} for bucket '
                    f'[{begin_case_id}, {end_case_id}). '
                    f'{_MAX_WORKER_RETRIES} retries exhausted. '
                    f'Skipping to {prev_begin_case_id + 1}.')
                next_begin_case_id = prev_begin_case_id + 1
                retries_without_progress = 0
            else:
                # Reset to retry from the same position.
                next_begin_case_id = prev_begin_case_id
        else:
            retries_without_progress = 0
            # TODO: Need to differentiate the case where time_budget has been reached vs a worker crashing.
            # If the worker crashed, we should retry the bucket from the beginning.
            # If the time_budget has been reached, we should yield to other BuildKite job.
            if next_begin_case_id < end_case_id:
                logger.info(
                    f'Bucket [{begin_case_id}, {end_case_id}) was partially processed. '
                    f'Continuing from case {next_begin_case_id}.')
            else:
                logger.info(
                    f'Bucket [{begin_case_id}, {end_case_id}) was fully processed.'
                )


def generate_and_partition_cases(kernel_tuner, storage_manager,
                                 optimizer) -> list[tuple[int, int]]:
    """Generates cases, persists them to storage, and partitions into buckets.

    This logic was previously in KernelTunerBase._generate_tuning_jobs().
    Moved here because it is storage orchestration, not kernel definition.

    Args:
        kernel_tuner: KernelTunerBase instance (for generate_cases, config).
        storage_manager: Storage manager for persisting cases.
        optimizer: TuningOptimizer for partitioning cases into buckets.

    Returns:
        A list of (begin_case_id, end_case_id) tuples.
    """
    from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase
    run_config = kernel_tuner.run_config

    try:
        if KernelTunerBase.init_case_set(storage_manager, run_config):
            start_time = time.perf_counter()
            if (kernel_tuner.tuner_config.support_autotune
                    and run_config.autotune_mode):
                cases, _ = kernel_tuner.generate_autotune_cases(
                    storage_manager)
            else:
                cases = kernel_tuner.generate_cases()
            total_cases = len(cases)
            for case_id, case_str in enumerate(map(str, cases)):
                storage_manager.add_tuner_case(run_config.case_set_id,
                                               case_id,
                                               case_str,
                                               tpu=run_config.tpu_queue_multi)
            storage_manager.flush()
            duration_sec = int(time.perf_counter() - start_time)
            storage_manager.finish_case_set(run_config.case_set_id,
                                            total_cases, 0, duration_sec * 1.0)
            logger.info(
                f"Complete Generate Tuning Cases for {run_config.case_set_id}, "
                f"Valid Cases: {total_cases} | Duration: {duration_sec}s")

        # Read back all the cases and partition them into buckets.
        cases = storage_manager.get_all_cases(run_config.case_set_id)
        assert len(cases) > 0, (
            f"No cases found for CaseSetId {run_config.case_set_id}. "
            "This should not happen as the cases should have been generated "
            "and stored in the storage manager before.")
        # Optimizer decides the bucket assignment for the cases. Different optimizer
        # has different bucket assignment strategies. e.g. sweep optimizer
        # splits the cases into equal size buckets, while Bayesian optimizer
        # splits the cases based on tuning key since for a optimzer's object is
        # specific to a tuning key.
        buckets = optimizer.generate_tuning_jobs(cases)
        logger.info(
            f'Total cases: {len(cases)}, total buckets: {len(buckets)}')
        return buckets
    except Exception as e:
        logger.error(
            f"Error initializing case set {run_config.case_set_id}: {e}")
        raise e


def _handle_buildkite_pipeline_generation(run_config: RunConfig):
    logger.info(
        'Generating Buildkite pipeline YAML. No tuning jobs will be run.')
    assert BEGIN_CASE_ID.value is None and END_CASE_ID.value is None, \
        'When GENERATE_BUILDKITE_PIPELINE is true, BEGIN_CASE_ID and END_CASE_ID should never be set.'
    storage_manager = create_storage_manager(run_config)
    buckets = _get_tuning_buckets(run_config, storage_manager)
    kernel_tuner.generate_buildkite_pipeline(buckets, storage_manager)


def _get_tuning_buckets(
        run_config: RunConfig,
        storage_manager: StorageManager | None = None
) -> list[tuple[int, int]]:
    # When run through BuildKite in TPU VMs Pool, each job sent to BuildKite in YML
    # is represented as a bucket [begin_case_id, end_case_id). The case set is already
    # created and the cases are already persisted to the storage manager when the YML
    # is generated. When GENERATE_BUILDKITE_PIPELINE is true, BEGIN_CASE_ID and
    # END_CASE_ID should never be set.
    if BEGIN_CASE_ID.value is not None:
        return [(BEGIN_CASE_ID.value, END_CASE_ID.value)]

    # This only happens when run on local TPU VM. This will create the full
    # case set and then partition them into buckets.
    kernel_tuner = create_kernel_tuner(KERNEL_TUNER_NAME.value, run_config)
    if storage_manager is None:
        storage_manager = create_storage_manager(run_config)

    # Create optimizer for partitioning only (no executor_mgr needed)
    if kernel_tuner.use_bayesian_optimization:
        from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
        optimizer = BayesianOptimizer(kernel_tuner)
    else:
        from tools.kernel.tuner.v1.optimizer import SweepOptimizer
        optimizer = SweepOptimizer(kernel_tuner)

    buckets = generate_and_partition_cases(kernel_tuner, storage_manager,
                                           optimizer)
    del kernel_tuner
    gc.collect()  # Force CPython to finalize and release JAX/TPU buffers.
    return buckets


def _run_tuning_jobs(run_config: RunConfig):
    if not RUN_LOCALLY.value:
        assert BEGIN_CASE_ID.value is not None and END_CASE_ID.value is not None, 'BEGIN_CASE_ID and END_CASE_ID must be specified when RUN_LOCALLY is False'
        logger.debug(
            f'Running tuning jobs directly. Skipping Buildkite pipeline generation. Bucket [{BEGIN_CASE_ID.value}, {END_CASE_ID.value}).'
        )

    buckets = _get_tuning_buckets(run_config)
    for begin_case_id, end_case_id in buckets:
        _run_bucket(run_config, begin_case_id, end_case_id)


def main(argv):
    del argv  # Unused.
    absl_logging.get_absl_handler().setFormatter(
        logging.Formatter(
            '%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%m%d %H:%M:%S',
        ))

    run_config = create_run_config_from_flags()

    if GENERATE_BUILDKITE_PIPELINE.value:
        _handle_buildkite_pipeline_generation(run_config)
        return

    _run_tuning_jobs(run_config)


if __name__ == '__main__':
    app.run(main)
