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
"""Shared factory functions for kernel tuner runner and worker.

This module owns the logic for loading kernel tuner classes and creating
RunConfig / kernel tuner instances.  Both ``kernel_tuner_runner`` and
``kernel_tuner_worker`` import from here, keeping them decoupled from
each other.
"""

import dataclasses
import importlib
import json
import logging
from datetime import datetime

from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.kernel_tuner_flags import (
    AUTOTUNE_MODE, CASE_SET_DESC, CASE_SET_ID, GCP_PROJECT_ID, JOB_PRIORITY,
    MAX_EXECUTION_MINUTES, MIN_CASES_FOR_BAYESIAN, N_BAYESIAN_TRIALS, RUN_ID,
    RUN_LOCALLY, SPANNER_DATABASE_ID, SPANNER_INSTANCE_ID, TPU_CORES,
    TPU_QUEUE_MULTI, TPU_VERSION, USE_BAYESIAN_OPTIMIZATION, WORKER_ID)
from tools.kernel.tuner.v1.utils import (get_tpu_queue_by_version_and_cores,
                                         get_worker_id)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_kernel_tuner_class(kernel_tuner_name: str):
    """Dynamically loads a kernel tuner class by its name.

    Args:
        kernel_tuner_name: The snake_case name of the kernel tuner module
            (e.g. 'batched_rpa_kernel_tuner').

    Returns:
        The kernel tuner class (a subclass of KernelTunerBase).

    Raises:
        ValueError: If the module or class cannot be found.
    """
    module_name = f'tools.kernel.tuner.v1.{kernel_tuner_name}'
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


def create_kernel_tuner(kernel_tuner_name: str,
                        run_config: RunConfig,
                        lightweight: bool = False):
    """Instantiates a kernel tuner by name with the given RunConfig.

    Args:
        kernel_tuner_name: The snake_case name of the kernel tuner module.
        run_config: The RunConfig for this tuning run.
        lightweight: If True, creates a lightweight instance that skips
            expensive initialization (JAX device setup, etc.).  Useful
            when only config / search-space access is needed.
    """
    kernel_tuner_cls = _load_kernel_tuner_class(kernel_tuner_name)
    return kernel_tuner_cls(run_config=run_config, lightweight=lightweight)


def create_run_config_from_flags() -> RunConfig:
    """Builds a RunConfig by reading values from absl flags.

    This is the primary way the *runner* creates a RunConfig when invoked
    from the command line.
    """
    case_set_id = CASE_SET_ID.value
    run_id = RUN_ID.value
    case_set_desc = CASE_SET_DESC.value
    assert case_set_id, 'case_set_id is required. Please specify it through --case_set_id flag.'
    assert run_id, 'run_id is required. Please specify it through --run_id flag.'
    logger.info(
        f'Using case_set_id: {case_set_id}, run_id: {run_id}, case_set_desc: {case_set_desc} for this tuning run.'
    )

    tpu_version = TPU_VERSION.value
    tpu_cores = TPU_CORES.value
    tpu_queue_multi = TPU_QUEUE_MULTI.value

    tpu_queue_multi = get_tpu_queue_by_version_and_cores(
        tpu_version, tpu_cores, tpu_queue_multi)

    run_config = RunConfig(
        case_set_id=case_set_id,
        run_id=run_id,
        case_set_desc=case_set_desc,
        tpu_version=tpu_version,
        tpu_cores=tpu_cores,
        tpu_queue_multi=tpu_queue_multi,
        run_locally=RUN_LOCALLY.value,
        job_priority=JOB_PRIORITY.value,
        max_execution_minutes=MAX_EXECUTION_MINUTES.value,
        gcp_project_id=GCP_PROJECT_ID.value,
        spanner_instance_id=SPANNER_INSTANCE_ID.value,
        spanner_database_id=SPANNER_DATABASE_ID.value,
        worker_id=WORKER_ID.value,
        autotune_mode=AUTOTUNE_MODE.value,
        use_bayesian_optimization=USE_BAYESIAN_OPTIMIZATION.value,
        n_bayesian_trials=N_BAYESIAN_TRIALS.value,
        min_cases_for_bayesian=MIN_CASES_FOR_BAYESIAN.value)
    return run_config


def create_storage_manager(run_config: RunConfig):
    """Creates the appropriate StorageManager based on RunConfig.

    Extracted from KernelTunerBase.__init__() so that storage can be
    created independently by the runner or worker process.
    """
    if run_config.run_locally:
        from tools.kernel.tuner.v1.storage_management.local_db_manager import \
            LocalDbManager
        if not run_config.local_db_path:
            date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            desc = run_config.case_set_desc or run_config.case_set_id or 'run'
            run_config.local_db_path = f'/tmp/kernel_tuner_runner_{desc}_{date_str}'
        return LocalDbManager(worker_id=get_worker_id(run_config.worker_id),
                              db_path=run_config.local_db_path)
    else:
        from tools.kernel.tuner.v1.storage_management.spanner_database_manager import \
            SpannerStorageManager
        return SpannerStorageManager(
            worker_id=get_worker_id(run_config.worker_id),
            gcp_project_id=run_config.gcp_project_id,
            spanner_instance_id=run_config.spanner_instance_id,
            spanner_database_id=run_config.spanner_database_id)


def run_config_to_json(run_config: RunConfig) -> str:
    """Serializes a RunConfig to a JSON string."""
    return json.dumps(dataclasses.asdict(run_config))


def run_config_from_json(json_str: str) -> RunConfig:
    """Deserializes a RunConfig from a JSON string."""
    return RunConfig(**json.loads(json_str))
