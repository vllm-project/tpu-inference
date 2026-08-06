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
import time
from abc import ABC, abstractmethod
from typing import Any

import jax
import yaml

# isort: off
from tools.kernel.tuner.v1.common.tuner_datatypes import (
    CaseResult, RunConfig, TunableParams, TunerConfig, TuningCase, TuningKey,
    TuningStatus)
# isort: on

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteralString(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(LiteralString, _literal_representer)


class ProcessedCasesTracker:
    """Tracks evaluated case IDs and their execution statuses to manage state and OOM early-pruning."""

    def __init__(self, initial_processed_ids: set[int] = None):
        self.processed_ids: set[int] = set(initial_processed_ids or [])
        self.history: dict[Any, list[tuple[TunableParams, TuningStatus]]] = {}

    def __contains__(self, case_id: int) -> bool:
        return case_id in self.processed_ids

    def record(self, case_id: int, tuning_key: TuningKey,
               tunable_params: TunableParams, status: TuningStatus) -> None:
        """Records a case ID as processed and tracks its tuning status."""
        self.processed_ids.add(case_id)
        self.history.setdefault(tuning_key, []).append(
            (tunable_params, status))

    def is_oom_expected(self, tuning_key: TuningKey,
                        tunable_params: TunableParams) -> bool:
        """Returns True if a smaller configuration for the same tuning key previously failed with OOM."""
        for p, s in self.history.get(tuning_key, []):
            if s == TuningStatus.FAILED_OOM and p <= tunable_params:
                return True
        return False


class KernelTunerBase(ABC):
    """
    Base class for kernel tuner runner. The kernel tuner runner is responsible for generating the tuning cases, partitioning the cases into buckets, generating the Buildkite pipeline, and measuring the latency of the cases. The specific kernel tuner runner should inherit from this base class and implement the generate_cases, generate_inputs, and run methods.
    Subclass should also define the TuningKey and TunableParams dataclasses according to the kernel's tuning space.
    The tuning cases, tuning results, and other metadata will be persisted in local file or database using storage_management module, which is abstracted by the StorageManager class. The specific implementation of StorageManager can be LocalDbManager for local JSON-file-backed storage or SpannerDbManager for Google Spanner-backed storage.
    The kernel tuner runner will be executed in a distributed manner, where each worker will claim a bucket of cases to process, run the kernel with the corresponding tuning key and tunable params, measure the latency, and save the results back to the storage manager. The Buildkite pipeline will be generated to orchestrate the distributed execution of the kernel tuner runner.

    Subclass should implement the following methods:
    - generate_cases: Generate the tuning cases for the given case_set_id and desc passed through the config, and return a list of TuningCase objects representing the tuning cases.
    - generate_inputs: Generate the kernel inputs for the given tuning key with caching, and return a dictionary of kernel inputs.
    - run: Execute the kernel with the given tuning key and tunable params for a certain number of iterations, measure the latency, and return the tuning status, average latency, and total latency.

    Subclass must call super().__init__(tuner_config=tuner_config, run_config=run_config) in the __init__ method to initialize the base class.

    """

    def __init__(self,
                 *,
                 tuner_config: TunerConfig = None,
                 run_config: RunConfig = None):
        assert tuner_config is not None, "tuner_config must be specified"
        assert run_config is not None, "run_config must be specified"
        assert tuner_config.tuning_key_class is not None and issubclass(
            tuner_config.tuning_key_class, TuningKey
        ), (f"tuner_config.tuning_key_class ({tuner_config.tuning_key_class}) "
            "must satisfy the TuningKey protocol (hashable/frozen).")
        assert tuner_config.tunable_params_class is not None and issubclass(
            tuner_config.tunable_params_class, TunableParams
        ), (f"tuner_config.tunable_params_class ({tuner_config.tunable_params_class}) "
            "must satisfy the TunableParams protocol (__hash__, __le__, __ge__)."
            )
        assert tuner_config.kernel_tuner_name is not None, "kernel_tuner_name must be specified, which will be used as the identifier for this kernel tuner in the Buildkite pipeline generation and execution. It should match the key in the KERNEL_TUNER_REGISTRY in kernel_tuner_runner.py to ensure the correct kernel tuner is called during execution."
        # lazy import the storage manager to avoid import spanner when running locally
        if run_config.run_locally:
            from tools.kernel.tuner.v1.storage_management.local_db_manager import \
                LocalDbManager
            self.storage_manager = LocalDbManager(
                db_path=f'/tmp/kernel_tuner_runner_{run_config.case_set_desc}')
        else:
            from tools.kernel.tuner.v1.storage_management.spanner_database_manager import \
                SpannerStorageManager
            self.storage_manager = SpannerStorageManager(
                gcp_project_id=run_config.gcp_project_id,
                spanner_instance_id=run_config.spanner_instance_id,
                spanner_database_id=run_config.spanner_database_id)
        self._kernel_inputs_cache = {}
        self._tuning_key = None
        self.tuner_config = tuner_config
        self.run_config = run_config
        if run_config.n_bayesian_trials is not None:
            self.tuner_config.n_bayesian_trials = run_config.n_bayesian_trials
        self.worker_id = run_config.worker_id or 'unknown_worker'
        self.xprof_dir = os.path.join("/tmp/kernel_tuning",
                                      self.tuner_config.kernel_tuner_name,
                                      "xprof")
        self.use_bayesian_optimization = tuner_config.support_bayesian_optimization and run_config.use_bayesian_optimization
        if run_config.use_bayesian_optimization and not tuner_config.support_bayesian_optimization:
            logger.info(
                f'{tuner_config.kernel_tuner_name} does not support Bayesian Optimization, falls back to full sweep.'
            )
        # Control number of iterations for measuring kernel latency.
        self._measurement_iters = 5 if self.tuner_config.jit_kernel_pattern else 100

        if self.use_bayesian_optimization:
            from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
            self.optimizer = BayesianOptimizer(self)
        else:
            from tools.kernel.tuner.v1.optimizer import SweepOptimizer
            self.optimizer = SweepOptimizer(self)

    def _init_case_set(self) -> bool:
        """Initialize the case set which will be used for tuning. The case set will be written to the storage manager. This will be called when the caseset_id is new.

        Returns:
            True if tuning cases were initialized so in _generate_tuning_jobs we don't need to regenerate them, False otherwise.

        """
        # check case_set_id exists in storage manager, if not exist, create a new case set with the given case_set_id and desc.
        # if exist, check whether the desc is the same as the existing one, if not, raise an error.
        if self.storage_manager.case_set_id_exists(
                self.run_config.case_set_id):
            existing_desc = self.storage_manager.get_case_set_desc(
                self.run_config.case_set_id)
            if existing_desc != self.run_config.case_set_desc:
                raise ValueError(
                    f"CaseSetId {self.run_config.case_set_id} already exists with a different description. Existing desc: {existing_desc}, new desc: {self.run_config.case_set_desc}. If you intend to create new case set, please use a new case set id. Updating comment of an existing case set is not allowed. Please use a different CaseSetId or update the description to match the existing one."
                )
            else:
                logger.info(
                    f"CaseSetId {self.run_config.case_set_id} already exists with the same description. Proceeding with the existing case set."
                )
        else:
            self.storage_manager.init_case_set(
                self.run_config.case_set_id,
                scan_space=0,
                desc=self.run_config.case_set_desc)
            logger.info(
                f"CaseSet with ID: {self.run_config.case_set_id} and description: {self.run_config.case_set_desc} initialized."
            )
            return True
        return False

    def generate_autotune_cases(self) -> list[TuningCase]:
        tuning_set = []
        # The case_set_id is constructed as {kernel_tuner_name}_{autotune_case_set_id} in the bootstrap_kernel_tuners.py
        autotune_case_set_id = self.run_config.case_set_id.removeprefix(
            f'{self.tuner_config.kernel_tuner_name}_')
        autotune_cases = self.storage_manager.read_autotune_cases(
            case_set_id=autotune_case_set_id,
            kernel_tuner_name=self.tuner_config.kernel_tuner_name,
            tpu=self.run_config.tpu_version)
        bucket_by_key = []
        for row in autotune_cases:
            case_key_value = row['CaseKeyValue']
            tuning_case = TuningCase.from_string(
                case_key_value, self.tuner_config.tuning_key_class,
                self.tuner_config.tunable_params_class)

            start_case_id = len(tuning_set)
            tuning_set.append(tuning_case)
            tuning_key = tuning_case.tuning_key
            search_space = self.get_search_space(tuning_key)
            if not isinstance(search_space, dict):
                raise ValueError(
                    f"get_search_space should return a dictionary, but got {type(search_space)}"
                )

            def all_combinations(remain_keys, current_combination):
                if not remain_keys:
                    # tunable_params_list.append(TunableParams.from_dict(current_combination))
                    if not current_combination:
                        return
                    yield self.tuner_config.tunable_params_class(
                        **current_combination)
                    return
                key = remain_keys[0]
                for value in search_space[key]:
                    new_combination = current_combination.copy()
                    new_combination[key] = value
                    yield from all_combinations(remain_keys[1:],
                                                new_combination)

            for tunable_params in all_combinations(list(search_space.keys()),
                                                   {}):
                tuning_set.append(
                    TuningCase(tuning_key, tunable_params, is_baseline=False))
            end_case_id = len(tuning_set)
            bucket_by_key.append(
                (start_case_id,
                 end_case_id))  # [Include start_case_id, Exclude end_case_id)

        logger.info(
            f"Retrieved {len(tuning_set)} autotune cases for CaseSetId: {self.run_config.case_set_id} from Spanner."
        )
        return tuning_set, bucket_by_key

    @abstractmethod
    def generate_cases(self) -> list[TuningCase]:
        """Generate the cases for the given case_set_id.
        This should not raise any exception, all exceptions should be caught and handled internally. The generated cases will be persisted in local file or database using storage_management module, where each case is represented as a TuningCase object and stored as a string. The case_id is the index of the case in the generated case list.

        Returns: A list of TuningCase objects representing the tuning cases to be processed.
        """
        raise NotImplementedError(
            "Specific kernel tuner should implement this to generate the cases for the given case_set_id and desc, and return a list of TuningCase objects representing the tuning cases."
        )

    def get_search_space(self, tuning_key: TuningKey) -> dict:
        """Get the search space for the given kernel tuner with the specified tuning key. The search space is a dictionary where the keys are the tunable parameter names and the values are lists of possible values for each parameter.

        For example, for a kernel tuner that TunableParams has two tunable parameters 'tile_size' and 'unroll_factor', the search space could be represented as:
        {
            'tile_size': [16, 32, 64],
            'unroll_factor': [1, 2, 4]
        }

        Returns:
            A dictionary representing the search space for the kernel tuner.
        """
        return {}

    def _generate_tuning_jobs(self) -> list[tuple[int, int]]:
        """Partitions the full case set into fixed-size work buckets.

        Calls `generate_cases` to determine the total number of cases, then
        splits them into contiguous ranges of at most `self.run_config.job_bucket_size` cases each.
        Buckets are intended to be dispatched and executed in parallel; result
        ordering is not guaranteed. Each bucket is identified by a half-open
        interval [begin_case_id, end_case_id).

        Returns:
            A list of [begin_case_id, end_case_id] pairs covering all cases.
        """
        try:
            if self._init_case_set():
                start_time = time.perf_counter()
                if self.tuner_config.support_autotune and self.run_config.autotune_mode:
                    cases, _ = self.generate_autotune_cases()
                else:
                    cases = self.generate_cases()
                total_cases = len(cases)
                for case_id, case_str in enumerate(map(str, cases)):
                    self.storage_manager.add_tuner_case(
                        self.run_config.case_set_id,
                        case_id,
                        case_str,
                        tpu=self.run_config.tpu_queue_multi)
                self.storage_manager.flush()
                duration_sec = int(time.perf_counter() - start_time)
                self.storage_manager.finish_case_set(
                    self.run_config.case_set_id,
                    total_cases,
                    0,  # invalid case count, doesn't matter here
                    duration_sec * 1.0)
                logger.info(
                    f"Complete Generate Tuning Cases for {self.run_config.case_set_id}, Valid Cases: {total_cases} | Duration: {duration_sec}s"
                )
            # read back all the cases and partition them into buckets for parallel execution
            cases = self.storage_manager.get_all_cases(
                self.run_config.case_set_id)
            assert len(
                cases
            ) > 0, f"No cases found for CaseSetId {self.run_config.case_set_id}. This should not happen as the cases should have been generated and stored in the storage manager before."
            buckets = self.optimizer.generate_tuning_jobs(cases)
            logger.info(
                f'total cases: {len(cases)}, total buckets: {len(buckets)}')
            return buckets
        except Exception as e:
            logger.error(
                f"Error initializing case set {self.run_config.case_set_id}: {e}"
            )
            raise e

    def _build_step(self, case_id_start: int, case_id_end: int,
                    parent_step_key: str) -> dict:
        step_key = f'{self.tuner_config.kernel_tuner_name}_{self.run_config.case_set_id}_{self.run_config.run_id}_{case_id_start}_{case_id_end}'
        return {
            "label":
            f"cs_id={self.run_config.case_set_id} rid={self.run_config.run_id} Bucket([{case_id_start}, {case_id_end}))",
            "key":
            step_key,
            "depends_on":
            parent_step_key,
            "agents": {
                "queue": self.run_config.tpu_queue_multi
            },
            "env": {
                "USE_PREBUILT_IMAGE": "1",
                "TPU_VERSION": self.run_config.tpu_version
            },
            "commands": [
                LiteralString(
                    'rm -f /tmp/kernel_tuning/generated_pipeline.yml'),
                LiteralString(
                    '.buildkite/scripts/run_in_docker.sh bash -c \''
                    'pip install -r tools/kernel/tuner/v1/storage_management/requirements.txt && '
                    'python -m tools.kernel.tuner.v1.kernel_tuner_runner '
                    f'--kernel_tuner_name={self.tuner_config.kernel_tuner_name} '
                    f'  --case_set_id={self.run_config.case_set_id} --run_id={self.run_config.run_id} '
                    f'  --tpu_version={self.run_config.tpu_version} '
                    f'  --tpu_cores={self.run_config.tpu_cores} '
                    f'  --case_set_desc=\"{self.run_config.case_set_desc}\" '
                    f'  --use_bayesian_optimization={self.run_config.use_bayesian_optimization} '
                    f'  --n_bayesian_trials={self.tuner_config.n_bayesian_trials} '
                    f'  --run_locally=False '
                    f'  --tpu_queue_multi={self.run_config.tpu_queue_multi} '
                    f'  --max_execution_minutes={self.run_config.max_execution_minutes} '
                    f'  --job_priority={self.run_config.job_priority} '
                    f'  --begin_case_id={case_id_start} --end_case_id={case_id_end}\''
                ),
                LiteralString(
                    f'if [ -f /tmp/kernel_tuning/generated_pipeline.yml ]; then '
                    f'  buildkite-agent artifact upload /tmp/kernel_tuning/generated_pipeline.yml && '
                    f'  echo \"Upload generated pipeline YAML to Buildkite artifacts with priority {self.run_config.job_priority}\" && '
                    f'  {{ '
                    f'      echo \"priority: {self.run_config.job_priority}\"; '
                    f'      cat /tmp/kernel_tuning/generated_pipeline.yml; '
                    f'  }} | buildkite-agent pipeline upload; '
                    f'  else '
                    f'      echo \"File /tmp/kernel_tuning/generated_pipeline.yml does not exist. Exiting successfully.\"; '
                    f'fi')
            ]
        }

    def generate_buildkite_pipeline_subbucket(self, start: int, end: int,
                                              parent_step_key: str):
        """Generate the Buildkite pipeline for a sub-bucket of tuning jobs.

        Args:
            start: The starting case_id of the sub-bucket (inclusive).
            end: The ending case_id of the sub-bucket (exclusive).
            parent_step_key: The key of the parent step in the Buildkite pipeline.
        """
        assert parent_step_key is not None, "parent_step_key must be specified for the sub-bucket pipeline generation to set the correct dependency in the Buildkite pipeline."
        assert start < end, f"Invalid sub-bucket range: start {start} should be less than end {end}."
        output_path = "/tmp/kernel_tuning/generated_pipeline.yml"
        if os.path.exists(output_path):
            # clean up the existing one
            os.remove(output_path)
        step = self._build_step(start, end, parent_step_key=parent_step_key)
        pipeline = {"group": 'Kernel Sweeping Group', "steps": [step]}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
        logger.info(
            f"Generated Buildkite pipeline YAML for sub-bucket [{start}, {end}) saved to {output_path} in Docker"
        )

    def generate_buildkite_pipeline(self) -> str:
        """
        Generate the Buildkite pipeline for the given tuning jobs. Each tuning job 
        will be represented as a Buildkite step that calls the measure_latency function
        with the corresponding case_id range.
        """
        output_path = "/tmp/kernel_tuning/generated_pipeline.yml"
        if os.path.exists(output_path):
            # clean up the existing one
            os.remove(output_path)
        buckets = self._generate_tuning_jobs()
        # The Buildkite pipeline YAML will be generated in the format of:
        # steps:
        #   - label: "Measure latency for cases [begin_case_id, end_case_id)"
        #     command: "python -m tools.kernel.tuner.v1.kernel_tuner_runner\
        #               --worker_id=WORKER_ID\
        #               --case_set_id=CASE_SET_ID\
        #               --run_id=RUN_ID\
        #               --begin_case_id=BEGIN_CASE_ID\
        #               --end_case_id=END_CASE_ID"
        pipeline = {"steps": []}

        for enum_bucket_id, (case_id_start, case_id_end) in enumerate(buckets):
            step = self._build_step(case_id_start,
                                    case_id_end,
                                    parent_step_key=os.environ.get(
                                        'BUILDKITE_STEP_KEY', None))
            # In Bayesian mode each bucket covers exactly one TuningKey and its
            # begin case_id is a stable unique identifier, so we use it as the
            # bucket_id to keep generate_buildkite_pipeline and measure_latency
            # consistent.  In sweep mode we continue using the enumerate index.
            bucket_id = (case_id_start
                         if self.use_bayesian_optimization else enum_bucket_id)
            logger.info(
                f"Adding Buildkite step for bucket {bucket_id}: cases [{case_id_start}, {case_id_end})"
            )
            pipeline["steps"].append(step)
            # (TODO): Check (case_set_id, run_id) exists in the storage or not first
            self.storage_manager.create_bucket_for_run(
                self.run_config.case_set_id,
                self.run_config.run_id,
                bucket_id,
                case_id_start,
                case_id_end,
                tpu=self.run_config.tpu_queue_multi)

        if self.use_bayesian_optimization:
            group_name = f'Bayesian Optimization Group[{self.tuner_config.kernel_tuner_name}]'
        else:
            group_name = f'Sweeping Group[{self.tuner_config.kernel_tuner_name}]'
        pipeline['steps'] = [{
            'group': group_name,
            'key':
            f'{self.tuner_config.kernel_tuner_name}_{self.run_config.tpu_version}_tuning_group',
            'steps': pipeline['steps']
        }]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
        logger.info(
            f"Generated Buildkite pipeline YAML saved to {output_path} in Docker"
        )

    def _evaluate_single_case(
        self,
        cid: int,
        tuning_key: TuningKey,
        tunable_params: TunableParams,
        tracker: ProcessedCasesTracker,
        log_prefix: str = "",
    ) -> tuple[TuningStatus, int]:
        """Evaluates a single tuning case (warmup, measurement run, xprof verification).

        Saves result to storage manager and updates the ProcessedCasesTracker state.

        Returns:
            tuple (status, average_latency_us). On failure, average_latency_us is 0.
        """
        worker_id = self.worker_id
        self._cleanup_xprof_dir()
        begin_case_perf = time.perf_counter_ns()

        # --- Warmup (1 iteration) ---
        status, warmup_ns, _ = self.run(tuning_key, tunable_params, iters=1)
        if status != TuningStatus.SUCCESS:
            logger.warning(
                f"{log_prefix}Case {cid} failed during warmup with status {status} for params {tunable_params}."
            )
            self.storage_manager.save_result(
                CaseResult(
                    case_set_id=self.run_config.case_set_id,
                    run_id=self.run_config.run_id,
                    case_id=cid,
                    processed_status=status.value,
                    worker_id=worker_id,
                    latency=0,
                    warmup_time=0,
                    total_time=0,
                    processed_at=self.storage_manager.get_timestamp_sec(),
                    tpu=self.run_config.tpu_queue_multi,
                ))
            tracker.record(cid, tuning_key, tunable_params, status)
            return status, 0

        warmup_us = int(warmup_ns // 1000)

        # --- Measurement Run ---
        if self.tuner_config.jit_kernel_pattern is not None:
            with jax.profiler.trace(self.xprof_dir,
                                    create_perfetto_link=False):
                status, avg_latency_ns, _ = self.run(
                    tuning_key, tunable_params, iters=self._measurement_iters)
        else:
            status, avg_latency_ns, _ = self.run(tuning_key,
                                                 tunable_params,
                                                 iters=self._measurement_iters)

        if status != TuningStatus.SUCCESS:
            logger.warning(
                f"{log_prefix}Case {cid} failed during measurement run with status {status} for params {tunable_params}."
            )
            self.storage_manager.save_result(
                CaseResult(
                    case_set_id=self.run_config.case_set_id,
                    run_id=self.run_config.run_id,
                    case_id=cid,
                    processed_status=status.value,
                    worker_id=worker_id,
                    latency=0,
                    warmup_time=warmup_us,
                    total_time=0,
                    processed_at=self.storage_manager.get_timestamp_sec(),
                    tpu=self.run_config.tpu_queue_multi,
                ))
            tracker.record(cid, tuning_key, tunable_params, status)
            return status, 0

        # --- Extract Latency (xprof or timer) ---
        if self.tuner_config.jit_kernel_pattern is not None:
            from tools.kernel.tuner.v1.common.utils import \
                find_events_by_pattern
            matching_events, average_latency_us = find_events_by_pattern(
                self.xprof_dir, self.tuner_config.jit_kernel_pattern)
            if len(matching_events) != self._measurement_iters:
                msg = (
                    f"{log_prefix}Expected {self._measurement_iters} matching events for "
                    f"pattern '{self.tuner_config.jit_kernel_pattern}' in xprof, but found "
                    f"{len(matching_events)}. Profiling/pattern-matching failure at {self.xprof_dir}."
                )
                logger.error(msg)
                self.storage_manager.save_result(
                    CaseResult(
                        case_set_id=self.run_config.case_set_id,
                        run_id=self.run_config.run_id,
                        case_id=cid,
                        processed_status=TuningStatus.XPROF_MEASUREMENT_ERROR.
                        value,
                        worker_id=worker_id,
                        latency=0,
                        warmup_time=warmup_us,
                        total_time=0,
                        processed_at=self.storage_manager.get_timestamp_sec(),
                        tpu=self.run_config.tpu_queue_multi,
                    ))
                tracker.record(cid, tuning_key, tunable_params,
                               TuningStatus.XPROF_MEASUREMENT_ERROR)
                raise RuntimeError(msg)
        else:
            average_latency_us = int(avg_latency_ns // 1000)

        total_time_us = int((time.perf_counter_ns() - begin_case_perf) // 1000)

        self.storage_manager.save_result(
            CaseResult(
                case_set_id=self.run_config.case_set_id,
                run_id=self.run_config.run_id,
                case_id=cid,
                processed_status=TuningStatus.SUCCESS.value,
                worker_id=worker_id,
                latency=average_latency_us,
                warmup_time=warmup_us,
                total_time=total_time_us,
                processed_at=self.storage_manager.get_timestamp_sec(),
                tpu=self.run_config.tpu_queue_multi,
            ))
        tracker.record(cid, tuning_key, tunable_params, TuningStatus.SUCCESS)

        return TuningStatus.SUCCESS, average_latency_us

    @abstractmethod
    def generate_inputs(self, tuning_key: TuningKey) -> dict:
        """Generates the kernel inputs for the given tuning key with caching.

        Args:
            tuning_key: Identifies the kernel shape / problem size for which
                inputs should be prepared.
        Returns:
            The kernel inputs corresponding to the given tuning key as a dictionary.
        """
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        raise NotImplementedError(
            "Specific kernel should implement this to generate the inputs to kernel based on the tuning key with caching."
        )

    @abstractmethod
    def run(self, tuning_key: TuningKey, tunable_params: TunableParams,
            iters: int) -> list[TuningStatus, int, int]:
        """Executes the kernel and measures its latency.

        Fetches inputs via `generate_inputs`, runs the kernel with the supplied
        tunable parameters for `iters` iterations, and returns timing results.
        OOM exceptions must be caught internally and return FAILED_OOM.
        Other exceptions must be logged and re-raised. These non OOM exception will be logged
        and stop the program since we should not fail silently.

        A common implementation pattern is:
        ```
            try:
                inputs_cache = self.generate_inputs(tuning_key)
            except Exception as e:
                logger.error(f"Error generating inputs for tuning key {tuning_key}: {e}")
                return TuningStatus.UNKNOWN_ERROR, 0, 0
            kernel_param_0 = inputs_cache['kernel_param_0']
            kernel_param_1 = inputs_cache['kernel_param_1']
            ...
            try:
                # Run the kernel with the tunable parameters and measure latency 
                start_time_ns = time.perf_counter_ns()
                for _ in range(iters):
                    # Call the kernel with kernel_param_0, kernel_param_1, ... and tunable_params
                end_time_ns = time.perf_counter_ns()
                average_latency_ns = (end_time_ns - start_time_ns) // iters
                return TuningStatus.SUCCESS, average_latency_ns, end_time_ns - start_time_ns
            except Exception as err:
                if "RESOURCE_EXHAUSTED:" in str(err):
                    logger.warning(
                        f"Kernel run failed with OOM for {tuning_key=}, {tunable_params=}"
                    )
                    return TuningStatus.FAILED_OOM, float("inf"), float("inf")
                logger.warning(
                    f"Failed with {tuning_key=}, {tunable_params=}, got error: {err=}"
                )
                raise Exception(
                    f"Kernel run failed with tuning key & tunable params:\nTuningKey=\n{tuning_key}, TunableParams=\n{tunable_params}, got error: {err=}"
                )
        ```

        Args:
            tuning_key: Identifies the kernel shape / problem size.
            tunable_params: Tile sizes and other parameters to evaluate.
            iters: Number of iterations to run for latency measurement.

        Returns:
            A three-element list of (status, average_latency_ns, total_latency_ns):
                - status: TuningStatus.SUCCESS on success,
                  TuningStatus.FAILED_OOM on out-of-memory, or
                  TuningStatus.UNKNOWN_ERROR for any other failure.
                - average_latency_ns: Mean per-iteration latency in nanoseconds,
                  or 0 on failure.
                - total_latency_ns: Cumulative latency across all iterations in
                  nanoseconds, or 0 on failure.
        """
        raise NotImplementedError(
            "Specific kernel should implement this to call the kernl with the inputs from generate_inputs"
        )

    def _cleanup_xprof_dir(self):
        """Clean up the xprof directory to avoid interference from previous runs."""
        if not os.path.isdir(self.xprof_dir):
            return
        try:
            import shutil
            for name in os.listdir(self.xprof_dir):
                path = os.path.join(self.xprof_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
        except Exception as e:
            logger.warning(
                f"Failed to clean up xprof dir {self.xprof_dir}: {e}")

    def measure_latency(self, begin_case_id: int, end_case_id: int):
        """Measure the latency of cases in the caseset with case_id in [begin_case_id, end_case_id)."""
        self.optimizer.measure_latency(begin_case_id, end_case_id)
