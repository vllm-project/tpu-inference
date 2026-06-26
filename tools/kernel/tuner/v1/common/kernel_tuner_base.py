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

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum

import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteralString(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(LiteralString, _literal_representer)


@dataclass
class TuningKey:
    # Specify the key for tuning case
    pass


@dataclass
class TunableParams:
    # Specify the tiles for tuning case
    pass


class TuningStatus(Enum):
    SUCCESS = 'SUCCESS'
    FAILED_OOM = 'FAILED_OOM'
    UNKNOWN_ERROR = 'UNKNOWN_ERROR'
    SKIPPED = 'SKIPPED'


class TuningCase:

    def __init__(self,
                 tuning_key: TuningKey,
                 tunable_params: TunableParams,
                 is_baseline: bool = False):
        self.tuning_key = tuning_key
        self.tunable_params = tunable_params
        self.is_baseline = is_baseline  # can be used to mark whether this case is the baseline case for the tuning key, which can be used for comparison in the analysis.

    def __str__(self):
        return json.dumps({
            'tuning_key': asdict(self.tuning_key),
            'tunable_params': asdict(self.tunable_params),
            'is_baseline': self.is_baseline
        })

    @classmethod
    def from_string(cls, string, tuning_key_class, tunable_params_class):
        data = json.loads(string)
        tuning_key = tuning_key_class(**data['tuning_key'])
        tunable_params = tunable_params_class(**data['tunable_params'])
        case = TuningCase(tuning_key, tunable_params)
        case.is_baseline = data.get('is_baseline', False)
        return case


@dataclass
class TunerConfig:
    tuning_key_class: any = None
    tunable_params_class: any = None
    kernel_tuner_name: str = None
    # When support autotune and run_config.autotune_mode is True,
    # the kernel tuner will read the cases from spanner using the case_set_id and kernel_tuner_name
    support_autotune: bool = False
    support_bayesian_optimization: bool = False


@dataclass
class RunConfig:
    case_set_id: str = None
    run_id: str = None
    case_set_desc: str = None
    tpu_version: str = None
    tpu_cores: int = None
    tpu_queue_multi: str = None
    run_locally: bool = False
    job_priority: int = -10
    max_execution_minutes: int = 20
    job_bucket_size: int = 100
    gcp_project_id: str = None
    spanner_instance_id: str = None
    spanner_database_id: str = None
    worker_id: str = None
    autotune_mode: bool = False
    debug: bool = False


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
        assert tuner_config.tuning_key_class is not None, "tuning_key_class must be specified"
        assert tuner_config.tunable_params_class is not None, "tunable_params_class must be specified"
        assert tuner_config.kernel_tuner_name is not None, "kernel_tuner_name must be specified, which will be used as the identifier for this kernel tuner in the Buildkite pipeline generation and execution. It should match the key in the KERNEL_TUNER_REGISTRY in kernel_tuner_runner.py to ensure the correct kernel tuner is called during execution."
        # lazy import the storage manager to avoid import spanner when running locally
        if run_config.run_locally:
            from tools.kernel.tuner.v1.storage_management.local_db_manager import \
                LocalDbManager
            self.storage_manager = LocalDbManager()
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
        self.worker_id = run_config.worker_id or 'unknown_worker'

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
        # The case_set_id is constructed as {kernel_tuner_name}_{auto_tune_case_set_id} in the bootstrap_kernel_tuners.py
        auto_tune_case_set_id = self.run_config.case_set_id.lstrip(
            f'{self.tuner_config.kernel_tuner_name}_')
        autotune_cases = self.storage_manager.read_auto_tune_cases(
            case_set_id=auto_tune_case_set_id,
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
                    yield TunableParams(**current_combination)
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

    @abstractmethod
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
        raise NotImplementedError(
            "Specific kernel tuner must implement this to return the search space for the kernel tuner if it supports bayesian optimization."
        )

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
                    f"\nComplete Generate Tuning Cases for {self.run_config.case_set_id}, Valid Cases: {total_cases} | Duration: {duration_sec}s"
                )
            # read back all the cases and partition them into buckets for parallel execution
            cases = self.storage_manager.get_all_cases(
                self.run_config.case_set_id)
            assert len(
                cases
            ) > 0, f"No cases found for CaseSetId {self.run_config.case_set_id}. This should not happen as the cases should have been generated and stored in the storage manager before."
            if self.tuner_config.support_bayesian_optimization:
                # For Bayesian optimization, partition the cases by key.
                buckets = []
                previous_tuning_key = None
                for case_id, case in enumerate(cases):
                    if previous_tuning_key is None or case.tuning_key != previous_tuning_key:
                        buckets.append((case_id, case_id + 1))
                        previous_tuning_key = case.tuning_key
                    else:
                        buckets[-1] = (buckets[-1][0], case_id + 1)
            else:
                total_cases = len(cases)
                buckets = [(i,
                            min(i + self.run_config.job_bucket_size,
                                total_cases))
                           for i in range(0, total_cases,
                                          self.run_config.job_bucket_size)]
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
                    'pip install --upgrade google-cloud-spanner && '
                    'pip install --upgrade google-api-core && '
                    'pip install --upgrade google-auth && '
                    'pip install --upgrade absl-py && '
                    'python -m tools.kernel.tuner.v1.kernel_tuner_runner '
                    f'--kernel_tuner_name={self.tuner_config.kernel_tuner_name} '
                    f'  --case_set_id={self.run_config.case_set_id} --run_id={self.run_config.run_id} '
                    f'  --tpu_version={self.run_config.tpu_version} '
                    f'  --tpu_cores={self.run_config.tpu_cores} '
                    f'  --case_set_desc=\"{self.run_config.case_set_desc}\" '
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
        """Generate the Buildkite pipeline for the given tuning jobs. Each tuning job will be represented as a Buildkite step that calls the measure_latency function with the corresponding case_id range.
        """
        output_path = "/tmp/kernel_tuning/generated_pipeline.yml"
        if os.path.exists(output_path):
            # clean up the existing one
            os.remove(output_path)
        buckets = self._generate_tuning_jobs()
        # The Buildkite pipeline YAML will be generated in the format of:
        # steps:
        #   - label: "Measure latency for cases [begin_case_id, end_case_id)"
        #     command: "python -m tools.kernel.tuner.v1.kernel_tuner_runner --worker_id=WORKER_ID --case_set_id=CASE_SET_ID --run_id=RUN_ID --begin_case_id=BEGIN_CASE_ID --end_case_id=END_CASE_ID"
        pipeline = {"steps": []}

        for bucket_id, (case_id_start, case_id_end) in enumerate(buckets):
            step = self._build_step(case_id_start,
                                    case_id_end,
                                    parent_step_key=os.environ.get(
                                        'BUILDKITE_STEP_KEY', None))
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

        if self.tuner_config.support_bayesian_optimization:
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
        All exceptions must be caught internally; nothing should propagate to
        the caller.

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
            except OOMError as e:
                logger.warning(f"OOM error when running kernel for tuning key {tuning_key} with tunable params {tunable_params}: {e}")
                return TuningStatus.FAILED_OOM, 0, 0
            except Exception as e:
                logger.error(f"Unknown error when running kernel for tuning key {tuning_key} with tunable params {tunable_params}: {e}")
                return TuningStatus.UNKNOWN_ERROR, 0, 0
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

    def measure_latency(self, begin_case_id: int, end_case_id: int):
        """Measure the latency of cases in the caseset with case_id in [begin_case_id, end_case_id). The latency of each case will be persisted in local file or database using storage_management module.

        Args:
            begin_case_id: Start of the case_id range (inclusive) within the caseset to measure.
            end_case_id: End of the case_id range (exclusive) within the caseset to measure.
        """
        worker_id = self.worker_id
        bucket_id = begin_case_id // self.run_config.job_bucket_size
        logger.info(
            f"Worker [{worker_id}] Claimed CaseSetId: {self.run_config.case_set_id}, RunId: {self.run_config.run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing."
        )
        self.storage_manager.mark_bucket_in_progress(
            self.run_config.case_set_id, self.run_config.run_id, bucket_id)

        processed_ids = self.storage_manager.get_already_processed_ids(
            self.run_config.case_set_id, self.run_config.run_id, begin_case_id,
            end_case_id)
        all_configs = self.storage_manager.get_bucket_configs(
            self.run_config.case_set_id, begin_case_id, end_case_id)

        bucket_start_perf = time.perf_counter()
        results_buffer = []
        bucket_fully_processed = True
        last_processed_case_id = begin_case_id - 1
        for cid in range(begin_case_id, end_case_id):
            time_elapsed_minutes = (time.perf_counter() -
                                    bucket_start_perf) / 60
            logger.info(
                f"Worker [{worker_id}] Processing CaseId: {cid} in Bucket {bucket_id}, [{begin_case_id}-{end_case_id}) with elapsed time {time_elapsed_minutes:.2f} minutes."
            )
            if not self.run_config.run_locally and (
                    time_elapsed_minutes
                    > self.run_config.max_execution_minutes
            ) and not self.tuner_config.support_bayesian_optimization:
                logger.warning(
                    f"Worker [{worker_id}] has been processing bucket {bucket_id} for {time_elapsed_minutes:.2f} minutes, which exceeds the limit of {self.run_config.max_execution_minutes} minutes. Stopping processing more cases in this bucket to allow other jobs(like CICD jobs) in the queue to proceed."
                )
                parent_step_key = f'{self.tuner_config.kernel_tuner_name}_{self.run_config.case_set_id}_{self.run_config.run_id}_{begin_case_id}_{end_case_id}'
                self.generate_buildkite_pipeline_subbucket(
                    cid, end_case_id, parent_step_key=parent_step_key)
                bucket_fully_processed = False
                break
            last_processed_case_id = cid
            if cid in processed_ids:
                continue
            assert cid in all_configs, f"CaseId {cid} is missing in the configs retrieved from storage manager for CaseSetId {self.run_config.case_set_id}. This should not happen as the configs should have been generated and stored in the storage manager before."
            _, _, case_key_value = all_configs[cid]
            tuning_case = TuningCase.from_string(
                case_key_value, self.tuner_config.tuning_key_class,
                self.tuner_config.tunable_params_class)
            tuning_key, tunable_params, _ = tuning_case.tuning_key, tuning_case.tunable_params, tuning_case.is_baseline

            begin_case_id_time = time.perf_counter_ns()
            # status can be SUCCESS, FAILED_OOM, UNKNOWN_ERROR.
            status, warmup_ns, _ = self.run(tuning_key,
                                            tunable_params,
                                            iters=1)
            if status != TuningStatus.SUCCESS:
                results_buffer.append(
                    (self.run_config.case_set_id, self.run_config.run_id, cid,
                     status.value, worker_id, 0, 0, 0,
                     self.storage_manager.get_timestamp_sec(),
                     self.run_config.tpu_queue_multi))
                logger.warning(
                    f"Case {cid} failed during warmup with status: {status}. Skipping to next case."
                )
                continue
            warmup_us = int(warmup_ns // 1000)

            status, average_latency_ns, _ = self.run(tuning_key,
                                                     tunable_params,
                                                     iters=10)
            end_time = time.perf_counter_ns()
            total_time = end_time - begin_case_id_time
            if status != TuningStatus.SUCCESS:
                results_buffer.append(
                    (self.run_config.case_set_id, self.run_config.run_id, cid,
                     status.value, worker_id, warmup_us, 0, 0,
                     self.storage_manager.get_timestamp_sec(),
                     self.run_config.tpu_queue_multi))
                logger.warning(
                    f"Case {cid} failed during main run with status: {status}. Total time spent: {total_time/1e9:.2f}s."
                )
                continue

            average_latency_us = int(average_latency_ns // 1000)
            total_time_us = int(total_time // 1000)
            results_buffer.append(
                (self.run_config.case_set_id, self.run_config.run_id, cid,
                 status.value, worker_id, average_latency_us, warmup_us,
                 total_time_us, self.storage_manager.get_timestamp_sec(),
                 self.run_config.tpu_queue_multi))

            if self.run_config.debug:
                logger.info(
                    f"Case {cid} completed with AvgLat={average_latency_us}us, Warmup={warmup_us}us, Total={total_time_us}us"
                )

            if len(results_buffer) >= 10:
                self.storage_manager.save_results_batch(results_buffer)
                results_buffer = []

        self.storage_manager.save_results_batch(results_buffer)

        bucket_total_time_us = int(
            (time.perf_counter() - bucket_start_perf) * 1_000_000)
        self.storage_manager.add_bucket_processed_time_us(
            self.run_config.case_set_id, self.run_config.run_id, bucket_id,
            bucket_total_time_us)
        if bucket_fully_processed:
            self.storage_manager.mark_bucket_completed(
                self.run_config.case_set_id, self.run_config.run_id, bucket_id)
        logger.info(
            f"Worker [{worker_id}] Completed Bucket {bucket_id} [{begin_case_id}-{last_processed_case_id + 1}) for CaseSetId: {self.run_config.case_set_id}, RunId: {self.run_config.run_id}. Total time: {bucket_total_time_us/1e6:.2f}s."
        )
