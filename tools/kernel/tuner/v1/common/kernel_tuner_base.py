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
from datetime import datetime
from enum import Enum

import yaml
from absl import flags

from tools.kernel.tuner.v1.storage_management.storage_manager import \
    StorageManager

FLAGS = flags.FLAGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def __init__(self, tuning_key: TuningKey, tunable_params: TunableParams):
        self.tuning_key = tuning_key
        self.tunable_params = tunable_params

    def __str__(self):
        return json.dumps({
            'tuning_key': asdict(self.tuning_key),
            'tunable_params': asdict(self.tunable_params)
        })

    @classmethod
    def from_string(cls, string, tuning_key_class, tunable_params_class):
        data = json.loads(string)
        tuning_key = tuning_key_class(**data['tuning_key'])
        tunable_params = tunable_params_class(**data['tunable_params'])
        case = TuningCase(tuning_key, tunable_params)
        return case.tuning_key, case.tunable_params


class KernelTunerBase(ABC):
    """
    Base class for kernel tuner runner. The kernel tuner runner is responsible for generating the tuning cases, partitioning the cases into buckets, generating the Buildkite pipeline, and measuring the latency of the cases. The specific kernel tuner runner should inherit from this base class and implement the generate_cases, generate_inputs, and run methods.
    Subclass should also define the TuningKey and TunableParams dataclasses according to the kernel's tuning space.
    The tuning cases, tuning results, and other metadata will be persisted in local file or database using storage_management module, which is abstracted by the StorageManager class. The specific implementation of StorageManager can be LocalDbManager for local JSON-file-backed storage or SpannerDbManager for Google Spanner-backed storage.
    The kernel tuner runner will be executed in a distributed manner, where each worker will claim a bucket of cases to process, run the kernel with the corresponding tuning key and tunable params, measure the latency, and save the results back to the storage manager. The Buildkite pipeline will be generated to orchestrate the distributed execution of the kernel tuner runner.

    Subclass should implement the following methods:
    - generate_cases: Generate the tuning cases for the given case_set_id and desc, and return a list of TuningCase objects representing the tuning cases.
    - generate_inputs: Generate the kernel inputs for the given tuning key with caching, and return a dictionary of kernel inputs.
    - run: Execute the kernel with the given tuning key and tunable params for a certain number of iterations, measure the latency, and return the tuning status, average latency, and total latency.

    Subclass must call super().__init__(tuning_key_class=tuning_key_class, tunable_params_class=tunable_params_class, storage_manager=storage_manager) in the __init__ method to initialize the base class with the tuning key class, tunable params class, and storage manager.

    """

    def __init__(self,
                 *,
                 tuning_key_class=None,
                 tunable_params_class=None,
                 storage_manager: StorageManager = None,
                 job_bucket_size: int = 100,
                 kernel_tuner_name: str = None):
        assert tuning_key_class is not None, "tuning_key_class must be specified"
        assert tunable_params_class is not None, "tunable_params_class must be specified"
        assert storage_manager is not None, "storage_manager must be specified"
        assert kernel_tuner_name is not None, "kernel_tuner_name must be specified, which will be used as the identifier for this kernel tuner in the Buildkite pipeline generation and execution. It should match the key in the KERNEL_TUNER_REGISTRY in kernel_tuner_runner.py to ensure the correct kernel tuner is called during execution."
        self.tuning_key_class = tuning_key_class
        self.tunable_params_class = tunable_params_class
        self.storage_manager = storage_manager
        self._KERNEL_INPUTS_CACHE = {}
        self._TUNING_KEY = None
        self.job_bucket_size = job_bucket_size
        self.kernel_tuner_name = kernel_tuner_name

    def _init_case_set(self, case_set_id: str, desc: str) -> bool:
        """Initialize the case set with the given case_set_id and description. This will be called when the caseset_id is new or the caseset_id is not specified.

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            desc: A description for this case set, which will be persisted in local file or database using storage_management module.

        Returns:
            True if tuning cases were initialized so in _generate_tuning_jobs we don't need to regenerate them, False otherwise.

        """
        if case_set_id is None:
            case_set_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # check case_set_id exists in storage manager, if not exist, create a new case set with the given case_set_id and desc.
        # if exist, check whether the desc is the same as the existing one, if not, raise an error.
        if self.storage_manager.case_set_id_exists(case_set_id):
            existing_desc = self.storage_manager.get_case_set_desc(case_set_id)
            if existing_desc != desc:
                raise ValueError(
                    f"CaseSetId {case_set_id} already exists with a different description. Existing desc: {existing_desc}, new desc: {desc}. If you intend to create new case set, please use a new case set id. Updating comment of an existing case set is not allowed. Please use a different CaseSetId or update the description to match the existing one."
                )
            else:
                logger.info(
                    f"CaseSetId {case_set_id} already exists with the same description. Proceeding with the existing case set."
                )
        else:
            self.storage_manager.init_case_set(case_set_id,
                                               scan_space=0,
                                               desc=desc)
            logger.info(
                f"Initialized new CaseSet with ID: {case_set_id} and description: {desc}"
            )
            return True
        return False

    @abstractmethod
    def generate_cases(self) -> list[TuningCase]:
        """Generate the cases for the given case_set_id. This will be called when the caseset_id is new or the caseset_id is not specified.
        This should not raise any exception, all exceptions should be caught and handled internally. The generated cases will be persisted in local file or database using storage_management module, where each case is represented as a TuningCase object and stored as a string. The case_id is the index of the case in the generated case list.

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            desc: A description for this case set, which will be persisted in local file or database using storage_management module."""
        raise NotImplementedError(
            "Specific kernel should implement this to generate the cases for the given case_set_id and desc, and return a list of TuningCase objects representing the tuning cases."
        )

    def _generate_tuning_jobs(
        self,
        case_set_id: str,
        desc: str,
    ) -> list[tuple[int, int]]:
        """Partitions the full case set into fixed-size work buckets.

        Calls `generate_cases` to determine the total number of cases, then
        splits them into contiguous ranges of at most `self.job_bucket_size` cases each.
        Buckets are intended to be dispatched and executed in parallel; result
        ordering is not guaranteed. Each bucket is identified by a half-open
        interval [begin_case_id, end_case_id).

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            desc: A description for this case set, which will be persisted in local file or database using storage_management module.

        Returns:
            A list of [begin_case_id, end_case_id] pairs covering all cases.
        """
        try:
            if self._init_case_set(case_set_id, desc=desc):
                start_time = time.perf_counter()
                cases = self.generate_cases()
                total_cases = len(cases)
                for case_id, case_str in enumerate(map(str, cases)):
                    self.storage_manager.add_tuner_case(
                        case_set_id, case_id, case_str)
                self.storage_manager.flush()
                duration_sec = int(time.perf_counter() - start_time)
                self.storage_manager.finish_case_set(
                    case_set_id,
                    total_cases,
                    0,  # invalid case count, doesn't matter here
                    duration_sec * 1.0)
                logger.info(
                    f"\nComplete Generate Tuning Cases for {case_set_id}, Valid Cases: {total_cases} | Duration: {duration_sec}s"
                )
            else:
                # If the case set already exists, we assume the cases have been generated and we just need to generate the buckets for tuning jobs.
                total_cases = self.storage_manager.get_total_cases_in_case_set(
                    case_set_id)
            buckets = [(i, min(i + self.job_bucket_size, total_cases))
                       for i in range(0, total_cases, self.job_bucket_size)]
            return buckets
        except Exception as e:
            logger.error(f"Error initializing case set {case_set_id}: {e}")
            raise e

    def generate_buildkite_pipeline(self, case_set_id: str, run_id: str,
                                    desc: str, tpu_version: str,
                                    tpu_queue_multi: str) -> str:
        """Generate the Buildkite pipeline for the given tuning jobs. Each tuning job will be represented as a Buildkite step that calls the measure_latency function with the corresponding case_id range.

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            run_id: Identifies a run of the tuning pipeline. Can be used to distinguish different
                runs of the tuning pipeline with the same caseset_id — useful when the caseset has
                not changed but the KernelTunerRunner class has changed.

        Returns:
            A string representing the Buildkite pipeline configuration in YAML format.
        """
        output_path = "/tmp/kernel_tuning/generated_pipeline.yml"
        if os.path.exists(output_path):
            # clean up the existing one
            os.remove(output_path)
        buckets = self._generate_tuning_jobs(case_set_id, desc=desc)
        # The Buildkite pipeline YAML will be generated in the format of:
        # steps:
        #   - label: "Measure latency for cases [begin_case_id, end_case_id)"
        #     command: "python -m tools.kernel.tuner.v1.kernel_tuner_runner --worker_id=WORKER_ID --case_set_id=CASE_SET_ID --run_id=RUN_ID --begin_case_id=BEGIN_CASE_ID --end_case_id=END_CASE_ID"
        pipeline = {"steps": []}

        for bucket_id, (case_id_start, case_id_end) in enumerate(buckets):
            step = {
                "label":
                f"cs_id={case_set_id} rid={run_id} Bucket([{case_id_start}, {case_id_end}))",
                "depends_on":
                f"{tpu_version}_build_docker",
                "agents": {
                    "queue": tpu_queue_multi
                },
                "env": {
                    "USE_PREBUILT_IMAGE": "1",
                    "TPU_VERSION": tpu_version
                },
                "commands": [
                    f".buildkite/scripts/run_in_docker.sh bash -c \"pip install --upgrade google-cloud-spanner && pip install --upgrade google-api-core && pip install --upgrade google-auth && pip install --upgrade absl-py && python -m tools.kernel.tuner.v1.kernel_tuner_runner --kernel_tuner_name={self.kernel_tuner_name} --case_set_id={case_set_id} --run_id={run_id} --begin_case_id={case_id_start} --end_case_id={case_id_end}\""
                ]
            }
            pipeline["steps"].append(step)
            self.storage_manager.create_bucket_for_run(case_set_id, run_id,
                                                       bucket_id,
                                                       case_id_start,
                                                       case_id_end)

        pipeline['steps'] = [{
            'group': 'Kernel Sweeping Group',
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
        if self._TUNING_KEY and tuning_key == self._TUNING_KEY:
            return self._KERNEL_INPUTS_CACHE
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

    def measure_latency(self, caseset_id: str, run_id: str, begin_case_id: int,
                        end_case_id: int):
        """Measure the latency of cases in the caseset with case_id in [begin_case_id, end_case_id). The latency of each case will be persisted in local file or database using storage_management module.

        Args:
            caseset_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            run_id: Identifies a run of the tuning pipeline. Can be used to distinguish different
                runs of the tuning pipeline with the same caseset_id — useful when the caseset has
                not changed but the KernelTunerRunner class has changed.
            begin_case_id: Start of the case_id range (inclusive) within the caseset to measure.
            end_case_id: End of the case_id range (exclusive) within the caseset to measure.
        """
        bucket_id = begin_case_id // self.job_bucket_size
        logger.info(
            f"Worker [{FLAGS.worker_id}] Claimed CaseSetId: {caseset_id}, RunId: {run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing."
        )
        self.storage_manager.mark_bucket_in_progress(caseset_id, run_id,
                                                     bucket_id)

        processed_ids = self.storage_manager.get_already_processed_ids(
            caseset_id, run_id, begin_case_id, end_case_id)
        all_configs = self.storage_manager.get_bucket_configs(
            caseset_id, begin_case_id, end_case_id)

        bucket_start_perf = time.perf_counter()
        results_buffer = []
        for cid in range(begin_case_id, end_case_id):
            if cid in processed_ids:
                continue
            config = all_configs.get(cid)
            if not config:
                continue
            _, _, case_key_value = config
            tuning_key, tunable_params = TuningCase.from_string(
                case_key_value, self.tuning_key_class,
                self.tunable_params_class)

            begin_case_id_time = time.perf_counter_ns()
            # status can be SUCCESS, FAILED_OOM, UNKNOWN_ERROR.
            status, warmup_ns, _ = self.run(tuning_key,
                                            tunable_params,
                                            iters=1)
            if status != TuningStatus.SUCCESS:
                results_buffer.append(
                    (caseset_id, run_id, cid, status.value, FLAGS.worker_id, 0,
                     0, 0, self.storage_manager.get_timestamp_sec()))
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
                    (caseset_id, run_id, cid, status.value,
                     FLAGS.worker_id, warmup_us, 0, 0,
                     self.storage_manager.get_timestamp_sec()))
                logger.warning(
                    f"Case {cid} failed during main run with status: {status}. Total time spent: {total_time/1e9:.2f}s."
                )
                continue

            average_latency_us = int(average_latency_ns // 1000)
            total_time_us = int(total_time // 1000)
            results_buffer.append(
                (caseset_id, run_id, cid, status.value, FLAGS.worker_id,
                 average_latency_us, warmup_us, total_time_us,
                 self.storage_manager.get_timestamp_sec()))

            if FLAGS.debug:
                logger.info(
                    f"Case {cid} completed with AvgLat={average_latency_us}us, Warmup={warmup_us}us, Total={total_time_us}us"
                )

            if len(results_buffer) >= 10:
                self.storage_manager.save_results_batch(results_buffer)
                results_buffer = []

        self.storage_manager.save_results_batch(results_buffer)

        bucket_total_time_us = int(
            (time.perf_counter() - bucket_start_perf) * 1_000_000)
        self.storage_manager.mark_bucket_completed(caseset_id, run_id,
                                                   bucket_id,
                                                   bucket_total_time_us)
        logger.info(
            f"Worker [{FLAGS.worker_id}] Completed Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for CaseSetId: {caseset_id}, RunId: {run_id}. Total time: {bucket_total_time_us/1e6:.2f}s."
        )
