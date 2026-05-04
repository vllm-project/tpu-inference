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
                 kernel_tuner_name: str = None,
                 case_set_id: str = None,
                 run_id: str = None,
                 case_set_desc: str = None,
                 tpu_version: str = None,
                 tpu_cores: int = None,
                 tpu_queue_multi: str = None,
                 run_locally: bool = None,
                 kernel_tuning_job_priority: int = -10):
        assert tuning_key_class is not None, "tuning_key_class must be specified"
        assert tunable_params_class is not None, "tunable_params_class must be specified"
        assert storage_manager is not None, "storage_manager must be specified"
        assert kernel_tuner_name is not None, "kernel_tuner_name must be specified, which will be used as the identifier for this kernel tuner in the Buildkite pipeline generation and execution. It should match the key in the KERNEL_TUNER_REGISTRY in kernel_tuner_runner.py to ensure the correct kernel tuner is called during execution."
        assert case_set_id is not None, "case_set_id must be specified, which identifies a set of tuning cases."
        assert run_id is not None, "run_id must be specified, which identifies a tuning run for a given case_set_id. This allows multiple tuning runs for the same case_set_id to be distinguished and tracked separately."
        assert case_set_desc is not None, "case_set_desc must be specified, which provides a description for the case set."

        self.tuning_key_class = tuning_key_class
        self.tunable_params_class = tunable_params_class
        self.storage_manager = storage_manager
        self._KERNEL_INPUTS_CACHE = {}
        self._TUNING_KEY = None
        self.job_bucket_size = job_bucket_size
        self.kernel_tuner_name = kernel_tuner_name
        self.case_set_id = case_set_id
        self.run_id = run_id
        self.case_set_desc = case_set_desc
        self.tpu_queue_multi = tpu_queue_multi
        self.tpu_version = tpu_version
        self.tpu_cores = tpu_cores
        self.run_locally = run_locally
        self.run_at_most_minutes = 20
        self.kernel_tuning_job_priority = kernel_tuning_job_priority

    def _init_case_set(self) -> bool:
        """Initialize the case set with the given case_set_id and description. This will be called when the caseset_id is new or the caseset_id is not specified.

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            desc: A description for this case set, which will be persisted in local file or database using storage_management module.

        Returns:
            True if tuning cases were initialized so in _generate_tuning_jobs we don't need to regenerate them, False otherwise.

        """
        if self.case_set_id is None:
            self.case_set_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # check case_set_id exists in storage manager, if not exist, create a new case set with the given case_set_id and desc.
        # if exist, check whether the desc is the same as the existing one, if not, raise an error.
        if self.storage_manager.case_set_id_exists(self.case_set_id):
            existing_desc = self.storage_manager.get_case_set_desc(
                self.case_set_id)
            if existing_desc != self.case_set_desc:
                raise ValueError(
                    f"CaseSetId {self.case_set_id} already exists with a different description. Existing desc: {existing_desc}, new desc: {self.case_set_desc}. If you intend to create new case set, please use a new case set id. Updating comment of an existing case set is not allowed. Please use a different CaseSetId or update the description to match the existing one."
                )
            else:
                logger.info(
                    f"CaseSetId {self.case_set_id} already exists with the same description. Proceeding with the existing case set."
                )
        else:
            self.storage_manager.init_case_set(self.case_set_id,
                                               scan_space=0,
                                               desc=self.case_set_desc)
            logger.info(
                f"Initialized new CaseSet with ID: {self.case_set_id} and description: {self.case_set_desc}"
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

    def _generate_tuning_jobs(self) -> list[tuple[int, int]]:
        """Partitions the full case set into fixed-size work buckets.

        Calls `generate_cases` to determine the total number of cases, then
        splits them into contiguous ranges of at most `self.job_bucket_size` cases each.
        Buckets are intended to be dispatched and executed in parallel; result
        ordering is not guaranteed. Each bucket is identified by a half-open
        interval [begin_case_id, end_case_id).

        Returns:
            A list of [begin_case_id, end_case_id] pairs covering all cases.
        """
        try:
            if self._init_case_set():
                start_time = time.perf_counter()
                cases = self.generate_cases()
                total_cases = len(cases)
                for case_id, case_str in enumerate(map(str, cases)):
                    self.storage_manager.add_tuner_case(
                        self.case_set_id,
                        case_id,
                        case_str,
                        tpu=self.tpu_queue_multi)
                self.storage_manager.flush()
                duration_sec = int(time.perf_counter() - start_time)
                self.storage_manager.finish_case_set(
                    self.case_set_id,
                    total_cases,
                    0,  # invalid case count, doesn't matter here
                    duration_sec * 1.0)
                logger.info(
                    f"\nComplete Generate Tuning Cases for {self.case_set_id}, Valid Cases: {total_cases} | Duration: {duration_sec}s"
                )
            else:
                # If the case set already exists, we assume the cases have been generated and we just need to generate the buckets for tuning jobs.
                total_cases = self.storage_manager.get_total_cases_in_case_set(
                    self.case_set_id)
            buckets = [(i, min(i + self.job_bucket_size, total_cases))
                       for i in range(0, total_cases, self.job_bucket_size)]
            return buckets
        except Exception as e:
            logger.error(
                f"Error initializing case set {self.case_set_id}: {e}")
            raise e

    def _build_step(self, case_id_start: int, case_id_end: int) -> dict:
        return {
            "label":
            f"cs_id={self.case_set_id} rid={self.run_id} Bucket([{case_id_start}, {case_id_end}))",
            "depends_on":
            f"{self.tpu_version}_build_docker",
            "agents": {
                "queue": self.tpu_queue_multi
            },
            "env": {
                "USE_PREBUILT_IMAGE": "1",
                "TPU_VERSION": self.tpu_version
            },
            "commands": [
                f"- |"
                f"  rm -f \"/tmp/kernel_tuning/generated_pipeline.yml\""
                f"- |"
                f"  .buildkite/scripts/run_in_docker.sh bash -c \""
                f"  pip install --upgrade google-cloud-spanner && "
                f"  pip install --upgrade google-api-core && "
                f"  pip install --upgrade google-auth && "
                f"  pip install --upgrade absl-py && "
                f"  python -m tools.kernel.tuner.v1.kernel_tuner_runner "
                f"  --kernel_tuner_name={self.kernel_tuner_name} "
                f"  --case_set_id={self.case_set_id} --run_id={self.run_id} "
                f"  --tpu_version={self.tpu_version} "
                f"  --tpu_cores={self.tpu_cores} "
                f"  --case_set_desc=\"{self.case_set_desc}\" "
                f"  --run_locally={self.run_locally} "
                f"  --tpu_queue_multi={self.tpu_queue_multi} "
                f"  --begin_case_id={case_id_start} --end_case_id={case_id_end}\""
                f"- |"
                f"  if [ -f /tmp/kernel_tuning/generated_pipeline.yml ]; then "
                f"    buildkite-agent artifact upload \"/tmp/kernel_tuning/generated_pipeline.yml\" && "
                f"    echo \"Upload generated pipeline YAML to Buildkite artifacts with priority {self.kernel_tuning_job_priority}\" && "
                f"    {{ "
                f"      echo \"priority: {self.kernel_tuning_job_priority}\"; "
                f"      cat /tmp/kernel_tuning/generated_pipeline.yml; "
                f"    }} | buildkite-agent pipeline upload; "
                f"  else "
                f"    echo \"File /tmp/kernel_tuning/generated_pipeline.yml does not exist. Exiting successfully.\"; "
                f"  fi"
            ]
        }

    def generate_buildkite_pipeline_subbucket(self, start: int, end: int):
        """Generate the Buildkite pipeline for a sub-bucket of tuning jobs.

        Args:
            start: The starting case_id of the sub-bucket (inclusive).
            end: The ending case_id of the sub-bucket (exclusive).
        """
        output_path = "/tmp/kernel_tuning/generated_pipeline.yml"
        if os.path.exists(output_path):
            # clean up the existing one
            os.remove(output_path)
        next_group_label = f"Tune Case ID [{start}, {end})"
        step = self._build_step(start, end)
        pipeline = {"group": next_group_label, "steps": [step]}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
        logger.info(
            f"Generated Buildkite pipeline YAML for sub-bucket [{start}, {end}) saved to {output_path} in Docker"
        )

    def generate_buildkite_pipeline(self, ) -> str:
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
            step = self._build_step(case_id_start, case_id_end)
            pipeline["steps"].append(step)
            self.storage_manager.create_bucket_for_run(
                self.case_set_id,
                self.run_id,
                bucket_id,
                case_id_start,
                case_id_end,
                tpu=self.tpu_queue_multi)

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

    def measure_latency(self, begin_case_id: int, end_case_id: int):
        """Measure the latency of cases in the caseset with case_id in [begin_case_id, end_case_id). The latency of each case will be persisted in local file or database using storage_management module.

        Args:
            begin_case_id: Start of the case_id range (inclusive) within the caseset to measure.
            end_case_id: End of the case_id range (exclusive) within the caseset to measure.
        """
        bucket_id = begin_case_id // self.job_bucket_size
        logger.info(
            f"Worker [{FLAGS.worker_id}] Claimed CaseSetId: {self.case_set_id}, RunId: {self.run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing."
        )
        self.storage_manager.mark_bucket_in_progress(self.case_set_id,
                                                     self.run_id, bucket_id)

        processed_ids = self.storage_manager.get_already_processed_ids(
            self.case_set_id, self.run_id, begin_case_id, end_case_id)
        all_configs = self.storage_manager.get_bucket_configs(
            self.case_set_id, begin_case_id, end_case_id)

        bucket_start_perf = time.perf_counter()
        results_buffer = []
        bucket_fully_processed = True
        last_processed_case_id = None
        for cid in range(begin_case_id, end_case_id):
            time_elapsed_minutes = (time.perf_counter() -
                                    bucket_start_perf) / 60
            if not self.run_locally and (time_elapsed_minutes
                                         > self.run_at_most_minutes or
                                         (cid - begin_case_id) > 1):
                logger.warning(
                    f"Worker [{FLAGS.worker_id}] has been processing bucket {bucket_id} for {time_elapsed_minutes:.2f} minutes, which exceeds the limit of {self.run_at_most_minutes} minutes. Stopping processing more cases in this bucket to allow other jobs(like CICD jobs) in the queue to proceed."
                )
                self.generate_buildkite_pipeline_subbucket(cid, end_case_id)
                bucket_fully_processed = False
                break
            last_processed_case_id = cid
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
                    (self.case_set_id, self.run_id, cid, status.value,
                     FLAGS.worker_id, 0, 0, 0,
                     self.storage_manager.get_timestamp_sec(),
                     self.tpu_queue_multi))
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
                    (self.case_set_id, self.run_id, cid, status.value,
                     FLAGS.worker_id, warmup_us, 0, 0,
                     self.storage_manager.get_timestamp_sec(),
                     self.tpu_queue_multi))
                logger.warning(
                    f"Case {cid} failed during main run with status: {status}. Total time spent: {total_time/1e9:.2f}s."
                )
                continue

            average_latency_us = int(average_latency_ns // 1000)
            total_time_us = int(total_time // 1000)
            results_buffer.append(
                (self.case_set_id, self.run_id, cid, status.value,
                 FLAGS.worker_id, average_latency_us, warmup_us, total_time_us,
                 self.storage_manager.get_timestamp_sec(),
                 self.tpu_queue_multi))

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
        self.storage_manager.add_bucket_processed_time_us(
            self.case_set_id, self.run_id, bucket_id, bucket_total_time_us)
        if bucket_fully_processed:
            self.storage_manager.mark_bucket_completed(self.case_set_id,
                                                       self.run_id, bucket_id)
        logger.info(
            f"Worker [{FLAGS.worker_id}] Completed Bucket {bucket_id} ({begin_case_id}-{last_processed_case_id + 1}] for CaseSetId: {self.case_set_id}, RunId: {self.run_id}. Total time: {bucket_total_time_us/1e6:.2f}s."
        )
