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
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

import yaml
from absl import flags

FLAGS = flags.FLAGS

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


@dataclass
class RunResult:
    """Per-trial result from the smart-search path.

    Carries the kernel output(s) alongside latency so the verifier can run
    without a second kernel call. ``output`` is whatever the kernel returns
    (single array or tuple); ``bench`` is the optional ``BenchResult`` from
    ``bench.harness.measure``.
    """
    status: TuningStatus
    avg_latency_ns: float
    total_latency_ns: float
    output: Any = None
    bench: Any = None
    aux: dict = field(default_factory=dict)


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


@dataclass
class TunerConfig:
    tuning_key_class: any = None
    tunable_params_class: any = None
    kernel_tuner_name: str = None


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
            self.storage_manager = SpannerStorageManager()
        self._kernel_inputs_cache = {}
        self._tuning_key = None
        self.tuner_config = tuner_config
        self.run_config = run_config

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
                f"Initialized new CaseSet with ID: {self.run_config.case_set_id} and description: {self.run_config.case_set_desc}"
            )
            return True
        return False

    @abstractmethod
    def generate_cases(self) -> list[TuningCase]:
        """Generate the cases for the given case_set_id.
        This should not raise any exception, all exceptions should be caught and handled internally. The generated cases will be persisted in local file or database using storage_management module, where each case is represented as a TuningCase object and stored as a string. The case_id is the index of the case in the generated case list.

        Returns: A list of TuningCase objects representing the tuning cases to be processed.
        """
        raise NotImplementedError(
            "Specific kernel should implement this to generate the cases for the given case_set_id and desc, and return a list of TuningCase objects representing the tuning cases."
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
            else:
                # If the case set already exists, we assume the cases have been generated and we just need to generate the buckets for tuning jobs.
                total_cases = self.storage_manager.get_total_cases_in_case_set(
                    self.run_config.case_set_id)
            buckets = [
                (i, min(i + self.run_config.job_bucket_size, total_cases))
                for i in range(0, total_cases, self.run_config.job_bucket_size)
            ]
            return buckets
        except Exception as e:
            logger.error(
                f"Error initializing case set {self.run_config.case_set_id}: {e}"
            )
            raise e

    def _build_step(self,
                    case_id_start: int,
                    case_id_end: int,
                    parent_step_key: str = None) -> dict:
        step_key = f'{self.tuner_config.kernel_tuner_name}_{self.run_config.case_set_id}_{self.run_config.run_id}_{case_id_start}_{case_id_end}'
        parent_step_key = parent_step_key or 'generate_tuning_cases_and_yml'
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
            step = self._build_step(case_id_start, case_id_end)
            pipeline["steps"].append(step)
            self.storage_manager.create_bucket_for_run(
                self.run_config.case_set_id,
                self.run_config.run_id,
                bucket_id,
                case_id_start,
                case_id_end,
                tpu=self.run_config.tpu_queue_multi)

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

    # ------------------------------------------------------------------
    # Smart-search extension points (Phase 0 + 1).
    #
    # Existing v1 subclasses are not required to override any of these;
    # the legacy ``measure_latency`` grid loop above keeps working unchanged.
    # Subclasses that opt into ``kernel_tuner_runner --search-strategy``
    # override these hooks to plug into the new verifier/search pipeline.
    # ------------------------------------------------------------------

    def get_default_tuning_key(self):
        """Return the ``TuningKey`` that smart-search should explore.

        Smart-search tunes a single ``TuningKey`` at a time (the search space
        is over ``TunableParams``). Subclasses return the canonical key for
        their workload; the runner uses ``generate_inputs(key)`` to pin inputs.

        Default returns ``None``; subclasses that haven't opted in cannot be
        driven by the smart-search runner.
        """
        return None

    def get_search_space(self):
        """Return the typed ``SearchSpace`` for ``TunableParams``.

        Mapping: each key matches a field of the subclass's ``TunableParams``
        dataclass; each value is a ``ParamRange`` from
        ``tools.kernel.tuner.v1.search.strategy``.
        """
        return None

    def get_oracle(self):
        """Return a ``ReferenceOracle`` for ``verify``. Default: ``None``."""
        return None

    def get_cost_model(self):
        """Return a ``bench.cost_estimate.CostModel`` or ``None``."""
        return None

    def build_kernel_fn(
        self,
        tuning_key,
        tunable_params,
        inputs: dict,
    ) -> Callable[[], Any] | None:
        """Return a zero-arg callable that runs the kernel with given inputs.

        The returned callable is fed to ``bench.harness.measure``. Capturing
        ``inputs`` in a closure lets the harness time only the kernel call,
        not Python overhead.

        Default ``None`` means the smart-search loop falls back to the
        subclass's ``run`` for timing (in which case ``output`` will not be
        available for verification).
        """
        return None

    def run_with_outputs(
        self,
        tuning_key,
        tunable_params,
        iters: int,
    ) -> RunResult:
        """Smart-search entry that returns latency *and* outputs.

        Default implementation delegates to ``run`` and reports
        ``output=None`` — useful for subclasses that haven't been ported but
        still want to participate in the new search loop without verification.
        """
        status, avg, total = self.run(tuning_key, tunable_params, iters)
        return RunResult(
            status=status,
            avg_latency_ns=avg,
            total_latency_ns=total,
            output=None,
        )

    def verify(self, tuning_key, tunable_params, output, *, inputs=None):
        """Validate the kernel output against the oracle.

        Returns a ``NumericsReport`` (truthy on success). Default uses
        ``get_oracle()`` and ``check_many`` with dtype-tier tolerances.
        Subclasses can override to inject custom semantic kwargs.
        """
        # Late import to avoid pulling the verifier package at v1 module load.
        from tools.kernel.tuner.v1.verifier.numerics import (NumericsReport,
                                                             check_many)

        oracle = self.get_oracle()
        if oracle is None or output is None or inputs is None:
            return NumericsReport(
                passed=True,
                max_abs_diff=0.0,
                cosine=1.0,
                nan_count=0,
                inf_count=0,
            )
        reference = oracle.compute(inputs)
        actual = output if isinstance(output, (tuple, list)) else (output, )
        reference = (reference if isinstance(reference, (tuple, list)) else
                     (reference, ))
        atol, rtol = oracle.dtype_tolerance(actual[0].dtype)
        return check_many(actual, reference, atol=atol, rtol=rtol)

    def measure_latency(self, begin_case_id: int, end_case_id: int):
        """Measure the latency of cases in the caseset with case_id in [begin_case_id, end_case_id). The latency of each case will be persisted in local file or database using storage_management module.

        Args:
            begin_case_id: Start of the case_id range (inclusive) within the caseset to measure.
            end_case_id: End of the case_id range (exclusive) within the caseset to measure.
        """
        bucket_id = begin_case_id // self.run_config.job_bucket_size
        logger.info(
            f"Worker [{FLAGS.worker_id}] Claimed CaseSetId: {self.run_config.case_set_id}, RunId: {self.run_config.run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing."
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
                f"Worker [{FLAGS.worker_id}] Processing CaseId: {cid} in Bucket {bucket_id}, [{begin_case_id}-{end_case_id}) with elapsed time {time_elapsed_minutes:.2f} minutes."
            )
            if not self.run_config.run_locally and (
                    time_elapsed_minutes
                    > self.run_config.max_execution_minutes):
                logger.warning(
                    f"Worker [{FLAGS.worker_id}] has been processing bucket {bucket_id} for {time_elapsed_minutes:.2f} minutes, which exceeds the limit of {self.run_config.max_execution_minutes} minutes. Stopping processing more cases in this bucket to allow other jobs(like CICD jobs) in the queue to proceed."
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
            tuning_key, tunable_params = TuningCase.from_string(
                case_key_value, self.tuner_config.tuning_key_class,
                self.tuner_config.tunable_params_class)

            begin_case_id_time = time.perf_counter_ns()
            # status can be SUCCESS, FAILED_OOM, UNKNOWN_ERROR.
            status, warmup_ns, _ = self.run(tuning_key,
                                            tunable_params,
                                            iters=1)
            if status != TuningStatus.SUCCESS:
                results_buffer.append(
                    (self.run_config.case_set_id, self.run_config.run_id, cid,
                     status.value, FLAGS.worker_id, 0, 0, 0,
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
                     status.value, FLAGS.worker_id, warmup_us, 0, 0,
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
                 status.value, FLAGS.worker_id, average_latency_us, warmup_us,
                 total_time_us, self.storage_manager.get_timestamp_sec(),
                 self.run_config.tpu_queue_multi))

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
            self.run_config.case_set_id, self.run_config.run_id, bucket_id,
            bucket_total_time_us)
        if bucket_fully_processed:
            self.storage_manager.mark_bucket_completed(
                self.run_config.case_set_id, self.run_config.run_id, bucket_id)
        logger.info(
            f"Worker [{FLAGS.worker_id}] Completed Bucket {bucket_id} [{begin_case_id}-{last_processed_case_id + 1}) for CaseSetId: {self.run_config.case_set_id}, RunId: {self.run_config.run_id}. Total time: {bucket_total_time_us/1e6:.2f}s."
        )
