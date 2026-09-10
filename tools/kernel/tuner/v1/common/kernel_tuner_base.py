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
from abc import ABC, abstractmethod

import yaml

# isort: off
from tools.kernel.tuner.v1.common.tuner_datatypes import (
    RunConfig, TunableParams, TunerConfig, TuningCase, TuningKey, TuningStatus)
# isort: on
from tools.kernel.tuner.v1.common.utils import safe_remove_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteralString(str):
    pass


def _literal_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(LiteralString, _literal_representer)


def _embed_flag_in_bash_c(arg: str) -> str:
    """Double-quote a '--name=value' arg so the inner shell of the generated
    bash -c command keeps the value as one word."""
    name, sep, value = arg.partition('=')
    if not sep:
        return arg  # boolean form '--name' / '--noname'
    assert "'" not in value, (
        f'{name} value must not contain single quotes when generating '
        'Buildkite steps (it would terminate the bash -c quoting)')
    escaped = (value.replace('\\', '\\\\').replace('"', '\\"').replace(
        '$', '\\$').replace('`', '\\`'))
    return f'{name}="{escaped}"'


class ProcessedCasesTracker:
    """Tracks evaluated case IDs and their execution statuses to manage state and OOM early-pruning."""

    def __init__(self, storage_manager, tuner_config: 'TunerConfig',
                 run_config: 'RunConfig', begin_case_id: int,
                 end_case_id: int):

        processed_ids_status = storage_manager.get_already_processed_ids(
            run_config.case_set_id, run_config.run_id, begin_case_id,
            end_case_id)
        self.processed_ids = set(
            [item.case_id for item in processed_ids_status])

        self.history: dict[TuningKey, list[tuple[TunableParams,
                                                 TuningStatus]]] = {}
        # TODO: refactor this to use the begin_case_id and end_case_id to limit the range of cases to read
        all_cases_id_case_key_value = storage_manager.get_all_cases(
            run_config.case_set_id)
        processed_ids_status_dict = {
            item.case_id: item.status
            for item in processed_ids_status
        }
        for case_id, case_key_value in all_cases_id_case_key_value:
            tuning_case = TuningCase.from_string(
                case_key_value, tuner_config.tuning_key_class,
                tuner_config.tunable_params_class)
            if case_id not in processed_ids_status_dict:
                continue
            self.history.setdefault(tuning_case.tuning_key, []).append(
                (tuning_case.tunable_params,
                 TuningStatus(processed_ids_status_dict.get(case_id))))

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
    """Pure kernel definition base class.

    Subclasses define the kernel's tuning space and execution logic by
    implementing ``generate_cases``, ``generate_inputs``, ``run``, and
    optionally ``get_search_space``.

    This class intentionally has **no** optimizer and **no** storage manager.
    Those concerns are owned by the runner and worker processes respectively.

    Args:
        tuner_config: Static configuration for this kernel tuner.
        run_config: Runtime configuration for the current tuning run.
        lightweight: If True, skip expensive initialization (JAX device
            setup, xprof directory, etc.).  Used by the worker process
            which only needs config and search-space access.
    """

    def __init__(self,
                 *,
                 tuner_config: TunerConfig = None,
                 run_config: RunConfig = None,
                 lightweight: bool = False):
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

        self.lightweight = lightweight
        self._kernel_inputs_cache = {}
        self._tuning_key = None
        self.tuner_config = tuner_config
        self.run_config = run_config
        if run_config.n_bayesian_trials is not None:
            self.tuner_config.n_bayesian_trials = run_config.n_bayesian_trials
        if run_config.min_cases_for_bayesian is not None:
            self.tuner_config.min_cases_for_bayesian = run_config.min_cases_for_bayesian
        self.use_bayesian_optimization = tuner_config.support_bayesian_optimization and run_config.use_bayesian_optimization

        if run_config.use_bayesian_optimization and not tuner_config.support_bayesian_optimization:
            logger.info(
                f'{tuner_config.kernel_tuner_name} does not support Bayesian Optimization, falls back to full sweep.'
            )

        self.xprof_dir = os.path.join("/tmp/kernel_tuning",
                                      self.tuner_config.kernel_tuner_name,
                                      "xprof")
        # Control number of iterations for measuring kernel latency.
        self._measurement_iters = 5 if self.tuner_config.jit_kernel_pattern else 100

    @property
    def worker_id(self) -> str:
        from tools.kernel.tuner.v1.utils import get_worker_id
        return get_worker_id(self.run_config.worker_id)

    @staticmethod
    def init_case_set(storage_manager, run_config: RunConfig) -> bool:
        """Initialize the case set in storage.

        Returns True if a new case set was created (cases need to be
        generated), False if the case set already exists.
        """
        if storage_manager.case_set_id_exists(run_config.case_set_id):
            existing_desc = storage_manager.get_case_set_desc(
                run_config.case_set_id)
            if existing_desc != run_config.case_set_desc:
                raise ValueError(
                    f"CaseSetId {run_config.case_set_id} already exists with a different description. Existing desc: {existing_desc}, new desc: {run_config.case_set_desc}. If you intend to create new case set, please use a new case set id. Updating comment of an existing case set is not allowed. Please use a different CaseSetId or update the description to match the existing one."
                )
            else:
                logger.info(
                    f"CaseSetId {run_config.case_set_id} already exists with the same description. Proceeding with the existing case set."
                )
        else:
            storage_manager.init_case_set(run_config.case_set_id,
                                          scan_space=0,
                                          desc=run_config.case_set_desc)
            logger.info(
                f"CaseSet with ID: {run_config.case_set_id} and description: {run_config.case_set_desc} initialized."
            )
            return True
        return False

    def _resolve_kernel_pattern(self, tuning_key: TuningKey) -> str:
        if callable(self.tuner_config.jit_kernel_pattern):
            return self.tuner_config.jit_kernel_pattern(tuning_key)
        else:
            return self.tuner_config.jit_kernel_pattern

    def generate_autotune_cases(self, storage_manager) -> list[TuningCase]:
        """Generate autotune cases by reading from storage.

        Args:
            storage_manager: The storage manager to read autotune cases from.
        """
        tuning_set = []
        # The case_set_id is constructed as {kernel_tuner_name}_{autotune_case_set_id} in the bootstrap_kernel_tuners.py
        autotune_case_set_id = self.run_config.case_set_id.removeprefix(
            f'{self.tuner_config.kernel_tuner_name}_')
        autotune_cases = storage_manager.read_autotune_cases(
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

    def _build_step(self, case_id_start: int, case_id_end: int,
                    parent_step_key: str) -> dict:
        step_key = f'{self.tuner_config.kernel_tuner_name}_{self.run_config.case_set_id}_{self.run_config.run_id}_{case_id_start}_{case_id_end}'
        yml_file_path = self.run_config.subbucket_yml_path(case_id_end)
        safe_remove_files(yml_file_path)
        from tools.kernel.tuner.v1.kernel_tuner_flags import \
            get_present_flag_args
        extra_flags = get_present_flag_args(
            exclude_flags={
                'run_locally',
                'begin_case_id',
                'end_case_id',
                'worker_id',  # keep per-agent env resolution; don't stamp the generator's id
            })
        extra_flags_str = ''.join(f'  {_embed_flag_in_bash_c(a)}'
                                  for a in extra_flags)

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
                # For a single step, it might generate subbucket job
                LiteralString(f'rm -f {yml_file_path}'),
                LiteralString(
                    '.buildkite/scripts/run_in_docker.sh bash -c \''
                    'pip install -r tools/kernel/tuner/v1/storage_management/requirements.txt && '
                    'python -m tools.kernel.tuner.v1.kernel_tuner_runner '
                    f'  --run_locally=False '
                    f'{extra_flags_str}'
                    f'  --begin_case_id={case_id_start} --end_case_id={case_id_end}\''
                ),
                LiteralString(
                    f'if [ -f {yml_file_path} ]; then '
                    f'  buildkite-agent artifact upload {yml_file_path} && '
                    f'  echo \"Upload generated pipeline YAML to Buildkite artifacts with priority {self.run_config.job_priority}\" && '
                    f'  {{ '
                    f'      echo \"priority: {self.run_config.job_priority}\"; '
                    f'      cat {yml_file_path}; '
                    f'  }} | buildkite-agent pipeline upload; rm -f {yml_file_path}'
                    f'  else '
                    f'      echo \"File {yml_file_path} does not exist. It is either this bucket is completely processed or encounters an issue that requires a bucket level retry.\"; '
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
        subbucket_yml_path = self.run_config.subbucket_yml_path(end)
        safe_remove_files(subbucket_yml_path)
        step = self._build_step(start, end, parent_step_key=parent_step_key)
        pipeline = {"group": 'Kernel Sweeping Group', "steps": [step]}
        os.makedirs(os.path.dirname(subbucket_yml_path), exist_ok=True)
        with open(subbucket_yml_path, "w") as f:
            yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
        logger.info(
            f"Generated Buildkite pipeline YAML for sub-bucket [{start}, {end}) saved to {subbucket_yml_path} in Docker"
        )

    def generate_buildkite_pipeline(self, buckets: list[tuple[int, int]],
                                    storage_manager) -> str:
        """Generate the Buildkite pipeline YAML for the given tuning buckets.

        The Buildkite pipeline YAML will be generated in the format of:
        steps:
          - label: "Measure latency for cases [begin_case_id, end_case_id)"
            command: "python -m tools.kernel.tuner.v1.kernel_tuner_runner\
                      --case_set_id=CASE_SET_ID\
                      --run_id=RUN_ID\
                      --begin_case_id=BEGIN_CASE_ID\
                      --end_case_id=END_CASE_ID\
                      <OTHER FLAGS DEFINED IN kernel_tuner_flags.py>

        Args:
            buckets: List of (begin_case_id, end_case_id) tuples.
            storage_manager: Storage manager for creating bucket records.
        """
        output_path = "/tmp/kernel_tuning/generated_pipeline.yml"
        safe_remove_files(output_path)
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
            storage_manager.create_bucket_for_run(
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

    # NOTE: _evaluate_single_case() has been moved to the optimizer layer.
    # It now uses ExecutorProcessManager for subprocess-isolated run() calls.

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
