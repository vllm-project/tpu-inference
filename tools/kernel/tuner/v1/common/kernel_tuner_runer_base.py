from dataclasses import dataclass, fields, asdict
import json
import time
from datetime import datetime
import logging
from absl import flags
from enum import Enum

from tools.kernel.tuner.v1.common.storage_manager import StorageManager
from tools.kernel.tuner.v1.common.utils import get_host_ip
from google.cloud import spanner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_DEBUG = flags.DEFINE_bool('debug', False, 'If true, prints results after each case iteration.')
_WORKER_ID = flags.DEFINE_string('worker_id', get_host_ip(), 'The worker id')

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

class CaseBase:
    def __init__(self, tuning_key: TuningKey, tunable_params: TunableParams):
        self.tuning_key = tuning_key
        self.tunable_params = tunable_params

    def __str__(self):
        return json.dumps({'tuning_key': asdict(self.tuning_key), 'tunable_params': asdict(self.tunable_params)})

    @classmethod
    def from_string(self, string, tuning_key_class, tunable_params_class):
        data = json.loads(string)
        self.tuning_key = tuning_key_class(**data['tuning_key'])
        self.tunable_params = tunable_params_class(**data['tunable_params'])
        return self.tuning_key, self.tunable_params

class KernelTunerBase:
    def __init__(self, tuning_key_class, tunable_params_class, storage_manager: StorageManager):
        self.tuning_key_class = tuning_key_class
        self.tunable_params_class = tunable_params_class
        self.storage_manager = storage_manager
        self._KERNEL_INPUTS_CACHE = {}
        self._TUNING_KEY = None

    def init_case_set(self, case_set_id: str, desc: str):
        """Initialize the case set with the given case_set_id and description. This will be called when the caseset_id is new or the caseset_id is not specified.

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            desc: A description for this case set, which will be persisted in local file or database using storage_management module.

        """
        if case_set_id is None:
            case_set_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # check case_set_id exists in storage manager, if not exist, create a new case set with the given case_set_id and desc.
        # if exist, check whether the desc is the same as the existing one, if not, raise an error.
        if self.storage_manager.case_set_id_exists(case_set_id):
            existing_desc = self.storage_manager.get_case_set_desc(case_set_id)
            if existing_desc != desc:
                raise ValueError(f"CaseSetId {case_set_id} already exists with a different description. Existing desc: {existing_desc}, new desc: {desc}. Please use a different CaseSetId or update the description to match the existing one.")
            else:
                logger.info(f"CaseSetId {case_set_id} already exists with the same description. Proceeding with the existing case set.")
        else:
            self.storage_manager.init_case_set(case_set_id, scan_space=0, desc=desc)
            logger.info(f"Initialized new CaseSet with ID: {case_set_id} and description: {desc}")
    
    def generate_cases(self) -> list[CaseBase]:
        """Generate the cases for the given case_set_id. This will be called when the caseset_id is new or the caseset_id is not specified.

        Args:
            case_set_id: Identifies a set of tuning cases. If specified when running the tuning
                pipeline, the caseset will only be regenerated when the caseset_id changes.
            desc: A description for this case set, which will be persisted in local file or database using storage_management module."""
        raise NotImplementedError("Specific kernel should implement this to generate the cases for the given case_set_id and desc, and return a list of CaseBase objects representing the tuning cases.")
    
    def generate_tuning_jobs(self, bucket_size: int = 100) -> list[tuple[int, int]]:
        """Generate the tuning jobs for the given cases. Each tuning job is represented as a tuple of (begin_case_id, end_case_id), where the tuning job will process the cases with case_id in [begin_case_id, end_case_id).

        Args:
            cases: A list of CaseBase objects representing the tuning cases.
            bucket_size: The number of cases to be processed in each tuning job.

        Returns:
            A list of tuples, where each tuple is (begin_case_id, end_case_id) representing a tuning job.
        """
        total_cases = len(self.generate_cases())
        buckets = [[i, min(i + bucket_size, total_cases)] for i in range(0, total_cases, bucket_size)]
        return buckets
    
    def generate_buildkite_pipeline(self, case_set_id: str, run_id: str, desc: str) -> str:
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
        try:
            self.init_case_set(case_set_id, desc=desc)
        except Exception as e:
            logger.error(f"Error initializing case set {case_set_id}: {e}")
            raise e
        raise NotImplementedError("Specific kernel should implement this to generate the Buildkite pipeline for the given case_set_id and run_id, and return a string representing the Buildkite pipeline configuration in YAML format.")

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
        raise NotImplementedError("Specific kernel should implement this to generate the inputs to kernel based on the tuning key with caching.")
        
    def run(self, tuning_key: TuningKey, tunable_params: TunableParams, iters: int) -> list[TuningStatus, int, int]:
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
        raise NotImplementedError("Specific kernel should implement this to call the kernl with the inputs from generate_inputs")

    def measure_latency(self, caseset_id, run_id, begin_case_id, end_case_id):
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
        bucket_id = f"{caseset_id}_{run_id}_{begin_case_id}_{end_case_id}"
        logger.info(f"[{_WORKER_ID.value}] Claimed CaseSetId: {caseset_id}, RunId: {run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing.")
        self.storage_manager.mark_bucket_in_progress(caseset_id, run_id, bucket_id)

        processed_ids = self.storage_manager.get_already_processed_ids(caseset_id, run_id, begin_case_id, end_case_id)
        all_configs = self.storage_manager.get_bucket_configs(caseset_id, begin_case_id, end_case_id)

        bucket_start_perf = time.perf_counter()
        results_buffer = []
        for cid in range(begin_case_id, end_case_id):
            if cid in processed_ids: continue
            config = all_configs.get(cid)
            if not config: continue
            _, _, case_key_value = config
            tuning_key, tunable_params = CaseBase.from_string(case_key_value, self.tuning_key_class, self.tunable_params_class)

            begin_case_id_time = time.perf_counter_ns()
            # status can be SUCCESS, FAILED_OOM, UNKNOWN_ERROR.
            status, warmup, _ = self.run(tuning_key, tunable_params, iters=1)
            if status != 'SUCCESS':
                results_buffer.append((caseset_id, run_id, cid, status, _WORKER_ID.value, 0, 0, spanner.COMMIT_TIMESTAMP))
                logger.warning(f"Case {cid} failed during warmup with status: {status}. Skipping to next case.")
                continue
            warmup_us = warmup // 1000

            status, average_latency, _ = self.run(tuning_key, tunable_params, iters=10)
            end_time = time.perf_counter_ns()
            total_time = end_time - begin_case_id_time
            if status != 'SUCCESS':
                results_buffer.append((caseset_id, run_id, cid, status, _WORKER_ID.value, warmup_us, 0, spanner.COMMIT_TIMESTAMP))
                logger.warning(f"Case {cid} failed during main run with status: {status}. Total time spent: {total_time/1e9:.2f}s.")
                continue
            
            average_latency_us = average_latency // 1000
            total_time_us = total_time // 1000
            results_buffer.append(
                (caseset_id, run_id, cid, status, _WORKER_ID.value, average_latency_us, warmup_us, total_time_us, spanner.COMMIT_TIMESTAMP)
            )

            if _DEBUG.value:
                logger.info(f"Case {cid} completed with AvgLat={average_latency_us}us, Warmup={warmup_us}us, Total={total_time_us}us")

            if len(results_buffer) >= 10:
                self.storage_manager.save_results_batch(results_buffer)
                results_buffer = []

        self.storage_manager.save_results_batch(results_buffer)
        
        bucket_total_time_us = int((time.perf_counter() - bucket_start_perf) * 1_000_000)
        self.storage_manager.mark_bucket_completed(caseset_id, run_id, bucket_id, bucket_total_time_us)