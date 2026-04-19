from dataclasses import dataclass, fields, asdict
import json
import time
import logging
from absl import flags

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

class CaseBase:
    def __init__(self, tuning_key_class: TuningKey, tunable_params_class: TunableParams):
        self.tuning_key_class = tuning_key_class
        self.tunable_params_class = tunable_params_class

    @classmethod
    def to_string(self, tuning_key, tunable_params):
        return json.dumps({'tuning_key': asdict(tuning_key), 'tunable_params': asdict(tunable_params)})

    @classmethod
    def from_string(self, string):
        data = json.loads(string)
        tuning_key = self.tuning_key_class(**data['tuning_key'])
        tunable_params = self.tunable_params_class(**data['tunable_params'])
        return tuning_key, tunable_params

class KernelTunerRunnerBase:
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager

    def setup_inputs(self, tuning_key: TuningKey):
        raise NotImplementedError("Specific kernel should implement this to initialize the inputs to kernel based on the tuning key")

    def run(self, tuning_key: TuningKey, tunable_params: TunableParams, iters: int) -> [int, int]: # type: ignore
        # return an integer represent the average kernel run time and the total time
        raise NotImplementedError("Specific kernel should implement this to call the kernl with the inputs from setup_inputs")

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

            begin_case_id_time = time.perf_counter_ns()
            # status can be SUCCESS, FAILED_OOM, UNKNOWN_ERROR.
            status, warmup, _ = self.run(case_key_value, iters=1)
            if status != 'SUCCESS':
                results_buffer.append((caseset_id, run_id, cid, status, _WORKER_ID.value, 0, 0, spanner.COMMIT_TIMESTAMP))
                logger.warning(f"Case {cid} failed during warmup with status: {status}. Skipping to next case.")
                continue
            warmup_us = warmup // 1000

            status, average_latency, _ = self.run(case_key_value,iters=10)
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