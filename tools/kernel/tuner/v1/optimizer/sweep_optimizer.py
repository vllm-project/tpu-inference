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
import time

from tools.kernel.tuner.v1.common.tuner_datatypes import (CaseResult,
                                                          TuningCase,
                                                          TuningStatus)
from tools.kernel.tuner.v1.optimizer.base_optimizer import TuningOptimizer

logger = logging.getLogger(__name__)


class SweepOptimizer(TuningOptimizer):
    """Optimizer strategy that performs sequential full sweeping across parameter combinations."""

    def generate_tuning_jobs(self, cases: list) -> list[tuple[int, int]]:
        total_cases = len(cases)
        bucket_size = self.tuner.run_config.job_bucket_size
        return [(i, min(i + bucket_size, total_cases))
                for i in range(0, total_cases, bucket_size)]

    def measure_latency(self, begin_case_id: int, end_case_id: int) -> None:
        from tools.kernel.tuner.v1.common.kernel_tuner_base import \
            ProcessedCasesTracker

        tuner = self.tuner
        worker_id = tuner.worker_id
        bucket_id = begin_case_id // tuner.run_config.job_bucket_size
        logger.info(
            f"Worker [{worker_id}] Claimed CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing."
        )
        tuner.storage_manager.mark_bucket_in_progress(
            tuner.run_config.case_set_id, tuner.run_config.run_id, bucket_id)

        tracker = ProcessedCasesTracker(
            tuner.storage_manager.get_already_processed_ids(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                begin_case_id, end_case_id))
        all_configs = tuner.storage_manager.get_bucket_configs(
            tuner.run_config.case_set_id, begin_case_id, end_case_id)

        bucket_start_perf = time.perf_counter()
        bucket_fully_processed = True
        last_processed_case_id = begin_case_id - 1
        try:
            for cid in range(begin_case_id, end_case_id):
                time_elapsed_minutes = (time.perf_counter() -
                                        bucket_start_perf) / 60
                logger.info(
                    f"Worker [{worker_id}] Processing CaseId: {cid} in Bucket {bucket_id}, [{begin_case_id}-{end_case_id}) with elapsed time {time_elapsed_minutes:.2f} minutes."
                )
                if not tuner.run_config.run_locally and (
                        time_elapsed_minutes
                        > tuner.run_config.max_execution_minutes
                ) and not tuner.use_bayesian_optimization:
                    logger.warning(
                        f"Worker [{worker_id}] has been processing bucket {bucket_id} for {time_elapsed_minutes:.2f} minutes, which exceeds the limit of {tuner.run_config.max_execution_minutes} minutes. Stopping processing more cases in this bucket to allow other jobs(like CICD jobs) in the queue to proceed."
                    )
                    parent_step_key = f'{tuner.tuner_config.kernel_tuner_name}_{tuner.run_config.case_set_id}_{tuner.run_config.run_id}_{begin_case_id}_{end_case_id}'
                    tuner.generate_buildkite_pipeline_subbucket(
                        cid, end_case_id, parent_step_key=parent_step_key)
                    bucket_fully_processed = False
                    break
                last_processed_case_id = cid
                if cid in tracker:
                    continue
                assert cid in all_configs, f"CaseId {cid} is missing in the configs retrieved from storage manager for CaseSetId {tuner.run_config.case_set_id}. This should not happen as the configs should have been generated and stored in the storage manager before."
                _, _, case_key_value = all_configs[cid]
                tuning_case = TuningCase.from_string(
                    case_key_value, tuner.tuner_config.tuning_key_class,
                    tuner.tuner_config.tunable_params_class)
                tuning_key, tunable_params, _ = tuning_case.tuning_key, tuning_case.tunable_params, tuning_case.is_baseline

                # check whether tuning_key is same as last one and if last one is OOM, then we can skip
                if tracker.is_oom_expected(tuning_key, tunable_params):
                    logger.warning(
                        f"Skipping CaseId {cid} with tuning key {tuning_key} and tunable params {tunable_params} because it is expected to fail with OOM based on previous cases."
                    )
                    tuner.storage_manager.save_result(
                        CaseResult(
                            case_set_id=tuner.run_config.case_set_id,
                            run_id=tuner.run_config.run_id,
                            case_id=cid,
                            processed_status=TuningStatus.SKIPPED.value,
                            worker_id=worker_id,
                            latency=0,
                            warmup_time=0,
                            total_time=0,
                            processed_at=tuner.storage_manager.
                            get_timestamp_sec(),
                            tpu=tuner.run_config.tpu_queue_multi,
                        ))
                    tracker.record(cid, tuning_key, tunable_params,
                                   TuningStatus.SKIPPED)
                    continue

                status, average_latency_us = tuner._evaluate_single_case(
                    cid=cid,
                    tuning_key=tuning_key,
                    tunable_params=tunable_params,
                    tracker=tracker,
                )

                if status == TuningStatus.SUCCESS:
                    source = 'xprof' if tuner.tuner_config.jit_kernel_pattern else 'timer'
                    logger.info(
                        f'Case {cid} average latency is {average_latency_us}us from {source}'
                    )
            if bucket_fully_processed:
                tuner.storage_manager.mark_bucket_completed(
                    tuner.run_config.case_set_id, tuner.run_config.run_id,
                    bucket_id)
        except Exception as e:
            logger.error(
                f"Error in sweeping for CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, Bucket {bucket_id}: {e}",
                exc_info=True,
            )
            raise
        finally:
            bucket_total_time_us = int(
                (time.perf_counter() - bucket_start_perf) * 1_000_000)
            tuner.storage_manager.add_bucket_processed_time_us(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                bucket_id, bucket_total_time_us)
            logger.info(
                f"Worker [{worker_id}] Completed Bucket {bucket_id} [{begin_case_id}-{last_processed_case_id + 1}) for CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}. Total time: {bucket_total_time_us/1e6:.2f}s."
            )
            tuner._cleanup_xprof_dir()
            tuner.storage_manager.flush_results()
