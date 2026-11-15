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

from tools.kernel.tuner.v1.common.tuner_datatypes import (BucketStatus,
                                                          CaseResult,
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

    def _evaluate_single_case(self,
                              cid,
                              tuning_key,
                              tunable_params,
                              tracker,
                              log_prefix=""):
        """Evaluates a single tuning case via the executor subprocess.

        Performs warmup + measurement, saves result to storage, and
        updates the tracker.

        Returns:
            tuple (status, average_latency_us). On failure, average_latency_us is 0.
        """
        tuner = self.tuner
        storage_manager = self.storage_manager
        executor_mgr = self.executor_mgr
        worker_id = tuner.worker_id
        begin_case_perf = time.perf_counter_ns()

        # --- Warmup (1 iteration) ---
        status, warmup_ns, _ = executor_mgr.execute_run(tuning_key,
                                                        tunable_params,
                                                        iters=1)
        if status != TuningStatus.SUCCESS:
            logger.warning(
                f"{log_prefix}Case {cid} failed during warmup with status "
                f"{status} for params {tunable_params}.")
            storage_manager.save_result(
                CaseResult(
                    case_set_id=tuner.run_config.case_set_id,
                    run_id=tuner.run_config.run_id,
                    case_id=cid,
                    processed_status=status.value,
                    worker_id=worker_id,
                    latency=0,
                    warmup_time=0,
                    total_time=0,
                    processed_at=storage_manager.get_timestamp_sec(),
                    tpu=tuner.run_config.tpu_queue_multi,
                ))
            tracker.record(cid, tuning_key, tunable_params, status)
            return status, 0

        warmup_us = int(warmup_ns // 1000)

        # --- Measurement Run ---
        use_xprof = tuner.tuner_config.jit_kernel_pattern is not None
        status, avg_latency_ns, _ = executor_mgr.execute_run(
            tuning_key,
            tunable_params,
            iters=tuner._measurement_iters,
            use_xprof=use_xprof)

        if status != TuningStatus.SUCCESS:
            logger.warning(
                f"{log_prefix}Case {cid} failed during measurement run with "
                f"status {status} for params {tunable_params}.")
            storage_manager.save_result(
                CaseResult(
                    case_set_id=tuner.run_config.case_set_id,
                    run_id=tuner.run_config.run_id,
                    case_id=cid,
                    processed_status=status.value,
                    worker_id=worker_id,
                    latency=0,
                    warmup_time=warmup_us,
                    total_time=0,
                    processed_at=storage_manager.get_timestamp_sec(),
                    tpu=tuner.run_config.tpu_queue_multi,
                ))
            tracker.record(cid, tuning_key, tunable_params, status)
            return status, 0

        # --- Extract Latency (xprof or timer) ---
        if use_xprof:
            from tools.kernel.tuner.v1.common.utils import \
                find_events_by_pattern
            measurement_iters = tuner._measurement_iters
            jit_kernel_pattern = tuner._resolve_kernel_pattern(tuning_key)
            # xprof dir is shared via filesystem with the executor
            matching_events, average_latency_us = find_events_by_pattern(
                tuner.xprof_dir, jit_kernel_pattern)
            if len(matching_events) != measurement_iters:
                msg = (
                    f"{log_prefix}Expected {measurement_iters} matching events "
                    f"for pattern '{jit_kernel_pattern}' in xprof, but found "
                    f"{len(matching_events)}. Profiling/pattern-matching failure."
                )
                logger.error(msg)
                storage_manager.save_result(
                    CaseResult(
                        case_set_id=tuner.run_config.case_set_id,
                        run_id=tuner.run_config.run_id,
                        case_id=cid,
                        processed_status=TuningStatus.XPROF_MEASUREMENT_ERROR.
                        value,
                        worker_id=worker_id,
                        latency=0,
                        warmup_time=warmup_us,
                        total_time=0,
                        processed_at=storage_manager.get_timestamp_sec(),
                        tpu=tuner.run_config.tpu_queue_multi,
                    ))
                tracker.record(cid, tuning_key, tunable_params,
                               TuningStatus.XPROF_MEASUREMENT_ERROR)
                return TuningStatus.XPROF_MEASUREMENT_ERROR, 0
        else:
            average_latency_us = int(avg_latency_ns // 1000)

        total_time_us = int((time.perf_counter_ns() - begin_case_perf) // 1000)

        storage_manager.save_result(
            CaseResult(
                case_set_id=tuner.run_config.case_set_id,
                run_id=tuner.run_config.run_id,
                case_id=cid,
                processed_status=TuningStatus.SUCCESS.value,
                worker_id=worker_id,
                latency=average_latency_us,
                warmup_time=warmup_us,
                total_time=total_time_us,
                processed_at=storage_manager.get_timestamp_sec(),
                tpu=tuner.run_config.tpu_queue_multi,
            ))
        tracker.record(cid, tuning_key, tunable_params, TuningStatus.SUCCESS)

        return TuningStatus.SUCCESS, average_latency_us

    # TODO: This function should take a time_budget parameter and yield based on this
    # time_budget so the worker thread can centralize/orchestrate with the jobs.
    def measure_latency(self,
                        begin_case_id: int,
                        end_case_id: int,
                        bucket_id: int | None = None) -> int:
        from tools.kernel.tuner.v1.common.kernel_tuner_base import \
            ProcessedCasesTracker

        tuner = self.tuner
        storage_manager = self.storage_manager
        worker_id = tuner.worker_id
        if bucket_id is None:
            bucket_id = (begin_case_id if tuner.use_bayesian_optimization else
                         begin_case_id // tuner.run_config.job_bucket_size)
        logger.info(
            f"Worker [{worker_id}] Claimed CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, Bucket {bucket_id} ({begin_case_id}-{end_case_id}) for processing."
        )

        tracker = ProcessedCasesTracker(storage_manager, tuner.tuner_config,
                                        tuner.run_config, begin_case_id,
                                        end_case_id)
        bucket_cases = storage_manager.get_bucket_configs(
            tuner.run_config.case_set_id, begin_case_id, end_case_id - 1)
        assert end_case_id - begin_case_id == len(
            bucket_cases
        ), f"The number of cases in the bucket ({len(bucket_cases)}) does not match the expected number of cases ({end_case_id - begin_case_id}). This should not happen as the cases should have been generated and stored in the storage manager before."

        storage_manager.update_bucket_status(tuner.run_config.case_set_id,
                                             tuner.run_config.run_id,
                                             bucket_id,
                                             BucketStatus.IN_PROGRESS)
        bucket_start_perf = time.perf_counter()
        last_processed_case_id = begin_case_id - 1
        try:
            for cid in range(begin_case_id, end_case_id):
                time_elapsed_minutes = (time.perf_counter() -
                                        bucket_start_perf) / 60
                logger.info(
                    f"Worker [{worker_id}] Processing CaseId: {cid} in Bucket {bucket_id}, [{begin_case_id}-{end_case_id}) @ time {time_elapsed_minutes:.2f} minutes."
                )
                if not tuner.run_config.run_locally and (
                        time_elapsed_minutes
                        > tuner.run_config.max_execution_minutes
                ) and not tuner.use_bayesian_optimization:
                    logger.info(
                        f"Worker [{worker_id}] has been processing bucket {bucket_id} for {time_elapsed_minutes:.2f} minutes, which exceeds the limit of {tuner.run_config.max_execution_minutes} minutes. Stopping processing more cases in this bucket to allow other jobs(like CICD jobs) in the queue to proceed."
                    )
                    parent_step_key = f'{tuner.tuner_config.kernel_tuner_name}_{tuner.run_config.case_set_id}_{tuner.run_config.run_id}_{begin_case_id}_{end_case_id}'
                    tuner.generate_buildkite_pipeline_subbucket(
                        cid, end_case_id, parent_step_key=parent_step_key)
                    storage_manager.update_bucket_status(
                        tuner.run_config.case_set_id, tuner.run_config.run_id,
                        bucket_id, BucketStatus.YIELDED)
                    break
                last_processed_case_id = cid
                if cid in tracker:
                    continue
                _, _, case_key_value = bucket_cases[cid]
                tuning_case = TuningCase.from_string(
                    case_key_value, tuner.tuner_config.tuning_key_class,
                    tuner.tuner_config.tunable_params_class)
                tuning_key, tunable_params, _ = tuning_case.tuning_key, tuning_case.tunable_params, tuning_case.is_baseline

                # check whether tuning_key is same as last one and if last one is OOM, then we can skip
                if tracker.is_oom_expected(tuning_key, tunable_params):
                    logger.warning(
                        f"Skipping CaseId {cid} with tuning key {tuning_key} and tunable params {tunable_params} because it is expected to fail with OOM based on previous cases."
                    )
                    storage_manager.save_result(
                        CaseResult(
                            case_set_id=tuner.run_config.case_set_id,
                            run_id=tuner.run_config.run_id,
                            case_id=cid,
                            processed_status=TuningStatus.SKIPPED.value,
                            worker_id=worker_id,
                            latency=0,
                            warmup_time=0,
                            total_time=0,
                            processed_at=storage_manager.get_timestamp_sec(),
                            tpu=tuner.run_config.tpu_queue_multi,
                        ))
                    tracker.record(cid, tuning_key, tunable_params,
                                   TuningStatus.SKIPPED)
                    continue

                status, average_latency_us = self._evaluate_single_case(
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
            if last_processed_case_id == end_case_id - 1:
                storage_manager.update_bucket_status(
                    tuner.run_config.case_set_id, tuner.run_config.run_id,
                    bucket_id, BucketStatus.COMPLETED)
        except Exception as e:
            logger.error(
                f"Error in sweeping for CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, Bucket {bucket_id}: {e}",
                exc_info=True,
            )
            storage_manager.update_bucket_status(
                tuner.run_config.case_set_id,
                tuner.run_config.run_id,
                bucket_id,
                BucketStatus.FAILED,
            )
            raise
        finally:
            # At this point, the bucket status should be COMPLETED, ERROR, or YIELDED.
            bucket_total_time_us = int(
                (time.perf_counter() - bucket_start_perf) * 1_000_000)
            storage_manager.add_bucket_processed_time_us(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                bucket_id, bucket_total_time_us)
            logger.info(
                f"Worker [{worker_id}] Completed Bucket {bucket_id} [{begin_case_id}-{last_processed_case_id + 1}) for CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}. Total time: {bucket_total_time_us/1e6:.2f}s."
            )
            storage_manager.flush_results()

        return last_processed_case_id + 1
