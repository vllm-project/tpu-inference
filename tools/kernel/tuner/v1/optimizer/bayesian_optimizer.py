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

try:
    import optuna
except ImportError:
    optuna = None

from tools.kernel.tuner.v1.common.tuner_datatypes import (CaseResult,
                                                          TunableParams,
                                                          TuningCase,
                                                          TuningStatus)
from tools.kernel.tuner.v1.optimizer.base_optimizer import TuningOptimizer

logger = logging.getLogger(__name__)


class RelativeEarlyStoppingCallback:
    """Callback to stop Optuna optimization when relative latency improvement is below min_delta_ratio for patience trials."""

    def __init__(self, patience: int = 10, min_delta_ratio: float = 0.05):
        self.patience = patience
        self.min_delta_ratio = min_delta_ratio
        self.best_value = None
        self.no_improvement_count = 0
        self.stopped = False

    def __call__(self, study: "optuna.Study",
                 trial: "optuna.trial.FrozenTrial") -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        current_val = trial.value
        if self.best_value is None:
            self.best_value = current_val
            return

        if self.best_value > 0 and (self.best_value - current_val) / float(
                self.best_value) >= self.min_delta_ratio:
            self.best_value = current_val
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            logger.info(
                f"Optuna early stopping triggered: Best latency ({self.best_value:.1f}us) "
                f"did not improve by >= {self.min_delta_ratio:.1%} for {self.patience} "
                f"consecutive trials. Stopping study.")
            self.stopped = True
            study.stop()


class BayesianOptimizer(TuningOptimizer):
    """Optimizer strategy using Optuna Bayesian Optimization (TPE sampler)."""

    def generate_tuning_jobs(self, cases: list) -> list[tuple[int, int]]:
        tuner = self.tuner
        buckets = []
        previous_tuning_key = None
        for idx, row in enumerate(cases):
            case_key_value = row[1]
            tuning_case = TuningCase.from_string(
                case_key_value, tuner.tuner_config.tuning_key_class,
                tuner.tuner_config.tunable_params_class)

            if previous_tuning_key is None or tuning_case.tuning_key != previous_tuning_key:
                buckets.append((idx, idx + 1))
                previous_tuning_key = tuning_case.tuning_key
            else:
                buckets[-1] = (buckets[-1][0], idx + 1)
        return buckets

    def measure_latency(self, begin_case_id: int, end_case_id: int) -> None:
        import optuna

        from tools.kernel.tuner.v1.common.kernel_tuner_base import \
            ProcessedCasesTracker
        from tools.kernel.tuner.v1.optimizer.sweep_optimizer import \
            SweepOptimizer
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        tuner = self.tuner
        worker_id = tuner.worker_id

        all_configs = tuner.storage_manager.get_bucket_configs(
            tuner.run_config.case_set_id, begin_case_id, end_case_id)

        if not all_configs:
            logger.warning(
                f"No configs found for [{begin_case_id}, {end_case_id}). "
                "Nothing to optimize.")
            bucket_id = begin_case_id
            tuner.storage_manager.mark_bucket_completed(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                bucket_id)
            return

        first_cid = min(all_configs.keys())
        _, _, first_case_kv = all_configs[first_cid]
        first_case = TuningCase.from_string(
            first_case_kv, tuner.tuner_config.tuning_key_class,
            tuner.tuner_config.tunable_params_class)
        tuning_key = first_case.tuning_key

        search_space = tuner.get_search_space(tuning_key)
        if not search_space:
            logger.warning(
                f"get_search_space returned empty dict for tuning key {tuning_key}. "
                "Cannot run Bayesian optimization; falling back to sequential sweep."
            )
            from tools.kernel.tuner.v1.optimizer.sweep_optimizer import \
                SweepOptimizer
            SweepOptimizer(tuner).measure_latency(begin_case_id, end_case_id)
            return

        # inital bucket tuning status
        bucket_id = begin_case_id
        logger.info(
            f"Worker [{worker_id}] Starting Bayesian optimization for "
            f"CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, "
            f"Bucket begin={begin_case_id} end={end_case_id}.")
        tuner.storage_manager.mark_bucket_in_progress(
            tuner.run_config.case_set_id, tuner.run_config.run_id, bucket_id)
        tracker = ProcessedCasesTracker(
            tuner.storage_manager.get_already_processed_ids(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                begin_case_id, end_case_id))

        params_to_case_id: dict[TunableParams, int] = {}
        for cid, (_, _, case_kv) in all_configs.items():
            tc = TuningCase.from_string(
                case_kv, tuner.tuner_config.tuning_key_class,
                tuner.tuner_config.tunable_params_class)
            params_to_case_id[tc.tunable_params] = cid

        if (tuner.tuner_config.min_cases_for_bayesian > 0
                and len(params_to_case_id)
                < tuner.tuner_config.min_cases_for_bayesian):
            logger.warning(
                f"Tuning key {tuning_key} search space contains {len(params_to_case_id)} cases, "
                f"which is less than min_cases_for_bayesian threshold ({tuner.tuner_config.min_cases_for_bayesian}). "
                "Falling back to sequential sweep.")
            from tools.kernel.tuner.v1.optimizer.sweep_optimizer import \
                SweepOptimizer
            SweepOptimizer(tuner).measure_latency(begin_case_id, end_case_id)
            return

        bucket_start_perf = time.perf_counter()

        int_param_sorted: dict[str, list] = {}
        for pname, pvalues in search_space.items():
            if pvalues and all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in pvalues):
                int_param_sorted[pname] = sorted(pvalues)

        def objective(trial: optuna.Trial) -> float:
            suggested: dict = {}
            for param_name, param_values in search_space.items():
                if param_name in int_param_sorted:
                    sorted_vals = int_param_sorted[param_name]
                    idx = trial.suggest_int(param_name, 0,
                                            len(sorted_vals) - 1)
                    suggested[param_name] = sorted_vals[idx]
                else:
                    suggested[param_name] = trial.suggest_categorical(
                        param_name, param_values)

            tunable_params = tuner.tuner_config.tunable_params_class(
                **suggested)
            cid = params_to_case_id.get(tunable_params, None)
            if cid is None:
                logger.warning(
                    f'Invalid TunableParam {tunable_params} is suggested for TuningKey {tuning_key}'
                )
                raise optuna.exceptions.TrialPruned()

            if tracker.is_oom_expected(tuning_key, tunable_params):
                logger.warning(
                    f"Trial {trial.number}: Skipping {tunable_params} "
                    "due to expected OOM from a smaller configuration.")
                if cid not in tracker:
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
                raise optuna.exceptions.TrialPruned()

            if cid in tracker:
                logger.info(
                    f'{cid=} {tuning_key=} {tunable_params=} is already processed; pruning duplicate Optuna trial.'
                )
                raise optuna.exceptions.TrialPruned()

            status, average_latency_us = tuner._evaluate_single_case(
                cid=cid,
                tuning_key=tuning_key,
                tunable_params=tunable_params,
                tracker=tracker,
                log_prefix=f"Trial {trial.number}: ",
            )

            if status != TuningStatus.SUCCESS:
                raise optuna.exceptions.TrialPruned()

            logger.info(f"Trial {trial.number}: params={suggested}, "
                        f"latency={average_latency_us}us")
            return float(average_latency_us)

        callbacks = []
        early_stopping_cb = None
        if (tuner.tuner_config.bayesian_early_stopping_patience is not None
                and tuner.tuner_config.bayesian_early_stopping_patience > 0):
            early_stopping_cb = RelativeEarlyStoppingCallback(
                patience=tuner.tuner_config.bayesian_early_stopping_patience,
                min_delta_ratio=tuner.tuner_config.
                bayesian_early_stopping_min_delta_ratio,
            )
            callbacks.append(early_stopping_cb)

        target_trials = min(tuner.tuner_config.n_bayesian_trials,
                            len(params_to_case_id))

        study = optuna.create_study(direction="minimize")
        try:
            while True:
                completed_trials = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if (len(completed_trials) >= target_trials
                        or (early_stopping_cb and early_stopping_cb.stopped)
                        or len(study.trials) >= 2 * target_trials):
                    logger.info(
                        f"len(study.trials): {len(study.trials)}, "
                        f"len(completed_trials): {len(completed_trials)}, "
                        f"target_trials: {target_trials}, "
                        f"early_stopping_cb.stopped: {early_stopping_cb.stopped if early_stopping_cb else 'None'}"
                    )
                    break
                study.optimize(
                    objective,
                    n_trials=1,
                    callbacks=callbacks,
                )
            tuner.storage_manager.mark_bucket_completed(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                bucket_id)
        except Exception as e:
            logger.error(
                f"Error in Bayesian optimization for CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, Bucket {bucket_id}: {e}",
                exc_info=True,
            )
            raise
        finally:
            tuner.storage_manager.flush_results()

            bucket_total_time_us = int(
                (time.perf_counter() - bucket_start_perf) * 1_000_000)
            tuner.storage_manager.add_bucket_processed_time_us(
                tuner.run_config.case_set_id, tuner.run_config.run_id,
                bucket_id, bucket_total_time_us)

        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed_trials:
            best_info = (f"Best latency: {study.best_value}us, "
                         f"best params: {study.best_params}. ")
        else:
            best_info = "No trials completed successfully. "
        logger.info(
            f"Worker [{worker_id}] Bayesian optimization complete for "
            f"CaseSetId: {tuner.run_config.case_set_id}, RunId: {tuner.run_config.run_id}, "
            f"tuning key: {tuning_key}. "
            f"Trials: {len(completed_trials)} completed / "
            f"{tuner.tuner_config.n_bayesian_trials} requested. "
            f"{best_info}"
            f"Total time: {bucket_total_time_us/1e6:.2f}s.")
