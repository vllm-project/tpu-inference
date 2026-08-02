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

import dataclasses
import unittest
from unittest import mock

import optuna

from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TunerConfig,
                                                          TuningCase,
                                                          TuningStatus)
from tools.kernel.tuner.v1.optimizer import (BayesianOptimizer, SweepOptimizer,
                                             TuningOptimizer)
from tools.kernel.tuner.v1.optimizer.bayesian_optimizer import \
    RelativeEarlyStoppingCallback


@dataclasses.dataclass(frozen=True)
class MockKey:
    key_id: int


@dataclasses.dataclass(frozen=True)
class MockParams:
    p1: int
    p2: int

    def __ge__(self, other) -> bool:
        return self.p1 >= other.p1 and self.p2 >= other.p2

    def __le__(self, other) -> bool:
        return self.p1 <= other.p1 and self.p2 <= other.p2


class TestOptimizerModule(unittest.TestCase):

    def test_optimizer_instantiation(self):
        mock_tuner = mock.MagicMock()
        mock_tuner.run_config.job_bucket_size = 5

        sweep_opt = SweepOptimizer(mock_tuner)
        self.assertIsInstance(sweep_opt, TuningOptimizer)

        bo_opt = BayesianOptimizer(mock_tuner)
        self.assertIsInstance(bo_opt, TuningOptimizer)

    def test_sweep_optimizer_generate_tuning_jobs(self):
        mock_tuner = mock.MagicMock()
        mock_tuner.run_config.job_bucket_size = 4

        cases = [(i, f"case_{i}") for i in range(10)]
        sweep_opt = SweepOptimizer(mock_tuner)
        jobs = sweep_opt.generate_tuning_jobs(cases)

        self.assertEqual(jobs, [(0, 4), (4, 8), (8, 10)])

    def test_relative_early_stopping_callback(self):
        cb = RelativeEarlyStoppingCallback(patience=3, min_delta_ratio=0.10)
        mock_study = mock.MagicMock()

        # Helper to create a fake completed trial
        def make_trial(val):
            t = mock.MagicMock()
            t.state = optuna.trial.TrialState.COMPLETE
            t.value = val
            return t

        # Trial 1: baseline = 100.0
        cb(mock_study, make_trial(100.0))
        self.assertFalse(cb.stopped)
        self.assertEqual(cb.no_improvement_count, 0)

        # Trial 2: 95.0 (5% improvement, required 10%) -> count=1
        cb(mock_study, make_trial(95.0))
        self.assertFalse(cb.stopped)
        self.assertEqual(cb.no_improvement_count, 1)

        # Trial 3: 94.0 -> count=2
        cb(mock_study, make_trial(94.0))
        self.assertFalse(cb.stopped)
        self.assertEqual(cb.no_improvement_count, 2)

        # Trial 4: 93.0 -> count=3 -> triggers early stopping
        cb(mock_study, make_trial(93.0))
        self.assertTrue(cb.stopped)
        mock_study.stop.assert_called_once()

    @mock.patch(
        "tools.kernel.tuner.v1.optimizer.sweep_optimizer.SweepOptimizer.measure_latency"
    )
    def test_bayesian_optimizer_fallback_empty_search_space(
            self, mock_sweep_measure):
        mock_tuner = mock.MagicMock()
        mock_tuner.worker_id = "test_worker"
        mock_tuner.run_config = RunConfig(
            case_set_id="cs",
            run_id="r1",
            case_set_desc="desc",
            tpu_version="v4",
            tpu_cores=8,
            tpu_queue_multi="queue",
        )
        mock_tuner.tuner_config = TunerConfig(
            tuning_key_class=MockKey,
            tunable_params_class=MockParams,
            kernel_tuner_name="mock_tuner",
            support_bayesian_optimization=True,
        )

        tc = TuningCase(MockKey(1), MockParams(1, 1))
        mock_tuner.storage_manager.get_already_processed_ids.return_value = set(
        )
        mock_tuner.storage_manager.get_bucket_configs.return_value = {
            0: ("cs", 0, str(tc))
        }
        mock_tuner.get_search_space.return_value = {}

        bo_opt = BayesianOptimizer(mock_tuner)
        bo_opt.measure_latency(0, 1)

        mock_sweep_measure.assert_called_once_with(0, 1)

    @mock.patch(
        "tools.kernel.tuner.v1.optimizer.sweep_optimizer.SweepOptimizer.measure_latency"
    )
    def test_bayesian_optimizer_fallback_min_cases(self, mock_sweep_measure):
        mock_tuner = mock.MagicMock()
        mock_tuner.worker_id = "test_worker"
        mock_tuner.run_config = RunConfig(
            case_set_id="cs",
            run_id="r1",
            case_set_desc="desc",
            tpu_version="v4",
            tpu_cores=8,
            tpu_queue_multi="queue",
        )
        mock_tuner.tuner_config = TunerConfig(
            tuning_key_class=MockKey,
            tunable_params_class=MockParams,
            kernel_tuner_name="mock_tuner",
            support_bayesian_optimization=True,
            min_cases_for_bayesian=20,
        )

        mock_tuner.storage_manager.get_already_processed_ids.return_value = set(
        )
        mock_tuner.storage_manager.get_bucket_configs.return_value = {
            i: ("cs", i, str(TuningCase(MockKey(1), MockParams(i, 1))))
            for i in range(5)
        }
        mock_tuner.get_search_space.return_value = {
            "p1": list(range(5)),
            "p2": [1]
        }

        bo_opt = BayesianOptimizer(mock_tuner)
        bo_opt.measure_latency(0, 5)

        mock_sweep_measure.assert_called_once_with(0, 5)

    def test_bayesian_optimizer_convergence(self):
        # Synthetic objective function: min latency at p1=3, p2=4
        def synthetic_kernel(p1, p2):
            return float((p1 - 3)**2 + (p2 - 4)**2 + 10)

        mock_tuner = mock.MagicMock()
        mock_tuner.worker_id = "test_worker"
        mock_tuner.run_config = RunConfig(
            case_set_id="cs_syn",
            run_id="r_syn",
            case_set_desc="desc",
            tpu_version="v4",
            tpu_cores=8,
            tpu_queue_multi="queue",
        )
        mock_tuner.tuner_config = TunerConfig(
            tuning_key_class=MockKey,
            tunable_params_class=MockParams,
            kernel_tuner_name="mock_tuner",
            support_bayesian_optimization=True,
            n_bayesian_trials=20,
            min_cases_for_bayesian=0,
        )

        search_space = {
            "p1": list(range(1, 20)),
            "p2": list(range(1, 20)),
        }
        mock_tuner.get_search_space.return_value = search_space

        all_configs = {}
        cid = 0
        for p1 in range(1, 20):
            for p2 in range(1, 20):
                tc = TuningCase(MockKey(1), MockParams(p1, p2))
                all_configs[cid] = ("cs_syn", cid, str(tc))
                cid += 1

        mock_tuner.storage_manager.get_already_processed_ids.return_value = set(
        )
        mock_tuner.storage_manager.get_bucket_configs.return_value = all_configs
        mock_tuner.storage_manager.get_timestamp_sec.return_value = 1000

        evaluations = []

        def mock_eval(cid, tuning_key, tunable_params, tracker, log_prefix=""):
            lat = synthetic_kernel(tunable_params.p1, tunable_params.p2)
            evaluations.append((tunable_params, lat))
            tracker.record(cid, tuning_key, tunable_params,
                           TuningStatus.SUCCESS)
            return TuningStatus.SUCCESS, lat

        mock_tuner._evaluate_single_case.side_effect = mock_eval

        bo_opt = BayesianOptimizer(mock_tuner)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        with mock.patch("optuna.create_study") as mock_create_study:
            mock_create_study.return_value = study
            bo_opt.measure_latency(0, len(all_configs))

        # Check that optimum (3, 4) with latency 10.0 was evaluated and best value is <= 12.0
        best_eval = min(evaluations, key=lambda x: x[1])
        self.assertLessEqual(best_eval[1], 12.0)
        self.assertLessEqual(best_eval[0].p1, 5)
        self.assertLessEqual(best_eval[0].p2, 5)


if __name__ == "__main__":
    unittest.main()
