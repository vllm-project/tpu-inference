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

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import tempfile
from dataclasses import dataclass
from unittest import mock

from absl.testing import absltest

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            TuningCase)
from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TunableParams,
                                                          TunerConfig,
                                                          TuningKey,
                                                          TuningStatus)
from tools.kernel.tuner.v1.storage_management.local_db_manager import \
    LocalDbManager


@dataclass(frozen=True)
class MockTuningKey:
    name: str
    size: int


@dataclass(frozen=True)
class MockTunableParams:
    size: int

    def __ge__(self, other: "MockTunableParams") -> bool:
        return self.size >= other.size

    def __le__(self, other: "MockTunableParams") -> bool:
        return self.size <= other.size


class MockKernelTuner(KernelTunerBase):

    def generate_cases(self) -> list[TuningCase]:
        tuning_key = MockTuningKey(name="same_key", size=1)
        return [
            TuningCase(tuning_key=tuning_key,
                       tunable_params=MockTunableParams(size=1),
                       is_baseline=False),
            TuningCase(tuning_key=tuning_key,
                       tunable_params=MockTunableParams(size=2),
                       is_baseline=False),
        ]

    def get_search_space(self, tuning_key: TuningKey) -> dict:
        return {"size": [1, 2]}

    def generate_inputs(self, tuning_key: TuningKey) -> dict:
        return {}

    def run(self, tuning_key: TuningKey, tunable_params: TunableParams,
            iters: int):
        raise NotImplementedError("This method should be mocked in tests")


class KernelTunerBaseTest(absltest.TestCase):

    def test_measure_latency_skips_larger_params_after_oom(self):
        tuner_config = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
        )
        run_config = RunConfig(
            case_set_id="test_case_set",
            run_id="test_run",
            case_set_desc="test description",
            tpu_version="tpu6e",
            tpu_cores=1,
            tpu_queue_multi="test_queue",
            run_locally=True,
            job_bucket_size=100,
        )

        kernel_tuner = MockKernelTuner(
            tuner_config=tuner_config,
            run_config=run_config,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_tuner.storage_manager = LocalDbManager(db_path=tmp_dir)

            with mock.patch.object(MockKernelTuner, "run",
                                   autospec=True) as run_mock:
                run_mock.side_effect = [
                    (TuningStatus.FAILED_OOM, 0, 0),
                ]

                buckets = kernel_tuner._generate_tuning_jobs()
                self.assertEqual(buckets, [(0, 2)])

                kernel_tuner.measure_latency(0, 2)

                results = kernel_tuner.storage_manager._read_table(
                    "CaseResults")
                case_status = {
                    result["CaseId"]: result["ProcessedStatus"]
                    for result in results
                }
                self.assertEqual(
                    case_status, {
                        0: TuningStatus.FAILED_OOM.value,
                        1: TuningStatus.SKIPPED.value
                    })
                self.assertEqual(run_mock.call_count, 1)

                run_mock.assert_called_once_with(
                    kernel_tuner,
                    MockTuningKey(name="same_key", size=1),
                    MockTunableParams(size=1),
                    iters=1,
                )

    def test_measure_latency_runs_both_cases_when_no_oom(self):
        tuner_config = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
        )
        run_config = RunConfig(
            case_set_id="test_case_set_no_oom",
            run_id="test_run_no_oom",
            case_set_desc="test description",
            tpu_version="tpu6e",
            tpu_cores=1,
            tpu_queue_multi="test_queue",
            run_locally=True,
            job_bucket_size=100,
        )

        kernel_tuner = MockKernelTuner(
            tuner_config=tuner_config,
            run_config=run_config,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_tuner.storage_manager = LocalDbManager(db_path=tmp_dir)

            with mock.patch.object(MockKernelTuner, "run",
                                   autospec=True) as run_mock:
                run_mock.side_effect = [
                    (TuningStatus.SUCCESS, 10, 10),
                    (TuningStatus.SUCCESS, 1000, 1000),
                    (TuningStatus.SUCCESS, 10, 10),
                    (TuningStatus.SUCCESS, 1000, 1000),
                ]

                buckets = kernel_tuner._generate_tuning_jobs()
                self.assertEqual(buckets, [(0, 2)])

                kernel_tuner.measure_latency(0, 2)

                results = kernel_tuner.storage_manager._read_table(
                    "CaseResults")
                case_status = {
                    result["CaseId"]: result["ProcessedStatus"]
                    for result in results
                }
                self.assertEqual(case_status, {
                    0: TuningStatus.SUCCESS.value,
                    1: TuningStatus.SUCCESS.value
                })
                self.assertEqual(run_mock.call_count, 4)

                run_mock.assert_any_call(
                    kernel_tuner,
                    MockTuningKey(name="same_key", size=1),
                    MockTunableParams(size=1),
                    iters=1,
                )
                run_mock.assert_any_call(
                    kernel_tuner,
                    MockTuningKey(name="same_key", size=1),
                    MockTunableParams(size=2),
                    iters=1,
                )

    def test_use_bayesian_optimization_flag_combinations(self):
        tc_support = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            support_bayesian_optimization=True,
        )
        tc_no_support = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            support_bayesian_optimization=False,
        )

        rc_enable = RunConfig(
            case_set_id="cs1",
            run_id="r1",
            case_set_desc="desc",
            use_bayesian_optimization=True,
        )
        rc_disable = RunConfig(
            case_set_id="cs2",
            run_id="r2",
            case_set_desc="desc",
            use_bayesian_optimization=False,
        )

        tuner_both = MockKernelTuner(tuner_config=tc_support,
                                     run_config=rc_enable)
        self.assertTrue(tuner_both.use_bayesian_optimization)

        tuner_rc_only = MockKernelTuner(tuner_config=tc_no_support,
                                        run_config=rc_enable)
        self.assertFalse(tuner_rc_only.use_bayesian_optimization)

        tuner_tc_only = MockKernelTuner(tuner_config=tc_support,
                                        run_config=rc_disable)
        self.assertFalse(tuner_tc_only.use_bayesian_optimization)

    def test_measure_latency_dispatches_correctly(self):
        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            support_bayesian_optimization=True,
        )

        from tools.kernel.tuner.v1.optimizer import (BayesianOptimizer,
                                                     SweepOptimizer)

        rc_bayesian = RunConfig(
            case_set_id="cs_bayesian",
            run_id="r1",
            case_set_desc="desc",
            use_bayesian_optimization=True,
        )
        tuner_bayesian = MockKernelTuner(tuner_config=tc,
                                         run_config=rc_bayesian)
        self.assertIsInstance(tuner_bayesian.optimizer, BayesianOptimizer)
        with mock.patch.object(tuner_bayesian.optimizer,
                               "measure_latency") as mock_measure:
            tuner_bayesian.measure_latency(0, 2)
            mock_measure.assert_called_once_with(0, 2)

        rc_sweep = RunConfig(
            case_set_id="cs_sweep",
            run_id="r1",
            case_set_desc="desc",
            use_bayesian_optimization=False,
        )
        tuner_sweep = MockKernelTuner(tuner_config=tc, run_config=rc_sweep)
        self.assertIsInstance(tuner_sweep.optimizer, SweepOptimizer)
        with mock.patch.object(tuner_sweep.optimizer,
                               "measure_latency") as mock_measure:
            tuner_sweep.measure_latency(0, 2)
            mock_measure.assert_called_once_with(0, 2)

    def test_bayesian_optimization_falls_back_when_search_space_empty(self):
        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            support_bayesian_optimization=True,
        )
        rc = RunConfig(
            case_set_id="cs_empty_space",
            run_id="r1",
            case_set_desc="desc",
            use_bayesian_optimization=True,
        )
        tuner = MockKernelTuner(tuner_config=tc, run_config=rc)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tuner.storage_manager = LocalDbManager(db_path=tmp_dir)
            tuner._generate_tuning_jobs()

            with mock.patch.object(tuner, "get_search_space", return_value={}), \
                 mock.patch("tools.kernel.tuner.v1.optimizer.sweep_optimizer.SweepOptimizer") as mock_sweep_cls:
                mock_sweep_inst = mock.MagicMock()
                mock_sweep_cls.return_value = mock_sweep_inst
                tuner.measure_latency(0, 2)
                mock_sweep_inst.measure_latency.assert_called_once_with(0, 2)

    def test_bayesian_optimization_runs_optuna_trials(self):
        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            support_bayesian_optimization=True,
            n_bayesian_trials=2,
        )
        rc = RunConfig(
            case_set_id="cs_optuna",
            run_id="r1",
            case_set_desc="desc",
            use_bayesian_optimization=True,
        )
        tuner = MockKernelTuner(tuner_config=tc, run_config=rc)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tuner.storage_manager = LocalDbManager(db_path=tmp_dir)
            tuner._generate_tuning_jobs()

            with mock.patch.object(MockKernelTuner, "run",
                                   autospec=True) as run_mock:
                run_mock.return_value = (TuningStatus.SUCCESS, 100, 100)
                tuner.measure_latency(0, 2)

                results = tuner.storage_manager._read_table("CaseResults")
                self.assertGreater(len(results), 0)
                for res in results:
                    self.assertEqual(res["ProcessedStatus"],
                                     TuningStatus.SUCCESS.value)

    def test_measure_latency_xprof_mismatch_raises_runtime_error(self):
        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            jit_kernel_pattern="test_kernel_pattern",
        )
        rc = RunConfig(
            case_set_id="cs_xprof",
            run_id="r1",
            case_set_desc="desc",
            run_locally=True,
        )
        tuner = MockKernelTuner(tuner_config=tc, run_config=rc)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tuner.storage_manager = LocalDbManager(db_path=tmp_dir)
            tuner._generate_tuning_jobs()

            with mock.patch.object(MockKernelTuner, "run", autospec=True) as run_mock, \
                 mock.patch("tools.kernel.tuner.v1.common.utils.find_events_by_pattern", return_value=([], 0)):
                run_mock.return_value = (TuningStatus.SUCCESS, 100, 100)
                with self.assertRaises(RuntimeError) as cm:
                    tuner.measure_latency(0, 1)
                self.assertIn("matching events", str(cm.exception))

    def test_bayesian_optimization_outperforms_randomization(self):
        """Verifies that Bayesian Optimization (TPESampler) achieves lower latency than Random Search (RandomSampler) under the same trial budget."""
        import optuna

        @dataclass(frozen=True)
        class GridTuningKey:
            name: str

        @dataclass(frozen=True)
        class GridTunableParams:
            param_a: int
            param_b: int

            def __ge__(self, other: "GridTunableParams") -> bool:
                return (self.param_a, self.param_b) >= (other.param_a,
                                                        other.param_b)

            def __le__(self, other: "GridTunableParams") -> bool:
                return (self.param_a, self.param_b) <= (other.param_a,
                                                        other.param_b)

        class GridKernelTuner(KernelTunerBase):

            def generate_cases(self) -> list[TuningCase]:
                key = GridTuningKey(name="grid_key")
                cases = []
                for a in range(10):
                    for b in range(10):
                        cases.append(
                            TuningCase(
                                tuning_key=key,
                                tunable_params=GridTunableParams(param_a=a,
                                                                 param_b=b),
                                is_baseline=False,
                            ))
                return cases

            def get_search_space(self, tuning_key: TuningKey) -> dict:
                return {
                    "param_a": list(range(10)),
                    "param_b": list(range(10)),
                }

            def generate_inputs(self, tuning_key: TuningKey) -> dict:
                return {}

            def run(self, tuning_key: TuningKey, tunable_params: TunableParams,
                    iters: int):
                # Synthetic latency bowl with optimal minimum latency=10.0us at param_a=7, param_b=3
                a = tunable_params.param_a
                b = tunable_params.param_b
                latency_us = 10.0 + 5.0 * (a - 7)**2 + 5.0 * (b - 3)**2
                latency_ns = int(latency_us * 1000)
                return TuningStatus.SUCCESS, latency_ns, latency_ns

        n_trials = 35
        tuner_config = TunerConfig(
            tuning_key_class=GridTuningKey,
            tunable_params_class=GridTunableParams,
            kernel_tuner_name="grid_kernel_tuner",
            support_bayesian_optimization=True,
            n_bayesian_trials=n_trials,
        )

        orig_create_study = optuna.create_study

        # 1. Run Bayesian Optimization (TPESampler)
        rc_bo = RunConfig(
            case_set_id="cs_bo",
            run_id="r_bo",
            case_set_desc="bo test",
            use_bayesian_optimization=True,
        )
        bo_tuner = GridKernelTuner(tuner_config=tuner_config, run_config=rc_bo)
        with tempfile.TemporaryDirectory() as tmp_dir:
            bo_tuner.storage_manager = LocalDbManager(db_path=tmp_dir)
            bo_tuner._generate_tuning_jobs()

            def bo_study_factory(*args, **kwargs):
                kwargs["sampler"] = optuna.samplers.TPESampler(
                    n_startup_trials=5, seed=42)
                return orig_create_study(*args, **kwargs)

            with mock.patch("optuna.create_study",
                            side_effect=bo_study_factory):
                bo_tuner.measure_latency(0, 100)

            bo_results = bo_tuner.storage_manager._read_table("CaseResults")
            bo_min_latency = min(
                r["Latency"] for r in bo_results
                if r["ProcessedStatus"] == TuningStatus.SUCCESS.value)

        # 2. Run Random Search (RandomSampler)
        rc_rand = RunConfig(
            case_set_id="cs_rand",
            run_id="r_rand",
            case_set_desc="rand test",
            use_bayesian_optimization=True,
        )
        rand_tuner = GridKernelTuner(tuner_config=tuner_config,
                                     run_config=rc_rand)
        with tempfile.TemporaryDirectory() as tmp_dir:
            rand_tuner.storage_manager = LocalDbManager(db_path=tmp_dir)
            rand_tuner._generate_tuning_jobs()

            def rand_study_factory(*args, **kwargs):
                kwargs["sampler"] = optuna.samplers.RandomSampler(seed=42)
                return orig_create_study(*args, **kwargs)

            with mock.patch("optuna.create_study",
                            side_effect=rand_study_factory):
                rand_tuner.measure_latency(0, 100)

            rand_results = rand_tuner.storage_manager._read_table(
                "CaseResults")
            rand_min_latency = min(
                r["Latency"] for r in rand_results
                if r["ProcessedStatus"] == TuningStatus.SUCCESS.value)

        # Verify trial budget executed
        self.assertGreater(len(bo_results), 0)
        self.assertGreater(len(rand_results), 0)

        # Verify Bayesian Optimization achieves lower latency than Random Search
        self.assertLessEqual(bo_min_latency, rand_min_latency)
        self.assertEqual(bo_min_latency, 10.0)


class DataclassProtocolTest(absltest.TestCase):

    def test_tuning_key_and_params_protocols(self):
        key1 = MockTuningKey(name="test", size=16)
        key2 = MockTuningKey(name="test", size=16)
        params1 = MockTunableParams(size=8)
        params2 = MockTunableParams(size=16)

        # 1. Test protocol compliance
        self.assertTrue(isinstance(key1, TuningKey))
        self.assertTrue(isinstance(params1, TunableParams))
        self.assertTrue(issubclass(MockTuningKey, TuningKey))
        self.assertTrue(issubclass(MockTunableParams, TunableParams))

        # 2. Test hashability and dict/set usage
        self.assertEqual(hash(key1), hash(key2))
        self.assertEqual(hash(params1), hash(params1))
        d = {key1: params1}
        self.assertEqual(d[key2], params1)

        # 3. Test immutability (frozen dataclass)
        with self.assertRaises((AttributeError, TypeError, Exception)):
            key1.size = 32
        with self.assertRaises((AttributeError, TypeError, Exception)):
            params1.size = 32

        # 4. Test comparison operators
        self.assertTrue(params1 <= params2)
        self.assertFalse(params1 >= params2)
        self.assertTrue(params2 >= params1)
        self.assertFalse(params2 <= params1)


if __name__ == "__main__":
    absltest.main()
