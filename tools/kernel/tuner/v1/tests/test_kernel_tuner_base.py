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


def _make_storage_and_cases(tuner, tmp_dir):
    """Helper to create a storage manager, init cases, and return (storage_manager, buckets)."""
    storage_manager = LocalDbManager(db_path=tmp_dir)
    KernelTunerBase.init_case_set(storage_manager, tuner.run_config)
    cases = tuner.generate_cases()
    for case_id, case_str in enumerate(map(str, cases)):
        storage_manager.add_tuner_case(tuner.run_config.case_set_id,
                                       case_id,
                                       case_str,
                                       tpu=tuner.run_config.tpu_queue_multi)
    storage_manager.flush()
    storage_manager.finish_case_set(tuner.run_config.case_set_id, len(cases),
                                    0, 0.0)
    return storage_manager


class KernelTunerBaseTest(absltest.TestCase):

    def test_measure_latency_skips_larger_params_after_oom(self):
        """Tests OOM-based pruning: if a smaller config OOMs, larger configs are skipped."""
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
            lightweight=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_manager = _make_storage_and_cases(kernel_tuner, tmp_dir)

            # Mock executor that returns OOM for the first (smallest) params
            mock_executor = mock.MagicMock()
            mock_executor.execute_run.side_effect = [
                (TuningStatus.FAILED_OOM, 0, 0),
            ]

            from tools.kernel.tuner.v1.optimizer import SweepOptimizer
            optimizer = SweepOptimizer(kernel_tuner, storage_manager,
                                       mock_executor)

            cases = storage_manager.get_all_cases(run_config.case_set_id)
            buckets = optimizer.generate_tuning_jobs(cases)
            self.assertEqual(buckets, [(0, 2)])

            optimizer.measure_latency(0, 2)

            results = storage_manager._read_table("CaseResults")
            case_status = {
                result["CaseId"]: result["ProcessedStatus"]
                for result in results
            }
            self.assertEqual(case_status, {
                0: TuningStatus.FAILED_OOM.value,
                1: TuningStatus.SKIPPED.value
            })
            # Only 1 call to executor (warmup of case 0), case 1 was OOM-skipped
            self.assertEqual(mock_executor.execute_run.call_count, 1)

    def test_measure_latency_runs_both_cases_when_no_oom(self):
        """Tests that both cases run when neither causes OOM."""
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
            lightweight=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_manager = _make_storage_and_cases(kernel_tuner, tmp_dir)

            mock_executor = mock.MagicMock()
            mock_executor.execute_run.return_value = (TuningStatus.SUCCESS,
                                                      1000, 1000)

            from tools.kernel.tuner.v1.optimizer import SweepOptimizer
            optimizer = SweepOptimizer(kernel_tuner, storage_manager,
                                       mock_executor)

            cases = storage_manager.get_all_cases(run_config.case_set_id)
            buckets = optimizer.generate_tuning_jobs(cases)
            self.assertEqual(buckets, [(0, 2)])

            optimizer.measure_latency(0, 2)

            results = storage_manager._read_table("CaseResults")
            case_status = {
                result["CaseId"]: result["ProcessedStatus"]
                for result in results
            }
            self.assertEqual(case_status, {
                0: TuningStatus.SUCCESS.value,
                1: TuningStatus.SUCCESS.value
            })
            # 2 cases × 2 calls each (warmup + measurement) = 4
            self.assertEqual(mock_executor.execute_run.call_count, 4)

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

    def test_min_cases_for_bayesian_run_config_override(self):
        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
            min_cases_for_bayesian=200,
        )
        rc_override = RunConfig(
            case_set_id="cs_min_cases",
            run_id="r1",
            case_set_desc="desc",
            min_cases_for_bayesian=50,
        )
        tuner = MockKernelTuner(tuner_config=tc, run_config=rc_override)
        self.assertEqual(tuner.tuner_config.min_cases_for_bayesian, 50)

    def test_lightweight_mode_sets_xprof_dir(self):
        """Verifies lightweight mode sets xprof_dir and _measurement_iters."""

        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
        )
        rc = RunConfig(
            case_set_id="cs_lw",
            run_id="r_lw",
            case_set_desc="desc",
        )
        tuner = MockKernelTuner(tuner_config=tc,
                                run_config=rc,
                                lightweight=True)
        self.assertTrue(tuner.lightweight)
        self.assertTrue(hasattr(tuner, 'xprof_dir'))
        self.assertTrue(hasattr(tuner, '_measurement_iters'))

    def test_full_mode_sets_xprof_dir(self):
        """Verifies full (non-lightweight) mode sets xprof_dir."""
        tc = TunerConfig(
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams,
            kernel_tuner_name="mock_kernel_tuner",
        )
        rc = RunConfig(
            case_set_id="cs_full",
            run_id="r_full",
            case_set_desc="desc",
        )
        tuner = MockKernelTuner(tuner_config=tc,
                                run_config=rc,
                                lightweight=False)
        self.assertFalse(tuner.lightweight)
        self.assertTrue(hasattr(tuner, 'xprof_dir'))
        self.assertTrue(hasattr(tuner, '_measurement_iters'))
        self.assertEqual(tuner._measurement_iters, 100)

    def test_init_case_set_static_method(self):
        """Tests the static init_case_set method."""
        rc = RunConfig(
            case_set_id="cs_init_test",
            run_id="r1",
            case_set_desc="test desc",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_manager = LocalDbManager(db_path=tmp_dir)
            # First call should return True (new case set)
            result = KernelTunerBase.init_case_set(storage_manager, rc)
            self.assertTrue(result)
            # Second call should return False (already exists)
            result = KernelTunerBase.init_case_set(storage_manager, rc)
            self.assertFalse(result)

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
        kernel_tuner = MockKernelTuner(tuner_config=tc,
                                       run_config=rc,
                                       lightweight=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_manager = _make_storage_and_cases(kernel_tuner, tmp_dir)

            mock_executor = mock.MagicMock()
            mock_executor.execute_run.return_value = (TuningStatus.SUCCESS,
                                                      100, 100)

            from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
            optimizer = BayesianOptimizer(kernel_tuner, storage_manager,
                                          mock_executor)

            optimizer.measure_latency(0, 2)

            results = storage_manager._read_table("CaseResults")
            self.assertGreater(len(results), 0)
            for res in results:
                self.assertEqual(res["ProcessedStatus"],
                                 TuningStatus.SUCCESS.value)

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

        # Helper to create storage+cases for the GridKernelTuner
        def _setup_grid(case_set_id, run_id, desc, tmp_dir):
            rc = RunConfig(
                case_set_id=case_set_id,
                run_id=run_id,
                case_set_desc=desc,
                use_bayesian_optimization=True,
            )
            tuner = GridKernelTuner(tuner_config=tuner_config,
                                    run_config=rc,
                                    lightweight=True)
            sm = _make_storage_and_cases(tuner, tmp_dir)

            # Create a mock executor that invokes the real run()
            # We create a real (non-lightweight) tuner for the actual run
            real_tuner = GridKernelTuner(tuner_config=tuner_config,
                                         run_config=rc,
                                         lightweight=False)
            mock_executor = mock.MagicMock()
            mock_executor.execute_run.side_effect = lambda tk, tp, iters, **kw: real_tuner.run(
                tk, tp, iters)
            return tuner, sm, mock_executor

        # 1. Run Bayesian Optimization (TPESampler)
        with tempfile.TemporaryDirectory() as tmp_dir:
            bo_tuner, bo_sm, bo_exec = _setup_grid("cs_bo", "r_bo", "bo test",
                                                   tmp_dir)

            from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
            bo_optimizer = BayesianOptimizer(bo_tuner, bo_sm, bo_exec)

            def bo_study_factory(*args, **kwargs):
                kwargs["sampler"] = optuna.samplers.TPESampler(
                    n_startup_trials=5, seed=42)
                return orig_create_study(*args, **kwargs)

            with mock.patch("optuna.create_study",
                            side_effect=bo_study_factory):
                bo_optimizer.measure_latency(0, 100)

            bo_results = bo_sm._read_table("CaseResults")
            bo_min_latency = min(
                r["Latency"] for r in bo_results
                if r["ProcessedStatus"] == TuningStatus.SUCCESS.value)

        # 2. Run Random Search (RandomSampler)
        with tempfile.TemporaryDirectory() as tmp_dir:
            rand_tuner, rand_sm, rand_exec = _setup_grid(
                "cs_rand", "r_rand", "rand test", tmp_dir)

            rand_optimizer = BayesianOptimizer(rand_tuner, rand_sm, rand_exec)

            def rand_study_factory(*args, **kwargs):
                kwargs["sampler"] = optuna.samplers.RandomSampler(seed=42)
                return orig_create_study(*args, **kwargs)

            with mock.patch("optuna.create_study",
                            side_effect=rand_study_factory):
                rand_optimizer.measure_latency(0, 100)

            rand_results = rand_sm._read_table("CaseResults")
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


class WorkerIdResolutionTest(absltest.TestCase):

    def test_explicit_worker_id(self):
        from tools.kernel.tuner.v1.utils import get_worker_id
        self.assertEqual(get_worker_id('custom_worker_123'),
                         'custom_worker_123')
        self.assertEqual(get_worker_id(42), '42')

    def test_env_worker_id_resolution(self):
        from tools.kernel.tuner.v1.utils import get_worker_id
        with mock.patch.dict(os.environ, {'TPU_WORKER_ID': 'tpu_worker_5'},
                             clear=True):
            self.assertEqual(get_worker_id(), 'tpu_worker_5')

        with mock.patch.dict(os.environ, {'HOST_NAME': 'host_alpha'},
                             clear=True):
            self.assertEqual(get_worker_id(), 'host_alpha')

        with mock.patch.dict(os.environ, {'HOSTNAME': 'host_beta'},
                             clear=True):
            self.assertEqual(get_worker_id(), 'host_beta')

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_worker_id(), '0')

    def test_component_worker_id_initialization(self):
        db_manager = LocalDbManager(worker_id="db_worker", dry_run=True)
        self.assertEqual(db_manager.worker_id, "db_worker")


class GetAlreadyProcessedIdsTest(absltest.TestCase):

    def test_local_db_manager_returns_processed_case_status_namedtuples(self):
        from tools.kernel.tuner.v1.common.tuner_datatypes import \
            ProcessedCaseStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            db_manager = LocalDbManager(db_path=tmpdir,
                                        worker_id="test_worker")
            # Insert dummy case result into local db
            table = [
                {
                    'ID': 'cs_1',
                    'RunId': 'r_1',
                    'CaseId': 10,
                    'ProcessedStatus': 'SUCCESS',
                },
                {
                    'ID': 'cs_1',
                    'RunId': 'r_1',
                    'CaseId': 20,
                    'ProcessedStatus': 'FAILED_OOM',
                },
            ]
            db_manager._write_table('CaseResults', table)

            res = db_manager.get_already_processed_ids('cs_1', 'r_1', 0, 100)
            self.assertEqual(len(res), 2)
            self.assertIsInstance(res[0], ProcessedCaseStatus)
            self.assertEqual(res[0].case_id, 10)
            self.assertEqual(res[0].status, 'SUCCESS')
            self.assertEqual(res[0][0], 10)
            self.assertEqual(res[0][1], 'SUCCESS')

            self.assertIsInstance(res[1], ProcessedCaseStatus)
            self.assertEqual(res[1].case_id, 20)
            self.assertEqual(res[1].status, 'FAILED_OOM')


if __name__ == "__main__":
    absltest.main()
