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
"""Smoke tests for kernel_tuner_runner.

Covers two execution modes:

* **Sweep mode** – every pre-generated case is evaluated sequentially.
  ``_run_tuner_smoke_test`` forces sweep mode and runs only the first case of
  the first bucket so the test completes quickly.

* **Bayesian optimization mode** – optuna (TPE sampler) selects which
  tunable-parameter combinations to evaluate.  ``_run_tuner_bayesian_smoke_test``
  forces Bayesian mode, caps ``n_bayesian_trials`` to 3 for speed, and runs the
  *full* first bucket (required so optuna has the complete search space for one
  TuningKey).

Environment variables (required):
    TPU_VERSION   -- e.g. "tpu6e" or "tpu7x"
    TPU_CORES     -- e.g. "1" (tpu6e) or "2" (tpu7x)
"""

import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

from absl.testing import absltest

# Importing kernel_tuner_flags registers the absl flags that
# kernel_tuner_worker reads at runtime.
import tools.kernel.tuner.v1.kernel_tuner_flags  # noqa: F401  # register flags
from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TuningStatus)
from tools.kernel.tuner.v1.kernel_tuner_factory import _load_kernel_tuner_class
from tools.kernel.tuner.v1.kernel_tuner_runner import \
    generate_and_partition_cases
from tools.kernel.tuner.v1.storage_management.local_db_manager import \
    LocalDbManager
from tools.kernel.tuner.v1.utils import get_tpu_queue_by_version_and_cores

# Statuses that are acceptable outcomes for a tuning case in a smoke test.
# SKIPPED is added because Bayesian optimization prunes cases that are
# expected to OOM based on smaller configurations that already failed.
_ACCEPTABLE_STATUSES = frozenset({
    TuningStatus.SUCCESS.value,
    TuningStatus.FAILED_OOM.value,
    TuningStatus.SKIPPED.value,
})


def _make_mock_executor(kernel_tuner):
    from unittest import mock

    import jax

    def _execute_run(tk, tp, iters, **kw):
        use_xprof = kw.get("use_xprof", False)
        if use_xprof and kernel_tuner.tuner_config.jit_kernel_pattern:
            kernel_tuner._cleanup_xprof_dir()
            with jax.profiler.trace(kernel_tuner.xprof_dir,
                                    create_perfetto_link=False):
                return kernel_tuner.run(tk, tp, iters)
        else:
            return kernel_tuner.run(tk, tp, iters)

    mock_executor = mock.MagicMock()
    mock_executor.execute_run.side_effect = _execute_run
    return mock_executor


class KernelTunerRunnerSmokeTest(absltest.TestCase):
    """Smoke tests ensuring each registered kernel tuner can run end-to-end."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tpu_env(self) -> tuple[str, int]:
        """Returns (tpu_version, tpu_cores) from the environment.

        Skips the current test if either variable is absent or empty.
        """
        tpu_version = os.environ.get("TPU_VERSION", "").strip()
        tpu_cores_str = os.environ.get("TPU_CORES", "").strip()
        if not tpu_version or not tpu_cores_str:
            self.skipTest(
                "TPU_VERSION and TPU_CORES environment variables must be set "
                "to run these tests (e.g. TPU_VERSION=tpu6e TPU_CORES=1).")
        try:
            tpu_cores = int(tpu_cores_str)
        except ValueError:
            self.skipTest(
                f"TPU_CORES must be an integer, got {tpu_cores_str!r}.")
        return tpu_version, tpu_cores

    def _make_run_config(self,
                         kernel_tuner_name: str,
                         use_bayesian_optimization: bool = False) -> RunConfig:
        """Builds a RunConfig with run_locally=True from TPU env vars."""
        tpu_version, tpu_cores = self._get_tpu_env()
        try:
            tpu_queue_multi = get_tpu_queue_by_version_and_cores(
                tpu_version, tpu_cores, "")
        except AssertionError as e:
            self.skipTest(
                f"Unsupported TPU_VERSION/TPU_CORES combination "
                f"({tpu_version!r}, {tpu_cores}): {e}. Supported combinations: "
                f"(tpu6e, 1), (tpu6e, 8), (tpu7x, 2), (tpu7x, 8), (tpu7x, 16)."
            )
        return RunConfig(
            case_set_id=f"test_{kernel_tuner_name}_{uuid.uuid4().hex[:8]}",
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            case_set_desc=f"Smoke test for {kernel_tuner_name}",
            tpu_version=tpu_version,
            tpu_cores=tpu_cores,
            tpu_queue_multi=tpu_queue_multi,
            run_locally=True,
            job_bucket_size=100,
            use_bayesian_optimization=use_bayesian_optimization,
        )

    def _run_tuner_smoke_test(self, kernel_tuner_name: str) -> None:
        """Sweep-mode smoke test shared by every per-tuner test method.

        Uses the new three-process architecture:
          1. Creates a full (non-lightweight) kernel tuner for run().
          2. Creates storage_manager independently.
          3. Creates a SweepOptimizer with a mock executor that delegates to real run().
        """
        run_config = self._make_run_config(kernel_tuner_name)
        kernel_tuner_cls = _load_kernel_tuner_class(kernel_tuner_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Full init — needed for run()/generate_inputs()
            kernel_tuner = kernel_tuner_cls(run_config=run_config)
            # Force sweep mode
            kernel_tuner.tuner_config.support_bayesian_optimization = False
            kernel_tuner.use_bayesian_optimization = False

            storage_manager = LocalDbManager(
                db_path=os.path.join(tmp_dir, "db"))

            mock_executor = _make_mock_executor(kernel_tuner)

            from tools.kernel.tuner.v1.optimizer import SweepOptimizer
            optimizer = SweepOptimizer(kernel_tuner, storage_manager,
                                       mock_executor)

            buckets = generate_and_partition_cases(kernel_tuner,
                                                   storage_manager, optimizer)
            self.assertGreater(
                len(buckets),
                0,
                msg=(f"{kernel_tuner_name}: generate_and_partition_cases() "
                     "returned no buckets"),
            )

            # Only run the first case of the first bucket.
            begin_case_id, _ = buckets[0]
            optimizer.measure_latency(begin_case_id, begin_case_id + 1)

            results = storage_manager._read_table("CaseResults")
            self.assertGreater(
                len(results),
                0,
                msg=(f"{kernel_tuner_name}: no results recorded after "
                     "measure_latency"),
            )

            for result in results:
                status = result["ProcessedStatus"]
                self.assertIn(
                    status,
                    _ACCEPTABLE_STATUSES,
                    msg=(f"{kernel_tuner_name}: case {result['CaseId']} "
                         f"returned unexpected status {status!r}. "
                         f"Acceptable statuses: {_ACCEPTABLE_STATUSES}"),
                )

    def _run_tuner_bayesian_smoke_test(self,
                                       kernel_tuner_name: str,
                                       n_trials: int = 3) -> None:
        """Bayesian-mode smoke test: runs one full bucket with a capped trial count."""
        run_config = self._make_run_config(kernel_tuner_name,
                                           use_bayesian_optimization=True)
        kernel_tuner_cls = _load_kernel_tuner_class(kernel_tuner_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_tuner = kernel_tuner_cls(run_config=run_config)
            kernel_tuner.tuner_config.n_bayesian_trials = n_trials

            storage_manager = LocalDbManager(
                db_path=os.path.join(tmp_dir, "db"))

            mock_executor = _make_mock_executor(kernel_tuner)

            from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
            optimizer = BayesianOptimizer(kernel_tuner, storage_manager,
                                          mock_executor)

            buckets = generate_and_partition_cases(kernel_tuner,
                                                   storage_manager, optimizer)
            self.assertGreater(
                len(buckets),
                0,
                msg=(f"{kernel_tuner_name}: generate_and_partition_cases() "
                     "returned no buckets in Bayesian mode"),
            )

            # Run the full first bucket
            begin_case_id, end_case_id = buckets[0]
            optimizer.measure_latency(begin_case_id, end_case_id)

            results = storage_manager._read_table("CaseResults")
            self.assertGreater(
                len(results),
                0,
                msg=(f"{kernel_tuner_name}: no results recorded after "
                     "Bayesian measure_latency"),
            )

            for result in results:
                status = result["ProcessedStatus"]
                self.assertIn(
                    status,
                    _ACCEPTABLE_STATUSES,
                    msg=(f"{kernel_tuner_name}: case {result['CaseId']} "
                         f"returned unexpected status {status!r}. "
                         f"Acceptable statuses: {_ACCEPTABLE_STATUSES}"),
                )

    # ------------------------------------------------------------------
    # One test method per entry in KERNEL_TUNER_REGISTRY
    # ------------------------------------------------------------------

    def test_example_kernel_tuner(self):
        self._run_tuner_smoke_test("example_kernel_tuner")

    def test_example_kernel_tuner_bayesian(self):
        """Smoke test for example_kernel_tuner in Bayesian optimization mode."""
        self._run_tuner_bayesian_smoke_test("example_kernel_tuner")

    def test_mla_kernel_tuner(self):
        self._run_tuner_smoke_test("mla_kernel_tuner")

    def test_mla_kernel_tuner_bayesian(self):
        """Smoke test for mla_kernel_tuner in Bayesian optimization mode."""
        self._run_tuner_bayesian_smoke_test("mla_kernel_tuner")

    def test_batched_rpa_kernel_tuner(self):
        self._run_tuner_smoke_test("batched_rpa_kernel_tuner")

    def test_gmm_v2_kernel_tuner(self):
        self._run_tuner_smoke_test("gmm_v2_kernel_tuner")

    def test_gmm_v2_kernel_tuner_bayesian(self):
        """Smoke test for gmm_v2_kernel_tuner in Bayesian optimization mode."""
        self._run_tuner_bayesian_smoke_test("gmm_v2_kernel_tuner")

    def test_flash_attention_kernel_tuner(self):
        self._run_tuner_smoke_test("flash_attention_kernel_tuner")

    def test_batched_rpa_kernel_tuner_bayesian(self):
        """Smoke test for batched_rpa_kernel_tuner in Bayesian optimization mode."""
        self._run_tuner_bayesian_smoke_test("batched_rpa_kernel_tuner")


class TuningCaseSerializationTest(absltest.TestCase):
    """Tests for TuningCase serialization and deserialization."""

    def test_from_string_returns_tuning_case(self):
        from dataclasses import dataclass

        from tools.kernel.tuner.v1.common.kernel_tuner_base import TuningCase

        @dataclass
        class MockTuningKey:
            name: str
            size: int

        @dataclass
        class MockTunableParams:
            batch_size: int

        # Given a string created by TuningCase.__str__
        original_case = TuningCase(
            tuning_key=MockTuningKey(name="test", size=128),
            tunable_params=MockTunableParams(batch_size=8),
            is_baseline=True)
        serialized_str = str(original_case)

        # When we deserialize it
        restored_case = TuningCase.from_string(
            serialized_str,
            tuning_key_class=MockTuningKey,
            tunable_params_class=MockTunableParams)

        # Then it should return a TuningCase (not a tuple as it did previously)
        self.assertIsInstance(restored_case, TuningCase)

        # And the data should match exactly
        self.assertEqual(restored_case.tuning_key.name, "test")
        self.assertEqual(restored_case.tuning_key.size, 128)
        self.assertEqual(restored_case.tunable_params.batch_size, 8)
        self.assertEqual(restored_case.is_baseline, True)


class ExampleKernelTunerSearchSpaceTest(absltest.TestCase):
    """Unit tests for get_search_space and generate_cases in ExampleKernelTuner."""

    def _make_tuner(self):
        from tools.kernel.tuner.v1.example_kernel_tuner import (
            ExampleKernelTuner, TuningKey)
        run_config = RunConfig(
            case_set_id=f"search_space_test_{uuid.uuid4().hex[:8]}",
            run_id="run_0",
            case_set_desc="search space unit test",
            tpu_version="tpu7x",
            tpu_cores=2,
            tpu_queue_multi="tpu_v7x_2_queue",
            run_locally=True,
            use_bayesian_optimization=True,
            job_bucket_size=100,
        )
        tuner = ExampleKernelTuner(run_config=run_config, lightweight=True)
        return tuner, TuningKey

    def test_get_search_space_returns_dict(self):
        tuner, TuningKey = self._make_tuner()
        space = tuner.get_search_space(TuningKey(key1=1, key2=4))
        self.assertIsInstance(space, dict)
        self.assertIn("param1", space)
        self.assertIn("param2", space)

    def test_get_search_space_all_values_are_lists(self):
        tuner, TuningKey = self._make_tuner()
        for key2 in [4, 8, 16]:
            space = tuner.get_search_space(TuningKey(key1=2, key2=key2))
            for param_name, values in space.items():
                self.assertIsInstance(
                    values,
                    list,
                    msg=
                    f"{param_name} values should be a list, got {type(values)}"
                )
                self.assertGreater(
                    len(values),
                    0,
                    msg=f"{param_name} values list must not be empty")

    def test_get_search_space_dynamic_for_large_key2(self):
        """param2 range should differ between key2=4 and key2>=8."""
        tuner, TuningKey = self._make_tuner()
        space_small = tuner.get_search_space(TuningKey(key1=1, key2=4))
        space_large = tuner.get_search_space(TuningKey(key1=1, key2=8))
        self.assertNotEqual(
            space_small["param2"],
            space_large["param2"],
            msg="param2 search space should be wider for key2 >= 8",
        )

    def test_generate_cases_uses_search_space(self):
        """Every case's tunable params must be in the corresponding search space."""
        tuner, TuningKey = self._make_tuner()
        cases = tuner.generate_cases()
        self.assertGreater(len(cases), 0)
        for case in cases:
            space = tuner.get_search_space(case.tuning_key)
            from dataclasses import asdict
            params = asdict(case.tunable_params)
            for param_name, value in params.items():
                self.assertIn(
                    value,
                    space[param_name],
                    msg=(f"case tuning_key={case.tuning_key}: "
                         f"{param_name}={value} not in search space "
                         f"{space[param_name]}"),
                )

    def test_generate_cases_count_matches_cartesian_product(self):
        """Total cases == sum over all (key1, key2) of |search_space| products."""
        import itertools
        tuner, TuningKey = self._make_tuner()
        key1_values = [1, 2, 4]
        key2_values = [4, 8, 16]
        expected = sum(
            len(
                list(
                    itertools.product(*tuner.get_search_space(
                        TuningKey(key1=k1, key2=k2)).values())))
            for k1, k2 in itertools.product(key1_values, key2_values))
        actual = len(tuner.generate_cases())
        self.assertEqual(actual, expected)

    def test_tuner_config_override_disables_bayesian(self):
        tuner, _ = self._make_tuner()
        tuner.tuner_config.support_bayesian_optimization = False
        self.assertFalse(tuner.tuner_config.support_bayesian_optimization)


class BayesianOptimizationValidationTest(absltest.TestCase):
    """Test validating that Bayesian Optimization reduces search steps by >= 80% while finding the correct result."""

    def _get_tpu_env(self) -> tuple[str, int]:
        tpu_version = os.environ.get("TPU_VERSION", "tpu7x").strip() or "tpu7x"
        tpu_cores_str = os.environ.get("TPU_CORES", "2").strip() or "2"
        return tpu_version, int(tpu_cores_str)

    def test_bayesian_optimization_step_reduction_and_accuracy(self):
        """Validates that BO reduces search steps by at least 80% and finds the exact optimal result."""
        import itertools
        from unittest import mock

        from tools.kernel.tuner.v1.common.tuner_datatypes import (TuningCase,
                                                                  TuningStatus)
        from tools.kernel.tuner.v1.example_kernel_tuner import (
            ExampleKernelTuner, TunableParams, TuningKey)

        tpu_version, tpu_cores = self._get_tpu_env()
        tpu_queue_multi = get_tpu_queue_by_version_and_cores(
            tpu_version, tpu_cores, "")

        param1_vals = range(4, 129, 4)  # [4, 8, ..., 128]
        param2_vals = range(4, 129, 4)  # [4, 8, ..., 128]

        def mock_get_search_space(self, tuning_key):
            return {
                'param1': param1_vals,
                'param2': param2_vals,
            }

        def mock_generate_cases(self):
            cases = []
            key = TuningKey(key1=1, key2=1)
            for p1, p2 in itertools.product(param1_vals, param2_vals):
                cases.append(
                    TuningCase(key, TunableParams(param1=p1, param2=p2)))
            return cases

        def mock_run(tuning_key, tunable_params, iters=1, **kwargs):
            golden_p1, golden_p2 = 32, 32
            base_latency_ns = 50_000.0
            dist_sq = (tunable_params.param1 -
                       golden_p1)**2 + (tunable_params.param2 - golden_p2)**2
            avg_latency_ns = base_latency_ns + dist_sq * 1000.0
            return TuningStatus.SUCCESS, avg_latency_ns, avg_latency_ns * iters

        with patch.object(ExampleKernelTuner, "get_search_space", mock_get_search_space), \
             patch.object(ExampleKernelTuner, "generate_cases", mock_generate_cases):

            # --- 1. Full Sweep Run ---
            with tempfile.TemporaryDirectory() as tmp_dir_sweep:
                sweep_run_config = RunConfig(
                    case_set_id=f"val_sweep_{uuid.uuid4().hex[:8]}",
                    run_id="run_sweep",
                    case_set_desc=os.path.join(tmp_dir_sweep, "db"),
                    tpu_version=tpu_version,
                    tpu_cores=tpu_cores,
                    tpu_queue_multi=tpu_queue_multi,
                    run_locally=True,
                    use_bayesian_optimization=False,
                    job_bucket_size=500,
                )
                tuner_sweep = ExampleKernelTuner(run_config=sweep_run_config,
                                                 lightweight=True)

                storage_sweep = LocalDbManager(
                    db_path=os.path.join(tmp_dir_sweep, "db"))

                mock_executor_sweep = mock.MagicMock()
                mock_executor_sweep.execute_run.side_effect = mock_run

                from tools.kernel.tuner.v1.optimizer import SweepOptimizer
                sweep_opt = SweepOptimizer(tuner_sweep, storage_sweep,
                                           mock_executor_sweep)
                buckets_sweep = generate_and_partition_cases(
                    tuner_sweep, storage_sweep, sweep_opt)
                for b_start, b_end in buckets_sweep:
                    sweep_opt.measure_latency(b_start, b_end)

                sweep_results = storage_sweep._read_table("CaseResults")
                total_sweep_steps = len(sweep_results)
                self.assertEqual(total_sweep_steps,
                                 len(param1_vals) * len(param2_vals))

                best_sweep = min(sweep_results, key=lambda r: r["Latency"])
                self.assertEqual(best_sweep["Latency"], 50)

            # --- 2. Bayesian Optimization Run ---
            steps_reduction_ratio = 0.9
            with tempfile.TemporaryDirectory() as tmp_dir_bo:
                bo_run_config = RunConfig(
                    case_set_id=f"val_bo_{uuid.uuid4().hex[:8]}",
                    run_id="run_bo",
                    case_set_desc=os.path.join(tmp_dir_bo, "db"),
                    tpu_version=tpu_version,
                    tpu_cores=tpu_cores,
                    tpu_queue_multi=tpu_queue_multi,
                    run_locally=True,
                    use_bayesian_optimization=True,
                    job_bucket_size=500,
                )
                tuner_bo = ExampleKernelTuner(run_config=bo_run_config,
                                              lightweight=True)
                tuner_bo.tuner_config.n_bayesian_trials = int(
                    len(param1_vals) * len(param2_vals) *
                    (1 - steps_reduction_ratio))

                storage_bo = LocalDbManager(
                    db_path=os.path.join(tmp_dir_bo, "db"))

                mock_executor_bo = mock.MagicMock()
                mock_executor_bo.execute_run.side_effect = mock_run

                from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
                bo_opt = BayesianOptimizer(tuner_bo, storage_bo,
                                           mock_executor_bo)
                buckets_bo = generate_and_partition_cases(
                    tuner_bo, storage_bo, bo_opt)
                for b_start, b_end in buckets_bo:
                    bo_opt.measure_latency(b_start, b_end)

                bo_results = storage_bo._read_table("CaseResults")
                total_bo_steps = len(bo_results)
                best_bo = min(bo_results, key=lambda r: r["Latency"])

            # --- 3. Assert >= 80% Step Reduction ---
            step_reduction = (total_sweep_steps -
                              total_bo_steps) / float(total_sweep_steps)
            self.assertGreaterEqual(
                step_reduction,
                steps_reduction_ratio,
                msg=
                f"Expected at least {steps_reduction_ratio:.2%} step reduction, got {step_reduction:.2%} "
                f"({total_bo_steps} BO steps vs {total_sweep_steps} sweep steps)."
            )

            # --- 4. Assert Near-Optimal Accuracy (Within 10us of Golden 50us Minimum) ---
            self.assertLessEqual(
                best_bo["Latency"],
                60,
                msg=
                f"Expected best BO latency to be <= 60us (near-optimal), got {best_bo['Latency']}us."
            )

    def test_bayesian_optimization_early_stopping(self):
        """Validates that early stopping halts Optuna optimization when relative improvement < min_delta_ratio for patience trials."""
        import itertools
        from unittest import mock

        from tools.kernel.tuner.v1.common.tuner_datatypes import (TuningCase,
                                                                  TuningStatus)
        from tools.kernel.tuner.v1.example_kernel_tuner import (
            ExampleKernelTuner, TunableParams, TuningKey)

        tpu_version, tpu_cores = self._get_tpu_env()
        tpu_queue_multi = get_tpu_queue_by_version_and_cores(
            tpu_version, tpu_cores, "")

        param1_vals = range(4, 65, 4)
        param2_vals = range(4, 65, 4)

        def mock_get_search_space(self, tuning_key):
            return {'param1': param1_vals, 'param2': param2_vals}

        def mock_generate_cases(self):
            cases = []
            key = TuningKey(key1=1, key2=1)
            for p1, p2 in itertools.product(param1_vals, param2_vals):
                cases.append(
                    TuningCase(key, TunableParams(param1=p1, param2=p2)))
            return cases

        def mock_run(tuning_key, tunable_params, iters=1, **kwargs):
            avg_latency_ns = 50_000.0
            return TuningStatus.SUCCESS, avg_latency_ns, avg_latency_ns * iters

        with patch.object(ExampleKernelTuner, "get_search_space", mock_get_search_space), \
             patch.object(ExampleKernelTuner, "generate_cases", mock_generate_cases):

            with tempfile.TemporaryDirectory() as tmp_dir:
                bo_run_config = RunConfig(
                    case_set_id=f"val_es_{uuid.uuid4().hex[:8]}",
                    run_id="run_es",
                    case_set_desc=os.path.join(tmp_dir, "db"),
                    tpu_version=tpu_version,
                    tpu_cores=tpu_cores,
                    tpu_queue_multi=tpu_queue_multi,
                    run_locally=True,
                    use_bayesian_optimization=True,
                    job_bucket_size=500,
                )
                tuner = ExampleKernelTuner(run_config=bo_run_config,
                                           lightweight=True)
                tuner.tuner_config.n_bayesian_trials = 200
                tuner.tuner_config.bayesian_early_stopping_patience = 5
                tuner.tuner_config.bayesian_early_stopping_min_delta_ratio = 0.10

                storage = LocalDbManager(db_path=os.path.join(tmp_dir, "db"))

                mock_executor = mock.MagicMock()
                mock_executor.execute_run.side_effect = mock_run

                from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
                optimizer = BayesianOptimizer(tuner, storage, mock_executor)
                buckets = generate_and_partition_cases(tuner, storage,
                                                       optimizer)
                for b_start, b_end in buckets:
                    optimizer.measure_latency(b_start, b_end)

                bo_results = storage._read_table("CaseResults")
                self.assertLessEqual(len(bo_results), 15)

    def test_exception_flushes_storage_manager(self):
        """Validates that raising an exception during tuning flushes buffered results and closes StorageManager."""
        from unittest import mock

        from tools.kernel.tuner.v1.common.tuner_datatypes import (TuningCase,
                                                                  TuningStatus)
        from tools.kernel.tuner.v1.example_kernel_tuner import (
            ExampleKernelTuner, TunableParams, TuningKey)

        tpu_version, tpu_cores = self._get_tpu_env()
        tpu_queue_multi = get_tpu_queue_by_version_and_cores(
            tpu_version, tpu_cores, "")

        def mock_generate_cases(self):
            key = TuningKey(key1=1, key2=1)
            return [
                TuningCase(key, TunableParams(param1=4, param2=4)),
                TuningCase(key, TunableParams(param1=8, param2=8)),
            ]

        call_count = [0]

        def mock_run(tuning_key, tunable_params, iters=1, **kwargs):
            call_count[0] += 1
            if tunable_params.param1 == 8 and call_count[0] > 2:
                raise RuntimeError("Fatal kernel error requested by user!")
            return TuningStatus.SUCCESS, 50_000.0, 50_000.0 * iters

        with patch.object(ExampleKernelTuner, "generate_cases",
                          mock_generate_cases):

            with tempfile.TemporaryDirectory() as tmp_dir:
                run_config = RunConfig(
                    case_set_id=f"val_err_{uuid.uuid4().hex[:8]}",
                    run_id="run_err",
                    case_set_desc=os.path.join(tmp_dir, "db"),
                    tpu_version=tpu_version,
                    tpu_cores=tpu_cores,
                    tpu_queue_multi=tpu_queue_multi,
                    run_locally=True,
                    use_bayesian_optimization=False,
                    job_bucket_size=500,
                )
                tuner = ExampleKernelTuner(run_config=run_config,
                                           lightweight=True)

                storage = LocalDbManager(db_path=os.path.join(tmp_dir, "db"))

                mock_executor = mock.MagicMock()
                mock_executor.execute_run.side_effect = mock_run

                from tools.kernel.tuner.v1.optimizer import SweepOptimizer
                optimizer = SweepOptimizer(tuner, storage, mock_executor)
                buckets = generate_and_partition_cases(tuner, storage,
                                                       optimizer)
                with self.assertRaises(RuntimeError):
                    for b_start, b_end in buckets:
                        optimizer.measure_latency(b_start, b_end)

                results = storage._read_table("CaseResults")
                self.assertGreaterEqual(len(results), 1)
                storage.close()
                self.assertTrue(storage._closed)


class KernelTunerRunnerInterruptTest(absltest.TestCase):

    @patch("tools.kernel.tuner.v1.kernel_tuner_runner.subprocess.Popen")
    @patch("tools.kernel.tuner.v1.kernel_tuner_runner.os.killpg")
    @patch("tools.kernel.tuner.v1.kernel_tuner_runner.os.getpgid")
    def test_invoke_worker_process_kills_process_group_on_interrupt(
            self, mock_getpgid, mock_killpg, mock_popen):
        import signal

        from tools.kernel.tuner.v1.common.tuner_datatypes import RunConfig
        from tools.kernel.tuner.v1.kernel_tuner_runner import \
            _invoke_worker_process

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.stdout = []
        mock_proc.wait.side_effect = KeyboardInterrupt("Ctrl+C pressed")
        mock_popen.return_value = mock_proc
        mock_getpgid.return_value = 12345

        run_config = RunConfig(
            case_set_id="test_cs",
            run_id="test_run",
            case_set_desc="test",
            tpu_version="tpu7x",
            tpu_cores=2,
            tpu_queue_multi="test_queue",
            run_locally=True,
        )

        with self.assertRaises(KeyboardInterrupt):
            _invoke_worker_process("example_kernel_tuner", run_config, 0, 10)

        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    @patch("tools.kernel.tuner.v1.kernel_tuner_runner.KERNEL_TUNER_NAME")
    @patch("tools.kernel.tuner.v1.kernel_tuner_runner._invoke_worker_process")
    @patch("tools.kernel.tuner.v1.kernel_tuner_runner.create_storage_manager")
    def test_run_bucket_exhausted_retries_saves_fallback_result(
            self, mock_create_storage, mock_invoke_worker,
            mock_tuner_name_flag):
        from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                                  TuningStatus)
        from tools.kernel.tuner.v1.kernel_tuner_runner import _run_bucket

        mock_tuner_name_flag.value = "example_kernel_tuner"
        mock_invoke_worker.return_value = 0  # No progress
        mock_storage = MagicMock()
        mock_create_storage.return_value = mock_storage

        run_config = RunConfig(
            case_set_id="test_cs_retry",
            run_id="test_run_retry",
            case_set_desc="test",
            tpu_version="tpu7x",
            tpu_cores=2,
            tpu_queue_multi="test_queue",
            run_locally=True,
        )

        _run_bucket(run_config, 0, 1)

        self.assertEqual(mock_invoke_worker.call_count, 3)
        mock_storage.save_result.assert_called_once()
        saved_case_result = mock_storage.save_result.call_args[0][0]
        self.assertEqual(saved_case_result.case_id, 0)
        self.assertEqual(saved_case_result.processed_status,
                         TuningStatus.UNKNOWN_ERROR.value)
        mock_storage.flush.assert_called_once()


if __name__ == "__main__":
    absltest.main()
