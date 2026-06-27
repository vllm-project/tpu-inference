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

from absl import flags
from absl.testing import absltest

from tools.kernel.tuner.v1.common.kernel_tuner_base import (RunConfig,
                                                            TuningStatus)
# Importing kernel_tuner_runner registers the absl flags FLAGS.debug and
# FLAGS.worker_id that KernelTunerBase.measure_latency reads at runtime.
from tools.kernel.tuner.v1.kernel_tuner_runner import (
    KERNEL_TUNER_REGISTRY, get_tpu_queue_by_version_and_cores)
from tools.kernel.tuner.v1.storage_management.local_db_manager import \
    LocalDbManager

FLAGS = flags.FLAGS

# Statuses that are acceptable outcomes for a tuning case in a smoke test.
# SKIPPED is added because Bayesian optimization prunes cases that are
# expected to OOM based on smaller configurations that already failed.
_ACCEPTABLE_STATUSES = frozenset({
    TuningStatus.SUCCESS.value,
    TuningStatus.FAILED_OOM.value,
    TuningStatus.SKIPPED.value,
})


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

    def _make_run_config(self, kernel_tuner_name: str) -> RunConfig:
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
        )

    def _run_tuner_smoke_test(self, kernel_tuner_name: str) -> None:
        """Sweep-mode smoke test shared by every per-tuner test method.

        Forces ``support_bayesian_optimization=False`` so that exactly one case
        (the first case of the first bucket) is evaluated in a single
        ``measure_latency`` call.  This keeps the test fast regardless of the
        tuner's default optimization mode.

        Steps:
          1. Construct a RunConfig from the TPU environment variables.
          2. Instantiate the kernel tuner class from KERNEL_TUNER_REGISTRY.
          3. Force sweep mode by setting
             ``tuner_config.support_bayesian_optimization = False``.
          4. Replace its storage_manager with a LocalDbManager backed by a
             temporary directory so each test run is fully isolated.
          5. Call _generate_tuning_jobs() to obtain the list of work buckets and
             persist the generated cases into the temp storage.
          6. Call measure_latency(begin, begin+1) to execute only the first case
             of the first bucket.
          7. Assert that every result stored in the temp DB has an acceptable
             status (SUCCESS, FAILED_OOM, or SKIPPED).
        """
        run_config = self._make_run_config(kernel_tuner_name)
        kernel_tuner_cls = KERNEL_TUNER_REGISTRY[kernel_tuner_name]

        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_tuner = kernel_tuner_cls(run_config=run_config)
            # Force sweep mode so we can evaluate just one case per bucket and
            # keep test time manageable, regardless of the tuner's default.
            kernel_tuner.tuner_config.support_bayesian_optimization = False
            # Redirect storage to the temp directory so tests are isolated and
            # do not leave state in the default /tmp/kernel_tuner_run_* path.
            kernel_tuner.storage_manager = LocalDbManager(db_path=tmp_dir)

            buckets = kernel_tuner._generate_tuning_jobs()
            self.assertGreater(
                len(buckets),
                0,
                msg=(f"{kernel_tuner_name}: _generate_tuning_jobs() returned "
                     "no buckets"),
            )

            # Only run the first case of the first bucket.
            begin_case_id, _ = buckets[0]
            kernel_tuner.measure_latency(begin_case_id, begin_case_id + 1)

            results = kernel_tuner.storage_manager._read_table("CaseResults")
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
        """Bayesian-mode smoke test: runs one full bucket with a capped trial count.

        Forces ``support_bayesian_optimization=True`` and sets
        ``n_bayesian_trials=n_trials`` (default 3) so optuna runs quickly.
        Unlike the sweep smoke test, the *full* first bucket is passed to
        ``measure_latency`` so that optuna has the complete search space for
        one TuningKey and can always map its suggestions to stored case IDs.

        Asserts:
          - At least one result is recorded in the DB.
          - No more than ``n_trials`` results are recorded (each trial produces
            at most one DB entry).
          - Every recorded status is acceptable (SUCCESS, FAILED_OOM, SKIPPED).
        """
        run_config = self._make_run_config(kernel_tuner_name)
        kernel_tuner_cls = KERNEL_TUNER_REGISTRY[kernel_tuner_name]

        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_tuner = kernel_tuner_cls(run_config=run_config)
            # Force Bayesian mode with a very small trial budget for speed.
            kernel_tuner.tuner_config.support_bayesian_optimization = True
            kernel_tuner.tuner_config.n_bayesian_trials = n_trials
            kernel_tuner.storage_manager = LocalDbManager(db_path=tmp_dir)

            buckets = kernel_tuner._generate_tuning_jobs()
            self.assertGreater(
                len(buckets),
                0,
                msg=(f"{kernel_tuner_name}: _generate_tuning_jobs() returned "
                     "no buckets in Bayesian mode"),
            )

            # Run the full first bucket so optuna can map every suggestion to
            # a pre-stored case_id.
            begin_case_id, end_case_id = buckets[0]
            kernel_tuner.measure_latency(begin_case_id, end_case_id)

            results = kernel_tuner.storage_manager._read_table("CaseResults")
            self.assertGreater(
                len(results),
                0,
                msg=(f"{kernel_tuner_name}: no results recorded after "
                     "Bayesian measure_latency"),
            )
            self.assertLessEqual(
                len(results),
                n_trials,
                msg=(
                    f"{kernel_tuner_name}: expected at most {n_trials} results "
                    f"(one per trial) but got {len(results)}"),
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
            job_bucket_size=100,
        )
        return ExampleKernelTuner(run_config=run_config), TuningKey

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


if __name__ == "__main__":
    absltest.main()
