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

Each test instantiates a registered kernel tuner (identified by its name in
KERNEL_TUNER_REGISTRY), generates tuning buckets, and runs only the first case
of the first bucket via measure_latency (to keep test time manageable).  A test
is considered passing when every recorded result has a status of SUCCESS or
FAILED_OOM.

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
_ACCEPTABLE_STATUSES = frozenset(
    {TuningStatus.SUCCESS.value, TuningStatus.FAILED_OOM.value})


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
        """Core smoke-test logic shared by every per-tuner test method.

        Steps:
          1. Construct a RunConfig from the TPU environment variables.
          2. Instantiate the kernel tuner class from KERNEL_TUNER_REGISTRY.
          3. Replace its storage_manager with a LocalDbManager backed by a
             temporary directory so each test run is fully isolated.
          4. Call _generate_tuning_jobs() to obtain the list of work buckets and
             to persist the generated cases into the temp storage.
          5. For each bucket, call measure_latency(begin, begin+1) to execute
             only the first case in the bucket (keeps test time manageable).
          6. Assert that every result stored in the temp DB has a status of
             SUCCESS or FAILED_OOM.
        """
        run_config = self._make_run_config(kernel_tuner_name)
        kernel_tuner_cls = KERNEL_TUNER_REGISTRY[kernel_tuner_name]

        with tempfile.TemporaryDirectory() as tmp_dir:
            kernel_tuner = kernel_tuner_cls(run_config=run_config)
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

            # Only run the first bucket's first case since we only interested
            # in a smoke test that exercises the code path, not a full tuning run.
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

    # ------------------------------------------------------------------
    # One test method per entry in KERNEL_TUNER_REGISTRY
    # ------------------------------------------------------------------

    def test_example_kernel_tuner(self):
        self._run_tuner_smoke_test("example_kernel_tuner")

    def test_mla_kernel_tuner(self):
        self._run_tuner_smoke_test("mla_kernel_tuner")

    def test_batched_rpa_kernel_tuner(self):
        self._run_tuner_smoke_test("batched_rpa_kernel_tuner")


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


if __name__ == "__main__":
    absltest.main()
