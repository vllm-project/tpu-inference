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


@dataclass(order=True, frozen=True)
class MockTunableParams:
    size: int


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


if __name__ == "__main__":
    absltest.main()
