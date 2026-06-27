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

import unittest
from unittest import mock

from tools.kernel.tuner.v1.optimizer import (BayesianOptimizer, SweepOptimizer,
                                             TuningOptimizer)


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


if __name__ == "__main__":
    unittest.main()
