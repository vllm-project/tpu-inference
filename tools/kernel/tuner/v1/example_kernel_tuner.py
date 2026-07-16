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
import itertools
import logging
import random
import time

from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase
from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TunerConfig,
                                                          TuningCase,
                                                          TuningStatus)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mock_kernel(key1, key2, param1, param2):
    # A mock kernel function that simulates a realistic latency surface so that
    # Bayesian optimization has something meaningful to minimize.  Each
    # (key1, key2) pair has an optimal (param1, param2) combination; deviating
    # from it increases latency.
    optimal_p1 = key1 * 16  # e.g. key1=1 -> 16, key1=2 -> 32, key1=4 -> 64
    optimal_p2 = key2 * 4  # e.g. key2=4 -> 16, key2=8 -> 32, key2=16 -> 64
    dist_p1 = abs(param1 - optimal_p1) / max(optimal_p1, 1)
    dist_p2 = abs(param2 - optimal_p2) / max(optimal_p2, 1)
    # Base 1 ms + up to ~20 ms penalty + small noise
    latency_s = 0.001 + (dist_p1 + dist_p2) * 0.01 + random.random() * 0.001
    time.sleep(latency_s)


@dataclasses.dataclass
class TuningKey:
    key1: int
    key2: int


@dataclasses.dataclass
class TunableParams:
    param1: int
    param2: int

    def __ge__(self, other) -> bool:
        return self.param1 >= other.param1 and self.param2 >= other.param2

    def __le__(self, other) -> bool:
        return self.param1 <= other.param1 and self.param2 <= other.param2


class ExampleKernelTuner(KernelTunerBase):
    # This is a reference implementation of a KernelTuner for testing purposes.
    # It defines a simple tuning key and tunable parameters, and simulates running
    # a kernel by sleeping for a random short duration. The latency returned is
    # not based on any real computation, but rather is just a placeholder to
    # demonstrate the tuning pipeline.

    def __init__(self, run_config: RunConfig):
        self.tuner_config = TunerConfig(
            tuning_key_class=TuningKey,
            tunable_params_class=TunableParams,
            kernel_tuner_name="example_kernel_tuner",
            support_bayesian_optimization=True,
            # Run 15 Bayesian trials per tuning-key bucket (search space has
            # 5×5 = 25 combinations, so 15 trials already demonstrate the
            # efficiency gain over a full sweep).
            n_bayesian_trials=15)
        self.run_config = run_config
        super().__init__(
            tuner_config=self.tuner_config,
            run_config=self.run_config)  # Use a small bucket size for testing

    def get_search_space(self, tuning_key: TuningKey) -> dict:
        # The search space can depend on the tuning key.  Here we use a larger
        # param2 range for bigger key2 values to make the space more realistic.
        param2_values = [8, 16, 32, 64, 128]
        if tuning_key.key2 >= 8:
            param2_values = [16, 32, 64, 128, 256]
        return {
            'param1': [8, 16, 32, 64, 128],
            'param2': param2_values,
        }

    def generate_cases(self) -> list[TuningCase]:
        # Generate tuning cases by taking the Cartesian product of all TuningKey
        # combinations and the per-key search space returned by get_search_space.
        # This supports both the original sweep mode and Bayesian optimization
        # (where measure_latency uses optuna to select a subset of these cases).
        key1_values = [1, 2, 4]
        key2_values = [4, 8, 16]
        cases = []
        for k1, k2 in itertools.product(key1_values, key2_values):
            tuning_key = TuningKey(key1=k1, key2=k2)
            search_space = self.get_search_space(tuning_key)
            for params_combo in itertools.product(*search_space.values()):
                params_dict = dict(zip(search_space.keys(), params_combo))
                tunable_params = TunableParams(**params_dict)
                cases.append(TuningCase(tuning_key, tunable_params))
        return cases

    def generate_inputs(self, tuning_key: TuningKey):
        # Generate some mock inputs for the kernel based on the tuning key.
        if self._tuning_key and tuning_key == self._tuning_key:
            return self._kernel_inputs_cache
        self._tuning_key = tuning_key
        self._kernel_inputs_cache = {
            'input1': tuning_key.key1,
            'input2': tuning_key.key2
        }
        return self._kernel_inputs_cache

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        # Run the mock kernel with the given tuning key and tunable params, and
        # return the latency.
        logger.debug(
            f"Running mock kernel with tuning_key={tuning_key}, tunable_params={tunable_params}, iters={iters}"
        )
        start_ns = time.perf_counter_ns()
        for i in range(iters):
            mock_kernel(tuning_key.key1, tuning_key.key2,
                        tunable_params.param1, tunable_params.param2)
        end_ns = time.perf_counter_ns()
        latency_ns = (end_ns - start_ns)
        return TuningStatus.SUCCESS, latency_ns / iters, latency_ns  # status, average latency, total latency
