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

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            TuningCase,
                                                            TuningStatus)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mock_kernel(key1, key2, param1, param2):
    # A mock kernel function that takes some time to execute and returns a
    # latency based on the input parameters.
    time.sleep(random.random() / 5)  # Simulate some computation time


@dataclasses.dataclass
class TuningKey:
    key1: int
    key2: int


@dataclasses.dataclass
class TunableParams:
    param1: int
    param2: int


class ExampleKernelTuner(KernelTunerBase):
    # This is a reference implementation of a KernelTuner for testing purposes.
    # It defines a simple tuning key and tunable parameters, and simulates running
    # a kernel by sleeping for a random short duration. The latency returned is
    # not based on any real computation, but rather is just a placeholder to
    # demonstrate the tuning pipeline.

    def __init__(self, storage_manager):
        super().__init__(tuning_key_class=TuningKey,
                         tunable_params_class=TunableParams,
                         storage_manager=storage_manager,
                         job_bucket_size=2,
                         kernel_tuner_name="example_kernel_tuner"
                         )  # Use a small bucket size for testing

    def generate_cases(self) -> list[TuningCase]:
        # Generate some mock tuning cases based on the case_set_id and desc.
        key1_values = [1, 2]
        key2_values = [4, 5]
        param1_values = [7]
        param2_values = [10, 11]
        cases = []
        for k1, k2, p1, p2 in itertools.product(key1_values, key2_values,
                                                param1_values, param2_values):
            tuning_key = TuningKey(key1=k1, key2=k2)
            tunable_params = TunableParams(param1=p1, param2=p2)
            cases.append(TuningCase(tuning_key, tunable_params))
        return cases

    def generate_inputs(self, tuning_key: TuningKey):
        # Generate some mock inputs for the kernel based on the tuning key.
        if self._TUNING_KEY and tuning_key == self._TUNING_KEY:
            return self._KERNEL_INPUTS_CACHE
        self._TUNING_KEY = tuning_key
        self._KERNEL_INPUTS_CACHE = {
            'input1': tuning_key.key1,
            'input2': tuning_key.key2
        }
        return self._KERNEL_INPUTS_CACHE

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
