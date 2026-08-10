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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.kernel.tuner.v1.common.kernel_tuner_base import KernelTunerBase


class TuningOptimizer(ABC):
    """Abstract base class for kernel tuning optimization strategies."""

    def __init__(self, tuner: "KernelTunerBase"):
        self.tuner = tuner

    @abstractmethod
    def generate_tuning_jobs(self, cases: list) -> list[tuple[int, int]]:
        """Partitions the cases into work buckets [begin_case_id, end_case_id).

        Args:
            cases: List of stored case tuples (CaseId, CaseKeyValue).

        Returns:
            A list of (begin_case_id, end_case_id) tuples.
        """
        pass

    @abstractmethod
    def measure_latency(self, begin_case_id: int, end_case_id: int) -> None:
        """Measures latency of cases within the given bucket range.

        Args:
            begin_case_id: Start case ID (inclusive).
            end_case_id: End case ID (exclusive).
        """
        pass
