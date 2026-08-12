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
    from tools.kernel.tuner.v1.executor_process_manager import \
        ExecutorProcessManager


class TuningOptimizer(ABC):
    """Abstract base class for kernel tuning optimization strategies.

    Args:
        tuner: KernelTunerBase instance (for tuner_config, search_space, etc.).
        storage_manager: Storage manager for persisting results. Optional when
            only ``generate_tuning_jobs`` is needed (e.g. in the runner process).
        executor_mgr: ExecutorProcessManager for subprocess-isolated run()
            calls. Optional when only ``generate_tuning_jobs`` is needed.
    """

    def __init__(self,
                 tuner: "KernelTunerBase",
                 storage_manager=None,
                 executor_mgr: "ExecutorProcessManager | None" = None):
        self.tuner = tuner
        self.storage_manager = storage_manager
        self.executor_mgr = executor_mgr

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
    def measure_latency(self,
                        begin_case_id: int,
                        end_case_id: int,
                        bucket_id: int | None = None) -> int:
        """Measures latency of cases within the given bucket range.

        Args:
            begin_case_id: Start case ID (inclusive).
            end_case_id: End case ID (exclusive).
            bucket_id: Optional bucket identifier. If None, derived from begin_case_id.

        Returns:
            The next case ID to process (for partial completion / retry).
        """
        pass
