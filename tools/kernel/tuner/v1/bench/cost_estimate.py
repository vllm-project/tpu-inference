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
"""Pre-filter for skipping infeasible kernel candidates before TPU runs.

Pallas does not yet expose a uniform cost-model API across kernels, so the
``CostModel`` here delegates to a per-kernel ``estimate`` callable that the
tuner supplies (typically a thin wrapper over the kernel's own
``get_vmem_estimate_bytes``-style helper).

A typical estimator returns a ``CostEstimate`` with VMEM/SMEM bytes; the
search loop skips candidates whose ``feasible`` is false before issuing a
compile or run.
"""

import dataclasses
from typing import Any, Callable


@dataclasses.dataclass
class CostEstimate:
    """Per-candidate feasibility / cost summary."""
    vmem_bytes: int | None = None
    smem_bytes: int | None = None
    flops: int | None = None
    reason: str | None = None  # populated when the candidate is infeasible

    @property
    def feasible(self) -> bool:
        return self.reason is None


class CostModel:
    """Wrap a per-kernel feasibility estimator."""

    def __init__(self, estimate: Callable[[Any, Any], CostEstimate]) -> None:
        self._estimate = estimate

    def estimate(self, tuning_key: Any, tunable_params: Any) -> CostEstimate:
        return self._estimate(tuning_key, tunable_params)

    def is_feasible(
        self,
        tuning_key: Any,
        tunable_params: Any,
    ) -> tuple[bool, str | None]:
        est = self.estimate(tuning_key, tunable_params)
        return est.feasible, est.reason
