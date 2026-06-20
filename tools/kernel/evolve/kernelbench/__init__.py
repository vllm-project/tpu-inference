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
"""KernelBench-style TPU/Pallas benchmark suite.

Stanford's KernelBench measures `fast_p`: pass-rate × p-speedup-over-baseline
across 250 tasks. This package ports a curated Level-1/Level-2 subset to
Pallas TPU as a public reproducibility target.
"""

from tools.kernel.evolve.kernelbench.runner import (KernelBenchResult,
                                                    TpuKernelBench, run_subset)
from tools.kernel.evolve.kernelbench.tasks import TASKS, KernelTask

__all__ = [
    "KernelBenchResult",
    "KernelTask",
    "TASKS",
    "TpuKernelBench",
    "run_subset",
]
