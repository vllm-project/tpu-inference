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
"""Reproducible benchmark utilities for kernel autotuning.

Modules:

* ``harness`` — fixed-warmup / fresh-input timer that returns ``BenchResult``
  with p50 / p95 / mean and the last output (for verifier hand-off).
* ``cost_estimate`` — pluggable feasibility filter that lets the search loop
  skip clearly bad candidates before spending TPU time.
"""

from tools.kernel.tuner.v1.bench.cost_estimate import CostEstimate, CostModel
from tools.kernel.tuner.v1.bench.harness import BenchResult, block_all, measure

__all__ = [
    "BenchResult",
    "CostEstimate",
    "CostModel",
    "block_all",
    "measure",
]
