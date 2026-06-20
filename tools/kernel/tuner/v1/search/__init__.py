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
"""Pluggable search strategies for kernel autotuning.

Strategy contract (``strategy.SearchStrategy``):

* ``suggest()`` returns a candidate parameter dict.
* ``observe(params, score, aux)`` feeds back a score (``inf`` for any
  failure mode — OOM, verification failure, cost-model rejection).
* ``done()`` signals termination; the runner checks this every iteration.

Three concrete strategies are shipped:

* ``GridSearch`` — exhaustive Cartesian product, back-compat with v1.
* ``TpeSearch`` — Tree-structured Parzen Estimator via Optuna.
* ``EvolutionarySearch`` — simple (μ+λ)-EA with tournament selection.
"""

from tools.kernel.tuner.v1.search.evolutionary import EvolutionarySearch
from tools.kernel.tuner.v1.search.grid import GridSearch
from tools.kernel.tuner.v1.search.strategy import (IntChoice, IntLog2Range,
                                                   ParamRange, SearchSpace,
                                                   SearchStrategy)
from tools.kernel.tuner.v1.search.tpe import TpeSearch

SEARCH_STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {
    "grid": GridSearch,
    "tpe": TpeSearch,
    "evolutionary": EvolutionarySearch,
}

__all__ = [
    "EvolutionarySearch",
    "GridSearch",
    "IntChoice",
    "IntLog2Range",
    "ParamRange",
    "SEARCH_STRATEGY_REGISTRY",
    "SearchSpace",
    "SearchStrategy",
    "TpeSearch",
]
