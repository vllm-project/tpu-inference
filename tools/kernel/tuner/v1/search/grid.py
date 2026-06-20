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
"""Exhaustive grid search — back-compat with v1 default behavior."""

import itertools
from typing import Any

from tools.kernel.tuner.v1.search.strategy import SearchSpace, SearchStrategy


class GridSearch(SearchStrategy):
    """Cartesian-product enumeration of the search space.

    Stops when either the product is exhausted or the trial budget is hit.
    Names are visited in sorted order to give deterministic behavior across
    runs. After exhaustion ``done()`` returns ``True`` immediately so the
    runner doesn't observe a duplicate fallback candidate.
    """

    def __init__(self, *, space: SearchSpace, trial_budget: int) -> None:
        super().__init__(space=space, trial_budget=trial_budget)
        self._names = sorted(space.keys())
        value_lists = [space[n].values() for n in self._names]
        full = [
            dict(zip(self._names, tup))
            for tup in itertools.product(*value_lists)
        ]
        self._pending: list[dict[str, Any]] = full[:trial_budget]
        self._fallback = {n: space[n].values()[0] for n in self._names}

    def suggest(self) -> dict[str, Any]:
        if not self._pending:
            return dict(self._fallback)
        return self._pending.pop(0)

    def done(self) -> bool:
        return (not self._pending) or super().done()
