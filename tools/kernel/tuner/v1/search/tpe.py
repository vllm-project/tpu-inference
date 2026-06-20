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
"""Tree-structured Parzen Estimator via Optuna.

Optuna's TPE handles the mixed-integer / categorical kernel tuning space
(``IntChoice``, ``IntLog2Range``) out of the box. We use the ask-and-tell API
so the search loop owns control flow.

Failed candidates (``score == inf``) are reported as ``PRUNED`` so they don't
poison the TPE density estimate.
"""

import logging
import math
from typing import Any

from tools.kernel.tuner.v1.search.strategy import (IntChoice, IntLog2Range,
                                                   ParamRange, SearchSpace,
                                                   SearchStrategy)

logger = logging.getLogger(__name__)


class TpeSearch(SearchStrategy):
    """TPE search wrapped to the SearchStrategy contract."""

    def __init__(
        self,
        *,
        space: SearchSpace,
        trial_budget: int,
        n_startup_trials: int = 10,
        seed: int | None = 1234,
    ) -> None:
        super().__init__(space=space, trial_budget=trial_budget)
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError as e:  # pragma: no cover - tested by import path
            raise RuntimeError(
                "TpeSearch requires Optuna. Install with `pip install optuna`."
            ) from e
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self._study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(n_startup_trials=n_startup_trials, seed=seed),
        )
        self._pending_trial = None

    def _suggest_one(self, trial, name: str, rng: ParamRange) -> Any:
        if isinstance(rng, (IntChoice, IntLog2Range)):
            return trial.suggest_categorical(name, rng.values())
        raise TypeError(
            f"TpeSearch does not support ParamRange type {type(rng).__name__} "
            f"(parameter {name!r}). Add a handler in TpeSearch._suggest_one.")

    def suggest(self) -> dict[str, Any]:
        self._pending_trial = self._study.ask()
        params: dict[str, Any] = {}
        for name in sorted(self.space.keys()):
            params[name] = self._suggest_one(self._pending_trial, name,
                                             self.space[name])
        return params

    def observe(
        self,
        params: dict[str, Any],
        score: float,
        aux: dict[str, Any] | None = None,
    ) -> None:
        super().observe(params, score, aux)
        trial = self._pending_trial
        self._pending_trial = None
        if trial is None:
            logger.warning(
                "TpeSearch.observe() called without a pending trial; "
                "params=%r score=%r", params, score)
            return
        if not math.isfinite(score):
            self._study.tell(trial, state=self._optuna.trial.TrialState.PRUNED)
        else:
            self._study.tell(trial, score)
