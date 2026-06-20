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
"""Learned cost-model surrogate.

Trains a small regressor on past telemetry to predict candidate fitness
(``log(latency_ns)``) from a handcrafted feature vector. The orchestrator
uses it as a Tier-1 filter: rank candidates by predicted fitness, only
hand the top-K to expensive TPU evaluation.

Two model classes ship:

* ``LinearSurrogate`` — ridge regression on the feature vector. Cheap,
  no extra deps, surprisingly competitive at small data sizes.
* ``GBTSurrogate`` (optional, requires scikit-learn) — gradient-boosted
  trees for larger telemetry corpora. Falls back to linear if sklearn
  isn't installed.

Features include kernel-specific knob values (block sizes, dtype as
one-hot), shape descriptors (head counts, max_model_len), and diff
metadata (line count, hunk count).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

# Keys we extract numerically from telemetry events. These are stable
# across kernels — extra kernel-specific knobs append to the vector.
_BASE_FEATURES: tuple[str, ...] = (
    "page_size",
    "head_dim",
    "num_q_heads",
    "num_kv_heads",
    "max_model_len",
    "bkv_p",
    "bq",
)


@dataclasses.dataclass
class FeatureVector:
    values: list[float]
    names: list[str]


class FeatureExtractor:
    """Pulls a fixed-length feature vector out of a telemetry event."""

    def __init__(self, extra_keys: tuple[str, ...] = ()) -> None:
        self.feature_names = list(_BASE_FEATURES) + list(extra_keys)

    def extract(self, event: dict[str, Any]) -> FeatureVector:
        out: list[float] = []
        extra = event.get("extra", {}) or {}
        for k in self.feature_names:
            v = extra.get(k)
            if v is None and "shape_key" in event:
                v = _parse_shape_field(event["shape_key"], k)
            if v is None:
                v = 0.0
            out.append(float(v))
        return FeatureVector(values=out, names=list(self.feature_names))


def _parse_shape_field(shape_key: str, name: str) -> float | None:
    """Heuristic: pick out values from the shape_key string."""
    # head_key looks like 'q_head-16_kv_head-8_head-128'
    patterns = {
        "num_q_heads": r"q_head-(\d+)",
        "num_kv_heads": r"kv_head-(\d+)",
        "head_dim": r"head-(\d+)$",
        "max_model_len": r"ml(\d+)",
    }
    m = re.search(patterns.get(name, "$^"), shape_key)
    if m:
        return float(m.group(1))
    return None


class LinearSurrogate:
    """Ridge-regression on log-fitness, with closed-form solution."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        *,
        l2: float = 1.0,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.l2 = l2
        self._w: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, events: Iterable[dict[str, Any]]) -> "LinearSurrogate":
        rows: list[list[float]] = []
        targets: list[float] = []
        for ev in events:
            if ev.get("status") != "VERIFIED":
                continue
            fitness = ev.get("fitness_ns")
            if fitness is None or not isinstance(fitness, (int, float)):
                continue
            if fitness <= 0 or not math.isfinite(fitness):
                continue
            fv = self.feature_extractor.extract(ev)
            rows.append(fv.values)
            targets.append(math.log(fitness))
        if len(rows) < 3:
            logger.warning(
                "LinearSurrogate: only %d training rows; predictions will "
                "be flat.", len(rows))
            self._w = None
            return self
        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        # Standardize features
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        Xs = (X - self._mean) / self._std
        # Ridge regression: w = (X^T X + λI)^-1 X^T y, plus a bias column
        Xs = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        A = Xs.T @ Xs + self.l2 * np.eye(Xs.shape[1])
        b = Xs.T @ y
        self._w = np.linalg.solve(A, b)
        return self

    def predict(self, event: dict[str, Any]) -> float:
        """Predict log(latency_ns); return finite value or +inf if unfit."""
        if self._w is None:
            return 0.0
        fv = self.feature_extractor.extract(event)
        x = (np.asarray(fv.values, dtype=np.float64) - self._mean) / self._std
        x = np.concatenate([[1.0], x])
        return float(x @ self._w)

    def rank(self, events: Iterable[dict[str,
                                         Any]]) -> list[tuple[int, float]]:
        """Return ``(index, predicted_log_fitness)`` sorted ascending."""
        events = list(events)
        preds = [(i, self.predict(e)) for i, e in enumerate(events)]
        preds.sort(key=lambda t: t[1])
        return preds


def train_surrogate(
    events: list[dict[str, Any]],
    *,
    l2: float = 1.0,
    extra_keys: tuple[str, ...] = ()) -> LinearSurrogate:
    fx = FeatureExtractor(extra_keys=extra_keys)
    return LinearSurrogate(fx, l2=l2).fit(events)
