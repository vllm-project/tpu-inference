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
"""Tests for the learned cost-model surrogate."""

import math
import random

from tools.kernel.evolve.cost_model.surrogate import (FeatureExtractor,
                                                      train_surrogate)


def _ev(*,
        bkv_p: int,
        bq: int,
        fitness: float | None,
        status: str = "VERIFIED") -> dict:
    return {
        "status": status,
        "fitness_ns": fitness,
        "shape_key": "q_head-16_kv_head-8_head-128/ml1024",
        "extra": {
            "page_size": 128,
            "head_dim": 128,
            "num_q_heads": 16,
            "num_kv_heads": 8,
            "max_model_len": 1024,
            "bkv_p": bkv_p,
            "bq": bq,
        },
    }


def test_feature_extractor_yields_base_vector():
    fx = FeatureExtractor()
    ev = _ev(bkv_p=8, bq=32, fitness=1.0)
    f = fx.extract(ev)
    assert "bkv_p" in f.names
    assert "bq" in f.names
    assert f.values[f.names.index("bkv_p")] == 8.0
    assert f.values[f.names.index("bq")] == 32.0


def test_linear_surrogate_fits_synthetic_quadratic():
    # Construct fitness ≈ exp(quadratic_in_bkv_p_and_bq) with a known optimum.
    optimum_bkv = 8
    optimum_bq = 32
    rng = random.Random(0)

    def fitness_for(b, q):
        # Convex around optimum; jitter. Scale by 1/500 keeps math.exp finite
        # even when (bkv_p, bq) is far from the optimum (e.g. (32, 128)).
        score = (b - optimum_bkv)**2 + (q - optimum_bq)**2
        return math.exp(score / 500.0 + rng.gauss(0, 0.05))

    events: list[dict] = []
    for bkv in (4, 8, 16, 32):
        for bq in (16, 32, 64, 128):
            events.append(_ev(bkv_p=bkv, bq=bq, fitness=fitness_for(bkv, bq)))
    surrogate = train_surrogate(events, l2=0.5)
    # Predicted fitness for the optimum should rank ≤ any non-optimum.
    opt = _ev(bkv_p=optimum_bkv, bq=optimum_bq, fitness=None, status="")
    bad = _ev(bkv_p=32, bq=128, fitness=None, status="")
    assert surrogate.predict(opt) < surrogate.predict(bad)


def test_surrogate_handles_empty_training_gracefully():
    surrogate = train_surrogate([], l2=1.0)
    # Should return a flat 0.0 prediction without error.
    assert surrogate.predict(_ev(bkv_p=8, bq=32, fitness=None,
                                 status="")) == 0.0


def test_surrogate_filters_non_verified_rows():
    events = [
        _ev(bkv_p=8, bq=32, fitness=100.0, status="VERIFIED"),
        _ev(bkv_p=16, bq=64, fitness=200.0, status="VERIFIED"),
        _ev(bkv_p=4, bq=16, fitness=None, status="FAILED_NUMERICS"),
        _ev(bkv_p=8, bq=128, fitness=400.0, status="FAILED_RUN"),
    ]
    s = train_surrogate(events)
    # Should still train on the two VERIFIED rows (will degenerate but
    # shouldn't crash). The predict call must run without raising.
    s.predict(_ev(bkv_p=8, bq=32, fitness=None, status=""))
