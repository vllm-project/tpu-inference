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
"""Multi-tier numerical comparison between a candidate output and an oracle.

Layers in order (first failure short-circuits and is reported):
    1. Shape match.
    2. NaN/Inf-free output.
    3. Cosine similarity above floor (catches a dropped layer that still
       passes a loose tolerance).
    4. dtype-aware ``allclose`` (atol, rtol from the oracle).

The default cosine floor (0.9999) was chosen so a single missing block in a
128-token attention output drops the similarity below the floor while still
being permissive of bf16/fp8 quantization noise.
"""

import dataclasses
import math
from typing import Any, Sequence

import jax
import numpy as np

# A candidate that scores below this on the (post-NaN/Inf) cosine check is
# rejected even if it would otherwise pass allclose with a loose tolerance.
COSINE_FLOOR_DEFAULT = 0.9999


@dataclasses.dataclass
class NumericsReport:
    """Summary of a single verification attempt."""
    passed: bool
    max_abs_diff: float
    cosine: float
    nan_count: int
    inf_count: int
    reason: str | None = None


def _to_np_f32(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=np.float32).reshape(-1)


def _cosine(a_flat: np.ndarray, b_flat: np.ndarray) -> float:
    na = float(np.linalg.norm(a_flat))
    nb = float(np.linalg.norm(b_flat))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (na * nb))


def check_one(
    actual: Any,
    reference: Any,
    *,
    atol: float,
    rtol: float,
    cosine_floor: float = COSINE_FLOOR_DEFAULT,
) -> NumericsReport:
    """Compare a single (actual, reference) pair against the multi-tier gate."""
    a_raw = np.asarray(jax.device_get(actual))
    r_raw = np.asarray(jax.device_get(reference))
    if a_raw.shape != r_raw.shape:
        return NumericsReport(
            passed=False,
            max_abs_diff=math.inf,
            cosine=0.0,
            nan_count=0,
            inf_count=0,
            reason=
            f"shape mismatch: actual {a_raw.shape} vs reference {r_raw.shape}",
        )
    nan_count = int(np.isnan(a_raw).sum())
    inf_count = int(np.isinf(a_raw).sum())
    if nan_count > 0:
        return NumericsReport(
            passed=False,
            max_abs_diff=math.inf,
            cosine=0.0,
            nan_count=nan_count,
            inf_count=inf_count,
            reason=f"actual has {nan_count} NaNs",
        )
    if inf_count > 0:
        return NumericsReport(
            passed=False,
            max_abs_diff=math.inf,
            cosine=0.0,
            nan_count=nan_count,
            inf_count=inf_count,
            reason=f"actual has {inf_count} Infs",
        )
    af = a_raw.astype(np.float32, copy=False).reshape(-1)
    rf = r_raw.astype(np.float32, copy=False).reshape(-1)
    max_abs_diff = float(np.max(np.abs(af - rf))) if af.size else 0.0
    cosine = _cosine(af, rf)
    if cosine < cosine_floor:
        return NumericsReport(
            passed=False,
            max_abs_diff=max_abs_diff,
            cosine=cosine,
            nan_count=nan_count,
            inf_count=inf_count,
            reason=f"cosine {cosine:.6f} < floor {cosine_floor}",
        )
    if not np.allclose(af, rf, atol=atol, rtol=rtol):
        rel = np.abs(af - rf) / (atol + rtol * np.abs(rf) + 1e-30)
        idx = int(np.argmax(rel))
        return NumericsReport(
            passed=False,
            max_abs_diff=max_abs_diff,
            cosine=cosine,
            nan_count=nan_count,
            inf_count=inf_count,
            reason=(f"allclose(atol={atol}, rtol={rtol}) failed at flat "
                    f"idx {idx}: actual={float(af[idx]):.6g} "
                    f"ref={float(rf[idx]):.6g}"),
        )
    return NumericsReport(
        passed=True,
        max_abs_diff=max_abs_diff,
        cosine=cosine,
        nan_count=0,
        inf_count=0,
    )


def check_many(
    actuals: Sequence[Any],
    references: Sequence[Any],
    *,
    atol: float,
    rtol: float,
    cosine_floor: float = COSINE_FLOOR_DEFAULT,
) -> NumericsReport:
    """Run the multi-tier check on each pair; short-circuit on first failure.

    On success, the aggregate report carries the worst observed
    ``max_abs_diff`` and the lowest cosine seen.
    """
    if len(actuals) != len(references):
        raise ValueError(f"actuals and references must have same length: "
                         f"{len(actuals)} vs {len(references)}")
    worst_mad = 0.0
    min_cosine = 1.0
    for a, r in zip(actuals, references):
        rep = check_one(a, r, atol=atol, rtol=rtol, cosine_floor=cosine_floor)
        if not rep.passed:
            return rep
        worst_mad = max(worst_mad, rep.max_abs_diff)
        min_cosine = min(min_cosine, rep.cosine)
    return NumericsReport(
        passed=True,
        max_abs_diff=worst_mad,
        cosine=min_cosine,
        nan_count=0,
        inf_count=0,
    )
