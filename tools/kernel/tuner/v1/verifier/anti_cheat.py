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
"""Anti-cheat detectors for kernel autotuning.

Failure modes documented in the Sakana AI CUDA Engineer and DeepReinforce
CUDA-L1 post-mortems:

* Kernel returns one of its input buffers verbatim.
* Kernel returns zeros or a constant (output dropped).
* Kernel returns before async work completes (caught by full
  ``block_until_ready`` in ``bench/harness.py``, not here).
* Kernel's output is independent of its inputs (caught by
  ``cross_trial_independence`` after a second seeded trial).

These detectors run per-trial against a single output buffer; the cross-trial
independence check requires two outputs from differently-seeded inputs.
"""

import dataclasses
from typing import Any, Iterable

import jax
import numpy as np


@dataclasses.dataclass
class AntiCheatReport:
    passed: bool
    reason: str | None = None


def detect_zero_output(actual: Any, *, tol: float = 1e-12) -> bool:
    """Return True if every element is within ``tol`` of zero."""
    a = np.asarray(jax.device_get(actual))
    return a.size > 0 and bool(np.all(np.abs(a) < tol))


def detect_constant_output(actual: Any, *, tol: float = 1e-6) -> bool:
    """Return True if the output is approximately constant (scale-invariant).

    Compares ``(max - min) / max(|max|, |min|)`` against ``tol``.
    """
    a = np.asarray(jax.device_get(actual), dtype=np.float64).reshape(-1)
    if a.size <= 1:
        return False
    mn = float(a.min())
    mx = float(a.max())
    scale = max(abs(mn), abs(mx), 1e-30)
    return (mx - mn) / scale < tol


def detect_returns_input(
        actual: Any,
        inputs: dict[str, Any],
        *,
        skip_keys: Iterable[str] = (),
) -> str | None:
    """If ``actual`` is bytewise identical to one of the inputs, return its key.

    Use ``skip_keys`` to mark inputs that may legitimately equal the output
    (e.g. ``kv_cache`` in an in-place update). The check tolerates only
    same-shape inputs; broadcasting-equal but different-shape inputs are
    ignored.
    """
    a = np.asarray(jax.device_get(actual))
    skip = set(skip_keys)
    for name, inp in inputs.items():
        if name in skip:
            continue
        if not hasattr(inp, "shape") or tuple(inp.shape) != tuple(a.shape):
            continue
        i = np.asarray(jax.device_get(inp))
        if a.dtype != i.dtype:
            continue
        if np.array_equal(a, i):
            return name
    return None


class AntiCheatGuard:
    """Bundle the single-trial anti-cheat checks into one inspection."""

    def __init__(
            self,
            *,
            check_zero: bool = True,
            check_constant: bool = True,
            check_returns_input: bool = True,
            constant_tol: float = 1e-6,
            zero_tol: float = 1e-12,
            input_skip_keys: Iterable[str] = (),
    ) -> None:
        self.check_zero = check_zero
        self.check_constant = check_constant
        self.check_returns_input = check_returns_input
        self.constant_tol = constant_tol
        self.zero_tol = zero_tol
        self.input_skip_keys = tuple(input_skip_keys)

    def inspect(
        self,
        actual: Any,
        inputs: dict[str, Any],
    ) -> AntiCheatReport:
        if self.check_zero and detect_zero_output(actual, tol=self.zero_tol):
            return AntiCheatReport(False, "output is all-zero")
        if self.check_constant and detect_constant_output(
                actual, tol=self.constant_tol):
            return AntiCheatReport(False, "output is constant")
        if self.check_returns_input:
            name = detect_returns_input(actual,
                                        inputs,
                                        skip_keys=self.input_skip_keys)
            if name is not None:
                return AntiCheatReport(
                    False,
                    f"output is bytewise identical to input '{name}'",
                )
        return AntiCheatReport(True)

    def inspect_many(
        self,
        actuals: Iterable[Any],
        inputs: dict[str, Any],
    ) -> AntiCheatReport:
        for a in actuals:
            rep = self.inspect(a, inputs)
            if not rep.passed:
                return rep
        return AntiCheatReport(True)


def cross_trial_independence(
    out_a: Any,
    out_b: Any,
    *,
    identical_tol: float = 1e-30,
) -> AntiCheatReport:
    """Verify that two outputs from differently-seeded inputs are not identical.

    A kernel that returns the same bytes regardless of input is ignoring its
    inputs entirely.
    """
    a = np.asarray(jax.device_get(out_a))
    b = np.asarray(jax.device_get(out_b))
    if a.shape != b.shape:
        return AntiCheatReport(True)
    if np.allclose(a, b, atol=identical_tol, rtol=identical_tol):
        return AntiCheatReport(
            False,
            "outputs from two different input seeds are bit-identical "
            "(kernel ignored its inputs)",
        )
    return AntiCheatReport(True)
