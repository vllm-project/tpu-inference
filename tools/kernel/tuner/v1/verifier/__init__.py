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
"""Correctness gate for kernel autotuning.

Composes three layers that an autotune candidate must pass before its
latency is recorded:

* ``reference_oracle`` — wraps the kernel's eager reference impl and exposes
  dtype-tier tolerances.
* ``numerics`` — NaN/Inf/shape sanity, cosine similarity, and dtype-aware
  ``allclose`` against the oracle's output.
* ``anti_cheat`` — single-trial sanity (zero/constant output, output equals
  an input) plus a cross-trial independence check.
* ``interpret_check`` — optional off-TPU correctness preview via
  ``pltpu.InterpretParams``.
* ``lm_eval_gate`` — optional outer eval-harness gate for the final winner.
"""

from tools.kernel.tuner.v1.verifier.anti_cheat import (
    AntiCheatGuard, AntiCheatReport, cross_trial_independence,
    detect_constant_output, detect_returns_input, detect_zero_output)
from tools.kernel.tuner.v1.verifier.numerics import (NumericsReport,
                                                     check_many, check_one)
from tools.kernel.tuner.v1.verifier.reference_oracle import (ReferenceOracle,
                                                             RpaV3Oracle,
                                                             rpa_v3_tolerance)

__all__ = [
    "AntiCheatGuard",
    "AntiCheatReport",
    "NumericsReport",
    "ReferenceOracle",
    "RpaV3Oracle",
    "check_many",
    "check_one",
    "cross_trial_independence",
    "detect_constant_output",
    "detect_returns_input",
    "detect_zero_output",
    "rpa_v3_tolerance",
]
