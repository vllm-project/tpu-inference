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
"""Off-TPU correctness pre-check using ``pltpu.InterpretParams``.

``InterpretParams`` (jax PR 25097) emulates HBM/VMEM and DMA semantics on CPU.
It catches race conditions, out-of-bounds reads, and missing DMA dependencies
that would otherwise burn a real TPU run. It does **not** verify timing.

Use this as a fast inner-loop sanity check before submitting a candidate to a
real-TPU run. Returns a ``NumericsReport`` for uniform composition with the
rest of the verifier stack.
"""

from typing import Any, Callable

from tools.kernel.tuner.v1.verifier.numerics import (COSINE_FLOOR_DEFAULT,
                                                     NumericsReport,
                                                     check_many)


def interpret_params_available() -> bool:
    """Return True if the running JAX exposes ``pltpu.InterpretParams``."""
    try:
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401
    except ImportError:
        return False
    return hasattr(pltpu, "InterpretParams")


def interpret_check(
    kernel_fn: Callable[[], Any],
    oracle,
    *,
    inputs: dict[str, Any],
    atol: float | None = None,
    rtol: float | None = None,
    cosine_floor: float = COSINE_FLOOR_DEFAULT,
) -> NumericsReport:
    """Run the kernel under interpret mode and compare against the oracle.

    Args:
        kernel_fn: zero-arg callable that runs the kernel with
            ``interpret=pltpu.InterpretParams(...)`` and returns its output.
            The caller is responsible for constructing the kernel with the
            interpret flag set.
        oracle: a ``ReferenceOracle`` whose ``compute(inputs)`` returns the
            reference output (single array or tuple).
        inputs: input dict passed to the oracle (so the same inputs gate both
            kernel and reference).
        atol / rtol: override the oracle's dtype tolerances; useful when the
            interpret mode's pseudo-DMA introduces extra rounding.
        cosine_floor: minimum cosine similarity to accept.
    """
    if not interpret_params_available():
        return NumericsReport(
            passed=False,
            max_abs_diff=float("inf"),
            cosine=0.0,
            nan_count=0,
            inf_count=0,
            reason="pltpu.InterpretParams unavailable in this JAX install",
        )
    actual = kernel_fn()
    reference = oracle.compute(inputs)
    if not isinstance(actual, (tuple, list)):
        actual = (actual, )
    if not isinstance(reference, (tuple, list)):
        reference = (reference, )
    if atol is None or rtol is None:
        atol_o, rtol_o = oracle.dtype_tolerance(actual[0].dtype)
        atol = atol if atol is not None else atol_o
        rtol = rtol if rtol is not None else rtol_o
    return check_many(actual,
                      reference,
                      atol=atol,
                      rtol=rtol,
                      cosine_floor=cosine_floor)
