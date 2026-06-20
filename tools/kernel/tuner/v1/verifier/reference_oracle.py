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
"""Reference oracles for kernel autotuning.

An oracle pairs an eager reference implementation with a dtype-aware tolerance
table. The tuner compares each candidate's output to ``oracle.compute(inputs)``
using ``oracle.dtype_tolerance(dtype)`` as the gate.
"""

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

# Tolerance table lifted verbatim from
# tests/kernels/ragged_paged_attention_kernel_v3_test.py:185-193 so the
# autotuner's pass/fail matches the kernel's own test rigor.
_RPA_V3_TOLS_PER_DTYPE_BITS = {32: 0.15, 16: 0.2, 8: 0.2, 4: 0.2}


def rpa_v3_tolerance(dtype: Any) -> tuple[float, float]:
    """Return ``(atol, rtol)`` for an RPA v3 output of the given dtype."""
    bits = jnp.dtype(dtype).itemsize * 8
    if bits not in _RPA_V3_TOLS_PER_DTYPE_BITS:
        raise ValueError(
            f"Unsupported dtype {dtype!r} for RPA v3 oracle (bits={bits})")
    tol = _RPA_V3_TOLS_PER_DTYPE_BITS[bits]
    return tol, tol


@runtime_checkable
class ReferenceOracle(Protocol):
    """Protocol for a reference oracle used to gate kernel candidates."""

    def compute(self, inputs: dict[str, Any]) -> Any:
        """Return the reference output(s) for the supplied inputs.

        The return type matches what the kernel under test returns: a single
        ``jax.Array`` or a tuple thereof. Tuples are compared element-wise.
        """
        ...

    def dtype_tolerance(self, dtype: Any) -> tuple[float, float]:
        """Return ``(atol, rtol)`` for the given output dtype."""
        ...


class RpaV3Oracle:
    """Oracle for ``ragged_paged_attention`` v3.

    Wraps ``ref_ragged_paged_attention`` from
    ``tpu_inference.kernels.ragged_paged_attention.v3.kernel`` — the same
    eager reference used by ``tests/kernels/ragged_paged_attention_kernel_v3_test.py``.

    Inputs dict keys (matching ``RpaV3KernelTuner.generate_inputs``):
        ``q``, ``k``, ``v``, ``kv_cache``, ``kv_lens``, ``page_indices``,
        ``cu_q_lens``, ``distribution``.
    """

    def __init__(self, semantic_kwargs: dict[str, Any] | None = None) -> None:
        self.semantic_kwargs = dict(semantic_kwargs or {})

    def compute(self, inputs: dict[str, Any]) -> jax.Array:
        # The reference returns ``(output, updated_kv_cache)``; we expose only
        # ``output`` because ``updated_kv_cache`` has NaN padding by design
        # (see tests/kernels/ragged_paged_attention_kernel_v3_test.py:194,
        # which masks NaNs before comparing kv_cache). Comparing the cache
        # bytewise would falsely flag every candidate.
        from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
            ref_ragged_paged_attention
        out, _ = ref_ragged_paged_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["kv_cache"],
            inputs["kv_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["distribution"],
            **self.semantic_kwargs,
        )
        return out

    def dtype_tolerance(self, dtype: Any) -> tuple[float, float]:
        return rpa_v3_tolerance(dtype)
