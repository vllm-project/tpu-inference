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
"""Generic kernel host adapters for non-RPA-v3 kernels.

Adding a new kernel target to the evolve loop used to require a full
``KernelHost`` subclass (~150 LOC of input generation, oracle wiring,
build_kernel_fn closures). ``GenericHost`` here lets you wire a new kernel
with ~20 LOC: provide the kernel source path, the kernel entry-point
name, a callable that builds inputs, a callable that computes the eager
reference, and (optionally) tolerances + anti-cheat-skip keys.

The host is decoupled from the kernel-tuner v1 machinery — no
``RpaV3KernelTuner`` style adapter is needed. This is the right
abstraction for `quantized_matmul`, `mla_v2`, `fused_moe_v1`, and any
future kernel that has a Python reference implementation.

Two concrete hosts ship today as worked examples:

* ``QuantizedMatmulHost`` — production kernel
  `tpu_inference.kernels.quantized_matmul.kernel.quantized_matmul_kernel`,
  reference = pure-JAX `x @ (w_q * w_scale).T`.
* ``FusedMoeV1Host`` — `tpu_inference.kernels.fused_moe.v1.kernel`,
  reference = exported `ref_moe`.

The MLA v2 + RPA v3 hosts live in their own modules because they need
specialized input generation and use ref impls that aren't simple
expressions.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclasses.dataclass
class _SimpleOracle:
    """Adapter that fits the ``ReferenceOracle`` Protocol the evaluator uses.

    ``compute`` builds a dict-keyed view of the inputs and runs the
    user-supplied reference function; tolerances are dtype-driven from a
    static map.
    """
    reference_fn: Callable[[dict], jax.Array | tuple[jax.Array, ...]]
    tol_by_bits: dict[int, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: {
            32: (0.15, 0.15),
            16: (0.2, 0.2),
            8: (0.5, 0.5),
            4: (0.5, 0.5),
        })

    def compute(self, inputs: dict[str, Any]) -> jax.Array:
        out = self.reference_fn(inputs)
        if isinstance(out, tuple):
            return out[0]
        return out

    def dtype_tolerance(self, dtype) -> tuple[float, float]:
        bits = jnp.dtype(dtype).itemsize * 8
        return self.tol_by_bits.get(bits, (0.5, 0.5))


class GenericHost:
    """``KernelHost``-shaped adapter parameterized by config.

    Required:
    * ``kernel_name``        — short identifier (also used as worktree
                                 cache-key prefix and CLI/file naming)
    * ``kernel_symbol``      — top-level callable exported from the
                                 mutated kernel module
    * ``baseline_path_rel``  — repo-relative path to the kernel source
                                 file the diff applies to
    * ``build_inputs``       — () -> dict of named jax arrays
    * ``reference_fn``       — fn(inputs_dict) -> jax.Array (or tuple
                                 whose [0] is the comparable output)
    * ``call_kernel``        — fn(kernel, inputs_dict) -> jax.Array (or
                                 tuple whose [0] is the comparable
                                 output)

    Optional:
    * ``anti_cheat_skip``    — tuple of input keys to skip in the
                                 input-aliasing detector (kernels that
                                 legitimately return one of their inputs
                                 as part of the output need this — see
                                 RPA v3's kv_cache)
    * ``tol_by_bits``        — override the default per-dtype-bit-width
                                 (atol, rtol) tolerance map
    """

    def __init__(
            self,
            *,
            kernel_name: str,
            kernel_symbol: str,
            baseline_path_rel: str,
            build_inputs: Callable[[], dict],
            reference_fn: Callable[[dict], Any],
            call_kernel: Callable[[Any, dict], Any],
            anti_cheat_skip: tuple[str, ...] = (),
            tol_by_bits: dict[int, tuple[float, float]] | None = None) -> None:
        self.kernel_name = kernel_name
        self.kernel_symbol = kernel_symbol
        self._baseline_path_rel = baseline_path_rel
        self._inputs = build_inputs()
        self._call_kernel = call_kernel
        if tol_by_bits is None:
            self._oracle = _SimpleOracle(reference_fn=reference_fn)
        else:
            self._oracle = _SimpleOracle(reference_fn=reference_fn,
                                         tol_by_bits=tol_by_bits)
        self._anti_cheat_skip = anti_cheat_skip

    @property
    def baseline_path(self) -> str:
        return self._baseline_path_rel

    def read_baseline_source(self) -> str:
        return (_REPO_ROOT / self._baseline_path_rel).read_text()

    @property
    def inputs(self):
        return self._inputs

    def get_oracle(self):
        return self._oracle

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        return self._anti_cheat_skip

    def build_kernel_fn(self, module: Any) -> Callable[[], Any]:
        kernel = getattr(module, self.kernel_symbol)
        call = self._call_kernel
        inputs = self._inputs

        def fn():
            out = call(kernel, inputs)
            if isinstance(out, tuple):
                return out[0]
            return out

        return fn


# ---------- Concrete configurations ----------


def make_quantized_matmul_host(
    *,
    n_batch: int = 1024,
    n_in: int = 4096,
    n_out: int = 4096,
    seed: int = 0,
) -> GenericHost:
    """Wire ``quantized_matmul_kernel`` into the evolve loop.

    Reference is pure-JAX dequant + matmul:  out = x @ (w_q * w_scale[:, None]).T
    """
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(0, 0.1,
                               size=(n_batch, n_in)).astype(np.float32),
                    dtype=jnp.bfloat16)
    w_q = jnp.asarray(
        rng.integers(-127, 127, size=(n_out, n_in)).astype(np.int8))
    w_scale = jnp.asarray(rng.normal(0, 0.05,
                                     size=(n_out, )).astype(np.float32),
                          dtype=jnp.bfloat16)

    def build_inputs():
        return {"x": x, "w_q": w_q, "w_scale": w_scale}

    def reference_fn(inputs):
        # Dequantize then matmul in bf16 (matches the kernel's compute path)
        w_deq = inputs["w_q"].astype(jnp.bfloat16) * inputs["w_scale"][:, None]
        return jnp.matmul(inputs["x"], w_deq.T)

    def call_kernel(kernel, inputs):
        return kernel(jnp.copy(inputs["x"]), jnp.copy(inputs["w_q"]),
                      jnp.copy(inputs["w_scale"]))

    return GenericHost(
        kernel_name="quantized_matmul",
        kernel_symbol="quantized_matmul_kernel",
        baseline_path_rel=("tpu_inference/kernels/quantized_matmul/"
                           "kernel.py"),
        build_inputs=build_inputs,
        reference_fn=reference_fn,
        call_kernel=call_kernel,
        # int8 weight × bf16 activation — relaxed tolerance per #1841
        tol_by_bits={
            8: (0.6, 0.6),
            16: (0.3, 0.3),
            32: (0.15, 0.15)
        },
    )


def make_fused_moe_v1_host(
    *,
    num_tokens: int = 128,
    hidden_size: int = 512,
    intermediate_size: int = 1024,
    num_experts: int = 8,
    topk: int = 2,
    seed: int = 0,
) -> GenericHost:
    """Wire fused_moe v1 into the evolve loop, using the exported
    ``ref_moe`` reference impl.

    The exact signature varies; this stub uses a minimal harness — the
    user may need to update ``call_kernel`` if the kernel signature drifts
    in the production source.
    """
    rng = np.random.default_rng(seed)
    hidden = jnp.asarray(rng.normal(0, 0.1,
                                    size=(num_tokens,
                                          hidden_size)).astype(np.float32),
                         dtype=jnp.bfloat16)
    w13 = jnp.asarray(rng.normal(0,
                                 0.05,
                                 size=(num_experts, hidden_size,
                                       intermediate_size * 2)).astype(
                                           np.float32),
                      dtype=jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0,
                                0.05,
                                size=(num_experts, intermediate_size,
                                      hidden_size)).astype(np.float32),
                     dtype=jnp.bfloat16)
    topk_indices = jnp.asarray(
        rng.integers(0, num_experts, size=(num_tokens, topk)).astype(np.int32))
    topk_weights = jnp.asarray(rng.uniform(0, 1,
                                           size=(num_tokens,
                                                 topk)).astype(np.float32),
                               dtype=jnp.bfloat16)

    def build_inputs():
        return {
            "hidden": hidden,
            "w13": w13,
            "w2": w2,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights
        }

    def reference_fn(inputs):
        # The exported ref_moe lives in the kernel module. We compute a
        # simple gather-+-matmul-+-combine pure-JAX reference here so
        # this Host doesn't depend on a specific export signature.
        from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
        return ref_moe(inputs["hidden"], inputs["w13"], inputs["w2"],
                       inputs["topk_indices"], inputs["topk_weights"])

    def call_kernel(kernel, inputs):
        # Production kernel signature — adjust here if it changes.
        return kernel(jnp.copy(inputs["hidden"]), jnp.copy(inputs["w13"]),
                      jnp.copy(inputs["w2"]), inputs["topk_indices"],
                      inputs["topk_weights"])

    return GenericHost(
        kernel_name="fused_moe_v1",
        kernel_symbol="fused_moe",
        baseline_path_rel="tpu_inference/kernels/fused_moe/v1/kernel.py",
        build_inputs=build_inputs,
        reference_fn=reference_fn,
        call_kernel=call_kernel,
    )
