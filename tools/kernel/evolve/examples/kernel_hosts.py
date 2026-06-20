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
    """Wire ``fused_ep_moe`` v1 into the evolve loop.

    The production kernel needs (a) a 2D JAX mesh with ``ep_axis_name``
    axis (single-device mesh so EP=1 for the bench), (b) ``w1`` shaped
    ``(num_experts, 2, hidden_size, intermediate_size)`` for fused
    gate/up, (c) ``gating_output`` of shape ``(num_tokens, num_experts)``
    — NOT top-k indices (those are computed inside).

    Shape constraints asserted by the kernel:
    * ``hidden_size % 128 == 0`` and ``intermediate_size % 128 == 0``
    * ``num_tokens % ep_size == 0`` (we use ep_size=1)
    """
    import jax
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError("fused_moe v1 requires hidden_size and "
                         "intermediate_size divisible by 128")
    rng = np.random.default_rng(seed)
    tokens = jnp.asarray(rng.normal(0, 0.1,
                                    size=(num_tokens,
                                          hidden_size)).astype(np.float32),
                         dtype=jnp.bfloat16)
    # w1[expert][0]=gate, w1[expert][1]=up
    w1 = jnp.asarray(rng.normal(0,
                                0.05,
                                size=(num_experts, 2, hidden_size,
                                      intermediate_size)).astype(np.float32),
                     dtype=jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(0,
                                0.05,
                                size=(num_experts, intermediate_size,
                                      hidden_size)).astype(np.float32),
                     dtype=jnp.bfloat16)
    # gating_output must share dtype with tokens (the kernel's internal
    # VMEM scratch for it is bf16 — DMA mismatch otherwise).
    gating_output = jnp.asarray(rng.normal(
        0, 1, size=(num_tokens, num_experts)).astype(np.float32),
                                dtype=jnp.bfloat16)
    # Single-device EP=1 mesh — ep_axis_name='model' matches the kernel
    # default. The kernel asserts a 2D mesh, so we include 'data' too.
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model"))

    def build_inputs():
        return {
            "tokens": tokens,
            "w1": w1,
            "w2": w2,
            "gating_output": gating_output,
            "top_k": topk,
            "mesh": mesh,
        }

    def reference_fn(inputs):
        from tpu_inference.kernels.fused_moe.v1.kernel import ref_moe
        return ref_moe(inputs["tokens"], inputs["w1"], inputs["w2"],
                       inputs["gating_output"], inputs["top_k"])

    def call_kernel(kernel, inputs):
        return kernel(inputs["mesh"], jnp.copy(inputs["tokens"]),
                      jnp.copy(inputs["w1"]), jnp.copy(inputs["w2"]),
                      inputs["gating_output"], inputs["top_k"])

    return GenericHost(
        kernel_name="fused_moe_v1",
        kernel_symbol="fused_ep_moe",
        baseline_path_rel="tpu_inference/kernels/fused_moe/v1/kernel.py",
        build_inputs=build_inputs,
        reference_fn=reference_fn,
        call_kernel=call_kernel,
        # MoE with bf16 weights + per-token routing → wider tolerance
        # because reference computes per-token loops while kernel batches.
        tol_by_bits={
            16: (0.5, 0.5),
            32: (0.15, 0.15)
        },
    )


def make_mla_v2_host(
    *,
    seq_lens: tuple[tuple[int, int], ...] = ((1, 128), (1, 256), (4, 384)),
    num_heads: int = 8,
    lkv_dim: int = 512,
    r_dim: int = 64,
    page_size: int = 64,
    seed: int = 0,
) -> GenericHost:
    """Wire MLA v2 ``mla_ragged_paged_attention`` into the evolve loop.

    MLA absorbs the up-projection into the attention computation so the
    KV cache is a single low-rank latent (lkv_dim) plus a rotary part
    (r_dim) rather than full k/v tensors. v2 also fuses cache writes
    into the same Pallas kernel (#1971).

    Reference is v1's ``ref_mla_ragged_paged_attention``. The v2 kernel
    expects ``ql_nope`` transposed to (num_heads, total_q_len, lkv_dim)
    while v1 takes (total_q_len, num_heads, lkv_dim) — we transpose at
    call time so the inputs dict stays in the v1 layout.

    DeepSeek V3 reference shapes: num_heads=128, lkv_dim=512, r_dim=64.
    Defaults here use a smaller fixture suitable for bench.
    """

    rng = np.random.default_rng(seed)

    def _cdiv(a, b):
        return (a + b - 1) // b

    def _align_to(x, a):
        return _cdiv(x, a) * a

    def gen(shape, dtype):
        return jnp.asarray(rng.random(size=shape, dtype=np.float32),
                           dtype=dtype)

    q_dtype = jnp.bfloat16
    kv_dtype = jnp.bfloat16
    packing = jnp.dtype(kv_dtype).itemsize  # bytes
    packing = max(1, 4 // packing)  # element packing for the cache layout
    padded_lkv = _align_to(lkv_dim, 128)
    padded_r = _align_to(r_dim, 128)
    padded_kv = padded_lkv + padded_r
    total_q = sum(s[0] for s in seq_lens)
    kv_lens_list = [s[1] for s in seq_lens]
    max_kv = max(kv_lens_list)
    pages_per_seq = _cdiv(max_kv, page_size)

    page_indices_list = []
    page_count = 0
    for kvl in kv_lens_list:
        n_pg = _cdiv(kvl, page_size)
        page_indices_list.extend(
            list(range(page_count, page_count + n_pg)) + [-1] *
            (pages_per_seq - n_pg))
        page_count += n_pg
    num_pages = max(64, page_count)

    cu_q_list = [0]
    for ql, _ in seq_lens:
        cu_q_list.append(cu_q_list[-1] + ql)
    num_decode = 0
    for ql, _ in seq_lens:
        if ql == 1:
            num_decode += 1
        else:
            break
    distribution = jnp.array(
        [num_decode, num_decode, len(seq_lens)], dtype=jnp.int32)

    ql_nope = gen((total_q, num_heads, lkv_dim), q_dtype)
    q_pe = gen((total_q, num_heads, r_dim), q_dtype)
    new_kv_c = gen((total_q, lkv_dim), kv_dtype)
    new_k_pe = gen((total_q, r_dim), kv_dtype)
    cache_kv = gen((num_pages, page_size // packing, packing, padded_kv),
                   kv_dtype)
    kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
    page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
    cu_q_lens = jnp.array(cu_q_list, dtype=jnp.int32)

    def build_inputs():
        return {
            "ql_nope": ql_nope,
            "q_pe": q_pe,
            "new_kv_c": new_kv_c,
            "new_k_pe": new_k_pe,
            "cache_kv": cache_kv,
            "kv_lens": kv_lens,
            "page_indices": page_indices,
            "cu_q_lens": cu_q_lens,
            "distribution": distribution,
        }

    def reference_fn(inputs):
        from tpu_inference.kernels.mla.v1.kernel import \
            ref_mla_ragged_paged_attention

        # v1 ref returns (out, updated_kv). The oracle compares the FIRST
        # element of a returned tuple, so we return as-is.
        return ref_mla_ragged_paged_attention(
            inputs["ql_nope"],
            inputs["q_pe"], inputs["new_kv_c"], inputs["new_k_pe"],
            jnp.copy(inputs["cache_kv"]), inputs["kv_lens"],
            inputs["page_indices"], inputs["cu_q_lens"],
            inputs["distribution"])

    def call_kernel(kernel, inputs):
        # v2 expects ql_nope transposed to (num_heads, total_q, lkv_dim)
        ql_nope_v2 = jnp.transpose(jnp.copy(inputs["ql_nope"]), (1, 0, 2))
        # v2 output is (num_heads, total_q, lkv_dim) — transpose back so
        # it matches the v1 reference's (total_q, num_heads, lkv_dim).
        # v2 requires the bench to choose block sizes (no auto-default).
        out, _ = kernel(ql_nope_v2,
                        jnp.copy(inputs["q_pe"]),
                        jnp.copy(inputs["new_kv_c"]),
                        jnp.copy(inputs["new_k_pe"]),
                        jnp.copy(inputs["cache_kv"]),
                        inputs["kv_lens"],
                        inputs["page_indices"],
                        inputs["cu_q_lens"],
                        inputs["distribution"],
                        num_kv_pages_per_block=4,
                        num_queries_per_block=8)
        return jnp.transpose(out, (1, 0, 2))

    return GenericHost(
        kernel_name="mla_v2",
        kernel_symbol="mla_ragged_paged_attention",
        baseline_path_rel="tpu_inference/kernels/mla/v2/kernel.py",
        build_inputs=build_inputs,
        reference_fn=reference_fn,
        call_kernel=call_kernel,
        # cache_kv is in/out (fused update) — anti-cheat would flag the
        # alias otherwise.
        anti_cheat_skip=("cache_kv", ),
        tol_by_bits={
            16: (0.3, 0.3),
            32: (0.15, 0.15)
        },
    )
