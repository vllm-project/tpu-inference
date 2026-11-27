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
"""Fused RMSNorm + FP8 dynamic quantization Pallas TPU kernel.

This module fuses RMSNorm (with optional residual addition) and FP8
(`float8_e4m3fn`) per-token or sub-channel dynamic quantization into a single
streaming HBM pass on TPU:
  1. Loads input activation tile `X [block_m, K]` (+ optional `residual`) into
     VMEM.
  2. Computes FP32 variance and RMSNorm:
     `x_norm = (x * rsqrt(mean(x^2) + eps)) * scale`.
  3. Computes dynamic quantization scale `s = max(|x_norm|) / 448.0` per token
     (or per `block_k` sub-channel).
  4. Clamps and casts directly to `float8_e4m3fn` in VMEM registers.
  5. Writes quantized activations `out_q [block_m, K]` and `out_scale`
     `[block_m, num_scales]` back to HBM as a `qwix.QArray`.
"""

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from qwix._src.core import qarray
from qwix._src.providers import ptq

try:
    # Newer qwix releases moved the PTQ einsum/dot_general activation path into
    # `boxed_param`; qwix <= 0.1.2 keeps it in `ptq`.
    from qwix._src.providers import boxed_param as _qwix_boxed_param
except ImportError:
    _qwix_boxed_param = None

# Tiled `block_m` candidates are multiples of 32 so that every
# `float8_e4m3fn` output block is aligned to its native (32, 128) TPU tiling.
_CANDIDATE_BLOCK_MS: tuple[int, ...] = (512, 256, 128, 64, 32)
# Inputs with fewer rows than the smallest tile run as a single block that
# spans all rows (block dims equal to the array dims are always legal).
_MIN_FULL_BLOCK_M: int = 8
_FP8_E4M3FN_MAX: float = 448.0
_MIN_SCALE: float = 1e-12
# Conservative VMEM model of one grid step: Pallas double-buffers every
# pipelined [block_m, K] block and Mosaic may keep a few f32 [block_m, K]
# temporaries live. Auto-selected blocks stay within this fraction of VMEM.
_NUM_F32_TEMPORARIES: int = 3
_VMEM_BUDGET_FRACTION: float = 0.75
# Used when the TPU info query is unavailable (smallest per-core VMEM: v7x).
_FALLBACK_VMEM_CAPACITY_BYTES: int = 64 * 1024 * 1024
_QWIX_PASSTHROUGH_INSTALLED: bool = False


def _is_fp8_e4m3fn(qtype: Any) -> bool:
    if qtype is None:
        return False
    try:
        return jnp.dtype(qtype) == jnp.float8_e4m3fn
    except TypeError:
        return False


def is_fusable_act_quant_rule(rule: Any) -> bool:
    """Returns whether `rule` quantizes activations exactly like this kernel.

    The kernel emits dynamic, per-token, absmax-calibrated `float8_e4m3fn`
    activations. Its `QArray` output is only a drop-in replacement for qwix's
    own activation quantization under rules that quantize the consuming
    einsum's weights (so that the einsum is quantized at all) and quantize its
    activations with that exact scheme: no static scale, no sub-channel tiling
    and plain `absmax` calibration.
    """
    if rule is None or getattr(rule, "weight_qtype", None) is None:
        return False
    if not _is_fp8_e4m3fn(getattr(rule, "act_qtype", None)):
        return False
    if getattr(rule, "act_static_scale", False):
        return False
    if getattr(rule, "tile_size", None) is not None:
        return False
    calibration = getattr(rule, "act_calibration_method", None) or "absmax"
    return calibration.strip().lower() == "absmax"


def _is_compatible_prequantized_act(
    array: qarray.QArray,
    how: qarray.HowToQuantize,
    rule: Any,
) -> bool:
    """Returns whether `array` matches what qwix would compute for `how`."""
    if getattr(rule, "act_static_scale", False):
        return False
    if (array.zero_point is not None
            or getattr(how, "noise_fn", None) is not None):
        return False
    if how.calibration_method.strip().lower() != "absmax":
        return False
    try:
        if jnp.dtype(array.qtype) != jnp.dtype(how.qtype):
            return False
        expected_scale_shape = tuple(qarray.get_scale_shape(array.shape, how))
    except (TypeError, ValueError):
        return False
    return tuple(array.scale.shape) == expected_scale_shape


def _wrap_quantize_act(
        quantize_act: Callable[..., qarray.QArray]) -> Callable[..., Any]:
    """Wraps qwix `quantize_act` to consume pre-quantized activations."""

    @functools.wraps(quantize_act)
    def _quantize_act(array, how, rule, *args, **kwargs):
        if isinstance(array, qarray.QArray):
            if _is_compatible_prequantized_act(array, how, rule):
                return array
            # The consuming op quantizes its activation differently (e.g.
            # another rule matched it), so requantize the dequantized values
            # instead of feeding it a mismatched QArray.
            array = qarray.dequantize(array)
        return quantize_act(array, how, rule, *args, **kwargs)

    return _quantize_act


def _wrap_promote_dtype(
        promote_dtype: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps qwix `promote_dtype` to leave `QArray` operands untouched."""

    @functools.wraps(promote_dtype)
    def _promote_dtype(self, *args, **kwargs):
        operands = (args[0] if len(args) == 1
                    and isinstance(args[0], Sequence) else args)
        if not any(isinstance(x, qarray.QArray) for x in operands):
            return promote_dtype(self, *args, **kwargs)
        # Pre-quantized activations are consumed as-is by the quantized
        # einsum/dot_general; only promote the remaining array operands.
        others = [
            None if isinstance(x, qarray.QArray) else x for x in operands
        ]
        promoted = (promote_dtype(self, others, **kwargs) if any(
            isinstance(x, jax.Array) for x in others) else others)
        return [
            x if isinstance(x, qarray.QArray) else p
            for x, p in zip(operands, promoted)
        ]

    return _promote_dtype


def enable_qwix_qarray_passthrough() -> None:
    """Lets qwix PTQ einsum/dot_general consume pre-quantized `QArray` inputs.

    Idempotent. Both hooks only change behavior for `qwix.QArray` operands and
    defer to qwix's original implementation for everything else:
      * `quantize_act` returns a pre-quantized activation unchanged when it
        matches how the consuming op quantizes activations, and requantizes
        its dequantized values otherwise.
      * `promote_dtype` leaves `QArray` operands untouched and promotes the
        remaining operands with qwix's implementation.
    Supports qwix <= 0.1.2 (`ptq`) and newer releases (`boxed_param`).
    """
    global _QWIX_PASSTHROUGH_INSTALLED
    if _QWIX_PASSTHROUGH_INSTALLED:
        return

    wrapped_quantize_acts: dict[int, Callable[..., Any]] = {}
    for module in (_qwix_boxed_param, ptq):
        orig_quantize_act = getattr(module, "quantize_act", None)
        if orig_quantize_act is None:
            continue
        if id(orig_quantize_act) not in wrapped_quantize_acts:
            wrapped_quantize_acts[id(orig_quantize_act)] = _wrap_quantize_act(
                orig_quantize_act)
        module.quantize_act = wrapped_quantize_acts[id(orig_quantize_act)]

    for provider_cls in (getattr(_qwix_boxed_param, "BoxedParamProvider",
                                 None), ptq.PtqProvider):
        if provider_cls is not None and "promote_dtype" in vars(provider_cls):
            provider_cls.promote_dtype = _wrap_promote_dtype(
                vars(provider_cls)["promote_dtype"])
    _QWIX_PASSTHROUGH_INSTALLED = True


def _vmem_capacity_bytes() -> int:
    """Returns the per-core VMEM capacity of the attached TPU."""
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except ValueError:  # Not running on a TPU backend.
        return _FALLBACK_VMEM_CAPACITY_BYTES


def max_block_m_for_vmem(
    k: int,
    *,
    x_itemsize: int = 2,
    residual_itemsize: int | None = None,
    return_residual: bool = False,
    vmem_capacity_bytes: int | None = None,
) -> int:
    """Returns the largest `block_m` whose estimated VMEM use fits the budget.

    Args:
      k: Hidden dimension.
      x_itemsize: Bytes per element of the input activations.
      residual_itemsize: Bytes per element of the residual, or None if there is
        no residual input.
      return_residual: Whether the updated (bfloat16) residual is written out.
      vmem_capacity_bytes: Per-core VMEM capacity. Defaults to
        `pltpu.get_tpu_info().vmem_capacity_bytes`.
    """
    if vmem_capacity_bytes is None:
        vmem_capacity_bytes = _vmem_capacity_bytes()
    # Pipelined blocks per row: activations in, float8_e4m3fn values out, plus
    # the optional residual in and bfloat16 residual out.
    io_bytes = x_itemsize + 1
    if residual_itemsize is not None:
        io_bytes += residual_itemsize
    if return_residual:
        io_bytes += 2
    bytes_per_row = k * (2 * io_bytes + 4 * _NUM_F32_TEMPORARIES)
    return int(vmem_capacity_bytes * _VMEM_BUDGET_FRACTION) // bytes_per_row


def select_block_m(
    m: int,
    preferred_block_m: int = 512,
    max_block_m: int | None = None,
) -> int | None:
    """Selects the Pallas `block_m` (rows per grid step) for `m` rows.

    Args:
      m: Number of rows (tokens) to process on a single device.
      preferred_block_m: Upper bound on `block_m` (default 512).
      max_block_m: Optional additional upper bound, e.g. from
        `max_block_m_for_vmem`.

    Returns:
      The largest candidate in `(512, 256, 128, 64, 32)` that is within both
      bounds and evenly divides `m`. Inputs with `8 <= m < 32` rows (a multiple
      of 8, within `max_block_m`) run as a single block spanning all rows.
      Returns None otherwise.
    """
    if m < _CANDIDATE_BLOCK_MS[-1]:
        fits = max_block_m is None or m <= max_block_m
        aligned = m >= _MIN_FULL_BLOCK_M and m % _MIN_FULL_BLOCK_M == 0
        return m if fits and aligned else None
    limit = (preferred_block_m if max_block_m is None else min(
        preferred_block_m, max_block_m))
    for candidate in _CANDIDATE_BLOCK_MS:
        if candidate <= limit and m % candidate == 0:
            return candidate
    return None


def _fused_rmsnorm_quant_kernel(
    *args,
    eps: float = 1e-6,
    quant_max: float = _FP8_E4M3FN_MAX,
    has_residual: bool = False,
    return_residual: bool = False,
    block_k: int | None = None,
):
    """Inner Pallas kernel executing on TPU VPU vector units."""
    arg_idx = 0
    x_ref = args[arg_idx]
    arg_idx += 1
    gamma_ref = args[arg_idx]
    arg_idx += 1
    if has_residual:
        residual_ref = args[arg_idx]
        arg_idx += 1
    else:
        residual_ref = None

    out_q_ref = args[arg_idx]
    arg_idx += 1
    out_scale_ref = args[arg_idx]
    arg_idx += 1
    if return_residual:
        out_res_ref = args[arg_idx]
        arg_idx += 1
    else:
        out_res_ref = None

    x = x_ref[...]
    if has_residual and residual_ref is not None:
        x = x + residual_ref[...]
        if return_residual and out_res_ref is not None:
            out_res_ref[...] = x.astype(jnp.bfloat16)

    # 1. High-precision FP32 variance and RMS normalization.
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    rsqrt_var = jax.lax.rsqrt(var + eps)
    gamma_f32 = gamma_ref[...].astype(jnp.float32)
    x_norm = (x_f32 * rsqrt_var) * gamma_f32

    # 2. Dynamic scale computation and clamping.
    # Round scale to bfloat16 before dividing x_norm to match qwix's
    # quantize_with_scale_zero_point divisor and downstream bfloat16 dequant.
    if block_k is None or block_k >= x_norm.shape[-1]:
        x_abs_max = jnp.max(jnp.abs(x_norm), axis=-1, keepdims=True)
        scale_bf16 = jnp.maximum(x_abs_max / quant_max,
                                 _MIN_SCALE).astype(jnp.bfloat16)
        x_scaled = x_norm / scale_bf16.astype(jnp.float32)
        x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
        out_scale_ref[...] = scale_bf16
    else:
        bm, k = x_norm.shape
        num_blocks = k // block_k
        x_norm_blocked = jnp.reshape(x_norm, (bm, num_blocks, block_k))
        x_abs_max = jnp.max(jnp.abs(x_norm_blocked), axis=-1, keepdims=True)
        scale_bf16 = jnp.maximum(x_abs_max / quant_max,
                                 _MIN_SCALE).astype(jnp.bfloat16)
        x_scaled = x_norm_blocked / scale_bf16.astype(jnp.float32)
        x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
        x_clamped = jnp.reshape(x_clamped, (bm, k))
        out_scale_ref[...] = jnp.reshape(scale_bf16, (bm, num_blocks))

    # 3. Direct cast to float8_e4m3fn in VMEM.
    out_q_ref[...] = x_clamped.astype(jnp.float8_e4m3fn)


def fused_rmsnorm_fp8_pallas(
    x: jax.Array,
    scale: jax.Array,
    residual: jax.Array | None = None,
    *,
    block_m: int = 256,
    block_k: int | None = None,
    eps: float = 1e-6,
    quant_max: float = _FP8_E4M3FN_MAX,
    return_residual: bool = False,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, ...]:
    """Fused RMSNorm and FP8 dynamic quantization Pallas TPU kernel on 2D inputs.

    `vmem_limit_bytes` defaults to the per-core VMEM capacity reported by
    `pltpu.get_tpu_info()`.
    """
    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D input x of shape (M, K), got {x.shape}.")
    if return_residual and residual is None:
        raise ValueError(
            "return_residual=True requires residual to be provided.")
    m, k = x.shape
    if m % block_m != 0:
        raise ValueError(
            f"Sequence length M ({m}) must be divisible by block_m ({block_m})."
        )
    if block_m != m and block_m % _CANDIDATE_BLOCK_MS[-1] != 0:
        raise ValueError(
            f"block_m ({block_m}) must be a multiple of"
            f" {_CANDIDATE_BLOCK_MS[-1]} (float8_e4m3fn tiling) or equal M"
            f" ({m}).")
    if block_k is not None and block_k < k and k % block_k != 0:
        raise ValueError(
            f"Hidden dimension K ({k}) must be divisible by block_k ({block_k})."
        )
    if vmem_limit_bytes is None:
        vmem_limit_bytes = _vmem_capacity_bytes()

    grid_m = m // block_m
    has_residual = residual is not None
    has_res_out = has_residual and return_residual
    num_scales = 1 if (block_k is None or block_k >= k) else (k // block_k)

    in_specs = [
        pl.BlockSpec((block_m, k), lambda i: (i, 0)),
        pl.BlockSpec((k, ), lambda i: (0, )),
    ]
    if has_residual:
        in_specs.append(pl.BlockSpec((block_m, k), lambda i: (i, 0)))

    out_specs = [
        pl.BlockSpec((block_m, k), lambda i: (i, 0)),
        pl.BlockSpec((block_m, num_scales), lambda i: (i, 0)),
    ]
    out_shapes = [
        jax.ShapeDtypeStruct((m, k), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((m, num_scales), jnp.bfloat16),
    ]

    if has_res_out:
        out_specs.append(pl.BlockSpec((block_m, k), lambda i: (i, 0)))
        out_shapes.append(jax.ShapeDtypeStruct((m, k), jnp.bfloat16))

    kernel_fn = functools.partial(
        _fused_rmsnorm_quant_kernel,
        eps=eps,
        quant_max=quant_max,
        has_residual=has_residual,
        return_residual=has_res_out,
        block_k=block_k,
    )

    args = [x, scale]
    if has_residual:
        args.append(residual)

    return pl.pallas_call(
        kernel_fn,
        grid=(grid_m, ),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", ),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
    )(*args)


def fused_rmsnorm_fp8_reference(
    x: jax.Array,
    scale: jax.Array,
    residual: jax.Array | None = None,
    *,
    block_k: int | None = None,
    eps: float = 1e-6,
    quant_max: float = _FP8_E4M3FN_MAX,
    return_residual: bool = False,
) -> tuple[jax.Array, ...]:
    """Plain-XLA reference for fused RMSNorm (+ residual) + FP8 dynamic quant."""
    if return_residual and residual is None:
        raise ValueError(
            "return_residual=True requires residual to be provided.")
    if residual is not None:
        x = (x + residual).astype(jnp.bfloat16)

    x_f32 = x.astype(jnp.float32)
    scale_f32 = scale.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    x_norm = (x_f32 * jax.lax.rsqrt(var + eps)) * scale_f32

    k = x_norm.shape[-1]
    if block_k is None or block_k >= k:
        abs_max = jnp.max(jnp.abs(x_norm), axis=-1, keepdims=True)
        q_scale_bf16 = jnp.maximum(abs_max / quant_max,
                                   _MIN_SCALE).astype(jnp.bfloat16)
        x_clamped = jnp.clip(x_norm / q_scale_bf16.astype(jnp.float32),
                             -quant_max, quant_max)
        out_scale = q_scale_bf16
    else:
        if k % block_k != 0:
            raise ValueError(
                f"Hidden dimension K ({k}) must be divisible by block_k ({block_k})."
            )
        leading = x_norm.shape[:-1]
        num_blocks = k // block_k
        x_norm_blocked = jnp.reshape(x_norm, (*leading, num_blocks, block_k))
        abs_max = jnp.max(jnp.abs(x_norm_blocked), axis=-1, keepdims=True)
        q_scale_bf16 = jnp.maximum(abs_max / quant_max,
                                   _MIN_SCALE).astype(jnp.bfloat16)
        x_clamped = jnp.clip(
            x_norm_blocked / q_scale_bf16.astype(jnp.float32),
            -quant_max,
            quant_max,
        )
        x_clamped = jnp.reshape(x_clamped, (*leading, k))
        out_scale = jnp.reshape(q_scale_bf16, (*leading, num_blocks))

    out_q = x_clamped.astype(jnp.float8_e4m3fn)
    if return_residual:
        return out_q, out_scale, x
    return out_q, out_scale


def _has_tpu_backend() -> bool:
    devices = jax.devices()
    return bool(devices) and all(d.platform == "tpu" for d in devices)


def fused_rmsnorm_fp8(
    x: jax.Array,
    scale: jax.Array,
    residual: jax.Array | None = None,
    *,
    eps: float = 1e-6,
    preferred_block_m: int = 512,
    block_k: int | None = None,
    return_residual: bool = False,
    implementation: Literal["auto", "pallas", "xla"] = "auto",
) -> qarray.QArray | tuple[qarray.QArray, jax.Array]:
    """Fuses RMSNorm (+ optional residual) and FP8 quantization into a `QArray`.

    With `implementation="auto"`, the Pallas kernel runs on TPU when `K` is a
    multiple of 128 and a valid `block_m` exists (see `select_block_m`), and
    the plain-XLA reference runs otherwise.
    """
    enable_qwix_qarray_passthrough()
    if implementation not in ("auto", "pallas", "xla"):
        raise ValueError(
            "implementation must be 'auto', 'pallas', or 'xla', got"
            f" {implementation!r}.")
    if return_residual and residual is None:
        raise ValueError(
            "return_residual=True requires residual to be provided.")
    if scale.ndim != 1 or scale.shape[0] != x.shape[-1]:
        raise ValueError(
            f"scale shape {scale.shape} must be 1D matching x.shape[-1]"
            f" ({x.shape[-1]}).")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {residual.shape} must match x shape {x.shape}.")

    leading_shape = x.shape[:-1]
    k = x.shape[-1]
    m = math.prod(leading_shape) if leading_shape else 1
    x_2d = jnp.reshape(x, (m, k))
    res_2d = jnp.reshape(residual, (m, k)) if residual is not None else None

    use_pallas = implementation == "pallas" or (implementation == "auto"
                                                and _has_tpu_backend()
                                                and k >= 128 and k % 128 == 0)
    block_m = None
    if use_pallas:
        block_m = select_block_m(
            m,
            preferred_block_m,
            max_block_m=max_block_m_for_vmem(
                k,
                x_itemsize=x.dtype.itemsize,
                residual_itemsize=(residual.dtype.itemsize
                                   if residual is not None else None),
                return_residual=return_residual,
            ),
        )
        if block_m is None and implementation == "pallas":
            raise ValueError(
                f"Token count M ({m}) is not divisible by any valid block_m <="
                f" {preferred_block_m}; pass implementation='xla' or pad M.")

    if block_m is not None:
        res = fused_rmsnorm_fp8_pallas(
            x_2d,
            scale,
            res_2d,
            block_m=block_m,
            block_k=block_k,
            eps=eps,
            return_residual=return_residual,
        )
    else:
        res = fused_rmsnorm_fp8_reference(
            x_2d,
            scale,
            res_2d,
            block_k=block_k,
            eps=eps,
            return_residual=return_residual,
        )

    num_scales = 1 if (block_k is None or block_k >= k) else (k // block_k)
    out_q = jnp.reshape(res[0], (*leading_shape, k))
    out_scale = jnp.reshape(res[1], (*leading_shape, num_scales))
    q_act = qarray.QArray(qvalue=out_q, scale=out_scale)
    if return_residual:
        updated_res = jnp.reshape(res[2], (*leading_shape, k))
        return q_act, updated_res
    return q_act
