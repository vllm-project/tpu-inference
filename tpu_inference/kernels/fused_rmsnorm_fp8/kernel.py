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
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from qwix._src.core import qarray
from qwix._src.providers import boxed_param, ptq

_CANDIDATE_BLOCK_MS: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8)
_FP8_E4M3FN_MAX: float = 448.0
_MIN_SCALE: float = 1e-12
_QWIX_PASSTHROUGH_INSTALLED: bool = False


def enable_qwix_qarray_passthrough() -> None:
    """Enables Qwix `QArray` activation passthrough for `boxed_param`."""
    global _QWIX_PASSTHROUGH_INSTALLED
    if _QWIX_PASSTHROUGH_INSTALLED:
        return

    orig_quantize_act = boxed_param.quantize_act

    @functools.wraps(orig_quantize_act)
    def _passthrough_quantize_act(
        array: jax.Array | qarray.QArray,
        how: qarray.HowToQuantize,
        rule: Any,
        act_name: str,
        *,
        _qarray_module=qarray,
    ) -> qarray.QArray:
        if isinstance(array, qarray.QArray):
            return array
        return orig_quantize_act(
            array,
            how,
            rule,
            act_name,
            _qarray_module=_qarray_module,
        )

    orig_promote_dtype = boxed_param.BoxedParamProvider.promote_dtype

    @functools.wraps(orig_promote_dtype)
    def _safe_promote_dtype(self, *args, **kwargs):
        seq_args = (args[0] if len(args) == 1
                    and isinstance(args[0], Sequence) else args)
        if kwargs.get("dtype") is None and not any(
                isinstance(x, jax.Array) for x in seq_args):
            return list(seq_args)
        return orig_promote_dtype(self, *args, **kwargs)

    boxed_param.quantize_act = _passthrough_quantize_act
    ptq.quantize_act = _passthrough_quantize_act
    boxed_param.BoxedParamProvider.promote_dtype = _safe_promote_dtype
    _QWIX_PASSTHROUGH_INSTALLED = True


def select_block_m(m: int, preferred_block_m: int = 512) -> int | None:
    """Selects the largest valid Pallas `block_m` dividing `m`.

    Args:
      m: Number of rows (tokens) to process on a single device.
      preferred_block_m: Upper bound on `block_m` (default 512).

    Returns:
      Largest candidate `block_m` in `(512, 256, 128, 64, 32, 16, 8)` that is
      `<= preferred_block_m` and evenly divides `m`, or `None` if `m < 8` or `m`
      is not divisible by any candidate.
    """
    if m < 8:
        return None
    for candidate in _CANDIDATE_BLOCK_MS:
        if candidate <= preferred_block_m and m % candidate == 0:
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
    if block_k is None or block_k >= x_norm.shape[-1]:
        x_abs_max = jnp.max(jnp.abs(x_norm), axis=-1, keepdims=True)
        scale = jnp.maximum(x_abs_max / quant_max, _MIN_SCALE)
        x_scaled = x_norm / scale
        x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
        out_scale_ref[...] = scale.astype(jnp.bfloat16)
    else:
        bm, k = x_norm.shape
        num_blocks = k // block_k
        x_norm_blocked = jnp.reshape(x_norm, (bm, num_blocks, block_k))
        x_abs_max = jnp.max(jnp.abs(x_norm_blocked), axis=-1, keepdims=True)
        scale = jnp.maximum(x_abs_max / quant_max, _MIN_SCALE)
        x_scaled = x_norm_blocked / scale
        x_clamped = jnp.clip(x_scaled, -quant_max, quant_max)
        x_clamped = jnp.reshape(x_clamped, (bm, k))
        out_scale_ref[...] = jnp.reshape(scale,
                                         (bm, num_blocks)).astype(jnp.bfloat16)

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
    vmem_limit_bytes: int = 128 * 1024 * 1024,
) -> tuple[jax.Array, ...]:
    """Fused RMSNorm and FP8 dynamic quantization Pallas TPU kernel on 2D inputs."""
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
    if block_k is not None and block_k < k and k % block_k != 0:
        raise ValueError(
            f"Hidden dimension K ({k}) must be divisible by block_k ({block_k})."
        )

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


fused_rmsnorm_fp8_quant = fused_rmsnorm_fp8_pallas


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
        q_scale = jnp.maximum(abs_max / quant_max, _MIN_SCALE)
        x_clamped = jnp.clip(x_norm / q_scale, -quant_max, quant_max)
        out_scale = q_scale.astype(jnp.bfloat16)
    else:
        if k % block_k != 0:
            raise ValueError(
                f"Hidden dimension K ({k}) must be divisible by block_k ({block_k})."
            )
        leading = x_norm.shape[:-1]
        num_blocks = k // block_k
        x_norm_blocked = jnp.reshape(x_norm, (*leading, num_blocks, block_k))
        abs_max = jnp.max(jnp.abs(x_norm_blocked), axis=-1, keepdims=True)
        q_scale = jnp.maximum(abs_max / quant_max, _MIN_SCALE)
        x_clamped = jnp.clip(x_norm_blocked / q_scale, -quant_max, quant_max)
        x_clamped = jnp.reshape(x_clamped, (*leading, k))
        out_scale = jnp.reshape(q_scale,
                                (*leading, num_blocks)).astype(jnp.bfloat16)

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
    """Fuses RMSNorm (+ optional residual) and FP8 quantization into a `QArray`."""
    enable_qwix_qarray_passthrough()
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

    chosen_block_m = select_block_m(m, preferred_block_m)
    if implementation == "pallas":
        if chosen_block_m is None:
            raise ValueError(
                f"Token count M ({m}) is not divisible by any valid block_m <="
                f" {preferred_block_m}; pass implementation='xla' or pad M.")
        res = fused_rmsnorm_fp8_pallas(
            x_2d,
            scale,
            res_2d,
            block_m=chosen_block_m,
            block_k=block_k,
            eps=eps,
            return_residual=return_residual,
        )
    elif implementation == "xla":
        res = fused_rmsnorm_fp8_reference(
            x_2d,
            scale,
            res_2d,
            block_k=block_k,
            eps=eps,
            return_residual=return_residual,
        )
    elif implementation == "auto":
        if (_has_tpu_backend() and chosen_block_m is not None and k >= 128
                and k % 128 == 0):
            res = fused_rmsnorm_fp8_pallas(
                x_2d,
                scale,
                res_2d,
                block_m=chosen_block_m,
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
    else:
        raise ValueError(
            "implementation must be 'auto', 'pallas', or 'xla', got"
            f" {implementation!r}.")

    num_scales = 1 if (block_k is None or block_k >= k) else (k // block_k)
    out_q = jnp.reshape(res[0], (*leading_shape, k))
    out_scale = jnp.reshape(res[1], (*leading_shape, num_scales))
    q_act = qarray.QArray(qvalue=out_q, scale=out_scale)
    if return_residual:
        updated_res = jnp.reshape(res[2], (*leading_shape, k))
        return q_act, updated_res
    return q_act
