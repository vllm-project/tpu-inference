# SPDX-License-Identifier: Apache-2.0
"""2D (Block-wise) Quantized matmul kernel.

This module implements "Block-wise" or "Sub-channel" quantization.
It contains:
1. Standard 2D Quantized Matmul (quantized_matmul_2d) - Reference
2. High-performance V7 Kernel (dispatch_w8a8_v7) - Optimized for Latency
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tpu_inference.kernels.quantized_matmul.util import next_multiple, unfold_args
from tpu_inference.kernels.quantized_matmul.tuned_block_sizes import get_device_vmem_limit

BLOCK_K = 128
SUPER_CHUNK = 16
TILE_K_SIZE = BLOCK_K * SUPER_CHUNK

# ==============================================================================
# 1. Reference Implementation (For Baseline/Verification)
# ==============================================================================

def _quantize_array_2d(x, x_abs_max, quant_dtype):
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    if is_float:
        dtype_info = jnp.finfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        if quant_dtype == jnp.float8_e4m3fn: dtype_max = 448.0
        dtype_min = float(dtype_info.min)
    else:
        dtype_info = jnp.iinfo(quant_dtype)
        dtype_max = float(dtype_info.max)
        dtype_min = float(dtype_info.min)

    scale_basis = jnp.maximum(x_abs_max, jnp.finfo(jnp.float32).tiny)
    scale = scale_basis / dtype_max
    x_scaled = x.astype(jnp.float32) / scale

    if not is_float:
        x_scaled = jnp.floor(x_scaled + 0.5)
        quantized_array = jnp.clip(x_scaled, dtype_min, dtype_max).astype(quant_dtype)
    else:
        quantized_array = x_scaled.astype(quant_dtype)
    return quantized_array, scale.astype(jnp.float32)

def _quantized_matmul_kernel_2d(
    x_ref, w_q_ref, w_scale_ref, x_abs_max_ref, out_ref,
    acc_scratch, x_q_scratch, x_scale_scratch,
    *, x_q_dtype, dot_dtype, save_acc, save_x_q
):
    out_idx, in_block_idx = pl.program_id(1), pl.program_id(2)
    n_in_blocks = pl.num_programs(2)
    quantize_activation = x_q_dtype != x_ref.dtype
    
    if save_x_q: quant = out_idx == 0
    else: quant = quantize_activation

    if save_acc:
        is_first_step, is_last_step = (in_block_idx == 0), (in_block_idx == (n_in_blocks - 1))
    else:
        is_first_step, is_last_step = True, True

    def matmul_body(quant, is_first_step, is_last_step):
        x_abs_max_row = x_abs_max_ref[in_block_idx]
        x_abs_max_current = x_abs_max_row[:, None]

        if quantize_activation:
            if quant:
                 x_q_tmp, x_scale_tmp = _quantize_array_2d(x_ref[...], x_abs_max_current, x_q_dtype)
                 if save_x_q: x_q_scratch[...], x_scale_scratch[...] = x_q_tmp, x_scale_tmp
            else:
                 x_q_tmp, x_scale_tmp = x_q_scratch[...], x_scale_scratch[...]
        else:
            x_q_tmp, x_scale_tmp = x_ref[...], 1.0 

        acc = jax.lax.dot_general(
            x_q_tmp, w_q_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=dot_dtype,
        )
        w_scale_current = w_scale_ref[in_block_idx][None, :]
        acc = acc.astype(jnp.float32)
        if quantize_activation: acc *= x_scale_tmp
        acc *= w_scale_current

        if not is_first_step: acc += acc_scratch[...]
        if is_last_step: out_ref[...] = acc.astype(x_ref.dtype)
        elif save_acc: acc_scratch[...] = acc

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)

@functools.partial(jax.jit, static_argnames=['x_q_dtype', 'quant_block_size', 'batch_block_size', 'out_block_size'])
def quantized_matmul_2d(x, w_q, w_scale, quant_block_size, x_q_dtype=None, *, batch_block_size=128, out_block_size=128):
    if x_q_dtype is None: x_q_dtype = x.dtype
    orig_n_batch, orig_n_in = x.shape
    orig_n_out, _ = w_q.shape

    padded_n_in = next_multiple(orig_n_in, quant_block_size)
    if orig_n_in < padded_n_in:
        padding_diff = padded_n_in - orig_n_in
        x = jnp.pad(x, ((0, 0), (0, padding_diff)))
        w_q = jnp.pad(w_q, ((0, 0), (0, padding_diff)))
    n_in_blocks = padded_n_in // quant_block_size

    x_blocked = x.reshape(orig_n_batch, n_in_blocks, quant_block_size)
    x_abs_max = jnp.max(jnp.abs(x_blocked), axis=-1).astype(jnp.float32)

    padded_n_batch, padded_n_out = next_multiple(orig_n_batch, batch_block_size), next_multiple(orig_n_out, out_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max, ((0, padded_n_batch - orig_n_batch), (0, 0)))
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, padded_n_out - orig_n_out), (0, 0)))

    x_abs_max_t, w_scale_t = jnp.transpose(x_abs_max), jnp.transpose(w_scale.astype(jnp.float32))
    n_batch, n_out = padded_n_batch // batch_block_size, padded_n_out // out_block_size
    save_acc = n_in_blocks > 1
    dot_dtype = jnp.int32 if (x_q_dtype != x.dtype and jnp.issubdtype(w_q.dtype, jnp.integer)) else jnp.float32

    kernel = pl.pallas_call(
        functools.partial(_quantized_matmul_kernel_2d, x_q_dtype=x_q_dtype, dot_dtype=dot_dtype, save_acc=save_acc, save_x_q=False),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((batch_block_size, quant_block_size), lambda b, o, i: (b, i)), pl.BlockSpec((out_block_size, quant_block_size), lambda b, o, i: (o, i)), pl.BlockSpec((n_in_blocks, out_block_size), lambda b, o, i: (0, o)), pl.BlockSpec((n_in_blocks, batch_block_size), lambda b, o, i: (0, b))],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[pltpu.VMEM((batch_block_size, out_block_size), jnp.float32) if save_acc else None, None, None],
            grid=(n_batch, n_out, n_in_blocks),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'arbitrary', 'arbitrary'), vmem_limit_bytes=get_device_vmem_limit()),
    )
    return kernel(x, w_q, w_scale_t, x_abs_max_t)[:orig_n_batch, :orig_n_out]

# ==============================================================================
# 2. Optimized V7 Implementation
# ==============================================================================

def _compute_quant_params_v7(x_slice, w_scale_row, max_val, dtype):
    """
    V7 Optimized Quantization.
    Features:
    1. FP32 Math for VPU Efficiency/Safety.
    2. Reciprocal Multiplication (replaces slow Division).
    """
    x_f32 = x_slice.astype(jnp.float32)
    x_abs = jnp.abs(x_f32)
    x_max = jnp.max(x_abs, axis=1, keepdims=True)
    x_max = jnp.maximum(x_max, 1e-6)

    # Optimization: Reciprocal Mul
    scale_to_int = max_val / x_max
    val = x_f32 * scale_to_int
    val_rounded = jnp.floor(val + 0.5)

    if dtype == jnp.int8:
        val_clipped = jnp.clip(val_rounded, -128.0, 127.0)
        x_q = val_clipped.astype(jnp.int8)
    else:
        val_clipped = jnp.clip(val_rounded, -448.0, 448.0)
        x_q = val_clipped.astype(dtype)

    scale_factor = (x_max / max_val) * w_scale_row
    return x_q, scale_factor

def _kernel_v7_optimized_entry(x_ref, w_ref, scales_ref, out_ref, acc_vmem, *,
                               batch_block_size, out_block_size, dtype):
    """
    V7 Kernel with Software Pipelining.
    Prologue -> Loop -> Epilogue structure effectively hides VPU accumulation latency.
    """
    acc_vmem[...] = jnp.zeros((batch_block_size, out_block_size), dtype=jnp.float32)

    if dtype == jnp.int8:
        max_val, dot_preferred = 127.0, jnp.int32
    else:
        max_val, dot_preferred = 448.0, jnp.float32

    def pipeline_step(x_chunk, w_chunk_std, s_chunk_transposed):
        # Prologue (Stage 0)
        x_0 = x_chunk[:, 0:BLOCK_K] 
        w_0 = w_chunk_std[:, 0:BLOCK_K]
        s_0 = s_chunk_transposed[0, :][None, :]

        x_q_curr, scale_curr = _compute_quant_params_v7(x_0, s_0, max_val, dtype)
        dot_curr = jax.lax.dot_general(x_q_curr, w_0, (((1,), (1,)), ((), ())), preferred_element_type=dot_preferred)
        
        dot_prev, scale_prev = dot_curr, scale_curr

        # Pipeline Loop (Stage 1..N-1)
        for j in range(1, SUPER_CHUNK):
            k_start, k_end = j * BLOCK_K, (j + 1) * BLOCK_K
            x_next, w_next, s_next = x_chunk[:, k_start:k_end], w_chunk_std[:, k_start:k_end], s_chunk_transposed[j, :][None, :]

            x_q_next, scale_next = _compute_quant_params_v7(x_next, s_next, max_val, dtype)
            dot_next = jax.lax.dot_general(x_q_next, w_next, (((1,), (1,)), ((), ())), preferred_element_type=dot_preferred)

            # Accumulate Prev (Parallel with Dot Next)
            acc_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev
            dot_prev, scale_prev = dot_next, scale_next

        # Epilogue (Stage N)
        acc_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev

    total_k = w_ref.shape[1]
    n_k_chunks = total_k // TILE_K_SIZE

    pltpu.emit_pipeline(
        pipeline_step,
        grid=(n_k_chunks,),
        in_specs=[
            pl.BlockSpec((batch_block_size, TILE_K_SIZE), lambda i: (0, i)),
            pl.BlockSpec((out_block_size, TILE_K_SIZE), lambda i: (0, i)),
            pl.BlockSpec((SUPER_CHUNK, out_block_size), lambda i: (i, 0))
        ]
    )(x_ref, w_ref, scales_ref)

    out_ref[...] = acc_vmem[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=['out_block_size', 'quant_dtype', 'override_block_b'])
def _dispatch_impl(x, w_q, w_scale, out_block_size, quant_dtype, override_block_b):
    bs, n_in = x.shape
    n_out, _ = w_q.shape
    CURRENT_BLOCK_B = override_block_b
    
    padded_bs = next_multiple(bs, CURRENT_BLOCK_B)
    padded_out = next_multiple(n_out, out_block_size)
    padded_in = next_multiple(n_in, TILE_K_SIZE)
    
    if padded_bs > bs: x = jnp.pad(x, ((0, padded_bs - bs), (0, 0)))
    if padded_in > n_in: x = jnp.pad(x, ((0, 0), (0, padded_in - n_in)))
        
    if padded_in > n_in:
        pad_k = padded_in - n_in
        w_q = jnp.pad(w_q, ((0, 0), (0, pad_k)))
        pad_blocks = pad_k // BLOCK_K
        w_scale = jnp.pad(w_scale, ((0, 0), (0, pad_blocks)), constant_values=1.0)
        
    if padded_out > n_out:
        pad_out = padded_out - n_out
        w_q = jnp.pad(w_q, ((0, pad_out), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, pad_out), (0, 0)))

    w_scale_t = w_scale.T

    n_batch_blocks, n_out_blocks = padded_bs // CURRENT_BLOCK_B, padded_out // out_block_size
    grid = (n_batch_blocks, n_out_blocks)
    
    in_specs = [
        pl.BlockSpec((CURRENT_BLOCK_B, padded_in), lambda b, o: (b, 0)),
        pl.BlockSpec((out_block_size, padded_in), lambda b, o: (o, 0)),
        pl.BlockSpec((w_scale_t.shape[0], out_block_size), lambda b, o: (0, o)),
    ]

    kernel = pl.pallas_call(
        functools.partial(_kernel_v7_optimized_entry, batch_block_size=CURRENT_BLOCK_B, out_block_size=out_block_size, dtype=quant_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=pl.BlockSpec((CURRENT_BLOCK_B, out_block_size), lambda b, o: (b, o)),
            grid=grid,
            scratch_shapes=[pltpu.VMEM((CURRENT_BLOCK_B, out_block_size), jnp.float32)]
        ),
        out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out), x.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_device_vmem_limit())
    )
    out = kernel(x, w_q, w_scale_t)
    return out[:bs, :n_out]


def dispatch_w8a8_v7(x, w_q, w_scale, out_block_size, quant_dtype):
    """
    Adaptive Dispatcher (Final Tuned):
    - Batch <= 48:  BLOCK_B=32  (Max Speedup: ~1.4x)
    - Batch <= 96:  BLOCK_B=64  (Max Speedup: ~1.1x)
    - Batch <= 192: BLOCK_B=128 (Parity w/ BF16)
    - Batch > 192:  BLOCK_B=256 (Compute Limited)
    """
    batch_size = x.shape[0]
    
    if batch_size <= 48:
        return _dispatch_impl(x, w_q, w_scale, out_block_size, quant_dtype, override_block_b=32)
    elif batch_size <= 96:
        return _dispatch_impl(x, w_q, w_scale, out_block_size, quant_dtype, override_block_b=64)
    elif batch_size <= 192:
        return _dispatch_impl(x, w_q, w_scale, out_block_size, quant_dtype, override_block_b=128)
    else:
        return _dispatch_impl(x, w_q, w_scale, out_block_size, quant_dtype, override_block_b=256)
    
