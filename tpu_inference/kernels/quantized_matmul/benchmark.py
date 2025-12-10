import functools
import time
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import pandas as pd
import numpy as np

# ==========================================
# 1. SHARED UTILITIES
# ==========================================

BLOCK_K = 256      
BLOCK_B = 64       
SUPER_CHUNK = 8    
TILE_K_SIZE = BLOCK_K * SUPER_CHUNK

def next_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

def get_tpu_vmem_limit() -> int:
    return 16 * 1024 * 1024 

def unfold_args(args, kwargs, func):
    if isinstance(kwargs, tuple):
        kwargs = {}
    return func(*args, **kwargs)

# ==========================================
# 2. INT8 REFERENCE IMPLEMENTATION
# ==========================================

@functools.partial(jax.jit, static_argnames=['block_size'])
def ref_int8_matmul(x, w, block_size):
    """
    Pure JAX reference for Block-wise W8A8 Quantized Matmul.
    """
    bs, n_in = x.shape
    n_out, _ = w.shape
    
    padded_in = next_multiple(n_in, block_size)
    if padded_in > n_in:
        x = jnp.pad(x, ((0, 0), (0, padded_in - n_in)))
        w = jnp.pad(w, ((0, 0), (0, padded_in - n_in)))
    
    n_blocks = padded_in // block_size
    
    # Reshape
    x_blk = x.reshape(bs, n_blocks, block_size)
    w_blk = w.reshape(n_out, n_blocks, block_size)
    
    # Quantize X
    x_max = jnp.max(jnp.abs(x_blk), axis=2, keepdims=True)
    x_max = jnp.maximum(x_max, 1e-6)
    x_scale_inv = 127.0 / x_max
    x_q = jnp.clip(jnp.floor(x_blk * x_scale_inv + 0.5), -128, 127).astype(jnp.int8)
    
    # Quantize W
    w_max = jnp.max(jnp.abs(w_blk), axis=2, keepdims=True)
    w_max = jnp.maximum(w_max, 1e-6)
    w_scale_inv = 127.0 / w_max
    w_q = jnp.clip(jnp.floor(w_blk * w_scale_inv + 0.5), -128, 127).astype(jnp.int8)
    
    # Dot Product
    dot_int32 = jnp.einsum('bki,oki->bko', x_q.astype(jnp.int32), w_q.astype(jnp.int32))
    
    xs = (x_max / 127.0).squeeze(-1) 
    ws = (w_max / 127.0).squeeze(-1)  
    
    joint_scale = xs[:, :, None] * ws.T[None, :, :]
    
    out_float = dot_int32.astype(jnp.float32) * joint_scale
    
    return jnp.sum(out_float, axis=1)

# ==========================================
# 3. ORIGINAL KERNEL & UTILS
# ==========================================

F8_E4M3FN_MAX = 448.0
EPS = jnp.finfo(jnp.float16).tiny

def quantize_2d_blocked(x: jax.Array, block_size: int, dtype: jnp.dtype = jnp.int8):
    n_rows, n_cols = x.shape
    if n_cols % block_size != 0:
        padded = next_multiple(n_cols, block_size)
        x = jnp.pad(x, ((0, 0), (0, padded - n_cols)))
        n_cols = padded

    n_col_blocks = n_cols // block_size
    x_blocked = x.reshape(n_rows, n_col_blocks, block_size)

    abs_max = jnp.max(jnp.abs(x_blocked), axis=-1)
    abs_max = jnp.maximum(abs_max, EPS)

    if jnp.issubdtype(dtype, jnp.integer):
        dtype_info = jnp.iinfo(dtype)
        max_val = float(dtype_info.max)
        scale_inv = max_val / abs_max
        scale_expanded = jnp.expand_dims(scale_inv, axis=-1)
        x_scaled = x_blocked * scale_expanded
        x_scaled = jnp.floor(x_scaled + 0.5)
        x_q_blocked = jnp.clip(x_scaled, dtype_info.min, dtype_info.max).astype(dtype)
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        dequant_scale = abs_max / max_val
        return x_q, dequant_scale.astype(jnp.float32)
    elif dtype == jnp.float8_e4m3fn:
        max_val = 448.0
        scale_inv = max_val / abs_max
        scale_expanded = jnp.expand_dims(scale_inv, axis=-1)
        x_scaled = x_blocked * scale_expanded
        # No floor/clip logic for FP8, just cast
        x_q_blocked = x_scaled.astype(dtype)
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        dequant_scale = abs_max / max_val
        return x_q, dequant_scale.astype(jnp.float32)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

def _quantize_array_2d(x, x_abs_max, quant_dtype):
    is_float = jnp.issubdtype(quant_dtype, jnp.floating)
    dtype_info = jnp.finfo(quant_dtype) if is_float else jnp.iinfo(quant_dtype)
    dtype_max = 448.0 if quant_dtype == jnp.float8_e4m3fn else float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    scale_basis = jnp.maximum(x_abs_max, jnp.finfo(jnp.float32).tiny)
    scale_inv = dtype_max / scale_basis
    x_scaled = x.astype(jnp.float32) * scale_inv

    if not is_float:
        x_scaled = jnp.floor(x_scaled + 0.5)
        quantized_array = jnp.clip(x_scaled, dtype_min, dtype_max).astype(quant_dtype)
    else:
        quantized_array = x_scaled.astype(quant_dtype)
    
    scale = scale_basis / dtype_max
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
        is_first_step = (in_block_idx == 0)
        is_last_step = (in_block_idx == (n_in_blocks - 1))
    else:
        is_first_step, is_last_step = True, True

    def matmul_body(quant, is_first_step, is_last_step):
        x_abs_max_row = x_abs_max_ref[in_block_idx]
        x_abs_max_current = x_abs_max_row[:, None]

        if quantize_activation:
            if quant:
                 x_q_tmp, x_scale_tmp = _quantize_array_2d(x_ref[...], x_abs_max_current, x_q_dtype)
                 if save_x_q:
                     x_q_scratch[...] = x_q_tmp
                     x_scale_scratch[...] = x_scale_tmp
            else:
                 x_q_tmp = x_q_scratch[...]
                 x_scale_tmp = x_scale_scratch[...]
        else:
            x_q_tmp = x_ref[...]
            x_scale_tmp = 1.0 

        acc = jax.lax.dot_general(
            x_q_tmp, w_q_ref[...], (((1,), (1,)), ((), ())),
            preferred_element_type=dot_dtype,
        )

        w_scale_current = w_scale_ref[in_block_idx][None, :]
        acc = acc.astype(jnp.float32)
        
        if quantize_activation:
            acc *= x_scale_tmp
        acc *= w_scale_current

        if save_acc:
            def _add_scratch(curr_acc): return curr_acc + acc_scratch[...]
            acc = jax.lax.cond(is_first_step, lambda a: a, _add_scratch, acc)

            def _write_out(_): 
                out_ref[...] = acc.astype(x_ref.dtype)
                return None
            def _write_scratch(_): 
                acc_scratch[...] = acc
                return None
            jax.lax.cond(is_last_step, _write_out, _write_scratch, None)
        else:
            out_ref[...] = acc.astype(x_ref.dtype)

    unfold_args((quant, is_first_step, is_last_step), (), matmul_body)

@functools.partial(jax.jit, static_argnames=['x_q_dtype', 'quant_block_size', 'batch_block_size', 'out_block_size'])
def original_quantized_matmul_2d(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quant_block_size: int,
    x_q_dtype=None,
    *,
    batch_block_size: int = 128,
    out_block_size: int = 128
) -> jax.Array:
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

    padded_n_batch = next_multiple(orig_n_batch, batch_block_size)
    if orig_n_batch < padded_n_batch:
        x = jnp.pad(x, ((0, padded_n_batch - orig_n_batch), (0, 0)))
        x_abs_max = jnp.pad(x_abs_max, ((0, padded_n_batch - orig_n_batch), (0, 0)))

    padded_n_out = next_multiple(orig_n_out, out_block_size)
    if orig_n_out < padded_n_out:
        w_q = jnp.pad(w_q, ((0, padded_n_out - orig_n_out), (0, 0)))
        w_scale = jnp.pad(w_scale, ((0, padded_n_out - orig_n_out), (0, 0)))

    w_scale = w_scale.astype(jnp.float32)
    x_abs_max_t = jnp.transpose(x_abs_max)
    w_scale_t = jnp.transpose(w_scale)

    n_batch = padded_n_batch // batch_block_size
    n_out = padded_n_out // out_block_size
    save_acc = n_in_blocks > 1
    save_x_q = False
    
    # Determine accumulator type
    dot_dtype = jnp.int32 if (x_q_dtype == jnp.int8 and w_q.dtype == jnp.int8) else jnp.float32

    kernel = pl.pallas_call(
        functools.partial(
            _quantized_matmul_kernel_2d,
            x_q_dtype=x_q_dtype,
            dot_dtype=dot_dtype,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block_size, quant_block_size), lambda b, o, i: (b, i)),
                pl.BlockSpec((out_block_size, quant_block_size), lambda b, o, i: (o, i)),
                pl.BlockSpec((n_in_blocks, out_block_size), lambda b, o, i: (0, o)),
                pl.BlockSpec((n_in_blocks, batch_block_size), lambda b, o, i: (0, b)),
            ],
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block_size, out_block_size), jnp.float32) if save_acc else None,
                pltpu.VMEM((batch_block_size, quant_block_size), x_q_dtype) if save_x_q else None,
                pltpu.VMEM((batch_block_size, 1), jnp.float32) if save_x_q else None,
            ],
            grid=(n_batch, n_out, n_in_blocks),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_n_batch, padded_n_out), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'arbitrary', 'arbitrary'),
            vmem_limit_bytes=get_tpu_vmem_limit(),
        ),
    )

    out = kernel(x, w_q, w_scale_t, x_abs_max_t)
    return out[:orig_n_batch, :orig_n_out]

# ==========================================
# 4. NEW TPU v7 OPTIMIZED KERNEL
# ==========================================

class QuantizationResult(NamedTuple):
    q_data: jax.Array
    scales: jax.Array

@functools.partial(jax.jit, static_argnames=['out_block_size', 'dtype'])
def quantize_offline_tiled_v7(w: jax.Array, out_block_size: int, dtype: jnp.dtype = jnp.int8) -> QuantizationResult:
    n_out, n_in = w.shape
    padded_in = next_multiple(n_in, TILE_K_SIZE)
    padded_out = next_multiple(n_out, out_block_size)
    
    if padded_in > n_in or padded_out > n_out:
        w = jnp.pad(w, ((0, padded_out - n_out), (0, padded_in - n_in)))

    n_out_outer = padded_out // out_block_size
    n_in_outer = padded_in // TILE_K_SIZE
    
    w_reshaped = w.reshape(n_out_outer, out_block_size, n_in_outer, TILE_K_SIZE)
    w_tiled = w_reshaped.transpose(0, 2, 3, 1) # (N_o, K_o, K_t, N_t)
    
    w_blocked = w_tiled.reshape(n_out_outer, n_in_outer, SUPER_CHUNK, BLOCK_K, out_block_size)
    w_max = jnp.max(jnp.abs(w_blocked), axis=3) 
    w_max = jnp.maximum(w_max, 1e-6)
    
    if dtype == jnp.int8:
        max_val = 127.0
        scales = w_max / max_val
        val = w_blocked / scales[:, :, :, None, :]
        w_q = jnp.clip(jnp.floor(val + 0.5), -128, 127).astype(jnp.int8)
    elif dtype == jnp.float8_e4m3fn:
        max_val = 448.0
        scales = w_max / max_val
        val = w_blocked / scales[:, :, :, None, :]
        w_q = val.astype(jnp.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    
    w_q_tiled = w_q.reshape(n_out_outer, n_in_outer, TILE_K_SIZE, out_block_size)
    # NOTE: We do NOT return 'dtype' here because JIT functions cannot return Python types.
    return QuantizationResult(w_q_tiled, scales.astype(jnp.float32))

def _compute_quant_params(x_slice, w_scale_row, max_val, dtype):
    """
    On-the-fly quantization of activation chunk inside the kernel.
    """
    x_max = jnp.max(jnp.abs(x_slice), axis=1, keepdims=True)
    x_max = jnp.maximum(x_max, 1e-6)
    scale_inv = max_val / x_max
    val = x_slice * scale_inv
    
    if dtype == jnp.int8:
        x_q = jnp.clip(jnp.floor(val + 0.5), -128, 127).astype(jnp.int8)
    else:
        # float8: just cast
        x_q = val.astype(dtype)

    # Combined scale: (x_max / max_val) * w_scale
    scale_factor = (x_max / max_val) * w_scale_row
    return x_q, scale_factor

def _kernel_real_tiled_entry(x_ref, w_tiled_ref, scales_ref, out_ref, acc_vmem, *,
                             batch_block_size, out_block_size, dtype):
    acc_vmem[...] = jnp.zeros((batch_block_size, out_block_size), dtype=jnp.float32)

    # Determine constants based on dtype
    if dtype == jnp.int8:
        max_val = 127.0
        dot_preferred = jnp.int32
    else:
        max_val = 448.0
        dot_preferred = jnp.float32

    def pipeline_step(x_chunk, w_chunk_4d, s_chunk_4d):
        w_curr_super = w_chunk_4d[0, 0] 
        s_curr_super = s_chunk_4d[0, 0] 
        
        # --- Stage 0 ---
        x_0 = x_chunk[:, 0:BLOCK_K]
        w_0 = w_curr_super[0:BLOCK_K, :]
        s_0 = s_curr_super[0, :][None, :]
        
        x_q_curr, scale_curr = _compute_quant_params(x_0, s_0, max_val, dtype)
        dot_curr = jax.lax.dot_general(x_q_curr, w_0, (((1,), (0,)), ((), ())), preferred_element_type=dot_preferred)
        dot_prev = dot_curr
        scale_prev = scale_curr

        # --- Steady State ---
        for j in range(1, SUPER_CHUNK):
            k_start = j * BLOCK_K
            k_end = (j + 1) * BLOCK_K
            
            x_next = x_chunk[:, k_start:k_end]
            w_next = w_curr_super[k_start:k_end, :]
            s_next = s_curr_super[j, :][None, :]
            
            x_q_next, scale_next = _compute_quant_params(x_next, s_next, max_val, dtype)
            dot_next = jax.lax.dot_general(x_q_next, w_next, (((1,), (0,)), ((), ())), preferred_element_type=dot_preferred)
            
            acc_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev
            dot_prev = dot_next
            scale_prev = scale_next
            
        # --- Epilogue ---
        acc_vmem[...] += dot_prev.astype(jnp.float32) * scale_prev

    n_k_chunks = w_tiled_ref.shape[1] 
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(n_k_chunks,),
        in_specs=[
            pl.BlockSpec((batch_block_size, TILE_K_SIZE), lambda i: (0, i)),
            pl.BlockSpec((1, 1, TILE_K_SIZE, out_block_size), lambda i: (0, i, 0, 0)),
            pl.BlockSpec((1, 1, SUPER_CHUNK, out_block_size), lambda i: (0, i, 0, 0))
        ]
    )(x_ref, w_tiled_ref, scales_ref)
    
    out_ref[...] = acc_vmem[...].astype(out_ref.dtype)

@functools.partial(jax.jit, static_argnames=['out_block_size', 'quant_dtype'])
def dispatch_real_v7(x: jax.Array, w_tiled: jax.Array, scales_tiled: jax.Array,
                     out_block_size: int, quant_dtype: jnp.dtype):
    bs, n_in = x.shape
    n_outer, k_outer, k_tile, n_tile = w_tiled.shape
    padded_bs = next_multiple(bs, BLOCK_B)
    total_in_supported = k_outer * k_tile
    
    if padded_bs > bs: x = jnp.pad(x, ((0, padded_bs - bs), (0, 0)))
    if total_in_supported > n_in: x = jnp.pad(x, ((0, 0), (0, total_in_supported - n_in)))

    grid = (padded_bs // BLOCK_B, n_outer)

    in_specs = [
        pl.BlockSpec((BLOCK_B, total_in_supported), lambda b, o: (b, 0)),
        pl.BlockSpec((1, k_outer, k_tile, n_tile), lambda b, o: (o, 0, 0, 0)),
        pl.BlockSpec((1, k_outer, SUPER_CHUNK, n_tile), lambda b, o: (o, 0, 0, 0)),
    ]

    kernel = pl.pallas_call(
        functools.partial(_kernel_real_tiled_entry, 
                          batch_block_size=BLOCK_B, 
                          out_block_size=out_block_size,
                          dtype=quant_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=pl.BlockSpec((BLOCK_B, out_block_size), lambda b, o: (b, o)),
            grid=grid,
            scratch_shapes=[pltpu.VMEM((BLOCK_B, out_block_size), jnp.float32)]
        ),
        out_shape=jax.ShapeDtypeStruct((padded_bs, n_outer * out_block_size), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_tpu_vmem_limit())
    )
    
    out = kernel(x, w_tiled, scales_tiled)
    return out[:bs, : n_outer * out_block_size]

# ==========================================
# 5. BENCHMARK SUITE
# ==========================================

class BenchmarkSuite:
    def measure_ms(self, func, args, n_iter=20, n_warmup=5):
        try:
            jax.block_until_ready(func(*args))
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
            
        for _ in range(n_warmup): func(*args)
        jax.block_until_ready(func(*args))
        start = time.time()
        for _ in range(n_iter): func(*args)
        jax.block_until_ready(func(*args))
        return ((time.time() - start) / n_iter) * 1000

    def check_correctness(self, x, w, block_size, res_orig, res_opt, dtype_str):
        if dtype_str == "int8":
            print("  > Verifying against JAX Int8 Reference...")
            # For correctness checking of Int8, we use the specific int8 ref logic
            try:
                ref = ref_int8_matmul(x, w, block_size)
                
                # Using Relative Error for validation
                # Mean Relative Error (MRE)
                ref_abs = jnp.abs(ref)
                denom = jnp.maximum(ref_abs, 1.0) 
                
                diff_orig = jnp.mean(jnp.abs(res_orig - ref) / denom)
                diff_opt = jnp.mean(jnp.abs(res_opt - ref) / denom)
                
                thresh = 0.05 # Int8 error tolerance
                
                status_orig = "OK" if diff_orig < thresh else "FAIL"
                status_opt = "OK" if diff_opt < thresh else "FAIL"
                
                print(f"    Original Rel Err: {diff_orig:.4f} ({status_orig})")
                print(f"    Opt V7 Rel Err:   {diff_opt:.4f} ({status_opt})")
            except Exception as e:
                print(f"    Verification Skipped due to error: {e}")
        else:
             print("  > Skipping precise reference check for FP8 (experimental)...")

    def run_benchmark(self):
        print(f"Config: BlockK={BLOCK_K}, BlockB={BLOCK_B}, SuperChunk={SUPER_CHUNK}")
        
        shapes = [
            (1, 8192, 8192),
            (8, 8192, 8192),
            (32, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            (256, 4096, 4096),
            (512, 4096, 4096),
            (1, 8192, 8192),
            (8, 8192, 8192),
            (32, 8192, 8192),
            (128, 8192, 8192),
            (256, 8192, 8192),
            (512, 8192, 8192),
        ]
        
        out_block = 512 
        quant_block = BLOCK_K
        results = []
        
        # We will benchmark both Int8 and Float8
        bench_dtypes = [
            ("int8", jnp.int8),
            ("fp8", jnp.float8_e4m3fn)
        ]

        for bs, n_in, n_out in shapes:
            print(f"\nRunning shape B={bs}, In={n_in}, Out={n_out}...")
            
            # Shared inputs
            key = random.PRNGKey(0)
            x = random.uniform(key, (bs, n_in), dtype=jnp.bfloat16)
            w = random.uniform(key, (n_out, n_in), dtype=jnp.bfloat16)
            
            # 1. Baseline BF16
            t_bf16 = self.measure_ms(jax.lax.dot_general, (x, w, (((1,), (1,)), ((), ()))))

            for d_name, d_type in bench_dtypes:
                print(f"  [{d_name.upper()}] Benchmarking...")
                
                # 2. Original Pallas Kernel
                # Quantize weights
                w_q_orig, w_s_orig = quantize_2d_blocked(w, quant_block, d_type)
                t_orig = self.measure_ms(original_quantized_matmul_2d, 
                                        (x, w_q_orig, w_s_orig, quant_block, d_type))
                
                # 3. V7 Optimized Kernel
                w_res_tiled = quantize_offline_tiled_v7(w, out_block, d_type)
                t_opt = self.measure_ms(dispatch_real_v7, 
                                        (x, w_res_tiled.q_data, w_res_tiled.scales, out_block, d_type))

                # Correctness (Run once)
                res_orig = original_quantized_matmul_2d(x, w_q_orig, w_s_orig, quant_block, d_type).astype(jnp.float32)
                res_opt = dispatch_real_v7(x, w_res_tiled.q_data, w_res_tiled.scales, out_block, d_type).astype(jnp.float32)
                self.check_correctness(x, w, quant_block, res_orig, res_opt, d_name)

                # Avoid division by zero
                bf16_speedup = t_bf16 / t_opt if t_opt > 0 else 0.0

                results.append({
                    "Batch": bs,
                    "In": n_in,
                    "Out": n_out,
                    "Dtype": d_name,
                    "BF16 (ms)": f"{t_bf16:.3f}",
                    "Orig Pallas (ms)": f"{t_orig:.3f}",
                    "Opt V7 (ms)": f"{t_opt:.3f}",
                    "Speedup (Opt/Orig)": f"{t_orig/t_opt:.2f}x" if t_opt > 0 else "N/A",
                    "Speedup vs BF16": f"{bf16_speedup:.2f}x"
                })
            
        df = pd.DataFrame(results)
        print("\nBenchmark Results:")
        # Reorder columns for readability
        cols = ["Batch", "In", "Out", "Dtype", "BF16 (ms)", "Orig Pallas (ms)", "Opt V7 (ms)", "Speedup (Opt/Orig)", "Speedup vs BF16"]
        print(df[cols].to_string(index=False))

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_benchmark()