import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import pandas as pd
import numpy as np

# --- 1. Utilities & Constants ---

def next_multiple(x: int, m: int) -> int:
    """Calculates the next multiple of m greater than or equal to x."""
    return ((x + m - 1) // m) * m

def get_tpu_vmem_limit() -> int:
    """Returns the TPU VMEM limit in bytes (96MB)."""
    return 96 * 1024 * 1024

FP8_MAX = 448.0
FP8_TYPE = jnp.float8_e4m3fn
INT8_MAX = 127.0

# --- 2. Quantization Helpers (Weights Only) ---

class QuantizationResult(NamedTuple):
    q_data: jax.Array
    scales_t: jax.Array

@functools.partial(jax.jit, static_argnames=['block_size'])
def quantize_offline_int8(w: jax.Array, block_size: int) -> QuantizationResult:
    """
    Quantizes weights to INT8 offline.
    """
    n_out, n_in = w.shape
    padded_in = next_multiple(n_in, block_size)
    if padded_in > n_in: w = jnp.pad(w, ((0, 0), (0, padded_in - n_in)))
    n_blocks = padded_in // block_size
    
    w_blocked = w.reshape(n_out, n_blocks, block_size)
    w_max = jnp.max(jnp.abs(w_blocked), axis=-1)
    scales = w_max / INT8_MAX
    
    val = w_blocked / scales[:, :, None]
    w_rounded = jnp.floor(val + 0.5)
    w_q = jnp.clip(w_rounded, -128, 127).astype(jnp.int8)
    
    w_q = w_q.reshape(n_out, padded_in)
    scales_t = jnp.copy(scales.T.astype(jnp.float32))
    return QuantizationResult(w_q, scales_t)

@functools.partial(jax.jit, static_argnames=['block_size'])
def quantize_offline_fp8(w: jax.Array, block_size: int) -> QuantizationResult:
    """
    Quantizes weights to FP8 offline.
    """
    n_out, n_in = w.shape
    padded_in = next_multiple(n_in, block_size)
    if padded_in > n_in: w = jnp.pad(w, ((0, 0), (0, padded_in - n_in)))
    n_blocks = padded_in // block_size
    
    w_blocked = w.reshape(n_out, n_blocks, block_size)
    w_max = jnp.max(jnp.abs(w_blocked), axis=-1)
    w_max = jnp.maximum(w_max, 1e-6)
    scales = w_max / FP8_MAX
    
    w_q = (w_blocked / scales[:, :, None]).astype(FP8_TYPE).reshape(n_out, padded_in)
    scales_t = jnp.copy(scales.T.astype(jnp.float32))
    return QuantizationResult(w_q, scales_t)


# --- 3. KERNEL BODIES (Fused Only) ---

def _body_fused_int8(x, w_q, w_s_t, out, *, bs, nb):
    """
    Fused INT8 Kernel Body.
    Performs online quantization of activations (x) and dot product with pre-quantized weights (w_q).
    """
    acc = jnp.zeros_like(out[...], dtype=jnp.float32)
    for i in range(nb):
        cs = i * bs
        x_sl = x[:, cs:cs+bs]
        w_sl = w_q[:, cs:cs+bs]
        
        # Fuse: Max, Scale, Round, Clip, Cast
        x_max = jnp.max(jnp.abs(x_sl), axis=1)
        x_max = jnp.maximum(x_max, 1e-6)
        x_scale = x_max / INT8_MAX
        
        val = x_sl / x_scale[:, None]
        x_rounded = jnp.floor(val + 0.5)
        x_q = jnp.clip(x_rounded, -128, 127).astype(jnp.int8)
        
        dot = jax.lax.dot_general(x_q, w_sl, (((1,), (1,)), ((), ())), preferred_element_type=jnp.int32)
        acc += dot.astype(jnp.float32) * x_scale[:, None] * w_s_t[i][None, :]
    out[...] = acc.astype(out.dtype)

def _body_fused_fp8(x, w_q, w_s_t, out, *, bs, nb):
    """
    Fused FP8 Kernel Body.
    Performs online quantization of activations (x) and dot product with pre-quantized weights (w_q).
    """
    acc = jnp.zeros_like(out[...], dtype=jnp.float32)
    for i in range(nb):
        cs = i * bs
        x_sl = x[:, cs:cs+bs]
        w_sl = w_q[:, cs:cs+bs]
        
        x_max = jnp.max(jnp.abs(x_sl), axis=1)
        x_scale = jnp.maximum(x_max, 1e-6) / FP8_MAX
        x_q = (x_sl / x_scale[:, None]).astype(FP8_TYPE)
        
        dot = jax.lax.dot_general(x_q, w_sl, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        acc += dot * x_scale[:, None] * w_s_t[i][None, :]
    out[...] = acc.astype(out.dtype)


# --- 4. Dispatcher ---

@functools.partial(jax.jit, static_argnames=['dtype_mode', 'quant_block_size', 'batch_block_size', 'out_block_size'])
def dispatch_fused_kernel(
    x: jax.Array, w_q: jax.Array, w_scales_t: jax.Array, 
    *,
    dtype_mode: str, 
    quant_block_size: int, 
    batch_block_size: int, 
    out_block_size: int
):
    """
    Dispatches the appropriate fused Pallas kernel based on dtype_mode.
    Handles padding and grid specification.
    """
    bs, n_in = x.shape
    n_out, _ = w_q.shape
    
    if dtype_mode == "int8": body = _body_fused_int8
    elif dtype_mode == "fp8": body = _body_fused_fp8
    else: raise ValueError("Invalid kernel config: dtype_mode must be 'int8' or 'fp8'")

    padded_bs = next_multiple(bs, batch_block_size)
    padded_out = next_multiple(n_out, out_block_size)
    padded_in = next_multiple(n_in, quant_block_size)
    n_blocks = padded_in // quant_block_size

    if padded_bs > bs: 
        x = jnp.pad(x, ((0, padded_bs - bs), (0, 0)))
    if padded_out > n_out: 
        w_q = jnp.pad(w_q, ((0, padded_out - n_out), (0, 0)))
        w_scales_t = jnp.pad(w_scales_t, ((0, 0), (0, padded_out - n_out)))
    
    w_scales_t = jnp.copy(w_scales_t)

    grid = (padded_bs // batch_block_size, padded_out // out_block_size)
    
    in_specs = [
        pl.BlockSpec((batch_block_size, padded_in), lambda b, o: (b, 0)),
        pl.BlockSpec((out_block_size, padded_in), lambda b, o: (o, 0)),
        pl.BlockSpec((n_blocks, out_block_size), lambda b, o: (0, o)),
    ]
    args = (x, w_q, w_scales_t)

    kernel = pl.pallas_call(
        functools.partial(body, bs=quant_block_size, nb=n_blocks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=pl.BlockSpec((batch_block_size, out_block_size), lambda b, o: (b, o)),
            grid=grid
        ),
        out_shape=jax.ShapeDtypeStruct((padded_bs, padded_out), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=get_tpu_vmem_limit())
    )
    return kernel(*args)[:bs, :n_out]

# --- 5. E2E Wrappers ---

def run_fused_int8(x, w_q, w_s_t, blk, bb, ob=512):
    return dispatch_fused_kernel(x, w_q, w_s_t, dtype_mode="int8", 
                                 quant_block_size=blk, batch_block_size=bb, out_block_size=ob)

def run_fused_fp8(x, w_q, w_s_t, blk, bb, ob=512):
    return dispatch_fused_kernel(x, w_q, w_s_t, dtype_mode="fp8", 
                                 quant_block_size=blk, batch_block_size=bb, out_block_size=ob)


# --- 6. Verification & Analysis Suite ---

def get_ref_dot(x, q_w, scales, blk):
    """Computes a reference dot product using dequantized weights and standard precision."""
    n_out, n_in = q_w.shape
    q_w_reshaped = q_w.reshape(n_out, n_in // blk, blk)
    scales_aligned = scales.T
    scales_expanded = scales_aligned[..., None] 
    w_dequant = (q_w_reshaped.astype(jnp.float32) * scales_expanded).reshape(n_out, n_in)
    x_f32 = x.astype(jnp.float32)
    return jax.lax.dot_general(x_f32, w_dequant, (((1,), (1,)), ((), ())))

def verify_all_kernels():
    print("\n=== Correctness Verification (2x2 Matrix) ===")
    bs, n_in, n_out = 16, 512, 512
    blk, bb = 128, 16
    
    key = random.PRNGKey(42)
    x = random.uniform(key, (bs, n_in), dtype=jnp.bfloat16)
    w = random.uniform(key, (n_out, n_in), dtype=jnp.bfloat16)
    
    w_res_i8 = quantize_offline_int8(w, blk)
    w_res_f8 = quantize_offline_fp8(w, blk)

    ref_i8 = get_ref_dot(x, w_res_i8.q_data, w_res_i8.scales_t, blk)
    ref_f8 = get_ref_dot(x, w_res_f8.q_data, w_res_f8.scales_t, blk)

    out = run_fused_int8(x, w_res_i8.q_data, w_res_i8.scales_t, blk, bb, ob=128)
    err = float(jnp.mean(jnp.abs(ref_i8 - out)))
    print(f"Fused Int8 Error: {err:.4f} {'[PASS]' if err < 0.5 else '[FAIL]'}")

    out = run_fused_fp8(x, w_res_f8.q_data, w_res_f8.scales_t, blk, bb, ob=128)
    err = float(jnp.mean(jnp.abs(ref_f8 - out)))
    print(f"Fused FP8 Error:  {err:.4f} {'[PASS]' if err < 0.5 else '[FAIL]'}")


class BenchmarkSuite:
    def measure_ms(self, func, args, n_iter=20, n_warmup=5):
        jax.block_until_ready(func(*args))
        for _ in range(n_warmup): func(*args)
        jax.block_until_ready(func(*args))
        start = time.time()
        for _ in range(n_iter): func(*args)
        jax.block_until_ready(func(*args))
        return ((time.time() - start) / n_iter) * 1000

    def get_optimal_tiling(self, bs):
        if bs <= 1: return (16, 128)
        if bs <= 32: return (32, 128)
        if bs <= 128: return (128, 256)
        return (128, 512)

    def run_general(self):
        print("\n=== General Benchmark (Ref vs Fused Kernels) ===")
        shapes = [
            (1, 4096, 4096),
            (32, 4096, 4096),
            (128, 4096, 4096),
            (1024, 4096, 4096)
        ]
        
        blk = 128
        dtype = jnp.bfloat16
        results = []

        for bs, n_in, n_out in shapes:
            key = random.PRNGKey(0)
            x = random.uniform(key, (bs, n_in), dtype=dtype)
            w = random.uniform(key, (n_out, n_in), dtype=dtype)
            
            bb, ob = self.get_optimal_tiling(bs)
            w_i8 = quantize_offline_int8(w, blk)
            w_f8 = quantize_offline_fp8(w, blk)

            # 1. Ref
            t_ref = self.measure_ms(jax.lax.dot_general, (x, w, (((1,), (1,)), ((), ()))))

            # 2. Kernels
            t_f_i8 = self.measure_ms(run_fused_int8, (x, w_i8.q_data, w_i8.scales_t, blk, bb, ob))
            t_f_f8 = self.measure_ms(run_fused_fp8, (x, w_f8.q_data, w_f8.scales_t, blk, bb, ob))

            results.append({
                "Batch": bs, 
                "Tiling": f"{bb}x{ob}",
                "Ref (ms)": t_ref,
                "Fused Int8 (x)": f"{t_ref / t_f_i8:.2f}x",
                "Fused FP8 (x)":  f"{t_ref / t_f_f8:.2f}x",
            })
        
        print(pd.DataFrame(results))

    def run_tuning(self):
        print("\n=== Tile Tuning Details (Top 10 Configs) ===")
        shapes = [(1, 4096, 4096), (32, 4096, 4096), (128, 4096, 4096), (1024, 4096, 4096)]
        blk = 128
        dtype = jnp.bfloat16
        
        tile_configs = [
            (16, 128), (32, 128), (128, 128), 
            (128, 256), (128, 512), 
            (256, 128), (256, 256)
        ]

        for bs, n_in, n_out in shapes:
            print(f"\n--- Shape: {bs} x {n_in} x {n_out} ---")
            key = random.PRNGKey(42)
            x = random.uniform(key, (bs, n_in), dtype=dtype)
            w = random.uniform(key, (n_out, n_in), dtype=dtype)
            
            w_i8 = quantize_offline_int8(w, blk)
            w_f8 = quantize_offline_fp8(w, blk)
            
            t_ref = self.measure_ms(jax.lax.dot_general, (x, w, (((1,), (1,)), ((), ()))))
            print(f"Ref (BF16): {t_ref:.4f} ms")
            
            results = []
            
            # Tuning loop for FUSED kernels only
            for name, func, w_res in [
                ("Fused Int8", run_fused_int8, w_i8),
                ("Fused FP8",  run_fused_fp8,  w_f8),
            ]:
                for bb, ob in tile_configs:
                    if bb > 128 and bs < 64: continue 
                    try:
                        t_ms = self.measure_ms(func, (x, w_res.q_data, w_res.scales_t, blk, bb, ob))
                        results.append({
                            "Kernel": name,
                            "Tile": f"{bb}x{ob}", 
                            "Time (ms)": t_ms, 
                            "Speedup": f"{t_ref/t_ms:.2f}x"
                        })
                    except: pass
            
            if results:
                df = pd.DataFrame(results).sort_values(by="Time (ms)", ascending=True)
                print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    verify_all_kernels()
    suite = BenchmarkSuite()
    suite.run_general()
    suite.run_tuning()