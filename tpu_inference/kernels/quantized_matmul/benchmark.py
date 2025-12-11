import functools
import time
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random

# Import your kernels
from tpu_inference.kernels.quantized_matmul.kernel_2d import (
    quantized_matmul_2d as original_quantized_matmul_2d,
    dispatch_real_v7
)

# ==========================================
# 1. SHARED UTILITIES
# ==========================================

BLOCK_K = 128
BLOCK_B = 64
SUPER_CHUNK = 16

def next_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

# ==========================================
# 2. INT8 REFERENCE (Verification)
# ==========================================

@functools.partial(jax.jit, static_argnames=['block_size'])
def ref_int8_matmul(x, w, block_size):
    """Pure JAX reference for Block-wise W8A8 Quantized Matmul."""
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
# 3. QUANTIZATION UTILS
# ==========================================

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
        x_q_blocked = x_scaled.astype(dtype)
        x_q = x_q_blocked.reshape(n_rows, n_cols)
        dequant_scale = abs_max / max_val
        return x_q, dequant_scale.astype(jnp.float32)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

# ==========================================
# 4. BENCHMARK SUITE
# ==========================================

class BenchmarkSuite:
    def measure_ms(self, func, args, n_iter=100, n_warmup=5):
        try:
            # Simple validity check
            jax.block_until_ready(func(*args))
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
            
        for _ in range(n_warmup): 
            jax.block_until_ready(func(*args))
            
        start = time.time()
        for _ in range(n_iter): 
            jax.block_until_ready(func(*args))
        end = time.time()
        
        return ((end - start) / n_iter) * 1000

    def check_correctness(self, x, w, block_size, res_orig, res_opt, dtype_str):
        if dtype_str == "int8":
            print("  > Verifying against JAX Int8 Reference...")
            try:
                ref = ref_int8_matmul(x, w, block_size)
                
                # Mean Relative Error (MRE)
                ref_abs = jnp.abs(ref)
                denom = jnp.maximum(ref_abs, 1.0) 
                
                diff_orig = jnp.mean(jnp.abs(res_orig - ref) / denom)
                diff_opt = jnp.mean(jnp.abs(res_opt - ref) / denom)
                
                thresh = 0.05
                
                status_orig = "OK" if diff_orig < thresh else "FAIL"
                status_opt = "OK" if diff_opt < thresh else "FAIL"
                
                print(f"    Original Rel Err: {diff_orig:.4f} ({status_orig})")
                print(f"    New V7 Rel Err:   {diff_opt:.4f} ({status_opt})")
            except Exception as e:
                print(f"    Verification Skipped due to error: {e}")
        else:
             print("  > Skipping precise reference check for FP8...")

    def run_benchmark(self):
        print(f"Config: BlockK={BLOCK_K}, BlockB={BLOCK_B}")
        
        shapes = [
            (1, 8192, 8192),
            (8, 8192, 8192),
            (16, 8192, 8192),
            (32, 8192, 8192),
            (64, 8192, 8192),
            (128, 8192, 8192),
            (1, 4096, 4096),
            (8, 4096, 4096),
            (16, 4096, 4096),
            (32, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
        ]
        
        out_block = 512 
        quant_block = BLOCK_K
        results = []
        
        bench_dtypes = [
            ("int8", jnp.int8),
            # ("fp8", jnp.float8_e4m3fn)
        ]

        for bs, n_in, n_out in shapes:
            print(f"\nRunning shape B={bs}, In={n_in}, Out={n_out}...")
            
            key = random.PRNGKey(0)
            x = random.uniform(key, (bs, n_in), dtype=jnp.bfloat16)
            w = random.uniform(key, (n_out, n_in), dtype=jnp.bfloat16)
            
            # 1. Baseline BF16 (Matmul only)
            t_bf16 = self.measure_ms(jax.lax.dot_general, (x, w, (((1,), (1,)), ((), ()))))

            for d_name, d_type in bench_dtypes:
                print(f"  [{d_name.upper()}] Benchmarking...")
                
                # Quantize weights (Shared standard format)
                # w_q_std: [N_Out, N_In]
                # w_s_std: [N_Out, N_Blocks]
                w_q_std, w_s_std = quantize_2d_blocked(w, quant_block, d_type)
                
                # 2. Original Pallas Kernel
                t_orig = self.measure_ms(original_quantized_matmul_2d, 
                                        (x, w_q_std, w_s_std, quant_block, d_type))
                
                # 3. New V7 Optimized Kernel
                # Note: We pass standard weights. The kernel handles scale transposition internally.
                t_opt = self.measure_ms(dispatch_real_v7, 
                                        (x, w_q_std, w_s_std, out_block, d_type))

                # Correctness (Run once)
                res_orig = original_quantized_matmul_2d(x, w_q_std, w_s_std, quant_block, d_type).astype(jnp.float32)
                res_opt = dispatch_real_v7(x, w_q_std, w_s_std, out_block, d_type).astype(jnp.float32)
                self.check_correctness(x, w, quant_block, res_orig, res_opt, d_name)

                bf16_speedup = t_bf16 / t_opt if t_opt > 0 else 0.0
                orig_speedup = t_orig / t_opt if t_opt > 0 else 0.0

                results.append({
                    "Batch": bs,
                    "In": n_in,
                    "Out": n_out,
                    "Dtype": d_name,
                    "BF16 (ms)": f"{t_bf16:.3f}",
                    "Orig (ms)": f"{t_orig:.3f}",
                    "V7 (ms)": f"{t_opt:.3f}",
                    "Bf16 Speedup": f"{bf16_speedup:.2f}x" if t_opt > 0 else "N/A",
                    "Orig Speedup": f"{orig_speedup:.2f}x" if t_opt > 0 else "N/A",
                })
            
        df = pd.DataFrame(results)
        print("\nBenchmark Results:")
        cols = ["Batch", "In", "Out", "Dtype", "BF16 (ms)", "Orig (ms)", "V7 (ms)", "Bf16 Speedup", "Orig Speedup"]
        print(df[cols].to_string(index=False))

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_benchmark()