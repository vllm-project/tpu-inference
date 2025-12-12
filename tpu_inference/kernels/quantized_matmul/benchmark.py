import functools
import time
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random

# Import your kernels
from tpu_inference.kernels.quantized_matmul.kernel_2d import quantized_matmul_2d

# ==========================================
# 1. SHARED UTILITIES
# ==========================================

QUANT_GROUP_SIZE = 128

def next_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

# ==========================================
# 2. QUANTIZATION UTILS
# ==========================================

EPS = jnp.finfo(jnp.float16).tiny

def quantize_2d_blocked(x: jax.Array, block_size: int, dtype: jnp.dtype = jnp.int8):
    """Simple quantizer for benchmarking setup."""
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
# 3. BENCHMARK SUITE
# ==========================================

class BenchmarkSuite:
    def measure_ms(self, func, args, n_iter=1000, n_warmup=100):
        try:
            # Simple validity check
            jax.block_until_ready(func(*args))
        except Exception as e:
            print(f"Error during execution: {e}")
            return 0.0
            
        for _ in range(n_warmup): 
            jax.block_until_ready(func(*args))
            
        start = time.time()
        for _ in range(n_iter): 
            jax.block_until_ready(func(*args))
        end = time.time()
        
        return ((end - start) / n_iter) * 1000

    def run_benchmark(self):
        print(f"Config: QuantGroupSize={QUANT_GROUP_SIZE}")
        
        shapes = [
            (1, 16384, 16384),
            (8, 16384, 16384),
            (16, 16384, 16384),
            (1, 8192, 8192),
            (8, 8192, 8192),
            (16, 8192, 8192),
            (32, 8192, 8192),
            (64, 8192, 8192),
            (128, 8192, 8192),
            (256, 8192, 8192),
            (512, 8192, 8192),
            (1, 4096, 4096),
            (8, 4096, 4096),
            (16, 4096, 4096),
            (32, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            (256, 4096, 4096),
            (512, 4096, 4096),
        ]
        
        out_block_size = 256 # Tuned for 256x256 MXU
        results = []
        
        bench_dtypes = [
            # ("int8", jnp.int8),
            ("fp8", jnp.float8_e4m3fn)
        ]

        for bs, n_in, n_out in shapes:
            print(f"\nRunning shape B={bs}, In={n_in}, Out={n_out}...")
            
            key = random.PRNGKey(0)
            x = random.uniform(key, (bs, n_in), dtype=jnp.bfloat16)
            w = random.uniform(key, (n_out, n_in), dtype=jnp.bfloat16)
            
            # 1. Baseline BF16 (Matmul only)
            t_bf16 = self.measure_ms(jax.lax.dot_general, (x, w, (((1,), (1,)), ((), ()))))

            for d_name, d_type in bench_dtypes:
                
                # Quantize weights (Setup cost, not measured)
                w_q_std, w_s_std = quantize_2d_blocked(w, QUANT_GROUP_SIZE, d_type)
                
                # 2. Optimized Kernel
                t_opt = self.measure_ms(quantized_matmul_2d, 
                                        (x, w_q_std, w_s_std, out_block_size, d_type))

                speedup_vs_bf16 = t_bf16 / t_opt if t_opt > 0 else 0.0

                results.append({
                    "Batch": bs,
                    "In": n_in,
                    "Out": n_out,
                    "Dtype": d_name,
                    "BF16 (ms)": f"{t_bf16:.3f}",
                    "Kernel (ms)": f"{t_opt:.3f}",
                    "Speedup": f"{speedup_vs_bf16:.2f}x",
                })

        df = pd.DataFrame(results)
        print("\nBenchmark Results:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_benchmark()