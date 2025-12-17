import functools
import time
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random

from tpu_inference.kernels.quantized_matmul.kernel_2d import quantized_matmul_2d, quantize_weights_2d, quantized_matmul_w8a16_2d

QUANT_GROUP_SIZE = 128 # Based on common model quantized block sizes
OUTPUT_LOAD_SIZE = 512 # Based on 256x256 MXU


class BenchmarkSuite:
    def measure_ms(self, func, args, n_iter=1000, n_warmup=100):
        try:
            out = func(*args)
            jax.block_until_ready(out)
            for _ in range(n_warmup):
                out = func(*args)
            jax.block_until_ready(out)
        except Exception as e:
            print(f"Error during execution: {e}")
            return 0.0
            
        start = time.time()
        for _ in range(n_iter): 
            out = func(*args)
        jax.block_until_ready(out) 
        end = time.time()
        
        return ((end - start) / n_iter) * 1000

    def run_benchmark(self):
        print(f"Config: QuantGroupSize={QUANT_GROUP_SIZE}, OutputLoadSize={OUTPUT_LOAD_SIZE}")
        
        # Format: (Batch, In_Features, Out_Features)
        shapes = [
            # ---------------------------------------------------------
            # Llama 3.1 405B (Flagship)
            # Hidden: 16,384 | FFN: 53,248 | KV-Proj: 1,024
            # ---------------------------------------------------------

            # Attention/Output (Square)
            (1, 16384, 16384),
            (8, 16384, 16384),
            (16, 16384, 16384),
            (32, 16384, 16384),
            (64, 16384, 16384),
            (128, 16384, 16384), 
            (256, 16384, 16384), 
            
            # # FFN Expansion
            # (1, 16384, 53248),
            # (8, 16384, 53248),
            # (16, 16384, 53248),
            # (32, 16384, 53248),
            # (64, 16384, 53248),
            # (128, 16384, 53248),
            # (256, 16384, 53248),
            
            # # FFN Contraction
            # (1, 53248, 16384),
            # (8, 53248, 16384),
            # (16, 53248, 16384),
            # (32, 53248, 16384),
            # (64, 53248, 16384),
            # (128, 53248, 16384),
            # (256, 53248, 16384),
            
            # # GQA KV Projections
            # (1, 16384, 1024),
            # (8, 16384, 1024),
            # (16, 16384, 1024),
            # (32, 16384, 1024),
            # (64, 16384, 1024),
            # (128, 16384, 1024),
            # (256, 16384, 1024),

            # # ---------------------------------------------------------
            # # Llama 3 70B
            # # Hidden: 8,192 | FFN: 28,672
            # # ---------------------------------------------------------

            # # Square
            # (1, 8192, 8192),
            # (8, 8192, 8192),
            # (16, 8192, 8192),
            # (32, 8192, 8192),
            # (64, 8192, 8192),
            # (128, 8192, 8192),
            # (256, 8192, 8192),

            # # FFN Expansion
            # (1, 8192, 28672),
            # (8, 8192, 28672),
            # (16, 8192, 28672),
            # (32, 8192, 28672),
            # (64, 8192, 28672),
            # (128, 8192, 28672),
            # (256, 8192, 28672),

            # # FFN Contraction
            # (1, 28672, 8192),
            # (8, 28672, 8192),
            # (16, 28672, 8192),
            # (32, 28672, 8192),
            # (64, 28672, 8192),
            # (128, 28672, 8192),
            # (256, 28672, 8192),

            # # ---------------------------------------------------------
            # # Llama 3 8B
            # # Hidden: 4,096 | FFN: 14,336
            # # ---------------------------------------------------------

            # # Square
            # (1, 4096, 4096),
            # (8, 4096, 4096),
            # (16, 4096, 4096),
            # (32, 4096, 4096),
            # (64, 4096, 4096),
            # (128, 4096, 4096),
            # (256, 4096, 4096),

            # # FFN Expansion
            # (1, 4096, 14336), 
            # (8, 4096, 14336), 
            # (16, 4096, 14336), 
            # (32, 4096, 14336), 
            # (64, 4096, 14336), 
            # (128, 4096, 14336), 
            # (256, 4096, 14336), 

            # # FFN Contraction
            # (1, 14336, 4096),
            # (8, 14336, 4096),
            # (16, 14336, 4096),
            # (32, 14336, 4096),
            # (64, 14336, 4096),
            # (128, 14336, 4096),
            # (256, 14336, 4096),

            # # Small square
            # (1, 2880, 2880),
            # (8, 2880, 2880),
            # (16, 2880, 2880),
            # (32, 2880, 2880),
            # (64, 2880, 2880),
            # (128, 2880, 2880),
            # (256, 2880, 2880),
        ]
        
        results = []
        
        bench_dtypes = [
            # ("int8", jnp.int8),
            ("fp8", jnp.float8_e4m3fn)
        ]

        for batch_size, n_in, n_out in shapes:
            print(f"\nShape B={batch_size}, In={n_in}, Out={n_out}")
            
            key = random.PRNGKey(0)
            activations = random.uniform(key, (batch_size, n_in), dtype=jnp.bfloat16)
            weights = random.uniform(key, (n_out, n_in), dtype=jnp.bfloat16)
            
            # Baseline
            t_bf16 = self.measure_ms(jax.lax.dot_general, (activations, weights, (((1,), (1,)), ((), ()))))
            t_native_fp8 = self.measure_ms(
                lambda a, w: jax.lax.dot_general(
                    a.astype(jnp.float8_e4m3fn), 
                    w.astype(jnp.float8_e4m3fn), 
                    (((1,), (1,)), ((), ())), 
                    preferred_element_type=jnp.float32
                ),
                (activations, weights)
            )

            for dt_name, dt_type in bench_dtypes:
                weights_q, scales = quantize_weights_2d(weights, QUANT_GROUP_SIZE, dt_type)
                
                # 1. Measure W8A8
                t_w8a8 = self.measure_ms(quantized_matmul_2d, 
                                         (activations, weights_q, scales, OUTPUT_LOAD_SIZE, dt_type))
                
                # 2. Measure W8A16
                t_w8a16 = self.measure_ms(quantized_matmul_w8a16_2d, 
                                          (activations, weights_q, scales, OUTPUT_LOAD_SIZE, dt_type))

                results.append({
                    "Batch": batch_size, "In": n_in, "Out": n_out, "Dtype": dt_name,
                    "BF16 (ms)": f"{t_bf16:.3f}",
                    "FP8 (ms)": f"{t_native_fp8:.3f}",
                    "W8A8 (ms)": f"{t_w8a8:.3f}",
                    "W8A16 (ms)": f"{t_w8a16:.3f}",
                    "Spdup A8": f"{t_bf16/t_w8a8:.2f}x" if t_w8a8 > 0 else "0.00x",
                    "Spdup A16": f"{t_bf16/t_w8a16:.2f}x" if t_w8a16 > 0 else "0.00x",
                    "Spdup A8-native_fp8": f"{t_native_fp8/t_w8a8:.2f}x" if t_w8a16 > 0 else "0.00x"
                })

        df = pd.DataFrame(results)
        print("\nBenchmark Results:")
        print(df.to_string(index=False))
if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_benchmark()