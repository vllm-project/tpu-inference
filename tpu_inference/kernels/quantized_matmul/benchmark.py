import functools
import time
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random

from tpu_inference.kernels.quantized_matmul.kernel_2d import quantized_matmul_2d, quantize_weights_2d
from tpu_inference.kernels.quantized_matmul.util import next_multiple

QUANT_GROUP_SIZE = 128 # Based on common model quantized block sizes
OUTPUT_LOAD_SIZE = 512 # Based on 256x256 MXU

@functools.partial(jax.jit, static_argnames=('dtype',))
def fp8_matmul(activations, weights, dtype=jnp.float8_e4m3fn):
    a_q = activations.astype(dtype)
    w_q = weights.astype(dtype)
    
    return jax.lax.dot_general(
        a_q, w_q,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32
    )

@functools.partial(jax.jit, static_argnames=('quant_group_size', 'quant_dtype'))
def quantized_matmul_reference(
    activations: jax.Array, 
    weights_quantized: jax.Array, 
    weight_scales: jax.Array, 
    quant_group_size: int = 128,
    quant_dtype: jnp.dtype = jnp.int8
):
    B, K = activations.shape
    N = weights_quantized.shape[0]
    n_groups = weight_scales.shape[1]
    
    K_padded = n_groups * quant_group_size
    if K_padded > K:
        activations = jnp.pad(activations, ((0, 0), (0, K_padded - K)))
        weights_quantized = jnp.pad(weights_quantized, ((0, 0), (0, K_padded - K)))

    x_reshaped = activations.reshape(B, n_groups, quant_group_size)
    w_reshaped = weights_quantized.reshape(N, n_groups, quant_group_size)

    x_abs = jnp.abs(x_reshaped)
    x_max = jnp.max(x_abs, axis=-1, keepdims=True)
    x_max = jnp.maximum(x_max, 1e-6)
    
    if quant_dtype == jnp.int8:
        quant_max = 127.0
        x_quant = jnp.clip(jnp.floor(x_reshaped * (quant_max / x_max) + 0.5), -128, 127).astype(jnp.int8)
        accum_dtype = jnp.int32
    else:
        quant_max = 448.0
        x_quant = jnp.clip(x_reshaped * (quant_max / x_max), -448, 448).astype(jnp.float8_e4m3fn)
        accum_dtype = jnp.float32

    dot_out = jax.lax.dot_general(
        x_quant,
        w_reshaped,
        dimension_numbers=(((2,), (2,)), ((1,), (1,))),
        preferred_element_type=accum_dtype
    )

    x_scale_vals = (x_max / quant_max).transpose(1, 0, 2)
    w_scale_vals = weight_scales.T[:, None, :]
    
    total_scale = x_scale_vals * w_scale_vals
    
    final_out = jnp.sum(dot_out.astype(jnp.float32) * total_scale, axis=0)
    return final_out.astype(activations.dtype)


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
    
    def verify_correctness(self):
        print("Running Correctness Check...")
        
        # Test Case Configuration
        B, K, N = 8, 4096, 4096
        dtype = jnp.float8_e4m3fn
        
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)
        
        activations = random.uniform(k1, (B, K), dtype=jnp.bfloat16)
        weights = random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.1 

        weights_q, scales = quantize_weights_2d(weights, QUANT_GROUP_SIZE, dtype)
        
        out_ref = quantized_matmul_reference(
            activations, weights_q, scales, QUANT_GROUP_SIZE, dtype
        )
        jax.block_until_ready(out_ref)
        
        out_kernel = quantized_matmul_2d(
            activations, weights_q, scales, OUTPUT_LOAD_SIZE, dtype
        )
        jax.block_until_ready(out_kernel)
        
        # Compare results
        diff = jnp.abs(out_ref - out_kernel)
        max_diff = jnp.max(diff)
        mean_diff = jnp.mean(diff)
        relative_error = max_diff / (jnp.max(jnp.abs(out_ref)) + 1e-6)

        print(f"Max Diff: {max_diff:.4}")
        print(f"Mean Diff: {mean_diff:.4}")
        print(f"Max Relative Error: {relative_error:.4}")

        if relative_error < 0.02: # < 2% error
            print("CORRECTNESS CHECK PASSED")
        else:
            print("CORRECTNESS CHECK FAILED")

    def run_benchmark(self):
        print(f"Config: QuantGroupSize={QUANT_GROUP_SIZE}, OutputLoadSize={OUTPUT_LOAD_SIZE}")

        self.verify_correctness()
        
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
            
            # FFN Expansion
            (1, 16384, 53248),
            (8, 16384, 53248),
            (16, 16384, 53248),
            (32, 16384, 53248),
            (64, 16384, 53248),
            (128, 16384, 53248),
            (256, 16384, 53248),
            
            # FFN Contraction
            (1, 53248, 16384),
            (8, 53248, 16384),
            (16, 53248, 16384),
            (32, 53248, 16384),
            (64, 53248, 16384),
            (128, 53248, 16384),
            (256, 53248, 16384),
            
            # GQA KV Projections
            (1, 16384, 1024),
            (8, 16384, 1024),
            (16, 16384, 1024),
            (32, 16384, 1024),
            (64, 16384, 1024),
            (128, 16384, 1024),
            (256, 16384, 1024),

            # ---------------------------------------------------------
            # Llama 3 70B
            # Hidden: 8,192 | FFN: 28,672
            # ---------------------------------------------------------

            # Square
            (1, 8192, 8192),
            (8, 8192, 8192),
            (16, 8192, 8192),
            (32, 8192, 8192),
            (64, 8192, 8192),
            (128, 8192, 8192),
            (256, 8192, 8192),

            # FFN Expansion
            (1, 8192, 28672),
            (8, 8192, 28672),
            (16, 8192, 28672),
            (32, 8192, 28672),
            (64, 8192, 28672),
            (128, 8192, 28672),
            (256, 8192, 28672),

            # FFN Contraction
            (1, 28672, 8192),
            (8, 28672, 8192),
            (16, 28672, 8192),
            (32, 28672, 8192),
            (64, 28672, 8192),
            (128, 28672, 8192),
            (256, 28672, 8192),

            # ---------------------------------------------------------
            # Llama 3 8B
            # Hidden: 4,096 | FFN: 14,336
            # ---------------------------------------------------------

            # Square
            (1, 4096, 4096),
            (8, 4096, 4096),
            (16, 4096, 4096),
            (32, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            (256, 4096, 4096),

            # FFN Expansion
            (1, 4096, 14336), 
            (8, 4096, 14336), 
            (16, 4096, 14336), 
            (32, 4096, 14336), 
            (64, 4096, 14336), 
            (128, 4096, 14336), 
            (256, 4096, 14336), 

            # FFN Contraction
            (1, 14336, 4096),
            (8, 14336, 4096),
            (16, 14336, 4096),
            (32, 14336, 4096),
            (64, 14336, 4096),
            (128, 14336, 4096),
            (256, 14336, 4096),

            # Small square
            (1, 2880, 2880),
            (8, 2880, 2880),
            (16, 2880, 2880),
            (32, 2880, 2880),
            (64, 2880, 2880),
            (128, 2880, 2880),
            (256, 2880, 2880),
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
            
            t_bf16 = self.measure_ms(jax.lax.dot_general, (activations, weights, (((1,), (1,)), ((), ()))))

            for dt_name, dt_type in bench_dtypes:
                weights_q, scales = quantize_weights_2d(weights, QUANT_GROUP_SIZE, dt_type)

                t_fp8 = self.measure_ms(fp8_matmul, (activations, weights, dt_type))
                
                t_ref = self.measure_ms(
                    quantized_matmul_reference, 
                    (activations, weights_q, scales, QUANT_GROUP_SIZE, dt_type)
                )

                t_w8a8 = self.measure_ms(
                    quantized_matmul_2d, 
                    (activations, weights_q, scales, OUTPUT_LOAD_SIZE, dt_type)
                )
                
                results.append({
                    "Batch": batch_size, "In": n_in, "Out": n_out, "Dtype": dt_name,
                    "BF16 (ms)": f"{t_bf16:.3f}",
                    "FP8 (ms)": f"{t_fp8:.3f}",
                    "Ref (ms)": f"{t_ref:.3f}",
                    "W8A8 (ms)": f"{t_w8a8:.3f}",
                    "Spdup vs Ref": f"{t_ref/t_w8a8:.2f}x" if t_w8a8 > 0 else "0.00x",
                    "Spdup vs BF16": f"{t_bf16/t_w8a8:.2f}x" if t_w8a8 > 0 else "0.00x",
                    "Spdup vs FP8": f"{t_fp8/t_w8a8:.2f}x" if t_w8a8 > 0 else "0.00x",
                })

        df = pd.DataFrame(results)
        print("\nBenchmark Results:")
        print(df.to_string(index=False))
        df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    suite = BenchmarkSuite()
    suite.run_benchmark()