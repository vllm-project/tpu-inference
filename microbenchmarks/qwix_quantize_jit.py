import functools
import time
from itertools import product

import jax
import jax.numpy as jnp
import qwix.pallas as qpl

NUM_ITERS = 50
NUM_WARMUP = 5

# Input Sizes: M: (1, 64K), N: (128, 64K)
INPUT_SIZES = [2**x for x in range(17)]
OUTPUT_SIZES = [2**x for x in range(7, 17)]
UNQUANT_TO_QUANT_DTYPES = [(jnp.bfloat16, jnp.float8_e4m3fn),
                           (jnp.bfloat16, jnp.float4_e2m1fn),
                           (jnp.float8_e4m3fn, jnp.float4_e2m1fn)]

INPUT_OUTPUT_SIZES_LIST = list(product(INPUT_SIZES, OUTPUT_SIZES))


@functools.partial(jax.jit, static_argnames=("quantized_dtype"))
def quantize_fn(x, quantized_dtype):
    # NOTE: we are hardocidng channelwise_axes right now, but can make it dynamic
    return qpl.quantize(x, quantized_dtype, channelwise_axes=[0])


@jax.jit
def dequantize_fn(quantized_x):
    return qpl.dequantize(quantized_x)


for tensor_shape in INPUT_OUTPUT_SIZES_LIST:
    for unquant_dtype, quant_dtype in UNQUANT_TO_QUANT_DTYPES:
        print(
            f"--- Benchmarking shape: {tensor_shape}, unquant dtype: {unquant_dtype}, quant dtype: {quant_dtype} ---"
        )
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, tensor_shape, dtype=unquant_dtype)

        # Warmup phase
        quantized_x_for_dequant_bench = None
        for _ in range(NUM_WARMUP):
            quantized_x_for_dequant_bench = quantize_fn(x, quant_dtype)

        for _ in range(NUM_WARMUP):
            dequantize_fn(quantized_x_for_dequant_bench)

        # Real quantization benchmark phase
        quantize_times = []
        for _ in range(NUM_ITERS):
            start_time = time.perf_counter()
            quantized_x = quantize_fn(x, quant_dtype)
            end_time = time.perf_counter()
            quantize_times.append(end_time - start_time)

        # Convert list to numpy array for easier stats, and get the average
        quantize_avg_time = jnp.mean(jnp.array(quantize_times))

        # Benchmark Dequantization
        dequantize_times = []
        for _ in range(NUM_ITERS):
            start_time = time.perf_counter()
            dequantize_fn(quantized_x).block_until_ready()
            end_time = time.perf_counter()
            dequantize_times.append(end_time - start_time)

        dequantize_avg_time = jnp.mean(jnp.array(dequantize_times))

        print(
            f"RESULTS: quantize time: {quantize_avg_time:.6f} s, dequantize time: {dequantize_avg_time:.6f} s\n"
        )
