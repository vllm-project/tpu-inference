import time
from itertools import product

import jax
import jax.numpy as jnp
import qwix.pallas as qpl

NUM_ITERS = 50

# Input Sizes: M: (1, 64K), N: (128, 64K)
INPUT_SIZES = [2**x for x in range(17)]
OUTPUT_SIZES = [2**x for x in range(7, 17)]
# TODO: (jnp.float8_e4m3fn, jnp.float4_e2m1fn)]
UNQUANT_TO_QUANT_DTYPES = [(jnp.bfloat16, jnp.float8_e4m3fn),
                           (jnp.bfloat16, jnp.float4_e2m1fn)]

INPUT_OUTPUT_SIZES_LIST = list(product(INPUT_SIZES, OUTPUT_SIZES))


def quantize_fn(x, quantized_dtype):
    # NOTE: we are hardocidng channelwise_axes right now, but can make it dynamic
    return qpl.quantize(x, quantized_dtype, channelwise_axes=[0])


def dequantize_fn(quantized_x):
    return qpl.dequantize(quantized_x)


for tensor_shape in INPUT_OUTPUT_SIZES_LIST:
    for unquant_dtype, quant_dtype in UNQUANT_TO_QUANT_DTYPES:
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, tensor_shape, dtype=unquant_dtype)

        # Real quantization benchmark phase
        quantize_times = []
        for _ in range(NUM_ITERS):
            start_time = time.perf_counter_ns()
            quantized_x = quantize_fn(x, quant_dtype)
            end_time = time.perf_counter_ns()
            quantize_times.append(end_time - start_time)

        # Convert list to numpy array for easier stats, and get the average
        quantize_avg_time = jnp.mean(jnp.array(quantize_times))

        # Benchmark Dequantization
        dequantize_times = []
        for _ in range(NUM_ITERS):
            start_time = time.perf_counter_ns()
            dequantize_fn(quantized_x).block_until_ready()
            end_time = time.perf_counter_ns()
            dequantize_times.append(end_time - start_time)

        dequantize_avg_time = jnp.mean(jnp.array(dequantize_times))

        print(
            f"Results for {tensor_shape}, unquant dtype: {unquant_dtype.__name__}, quant dtype: {quant_dtype.__name__} quantize time: {quantize_avg_time/1000:.6f} us, dequantize time: {dequantize_avg_time/1000:.6f} us"
        )
