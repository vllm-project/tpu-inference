import time

import jax
import jax.numpy as jnp
import qwix.pallas as qpl

NUM_ITERS = 50

for tensor_shape in [(1024, 1024), (1024, 1024, 1024)]:
    for quantized_dtype in [jnp.float8_e4m3fn]:
        quantize_total_time = 0
        dequantize_total_time = 0
        for _ in range(NUM_ITERS):
            x = jax.random.normal(jax.random.PRNGKey(0),
                                  tensor_shape,
                                  dtype=jnp.bfloat16)
            # quantize time
            quantize_start_time = time.perf_counter()
            quantized_x = qpl.quantize(x,
                                       quantized_dtype,
                                       channelwise_axes=[0])
            quantize_end_time = time.perf_counter()

            # dequantize time
            dequantize_start_time = time.perf_counter()
            dequantized_x = qpl.dequantize(quantized_x)
            dequantize_end_time = time.perf_counter()

            quantize_total_time += quantize_end_time - quantize_start_time
            dequantize_total_time += dequantize_end_time - dequantize_start_time

        print(
            f"{tensor_shape} {quantized_dtype} quantize time: {quantize_total_time / NUM_ITERS} dequantize time: {dequantize_total_time / NUM_ITERS}"
        )
