# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.quantized_matmul import (blockwise_kernel, kernel,
                                                    tuned_block_sizes, util)

xla_quantized_matmul = util.xla_quantized_matmul
per_channel_kernel = kernel.quantized_matmul_kernel
blockwise_kernel = blockwise_kernel.quantized_matmul_kernel
quantize_tensor = util.quantize_tensor
get_tuned_block_sizes = tuned_block_sizes.get_tuned_block_sizes
TunedValue = tuned_block_sizes.TunedValue

jax.config.parse_flags_with_absl()


def reference_block_quantized_matmul(x, w_q, w_scale, block_size, x_q_dtype):
    """Pure JAX reference for Block-wise Quantized Matmul.
    
    Mimics the logic of the Pallas kernel using the SAME weights/scales:
    1. Reshape inputs to blocks.
    2. Quantize X per-block (using kernel's utility).
    3. Use provided W_Q and W_Scale (static).
    4. Dot product and scale accumulation.
    """
    n_batch, n_in = x.shape
    n_out, _ = w_q.shape

    if n_in % block_size != 0:
        raise ValueError(
            f"Input dimension {n_in} not divisible by block_size {block_size}")

    x_reshaped = x.reshape(n_batch, -1, block_size)

    x_q, x_s = util.quantize_block(x_reshaped, axis=2, target_dtype=x_q_dtype)

    w_q_reshaped = w_q.reshape(n_out, -1, block_size)

    w_s_aligned = w_scale.transpose(1, 0, 2)
    x_q_f = x_q.astype(jnp.float32)
    w_q_f = w_q_reshaped.astype(jnp.float32)

    # Einsum: Batch(b), N_Blocks(n), Out(o), Block_Size(k)
    # x(bnk) @ w(onk).T -> dot(bno)
    dot_blocks = jnp.einsum('bnk, onk -> bno', x_q_f, w_q_f)
    scaled_blocks = dot_blocks * x_s * w_s_aligned

    out = jnp.sum(scaled_blocks, axis=1)

    return out.astype(x.dtype)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class QuantizedMatmulKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(6):
            self.skipTest("Expect TPUv6+")

    def _test_quantized_matmul(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
        tuned_value=None,
        atol=0.5,
        rtol=0.5,
        block_size: int | None = None,
        x_q_dtype: int | None = None,
    ):

        prng_key = jax.random.key(1234)
        k0, k1 = jax.random.split(prng_key, 2)
        x = jax.random.uniform(k0, (bs, n_input_features),
                               dtype=dtype,
                               minval=0,
                               maxval=1)
        w = jax.random.uniform(
            k1,
            (n_output_features, n_input_features),
            dtype=dtype,
            minval=-1,
            maxval=1,
        )

        w_q, w_scale = quantize_tensor(w, q_dtype, block_size=block_size)
        if block_size is None:
            w_scale = jnp.squeeze(w_scale)
            assert w_scale.shape == (n_output_features, )
        else:
            assert w_scale.shape == (n_input_features // block_size, 1,
                                     n_output_features)

        if x_q_dtype is None:
            x_q_dtype = w_q.dtype if quantize_activation else dtype

        kernel_fn = per_channel_kernel if block_size is None else blockwise_kernel

        output = kernel_fn(
            x,
            w_q,
            w_scale,
            block_size=block_size,
            x_q_dtype=x_q_dtype,
            tuned_value=tuned_value,
        )

        if block_size is None:
            w_scale_xla = jnp.squeeze(w_scale)
            expected = xla_quantized_matmul(
                x, w_q, w_scale_xla, quantize_activation=quantize_activation)
        else:
            expected = reference_block_quantized_matmul(
                x, w_q, w_scale, block_size, q_dtype)

        self.assertAllClose(output,
                            expected,
                            rtol=rtol,
                            atol=atol,
                            check_dtypes=True)

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        q_dtype=[jnp.int8, jnp.float8_e4m3fn],
        bs=[128, 256, 512],
        n_input_features=[128, 256, 512],
        n_output_features=[128, 256, 512],
        quantize_activation=[True],
    )
    def test_quantized_matmul_various_input_shapes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
    ):
        self._test_quantized_matmul(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        q_dtype=[jnp.int8, jnp.float8_e4m3fn],
        bs=[64, 192],
        n_input_features=[64, 192],
        n_output_features=[64, 192],
        quantize_activation=[True],
    )
    def test_quantized_matmul_unaligned_input_shapes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
    ):
        self._test_quantized_matmul(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )

    @parameterized.parameters(
        (jnp.bfloat16, jnp.int8, 128, 1280, 8192, True),
        (jnp.bfloat16, jnp.int8, 128, 28672, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 4096, 14336, True),
        (jnp.bfloat16, jnp.int8, 128, 4096, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 6144, 4096, True),
        (jnp.bfloat16, jnp.int8, 128, 7168, 8192, True),
        (jnp.bfloat16, jnp.int8, 128, 8192, 1024, True),
        (jnp.bfloat16, jnp.int8, 128, 8192, 3584, True),
    )
    def test_quantized_matmul_use_tuned_block_sizes(
        self,
        dtype: jnp.dtype,
        q_dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
        quantize_activation: bool,
    ):
        self._test_quantized_matmul(
            dtype,
            q_dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            tuned_value=None,
        )

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        bs=[512, 1024],
        n_input_features=[512, 1024],
        n_output_features=[512, 1024, 2048],
    )
    def test_quantized_matmul_blockwise_w8a8(
        self,
        dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
    ):
        self._test_quantized_matmul(
            dtype,
            jnp.float8_e4m3fn,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=True,
            tuned_value=TunedValue(512, 512, 512, 2),
            block_size=512,
            x_q_dtype=jnp.float8_e4m3fn,
        )

    @parameterized.parameters(
        (jnp.bfloat16, 512, 512, 1024),
        (jnp.bfloat16, 512, 512, 512),
        (jnp.bfloat16, 512, 1024, 512),
        (jnp.bfloat16, 512, 1024, 1024),
        (jnp.bfloat16, 512, 1024, 2048),
        (jnp.float32, 512, 512, 1024),
        (jnp.float32, 512, 512, 512),
        (jnp.float32, 512, 1024, 512),
        (jnp.float32, 512, 1024, 1024),
        (jnp.float32, 512, 1024, 2048),
    )
    def test_quantized_matmul_blockwise_w4a8(
        self,
        dtype: jnp.dtype,
        bs: int,
        n_input_features: int,
        n_output_features: int,
    ):
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        self._test_quantized_matmul(
            dtype,
            jnp.float4_e2m1fn,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=True,
            tuned_value=TunedValue(512, 512, 512, 2),
            block_size=512,
            atol=6,
            rtol=0.5,
            x_q_dtype=jnp.float8_e4m3fn,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
