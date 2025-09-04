# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_commons.kernels.quantized_matmul import (kernel, tuned_block_sizes,
                                                  util)

quantized_matmul_kernel = kernel.quantized_matmul_kernel
quantize_tensor = util.quantize_tensor
get_tuned_block_sizes = tuned_block_sizes.get_tuned_block_sizes

jax.config.parse_flags_with_absl()


@functools.partial(jax.jit, static_argnames=["quantize_activation"])
def reference_quantized_matmul(x, w_q, w_scale, quantize_activation=True):
    if quantize_activation:
        x_q, x_scale = quantize_tensor(x)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.int32,
        ).astype(jnp.float32)
        out *= x_scale
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1, ), (1, )), ((), ())),
            preferred_element_type=jnp.float32,
        )
    out *= jnp.expand_dims(w_scale, 0)
    return out.astype(x.dtype)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class QuantizedMatmulKernelTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        if not jtu.is_device_tpu_at_least(6):
            self.skipTest("Expect TPUv6+")

    def _test_quantized_matmul(
        self,
        dtype,
        bs,
        n_input_features,
        n_output_features,
        quantize_activation,
        batch_block_size=None,
        out_block_size=None,
        in_block_size=None,
        atol=2e-1,
        rtol=1e-1,
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
        w_q, w_scale = quantize_tensor(w)
        w_scale = jnp.squeeze(w_scale)
        assert w_scale.shape == (n_output_features, )

        output = quantized_matmul_kernel(
            x,
            w_q,
            w_scale,
            quantize_activation=quantize_activation,
            batch_block_size=batch_block_size,
            out_block_size=out_block_size,
            in_block_size=in_block_size,
        )
        expected = reference_quantized_matmul(
            x, w_q, w_scale, quantize_activation=quantize_activation)

        self.assertAllClose(output,
                            expected,
                            rtol=rtol,
                            atol=atol,
                            check_dtypes=True)

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        bs=[128, 256, 512],
        n_input_features=[128, 256, 512],
        n_output_features=[128, 256, 512],
        quantize_activation=[True],
    )
    def test_quantized_matmul_various_input_shapes(self, dtype, bs,
                                                   n_input_features,
                                                   n_output_features,
                                                   quantize_activation):
        self._test_quantized_matmul(
            dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            batch_block_size=128,
            out_block_size=128,
            in_block_size=128,
        )

    @parameterized.product(
        dtype=[jnp.bfloat16, jnp.float32],
        bs=[64, 192],
        n_input_features=[64, 192],
        n_output_features=[64, 192],
        quantize_activation=[True],
    )
    def test_quantized_matmul_unaligned_input_shapes(self, dtype, bs,
                                                     n_input_features,
                                                     n_output_features,
                                                     quantize_activation):
        self._test_quantized_matmul(
            dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            batch_block_size=128,
            out_block_size=128,
            in_block_size=128,
        )

    @parameterized.parameters(
        (jnp.bfloat16, 128, 1280, 8192, True),
        (jnp.bfloat16, 128, 28672, 4096, True),
        (jnp.bfloat16, 128, 4096, 14336, True),
        (jnp.bfloat16, 128, 4096, 4096, True),
        (jnp.bfloat16, 128, 6144, 4096, True),
        (jnp.bfloat16, 128, 7168, 8192, True),
        (jnp.bfloat16, 128, 8192, 1024, True),
        (jnp.bfloat16, 128, 8192, 3584, True),
    )
    def test_quantized_matmul_use_tuned_block_sizes(self, dtype, bs,
                                                    n_input_features,
                                                    n_output_features,
                                                    quantize_activation):
        self._test_quantized_matmul(
            dtype,
            bs,
            n_input_features,
            n_output_features,
            quantize_activation=quantize_activation,
            batch_block_size=None,
            out_block_size=None,
            in_block_size=None,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
