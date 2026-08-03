# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for the dense-path expert dequantization helper.

`dequantize_unfused_moe_weights` is what lets DENSE_MAT (the reference
backend) consume block-quantized expert kernels: it expands each (value,
scale) pair back to `cast_dtype` before the plain einsums run.
"""

import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common.process_weights.moe_weights import \
    UnfusedMoEWeights
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.jax.moe.dense_moe import \
    dequantize_unfused_moe_weights

E, K, N = 2, 8, 4
BLOCK = 4  # contracting-axis block size -> scale shape (E, K // BLOCK, N)


def _synthetic_weights(seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(-1.0, 1.0, size=(E, K, N)),
                       dtype=jnp.float32)


def _quantized_moe_weights():
    """Quantize three synthetic (E, K, N) kernels along the contracting axis."""
    originals = [_synthetic_weights(seed) for seed in (0, 1, 2)]
    pairs = [
        quantize_tensor(jnp.int8, w, axis=1, block_size=BLOCK)
        for w in originals
    ]
    weights = UnfusedMoEWeights(
        w1_weight=pairs[0][0],
        w1_weight_scale=pairs[0][1],
        w1_bias=None,
        w2_weight=pairs[1][0],
        w2_weight_scale=pairs[1][1],
        w2_bias=None,
        w3_weight=pairs[2][0],
        w3_weight_scale=pairs[2][1],
        w3_bias=None,
    )
    return originals, weights


def test_dequantize_round_trips_block_quantized_kernels():
    originals, weights = _quantized_moe_weights()
    assert weights.w1_weight_scale.shape == (E, K // BLOCK, N)

    result = dequantize_unfused_moe_weights(weights, jnp.float32)

    for original, dequantized in zip(
            originals, (result.w1_weight, result.w2_weight, result.w3_weight)):
        assert dequantized.dtype == jnp.float32
        assert dequantized.shape == (E, K, N)
        # int8 absmax quantization: error bounded by half a step per block.
        atol = float(jnp.max(jnp.abs(original))) / 127.0
        np.testing.assert_allclose(np.asarray(dequantized),
                                   np.asarray(original),
                                   atol=atol)
    # Scales are consumed: the dense einsums must see plain arrays.
    assert result.w1_weight_scale is None
    assert result.w2_weight_scale is None
    assert result.w3_weight_scale is None


def test_dequantize_casts_to_requested_dtype():
    _, weights = _quantized_moe_weights()
    result = dequantize_unfused_moe_weights(weights, jnp.bfloat16)
    assert result.w1_weight.dtype == jnp.bfloat16
    assert result.w2_weight.dtype == jnp.bfloat16
    assert result.w3_weight.dtype == jnp.bfloat16


def test_dequantize_passes_through_unquantized_weights():
    originals = [_synthetic_weights(seed) for seed in (3, 4, 5)]
    weights = UnfusedMoEWeights(
        w1_weight=originals[0],
        w1_weight_scale=None,
        w1_bias=None,
        w2_weight=originals[1],
        w2_weight_scale=None,
        w2_bias=None,
        w3_weight=originals[2],
        w3_weight_scale=None,
        w3_bias=None,
    )
    result = dequantize_unfused_moe_weights(weights, jnp.bfloat16)
    # All-None scales: returned unchanged, no cast, same object.
    assert result is weights
    assert result.w1_weight.dtype == jnp.float32


def test_dequantize_preserves_biases():
    _, weights = _quantized_moe_weights()
    bias = jnp.ones((E, N), dtype=jnp.float32)
    weights = UnfusedMoEWeights(
        w1_weight=weights.w1_weight,
        w1_weight_scale=weights.w1_weight_scale,
        w1_bias=bias,
        w2_weight=weights.w2_weight,
        w2_weight_scale=weights.w2_weight_scale,
        w2_bias=None,
        w3_weight=weights.w3_weight,
        w3_weight_scale=weights.w3_weight_scale,
        w3_bias=bias,
    )
    result = dequantize_unfused_moe_weights(weights, jnp.float32)
    assert result.w1_bias is bias
    assert result.w2_bias is None
    assert result.w3_bias is bias
