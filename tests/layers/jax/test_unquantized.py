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

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from vllm.config import ModelConfig, VllmConfig

from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedConfig, UnquantizedLinearMethod)


@pytest.fixture
def rng():
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestUnquantizedLinear:

    def test_unquantized_linear_shape(self):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization=None))

        quant_config = get_tpu_quantization_config(vllm_config)
        assert isinstance(quant_config, UnquantizedConfig)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_dim,
                          output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        assert isinstance(layer.quant_method, UnquantizedLinearMethod)

        hidden_states = jnp.ones((batch_size, input_dim))
        out = layer(hidden_states)
        assert out.shape == (batch_size, output_dim)

    def test_unquantized_linear_correctness(self, rng):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization=None))
        quant_config = get_tpu_quantization_config(vllm_config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        k1, k2 = jax.random.split(rng, 2)
        w_val = jax.random.uniform(k1, (input_dim, output_dim),
                                   dtype=jnp.float32)
        layer.weight.value = w_val

        hidden_states = jax.random.uniform(k2, (batch_size, input_dim),
                                           dtype=jnp.float32)

        expected = jnp.dot(hidden_states, w_val)

        out = layer(hidden_states)
        assert jnp.allclose(out, expected, rtol=1e-5)
