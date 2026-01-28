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
import pytest
from flax import nnx
from vllm.config import ModelConfig, VllmConfig

from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config


class TestJaxRmsNorm:
    """Test for JaxRmsNorm layer."""

    @pytest.mark.parametrize(
        "rng_key",
        [jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    @pytest.mark.parametrize("num_features", [16, 32, 64])
    @pytest.mark.parametrize(
        "dtype", [jax.numpy.float32, jax.numpy.float16, jax.numpy.bfloat16])
    def test_numerical_correctness_against_flax_unquantized(
            self, rng_key, num_features, dtype):
        """Run the same input through JaxRmsNorm vs. flax RMSNorm and compare outputs.
        """
        mesh = jax.make_mesh((4, ), ('model', ))
        unquantize_config = get_tpu_quantization_config(
            VllmConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B")),
            mesh)

        kwargs = {
            "num_features": num_features,
            "epsilon": 1e-6,
            "dtype": dtype,
            "param_dtype": dtype,
            "rngs": nnx.Rngs(0)
        }
        layer = JaxRmsNorm(quant_config=unquantize_config, **kwargs)
        layer.weight.value = jax.random.uniform(rng_key, (num_features, ),
                                                dtype=dtype)

        flax_layer = nnx.RMSNorm(**kwargs)
        flax_layer.scale.value = layer.weight.value

        x = jax.random.uniform(rng_key, (2, 8, num_features), dtype=dtype)

        jax_output = layer(x)
        flax_output = flax_layer(x)

        assert jax.numpy.allclose(jax_output,
                                  flax_output,
                                  rtol=1e-5,
                                  atol=1e-5)
