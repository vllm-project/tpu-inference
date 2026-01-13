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

from typing import Optional

import jax
from flax import nnx
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.vllm.quantization import JaxQuantizeMethodBase


class JaxQuantizedLinearMethod(JaxQuantizeMethodBase):
    """Quantization method for JAX linear layer.
    """

    def create_weights_jax(self, layer: JaxModule) -> None:
        """Create weights for JAX linear layer."""
        raise NotImplementedError()

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        raise NotImplementedError()


class JaxLinear(nnx.Linear, JaxModule):
    """Linear layer for JAX.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        use_bias: If false, skip adding bias.
        param_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
        prefix: Prefix for parameter names.
    """

    def __init__(self,
                 *args,
                 rngs,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        assert quant_config is not None, "Even though not a quantized model, VllmUnquantizedConfig must be provided."
        from tpu_inference.layers.vllm.quantization.common import \
            JaxQuantizationConfig
        assert isinstance(quant_config, JaxQuantizationConfig)

        nnx.Linear.__init__(self, *args, rngs=rngs, **kwargs)
        self.einsum_str = "mn,np->mp"
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.kernel` to
        # `self.weight` such that `named_parameters()` can match the names in HF models. We also
        # apply transpose here to match HF weight layout.
        self.weight = self.kernel
        delattr(self, 'kernel')
        # For compatibility with vLLM. e.g. `JaxCommonLinearConfig` is used in both flax_nnx
        # and vllm implementations, and it's looking for `layer.output_size` attribute.
        self.output_size = self.out_features

        quant_method = quant_config.get_quant_method(self, prefix=prefix)
        assert isinstance(quant_method, JaxQuantizedLinearMethod)
        self.quant_method = quant_method
        self.quant_method.create_weights_jax(self)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.quant_method.apply_jax(self, x)
