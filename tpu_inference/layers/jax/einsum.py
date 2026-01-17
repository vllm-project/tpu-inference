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


class JaxQuantizedEinsumMethod(JaxQuantizeMethodBase):
    """Quantization method for JAX Einsum layer.
    """

    def create_weights_jax(self, layer: JaxModule) -> None:
        """Create weights for JAX Einsum layer."""
        raise NotImplementedError()

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        raise NotImplementedError()


class JaxEinsum(nnx.Einsum, JaxModule):
    """Einsum layer for JAX.

    Args:
        einsum_str: a string to denote the einsum equation.
        kernel_shape: the shape of the kernel.
        bias_shape: the shape of the bias. If this is None, a bias won't be used.
        param_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
    """

    def __init__(self,
                 einsum_str: str,
                 kernel_shape: tuple[int, ...],
                 rngs,
                 bias_shape: Optional[tuple[int, ...]] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        nnx.Einsum.__init__(self,
                            rngs=rngs,
                            einsum_str=einsum_str,
                            kernel_shape=kernel_shape,
                            bias_shape=bias_shape,
                            **kwargs)
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.kernel` to
        # `self.weight` such that `named_parameters()` can match the names in HF models.
        self.weight = self.kernel
        delattr(self, 'kernel')
        # For compatibility with vLLM. e.g. `JaxCommonLinearConfig` is used in both flax_nnx
        # and vllm implementations, and it's looking for `layer.output_size` attribute.
        self.output_size = 0

        if quant_config is None:
            self.quant_method = None
        elif (quant_method := quant_config.get_quant_method(self,
                                                            prefix=prefix)):
            assert isinstance(quant_method, JaxQuantizedEinsumMethod)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self)
        else:
            self.quant_method = None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.quant_method is None:
            return super().__call__(inputs)
        return self.quant_method.apply_jax(self, inputs)
