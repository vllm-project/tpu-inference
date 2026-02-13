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

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


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
        self.kernel_init = kwargs.get("kernel_init",
                                      jax.nn.initializers.lecun_normal())
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.kernel` to
        # `self.weight` such that `named_parameters()` can match the names in HF models.
        self.weight = self.kernel
        delattr(self, 'kernel')
        if hasattr(self.weight, 'out_sharding'):
            self.weight.set_metadata('sharding', self.weight.out_sharding)
        logger.info(
            f'{self.weight.value.shape=} {self.weight.value.dtype=} {self.weight.sharding=}'
        )

        if quant_config is None:
            self.quant_method = None
        elif (quant_method := quant_config.get_quant_method(self,
                                                            prefix=prefix)):
            assert isinstance(quant_method, QuantizeMethodBase)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self, rngs=rngs)
        else:
            self.quant_method = None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.quant_method is not None:
            return self.quant_method.apply_jax(self, inputs)

        output = jax.numpy.einsum(self.einsum_str, inputs, self.weight.value)
        if self.bias is not None:
            output += self.bias
        return output


class JaxLinear(JaxEinsum):
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
                 input_size: int,
                 output_size: int,
                 rngs,
                 *,
                 use_bias: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        JaxEinsum.__init__(self,
                           rngs=rngs,
                           einsum_str="mn,np->mp",
                           kernel_shape=(input_size, output_size),
                           bias_shape=(output_size, ) if use_bias else None,
                           quant_config=quant_config,
                           prefix=prefix,
                           **kwargs)
