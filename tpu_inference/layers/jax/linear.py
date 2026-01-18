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

from tpu_inference.layers.jax.einsum import JaxEinsum, JaxQuantizedEinsumMethod
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig

JaxQuantizedLinearMethod = JaxQuantizedEinsumMethod


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
                 quant_config: Optional[VllmQuantConfig] = None,
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
