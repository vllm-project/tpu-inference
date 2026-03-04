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


class JaxRmsNorm(nnx.RMSNorm, JaxModule):
    """RmsNorm layer for JAX."""

    def __init__(self,
                 *args,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        nnx.RMSNorm.__init__(self, *args, **kwargs)
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.scale` to
        # `self.weight` such that `named_parameters()` can match the names in HF models. We also
        # apply transpose here to match HF weight layout.
        self.weight = self.scale
        delattr(self, 'scale')
        if hasattr(self.weight, 'out_sharding'):
            self.weight.set_metadata('sharding', self.weight.out_sharding)

        self.quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
            if quant_method is not None:
                assert isinstance(quant_method, QuantizeMethodBase)
                self.quant_method = quant_method
                quant_method.create_weights_jax(self)

    def __getattr__(self, name: str):
        if name == "scale":
            # nnx.RMSNorm needs to access self.scale
            return self.weight

    def __call__(self,
                 x: jax.Array,
                 mask: Optional[jax.Array] = None) -> jax.Array:
        if self.quant_method is None:
            return nnx.RMSNorm.__call__(self, x, mask=mask)
        return self.quant_method.apply_jax(self, x, mask=mask)
