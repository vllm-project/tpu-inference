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


class JaxEmbed(nnx.Embed, JaxModule):
    """Embedding layer for JAX."""

    def __init__(self,
                 *args,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        # nnx.Embed uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility, but forward it
        # as `param_dtype` so that weights are created with the correct dtype.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        nnx.Embed.__init__(self, *args, **kwargs)
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.embedding` to
        # `self.weight` such that `named_parameters()` can match the names in HF models.
        self.weight = self.embedding
        delattr(self, 'embedding')
        if hasattr(self.weight, "out_sharding"):
            self.weight.set_metadata('sharding', self.weight.out_sharding)

        self.quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
            if quant_method is not None:
                self.quant_method = quant_method
                assert isinstance(quant_method, QuantizeMethodBase)
                quant_method.create_weights_jax(self)

    def __getattr__(self, name: str):
        if name == "embedding":
            # nnx.Embed needs to access self.embedding
            return self.weight

    def __call__(self, x) -> jax.Array:
        if self.quant_method is not None:
            return self.quant_method.apply_jax(self, x)

        # Qwix manual logic wraps the QArray in a WithAux container
        q_container = self.weight
        if hasattr(q_container, 'array'):
            q_container = q_container.array

        # Check if we successfully unwrapped a QArray-like structure
        if hasattr(q_container, 'qvalue') and hasattr(q_container, 'scale'):
            # Fetch the quantized bytes (which are wrapped in nnx.Param)
            q_vals = jax.numpy.take(q_container.qvalue.value, x, axis=0)

            # Fetch scales (which are wrapped in nnx.Param)
            scales = q_container.scale.value
            # If the scale has length > 1 along the vocabulary axis, we must take it too
            if scales.shape[0] > 1:
                scales = jax.numpy.take(scales, x, axis=0)

            # Perform the dequantization dynamically
            return q_vals.astype(scales.dtype) * scales

        return super().__call__(x)

    def decode(self, x: jax.Array) -> jax.Array:
        q_container = self.weight
        if hasattr(q_container, 'array'):
            q_container = q_container.array

        if hasattr(q_container, 'qvalue') and hasattr(q_container, 'scale'):
            # Dequantize the full vocabulary weight matrix
            # qvalue is shape (V, D)
            q_vals = q_container.qvalue.value
            scales = q_container.scale.value
            weight_val = q_vals.astype(scales.dtype) * scales
            return jax.numpy.dot(x, weight_val.T)

        return jax.numpy.dot(x, self.weight.value.T)
