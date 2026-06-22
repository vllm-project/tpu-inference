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
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig


class JaxRmsNorm(JaxModule):
    """RmsNorm layer for JAX."""

    def __init__(self,
                 num_features: int,
                 *,
                 epsilon: float = 1e-6,
                 param_dtype: jax.numpy.dtype = jax.numpy.float32,
                 dtype: Optional[jax.numpy.dtype] = None,
                 use_scale: bool = True,
                 scale_init=nnx.initializers.uniform(),
                 rngs: nnx.Rngs,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        """Initializes the RmsNorm layer.
        
        Args:
            num_features: Number of features in the input.
            epsilon: A small float added to variance to avoid dividing by zero.
            param_dtype: The dtype of the parameters.
            dtype: The dtype of the output. If None, defaults to input dtype.
            use_scale: If True, a learnable scale parameter is used.
            scale_init: Initializer for the scale parameter.
            rngs: Random number generators for parameter initialization.
            quant_config: Optional quantization configuration.
            prefix: Prefix for the layer name.
        """
        self.epsilon = epsilon
        self.dtype = dtype
        if use_scale:
            self.weight = nnx.Param(
                scale_init(rngs.params(), (num_features, ), param_dtype))
        else:
            self.weight = nnx.data(None)
        self._layer_prefix = prefix

    def __call__(self,
                 x: jax.Array,
                 mask: Optional[jax.Array] = None) -> jax.Array:
        with jax.named_scope(self._layer_prefix):
            weight = self.weight[...] if self.weight is not None else None
            # Unlike nnx.RmsNorm, we do not want to upcast the output to float32, which causes
            # convert + reshape. Instead, we keep the output in the same dtype as the input for
            # better performance.
            out_dtype = self.dtype if self.dtype is not None else x.dtype

            var = jax.numpy.mean(jax.numpy.square(x),
                                 axis=-1,
                                 keepdims=True,
                                 where=mask)
            mul = jax.lax.rsqrt(var + self.epsilon)

            y = x * mul
            if weight is not None:
                y = y * jax.numpy.asarray(weight, out_dtype)
            return jax.numpy.asarray(y, out_dtype)


class JaxLayerNorm(nnx.LayerNorm, JaxModule):
    """LayerNorm layer that inherits from JaxModule and maps scale to weight for compatibility."""

    def __init__(self, *args, **kwargs):
        # nnx.LayerNorm uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        nnx.LayerNorm.__init__(self, *args, **kwargs)

        # For compatibility, alias scale to weight
        self.weight = self.scale
        delattr(self, 'scale')

    def __getattr__(self, name: str):
        if name == "scale":
            return self.weight
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
