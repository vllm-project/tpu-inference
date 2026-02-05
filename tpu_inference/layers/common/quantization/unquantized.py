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

from typing import Optional, Sequence

import jax
from jax import numpy as jnp

from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation


class UnquantizedLinearMethod:
    """Implements the forward method for unquantized linear layers.

    This class will be shared in both vLLM and jax path.
    """

    def __init__(self, linear_config: QuantLinearConfig):
        self.linear_config = linear_config

    def _apply_fused(self,
                     x_jax: jax.Array,
                     weight_jax: jax.Array,
                     bias_jax: Optional[jax.Array] = None,
                     einsum_str: str = "mn,pn->mp") -> jax.Array:
        outs = jnp.einsum(einsum_str, x_jax, weight_jax)
        if bias_jax is not None:
            outs += bias_jax

        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes,
            getattr(self.linear_config, 'n_shards', 1))
        out = jnp.concatenate(outs, axis=-1)
        return out

    def _apply_split(
            self,
            x_jax: jax.Array,
            weights: Sequence[jax.Array],
            bias_jax: Optional[Sequence[jax.Array]] = None) -> jax.Array:
        outs = []
        for i, weight_jax in enumerate(weights):
            out = jnp.einsum("mn,pn->mp", x_jax, weight_jax)
            if bias_jax is not None:
                out += bias_jax[i]

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return out
