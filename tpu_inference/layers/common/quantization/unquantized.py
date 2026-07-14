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

from tpu_inference.layers.common.linear import sharded_matmul
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation


class UnquantizedFusedMoEMethod:
    """Shared base for jax and vllm unquantized fused-MoE methods.

    Both paths track extra_backend_kwargs (e.g. ep_axis_name, block sizes)
    that flow through to the kernel call.  Keeping this here avoids
    duplicating the initialisation in each backend.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}


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
                     einsum_str: str | None = None) -> jax.Array:
        """Applies fused linear operation.

        Operates on a single large weight matrix that combines multiple logical
        operations (e.g., QKV projection).

        Args:
            x_jax: Input array of shape [..., hidden_dim].
            weight_jax: Weight array of shape [output_dim, hidden_dim].
            bias_jax: Optional bias array of shape [output_dim].
            einsum_str: Einsum string for the operation. If None, a standard matrix multiplication is performed.

        Returns:
            Output array of shape [..., total_output_dim].
        """
        if einsum_str:
            assert not self.linear_config.defer_all_reduce, \
                "Defer all-reduce is not compatible with custom einsum strings."
            outs = jnp.einsum(einsum_str, x_jax, weight_jax)
        else:
            assert weight_jax.ndim == 2, "Weight must be 2D for sharded_matmul."
            outs = sharded_matmul(
                x_jax,
                weight_jax,
                self.linear_config.weight_sharding,
                mesh=self.linear_config.mesh,
                defer_all_reduce=self.linear_config.defer_all_reduce,
            )

        if bias_jax is not None:
            outs += bias_jax
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        out = jnp.concatenate(outs, axis=-1)
        return out

    def _apply_split(
            self,
            x_jax: jax.Array,
            weights: Sequence[jax.Array],
            bias_jax: Optional[Sequence[jax.Array]] = None) -> jax.Array:
        """Applies split linear operation.

        Operates on a sequence of separate weight matrices, performing
        computation for each and concatenating the results.

        Args:
            x_jax: Input array of shape [..., hidden_dim].
            weights: Sequence of weight arrays, each of shape [output_dim_i, hidden_dim].
            bias_jax: Optional sequence of bias arrays, each of shape [output_dim_i].

        Returns:
            Output array of shape [..., total_output_dim].
        """
        outs = []
        for i, weight_jax in enumerate(weights):
            out = jnp.einsum("...k,kn->...n", x_jax, weight_jax)
            if bias_jax is not None:
                out += bias_jax[i]

            outs.append(out)
        out = jnp.concatenate(outs, axis=-1)
        return out
