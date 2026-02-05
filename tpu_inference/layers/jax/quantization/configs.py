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

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase

if TYPE_CHECKING:
    from tpu_inference.layers.jax.linear import JaxEinsum


class JaxQuantLinearConfig(QuantLinearConfig):
    """JAX-specific quantization linear config.

    Extracts configuration from a JaxEinsum layer, similar to how
    VllmQuantLinearConfig extracts config from LinearBase in vLLM.
    """

    def __init__(self, layer: "JaxEinsum", quant_config=None):
        # Parse einsum dimensions
        einsum_str = getattr(layer, 'einsum_str', '')
        weight_shape = layer.weight.shape

        self.input_size, self.output_size, self.contracting_dims, self.output_dims = \
            self._parse_einsum_dims(einsum_str, weight_shape)

        super().__init__(
            enable_sp=False,  # JAX doesn't use SP currently
            output_sizes=[self.output_size],
        )

        self.block_size = envs.REQUANTIZE_BLOCK_SIZE
        self.enable_quantized_matmul_kernel = self.block_size is not None

        # Block size from HF checkpoint config (for dequantizing pre-quantized
        # checkpoints before re-quantizing to a different block size)
        self.checkpoint_block_size = None
        if quant_config and hasattr(quant_config, 'weight_block_size'):
            self.checkpoint_block_size = quant_config.weight_block_size

        # Compute weight sharding from layer
        self.weight_sharding = self._compute_weight_sharding(layer)

    @staticmethod
    def _parse_einsum_dims(
        einsum_str: str,
        weight_shape: tuple[int,
                            ...]) -> tuple[int, int, list[int], list[int]]:
        """Parse einsum string to identify input/contracting and output dimensions.

        This is needed for FP8 quantization because the FP8 kernel requires 2D
        tensors, but JaxEinsum can have multi-dimensional weights (e.g., attention
        projections with shape (hidden, heads, head_dim)).

        Args:
            einsum_str: e.g. "TD,DNH->TNH" or "mn,np->mp"
            weight_shape: e.g. (4096, 32, 128) or (4096, 4096)

        Returns:
            input_size: Product of contracting dimensions.
            output_size: Product of output dimensions.
            contracting_dims: List of indices in weight_shape that are contracting.
            output_dims: List of indices in weight_shape that are output.
        """
        if not einsum_str or "->" not in einsum_str:
            # Fallback: dim 0 is input, rest output
            input_size = weight_shape[0]
            output_size = math.prod(weight_shape[1:])
            return input_size, output_size, [0], list(
                range(1, len(weight_shape)))

        lhs, rhs = einsum_str.split("->")
        inputs_str, weight_str = lhs.split(",")

        inputs_str = inputs_str.strip()
        weight_str = weight_str.strip()
        output_str = rhs.strip()

        input_indices = set(inputs_str)
        output_indices = set(output_str)
        weight_indices = list(weight_str)

        if len(weight_indices) != len(weight_shape):
            # Fallback if einsum string doesn't match weight shape
            input_size = weight_shape[0]
            output_size = math.prod(weight_shape[1:])
            return input_size, output_size, [0], list(
                range(1, len(weight_shape)))

        contracting_dims = []
        output_dims = []
        input_size = 1
        output_size = 1

        for i, char in enumerate(weight_indices):
            dim_size = weight_shape[i]
            if char in input_indices and char not in output_indices:
                # This dimension is contracted (appears in input but not output)
                contracting_dims.append(i)
                input_size *= dim_size
            else:
                # This dimension appears in output
                output_dims.append(i)
                output_size *= dim_size

        return input_size, output_size, contracting_dims, output_dims

    def _compute_weight_sharding(self, layer: "JaxEinsum") -> P:
        """Extract and adapt weight sharding for 2D quantized kernel.

        The FP8 kernel expects (Out, In) shaped weights, so we need to
        flatten multi-dimensional shardings to 2D and transpose.
        """
        sharding = getattr(layer.weight, 'sharding', None)
        if sharding is None:
            return P(None, None)

        # Extract the spec from various sharding types
        if isinstance(sharding, NamedSharding):
            spec = sharding.spec
        elif isinstance(sharding, (tuple, P)):
            spec = sharding
        else:
            return P(None, None)

        if not isinstance(spec, (tuple, P)) or len(spec) < 2:
            return P(None, None)

        # Find non-None sharding specs for contracting and output dims
        def get_spec_for_dims(dims):
            for i in dims:
                if i < len(spec) and spec[i] is not None:
                    return spec[i]
            return None

        in_spec = get_spec_for_dims(self.contracting_dims)
        out_spec = get_spec_for_dims(self.output_dims)

        # Kernel expects (Out, In) format
        return P(out_spec, in_spec)


class QuantizationConfig(ABC):

    @abstractmethod
    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        raise NotImplementedError
