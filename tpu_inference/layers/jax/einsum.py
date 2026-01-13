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
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

from tpu_inference.layers.jax.linear import JaxLinear


class JaxEinsum(JaxLinear):
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
                 bias_shape: Optional[tuple[int, ...]] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 **kwargs):
        # Only einsum_str that satisfies:
        #  * contracting dimensions are consecutive and in the same order in both inputs (i.e., 'abc,acd->bd' 'abc,cbd->ad' not supported)
        #  * contracting dimensions are the last dimensions of the first input (i.e. 'bca,bcd->ad' not supported)
        #  * non-contracting dimensions from the second input are consecutive and in the same order in the output (i.e. 'abc,dbce->ade' not supported)
        # is allowed.
        input_strs, output_str = einsum_str.strip(' ').split("->")
        input_a, input_b = input_strs.split(",")
        assert len(input_b) == len(
            kernel_shape
        ), f"Length of input_b ({len(input_b)}) must match the length of kernel_shape ({len(kernel_shape)})."

        # Find contracting dimensions
        if input_a[-1] not in input_b:
            raise ValueError(
                f"Invalid einsum_str: {einsum_str}, contracting dimensions must be the last dimensions of the first input."
            )
        idx_last_contract_b = input_b.index(input_a[-1])
        idx_first_contract_b = 0
        for i in range(idx_last_contract_b + 1):
            if input_b[i] in input_a:
                idx_first_contract_b = i
                break
        if input_b[idx_first_contract_b:idx_last_contract_b +
                   1] not in input_a:
            raise ValueError(
                f"Invalid einsum_str: {einsum_str}, contracting dimensions ({input_b[idx_first_contract_b:idx_last_contract_b + 1]}) must be in the same order and consecutive in both inputs."
            )
        if len(input_a) - 1 != idx_last_contract_b + 1 - idx_first_contract_b:
            raise NotImplementedError(
                f"Only support input has just one non-contracting dimension which is sequence dimension, got {input_a.strip(input_b[idx_first_contract_b:idx_last_contract_b + 1])}."
            )
        in_features, out_features = 1, 1
        for c in kernel_shape[idx_first_contract_b:idx_last_contract_b + 1]:
            in_features *= c
        if idx_first_contract_b == 0:
            out_features_str = input_b[idx_last_contract_b + 1:]
            self._out_reshape_dims = tuple(kernel_shape[idx_last_contract_b +
                                                        1:])
            for i, c in enumerate(kernel_shape[idx_last_contract_b + 1:]):
                out_features *= c
                assert bias_shape[
                    i] == c if bias_shape is not None else True, f"Bias shape {bias_shape} does not match output dimension {c}."
        elif idx_last_contract_b == len(input_b) - 1:
            out_features_str = input_b[:idx_first_contract_b]
            self._out_reshape_dims = tuple(kernel_shape[:idx_first_contract_b])
            for i, c in enumerate(kernel_shape[:idx_first_contract_b]):
                out_features *= c
                assert bias_shape[
                    i] == c if bias_shape is not None else True, f"Bias shape {bias_shape} does not match output dimension {c}."
        else:
            raise ValueError(
                f"Invalid einsum_str: {einsum_str}, contracting dimensions ({input_b[idx_first_contract_b:idx_last_contract_b + 1]}) must be either the first or last dimensions of the second input ({input_b})."
            )

        if out_features_str not in output_str:
            raise ValueError(
                f"Invalid einsum_str: {einsum_str}, non-contracting dimensions from the second input ({out_features_str}) must be in the same order and consecutive in the output ({output_str})."
            )
        elif output_str[0:len(out_features_str)] == out_features_str:
            self._out_reshape_dims += (-1, )
        else:
            self._out_reshape_dims = (-1, ) + self._out_reshape_dims

        JaxLinear.__init__(self,
                           in_features,
                           out_features,
                           use_bias=bias_shape is not None,
                           quant_config=quant_config,
                           **kwargs)
        self.einsum_str = einsum_str
        self.weight.value = self.weight.value.reshape(kernel_shape)
        if self.bias is not None:
            self.bias.value = self.bias.value.reshape(bias_shape)

        # TODO(lk-chen): reduce kernel's sharding

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.quant_method.apply_jax(self, inputs)
