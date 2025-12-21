# Copyright 2025 Google LLC
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

import jax.numpy as jnp
import torch
from jax._src import dtypes
from jax._src.numpy.scalar_types import _ScalarMeta
from torchax.ops.mappings import j2t_dtype, t2j_dtype

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Map vllm dtype string that doesn't exactly match jax dtype string name.
_VLLM_DTYPE_STR_TO_JAX_DTYPE = {
    "fp8": jnp.float8_e4m3fn.dtype,
    "fp8_e4m3": jnp.float8_e4m3fn.dtype,
    "fp8_e5m2": jnp.float8_e5m2.dtype,
}


def to_jax_dtype(dtype: str | jnp.dtype | torch.dtype) -> jnp.dtype:
    if isinstance(dtype, str):
        if dict_dtype := _VLLM_DTYPE_STR_TO_JAX_DTYPE.get(dtype, None):
            return dict_dtype
        return jnp.dtype(dtype)
    elif isinstance(dtype, torch.dtype):
        return t2j_dtype(dtype)
    elif isinstance(dtype, jnp.dtype):
        return dtype
    elif isinstance(dtype, _ScalarMeta):
        return dtype.dtype
    else:
        raise ValueError(f"Argument is unsupported data type {type(dtype)}")


def to_torch_dtype(dtype: str | jnp.dtype | torch.dtype) -> torch.dtype:
    # Use jax dtype as an intermediate dtype which we'll be used to convert it
    # into torch dtype.
    dtype = to_jax_dtype(dtype)
    return j2t_dtype(dtype)


def get_dtype_bitwidth(dtype):
    return (dtypes.bit_width(dtype)
            if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype))


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits
