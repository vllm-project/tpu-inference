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
"""Common utilities for GMM kernels."""

import re

import jax
import jax.numpy as jnp


def is_tpu() -> bool:
    return "TPU" in jax.devices()[0].device_kind


def tpu_kind() -> str:
    """Query identification string for the currently attached TPU."""
    return jax.devices()[0].device_kind


# Most TPU devices follow the pattern "TPU v{version}{variant}", e.g. "TPU v5p"
# TPU v7 has a different pattern (i.e. "TPU7x")
_TPU_KIND_PATTERN = re.compile(r"TPU( v)?(\d+)")


def tpu_generation() -> int:
    """Generation number of the currently attached TPU."""
    if version := _TPU_KIND_PATTERN.match(tpu_kind()):
        return int(version[2])
    raise NotImplementedError("only TPU devices are supported")


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
    if dtype not in [
            jnp.bfloat16,
            jnp.float32,
            jnp.float8_e4m3fn,
            jnp.float8_e5m2,
            jnp.int8,
            jnp.int4,
            jnp.float4_e2m1fn,
            jnp.uint4,
    ]:
        raise ValueError(f"No support for {dtype=}.")
