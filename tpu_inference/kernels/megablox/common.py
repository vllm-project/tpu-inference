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


<<<<<<< HEAD
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
=======
def supports_bfloat16_matmul() -> bool:
    """Does the currently attached CPU support bfloat16 inputs?"""
    return not is_tpu() or tpu_generation() >= 4


def assert_is_supported_dtype(dtype: jnp.dtype) -> None:
    if dtype not in [
            jnp.bfloat16, jnp.float32, jnp.float8_e4m3fn, jnp.float8_e5m2,
            jnp.int8, jnp.int4, jnp.float4_e2m1fn, jnp.uint4
    ]:
        raise ValueError(f"No support for {dtype=}.")


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
    """A type to which both input should be adapted to before dot product."""
    # bf16xbf16 matmul is only supported since TPUv4 generation. In case of mixed
    # input precision, we need to convert bf16 argument to fp32 beforehand.
    if (supports_bfloat16_matmul() and lhs.dtype == jnp.bfloat16
            and rhs.dtype == jnp.bfloat16):
        return jnp.bfloat16
    else:
        return jnp.float32
>>>>>>> 96a7a80d (Add fp4 gmm)
