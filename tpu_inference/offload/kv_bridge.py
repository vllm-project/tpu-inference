# Copyright 2026
# LMCache-on-TPU value bridge: jax.Array (host) <-> flat bytes buffer.
from __future__ import annotations
import dataclasses
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class KVBlockSpec:
    """Everything needed to reconstruct a JAX KV block from raw bytes."""
    shape: tuple
    dtype_str: str

    def to_dict(self) -> dict:
        return {"shape": list(self.shape), "dtype_str": self.dtype_str}

    @staticmethod
    def from_dict(d: dict) -> "KVBlockSpec":
        return KVBlockSpec(shape=tuple(d["shape"]), dtype_str=d["dtype_str"])


def jax_block_to_bytes(arr: Any):
    """Serialize a host jax.Array KV block to raw bytes + reconstruct spec.
    Raw byte view -> bfloat16/float16/int8/fp8 all roundtrip bit-exactly."""
    host = np.asarray(jax.device_get(arr))
    spec = KVBlockSpec(shape=tuple(host.shape), dtype_str=jnp.dtype(arr.dtype).name)
    raw = host.tobytes()
    return raw, spec


def bytes_to_jax_block(raw: bytes, spec: KVBlockSpec):
    """Reconstruct a host jax.Array from raw bytes + spec (bit-exact)."""
    jdt = jnp.dtype(spec.dtype_str)
    np_dt = np.dtype(jdt)
    flat = np.frombuffer(raw, dtype=np_dt)
    host = flat.reshape(spec.shape)
    return jnp.asarray(host)
