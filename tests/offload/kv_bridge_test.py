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
"""Bit-exact roundtrip tests for the LMCache-on-TPU value bridge.

Runs on jax[cpu]; no TPU required.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.offload.kv_bridge import (bytes_to_jax_block,
                                             jax_block_to_bytes)

KV_SHAPE = (1, 4, 16, 8, 2, 32)  # (1, num_layers, block, num_head, 2, head_dim)


def _bytes(x):
    return np.asarray(jax.device_get(x)).tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int8", "int32"])
def test_roundtrip_bit_exact(dtype):
    jdt = jnp.dtype(dtype)
    key = jax.random.PRNGKey(0)
    if jdt in (jnp.int8, jnp.int32):
        x = jax.random.randint(key, KV_SHAPE, -100, 100).astype(jdt)
    else:
        x = jax.random.normal(key, KV_SHAPE).astype(jdt)
    raw, spec = jax_block_to_bytes(x)
    y = bytes_to_jax_block(raw, spec)
    assert y.shape == x.shape
    assert jnp.dtype(y.dtype) == jnp.dtype(x.dtype)
    assert _bytes(y) == _bytes(x), f"bytes differ for {dtype}"


def test_bfloat16_edge_values():
    vals = jnp.array(
        [0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), 3.14159, 0.333333],
        dtype=jnp.bfloat16,
    )
    x = jnp.broadcast_to(vals.reshape(1, 1, 4, 1, 2, 1), (1, 2, 4, 2, 2, 8)).astype(jnp.bfloat16)
    raw, spec = jax_block_to_bytes(x)
    y = bytes_to_jax_block(raw, spec)
    assert _bytes(y) == _bytes(x)


def test_spec_serialization():
    from tpu_inference.offload.kv_bridge import KVBlockSpec
    s = KVBlockSpec(shape=(1, 4, 16, 8, 2, 32), dtype_str="bfloat16")
    s2 = KVBlockSpec.from_dict(s.to_dict())
    assert s2 == s
