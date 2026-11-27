# SPDX-License-Identifier: Apache-2.0
"""FUSED_MATMUL_REDUCE_SCATTER on the sequence-parallel row-parallel path of
sharded_quantized_matmul, on a multi-axis (attention-DP x TP) mesh."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.sharding import ShardingAxisName

MESH_AXES = ("data", "pcp", "attn_dp", "attn_dp_expert", "expert", "model",
             "dcp")


def _mesh(attn_dp, tp):
    devices = np.array(jax.devices()[:attn_dp * tp])
    return Mesh(devices.reshape(1, 1, attn_dp, 1, 1, tp, 1), MESH_AXES)


def _run(mesh, x, w_q, w_s, maybe_quantize_x):
    weight_spec = P(ShardingAxisName.ATTN_HEAD, None)  # row-parallel
    f = jax.jit(lambda x, w_q, w_s: sharded_quantized_matmul(
        x,
        w_q,
        w_s,
        weight_spec,
        mesh=mesh,
        maybe_quantize_x=maybe_quantize_x,
        sp_token_axes=ShardingAxisName.EXPERT_DATA))
    text = f.lower(x, w_q, w_s).as_text()
    return np.asarray(f(x, w_q, w_s), np.float32), text


@pytest.mark.parametrize("maybe_quantize_x", [True, False])
@pytest.mark.parametrize("attn_dp,tp", [(2, 2), (1, 4)])
def test_fused_path_used_and_matches(monkeypatch, maybe_quantize_x, attn_dp,
                                     tp):
    if jax.devices()[0].platform != "tpu" or jax.device_count() < 4:
        pytest.skip("needs 4 TPU devices")
    mesh = _mesh(attn_dp, tp)
    T, K, N = 2048, 8192, 1792
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.device_put(
        jax.random.normal(k0, (T, K), jnp.bfloat16),
        NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
    w_q = jax.random.randint(k1, (K, N), -127, 128, jnp.int8)
    w_s = jax.random.uniform(k2, (N, ), jnp.float32, 1e-3, 2e-3)

    monkeypatch.delenv("FUSED_MATMUL_REDUCE_SCATTER", raising=False)
    expected, text_off = _run(mesh, x, w_q, w_s, maybe_quantize_x)
    monkeypatch.setenv("FUSED_MATMUL_REDUCE_SCATTER", "1")
    out, text_on = _run(mesh, x, w_q, w_s, maybe_quantize_x)

    assert "quantized_matmul_reduce_scatter" not in text_off
    assert "quantized_matmul_reduce_scatter" in text_on
    err = np.max(np.abs(out - expected)) / np.max(np.abs(expected))
    assert err < 2e-2, err


def test_decode_batch_falls_back(monkeypatch):
    if jax.devices()[0].platform != "tpu" or jax.device_count() < 4:
        pytest.skip("needs 4 TPU devices")
    mesh = _mesh(2, 2)
    T, K, N = 64, 8192, 1792  # too few tokens for the kernel
    x = jnp.ones((T, K), jnp.bfloat16)
    w_q = jnp.ones((K, N), jnp.int8)
    w_s = jnp.full((N, ), 1e-3, jnp.float32)
    monkeypatch.setenv("FUSED_MATMUL_REDUCE_SCATTER", "1")
    out, text = _run(mesh, x, w_q, w_s, True)
    assert "quantized_matmul_reduce_scatter" not in text
    np.testing.assert_allclose(out, np.full((T, N), K * 1e-3), rtol=2e-2)
