# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for the canonical-layout weight-sync helpers of the torchax path.

The end-to-end check (MaxText Qwen3.5-35B-A3B -> Tunix `VllmSampler.update_params`
-> torchax model, compared against the HF-loaded model) lives with the MaxText
mapping; these tests cover the layout helpers that do not need a TPU model.
"""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from tpu_inference.models.vllm import weight_sync


def test_strip_prefix():
    assert weight_sync._strip_prefix("vllm_model.model.layers.0.x") == \
        "model.layers.0.x"
    assert weight_sync._strip_prefix("model.layers.0.x") == "model.layers.0.x"


def _fake_qkv(total_heads, total_kv, head_size, replicas):
    return SimpleNamespace(total_num_heads=total_heads,
                           total_num_kv_heads=total_kv,
                           head_size=head_size,
                           num_kv_head_replicas=replicas)


def test_replicate_kv_heads_repeats_each_kv_head_consecutively():
    # 4 q heads, 2 kv heads, head_size 3, tp=8 -> each kv head 4 times in a row,
    # matching vLLM's per-rank `qkv_proj` layout (rank r owns kv head r // 4).
    hs, in_dim = 3, 5
    q = np.arange(4 * hs * in_dim, dtype=np.float32).reshape(4 * hs, in_dim)
    k = 100 + np.arange(2 * hs * in_dim, dtype=np.float32).reshape(2 * hs, in_dim)
    v = 200 + np.arange(2 * hs * in_dim, dtype=np.float32).reshape(2 * hs, in_dim)
    canon = jnp.asarray(np.concatenate([q, k, v], 0))
    out = np.asarray(
        weight_sync._replicate_kv_heads(_fake_qkv(4, 2, hs, 4), canon))
    assert out.shape == ((4 + 2 * 4 + 2 * 4) * hs, in_dim)
    np.testing.assert_array_equal(out[:4 * hs], q)
    k_rep = out[4 * hs:4 * hs + 8 * hs].reshape(8, hs, in_dim)
    v_rep = out[4 * hs + 8 * hs:].reshape(8, hs, in_dim)
    for r in range(8):
        np.testing.assert_array_equal(k_rep[r], k.reshape(2, hs, in_dim)[r // 4])
        np.testing.assert_array_equal(v_rep[r], v.reshape(2, hs, in_dim)[r // 4])


def test_replicate_kv_heads_is_identity_without_replicas():
    canon = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)
    out = weight_sync._replicate_kv_heads(_fake_qkv(2, 2, 1, 1), canon)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(canon))


def test_replicate_kv_heads_handles_1d_bias():
    hs = 2
    canon = jnp.asarray(np.arange((4 + 2 + 2) * hs, dtype=np.float32))
    out = np.asarray(
        weight_sync._replicate_kv_heads(_fake_qkv(4, 2, hs, 2), canon))
    assert out.shape == ((4 + 4 + 4) * hs, )
    k = np.asarray(canon)[4 * hs:6 * hs].reshape(2, hs)
    np.testing.assert_array_equal(out[4 * hs:8 * hs].reshape(4, hs),
                                  np.repeat(k, 2, axis=0))
