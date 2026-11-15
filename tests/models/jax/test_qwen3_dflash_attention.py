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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.dflash_attention_interface import \
    dflash_concat_attention
from tpu_inference.models.jax.qwen3_dflash import Qwen3DFlashAttention


def _make_attention_metadata(query_start_loc: list[int]) -> AttentionMetadata:
    query_start_loc = np.asarray(query_start_loc, dtype=np.int32)
    seq_lens = np.diff(query_start_loc)
    total_tokens = int(query_start_loc[-1])
    return AttentionMetadata(
        input_positions=jnp.arange(total_tokens, dtype=jnp.int32),
        block_tables=jnp.zeros((max(1, total_tokens), ), dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
        query_start_loc=jnp.asarray(query_start_loc, dtype=jnp.int32),
        request_distribution=jnp.asarray([0, 0, len(seq_lens)],
                                         dtype=jnp.int32),
    )


def _dense_reference_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    sm_scale: float,
) -> jax.Array:
    logits = jnp.einsum("qnh,knh->nqk", q.astype(jnp.float32),
                        k.astype(jnp.float32))
    logits = logits * sm_scale
    probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
    return jnp.einsum("nqk,knh->qnh", probs, v)


def _build_stub_attention(impl: str) -> Qwen3DFlashAttention:
    attention = object.__new__(Qwen3DFlashAttention)
    # nnx.Module is frozen in flax >= 0.12.4; bypass the freeze for the stub.
    _set = object.__setattr__.__get__(attention)
    _set("q_proj", lambda x: x)
    _set("q_norm", lambda x: x)
    _set("k_proj", lambda x: x)
    _set("k_norm", lambda x: x)
    _set("v_proj", lambda x: x)
    _set("o_proj", lambda x: x)
    _set("head_dim_original", 1)
    _set("head_dim", 1)
    _set("num_heads", 1)
    _set("rope_theta", 10000.0)
    _set("rope_scaling", None)
    _set("mesh", object())
    _set("dflash_attention_impl", impl)
    _set("max_query_len", 2)
    _set("kv_cache_quantized_dtype", None)
    _set("_k_scale", 1.0)
    _set("_v_scale", 1.0)
    return attention


def test_dflash_concat_attention_matches_concat_reference():
    q = jnp.array([[[1.0]], [[2.0]]], dtype=jnp.float32)
    k_ctx = jnp.array([[[1.0]], [[0.0]]], dtype=jnp.float32)
    k_noise = jnp.array([[[0.0]], [[1.0]]], dtype=jnp.float32)
    v_ctx = jnp.array([[[10.0]], [[20.0]]], dtype=jnp.float32)
    v_noise = jnp.array([[[100.0]], [[200.0]]], dtype=jnp.float32)
    md = _make_attention_metadata([0, 2])
    sm_scale = 1.0

    output = dflash_concat_attention(
        q,
        k_ctx,
        k_noise,
        v_ctx,
        v_noise,
        md,
        md.query_start_loc,
        max_query_len=2,
        sm_scale=sm_scale,
    )

    k_cat = jnp.concatenate([k_ctx, k_noise], axis=0)
    v_cat = jnp.concatenate([v_ctx, v_noise], axis=0)
    expected = _dense_reference_attention(q, k_cat, v_cat, sm_scale=sm_scale)
    np.testing.assert_allclose(np.asarray(output),
                               np.asarray(expected),
                               rtol=1e-5,
                               atol=1e-5)

    additive_expected = _dense_reference_attention(q,
                                                   k_ctx + k_noise,
                                                   v_ctx + v_noise,
                                                   sm_scale=sm_scale)
    assert not np.allclose(np.asarray(output), np.asarray(additive_expected))


def test_dflash_concat_attention_repeats_kv_heads_for_gqa():
    q = jnp.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
        ],
        dtype=jnp.float32,
    )
    k_ctx = jnp.array([[[0.5]], [[1.5]]], dtype=jnp.float32)
    k_noise = jnp.array([[[2.0]], [[3.0]]], dtype=jnp.float32)
    v_ctx = jnp.array([[[11.0]], [[13.0]]], dtype=jnp.float32)
    v_noise = jnp.array([[[17.0]], [[19.0]]], dtype=jnp.float32)
    md = _make_attention_metadata([0, 2])

    output = dflash_concat_attention(
        q,
        k_ctx,
        k_noise,
        v_ctx,
        v_noise,
        md,
        md.query_start_loc,
        max_query_len=2,
        sm_scale=1.0,
    )

    k_cat = jnp.concatenate(
        [jnp.repeat(k_ctx, 2, axis=1),
         jnp.repeat(k_noise, 2, axis=1)],
        axis=0,
    )
    v_cat = jnp.concatenate(
        [jnp.repeat(v_ctx, 2, axis=1),
         jnp.repeat(v_noise, 2, axis=1)],
        axis=0,
    )
    expected = _dense_reference_attention(q, k_cat, v_cat, sm_scale=1.0)
    np.testing.assert_allclose(np.asarray(output),
                               np.asarray(expected),
                               rtol=1e-5,
                               atol=1e-5)


def test_qwen3_dflash_attention_concat_impl(monkeypatch):
    md = _make_attention_metadata([0, 2])
    hidden_states = jnp.array([[[1.0]], [[2.0]]], dtype=jnp.float32)
    target_hidden_states = jnp.array([[[3.0]], [[4.0]]], dtype=jnp.float32)
    kv_cache = jnp.array([0.0], dtype=jnp.float32)

    attention_calls = []

    def fake_attention(
        kv_cache,
        q,
        k,
        v,
        _md,
        _mesh,
        _head_dim_original,
        **_kwargs,
    ):
        attention_calls.append({
            "q":
            np.asarray(q),
            "k":
            np.asarray(k),
            "v":
            np.asarray(v),
            "update_kv_cache":
            _kwargs.get("update_kv_cache", True),
            "use_causal_mask":
            _kwargs.get("use_causal_mask", True),
        })
        if len(attention_calls) == 1:
            return kv_cache + 1.0, jnp.full_like(q, -5.0)
        else:
            return kv_cache + 2.0, jnp.full_like(q, 7.0)

    monkeypatch.setattr("tpu_inference.models.jax.qwen3_dflash.apply_rope",
                        lambda x, *_args, **_kwargs: x)
    monkeypatch.setattr("tpu_inference.models.jax.qwen3_dflash.attention",
                        fake_attention)

    attention = _build_stub_attention("concat_dense")
    new_kv_cache, output = attention(
        kv_cache=kv_cache,
        hidden_states=hidden_states,
        target_hidden_states=target_hidden_states,
        attention_metadata=md,
        target_query_start_loc=md.query_start_loc,
        target_positions=jnp.array([1, 2], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(output), np.full((2, 1, 1), 7.0))
    np.testing.assert_allclose(np.asarray(new_kv_cache), np.array([3.0]))
    assert len(attention_calls) == 2

    # Step 1 Target write assertions
    np.testing.assert_allclose(attention_calls[0]["q"],
                               np.zeros_like(hidden_states))
    np.testing.assert_allclose(attention_calls[0]["k"],
                               np.asarray(target_hidden_states))
    np.testing.assert_allclose(attention_calls[0]["v"],
                               np.asarray(target_hidden_states))
    assert attention_calls[0]["update_kv_cache"] is True
    assert attention_calls[0]["use_causal_mask"] is True

    # Step 2 Noise attention assertions
    np.testing.assert_allclose(attention_calls[1]["q"],
                               np.asarray(hidden_states))
    np.testing.assert_allclose(attention_calls[1]["k"],
                               np.asarray(hidden_states))
    np.testing.assert_allclose(attention_calls[1]["v"],
                               np.asarray(hidden_states))
    assert attention_calls[1]["update_kv_cache"] is True
    assert attention_calls[1]["use_causal_mask"] is False


def test_qwen3_dflash_attention_additive_legacy_impl(monkeypatch):
    md = _make_attention_metadata([0, 2])
    hidden_states = jnp.array([[[1.0]], [[2.0]]], dtype=jnp.float32)
    target_hidden_states = jnp.array([[[3.0]], [[4.0]]], dtype=jnp.float32)
    kv_cache = jnp.array([0.0], dtype=jnp.float32)

    calls = {}

    def fake_attention(
        kv_cache,
        q,
        k,
        v,
        _md,
        _mesh,
        _head_dim_original,
        **_kwargs,
    ):
        calls["q"] = np.asarray(q)
        calls["k"] = np.asarray(k)
        calls["v"] = np.asarray(v)
        return kv_cache + 2.0, jnp.full_like(q, 3.0)

    def fail_concat(*_args, **_kwargs):
        raise AssertionError("concat path should not run for additive_legacy")

    monkeypatch.setattr("tpu_inference.models.jax.qwen3_dflash.apply_rope",
                        lambda x, *_args, **_kwargs: x)
    monkeypatch.setattr("tpu_inference.models.jax.qwen3_dflash.attention",
                        fake_attention)

    attention = _build_stub_attention("additive_legacy")
    new_kv_cache, output = attention(
        kv_cache=kv_cache,
        hidden_states=hidden_states,
        target_hidden_states=target_hidden_states,
        attention_metadata=md,
        target_query_start_loc=md.query_start_loc,
        target_positions=jnp.array([1, 2], dtype=jnp.int32),
    )

    expected_k = np.asarray(target_hidden_states + hidden_states)
    np.testing.assert_allclose(calls["k"], expected_k)
    np.testing.assert_allclose(calls["v"], expected_k)
    np.testing.assert_allclose(np.asarray(output), np.full((2, 1, 1), 3.0))
    np.testing.assert_allclose(np.asarray(new_kv_cache), np.array([2.0]))


def test_qwen3_dflash_attention_unknown_impl_raises(monkeypatch):
    md = _make_attention_metadata([0, 2])
    hidden_states = jnp.array([[[1.0]], [[2.0]]], dtype=jnp.float32)
    target_hidden_states = jnp.array([[[3.0]], [[4.0]]], dtype=jnp.float32)
    kv_cache = jnp.array([0.0], dtype=jnp.float32)

    monkeypatch.setattr("tpu_inference.models.jax.qwen3_dflash.apply_rope",
                        lambda x, *_args, **_kwargs: x)

    attention = _build_stub_attention("bad_impl")
    with pytest.raises(ValueError, match="Unsupported"):
        attention(
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_metadata=md,
            target_query_start_loc=md.query_start_loc,
            target_positions=jnp.array([1, 2], dtype=jnp.int32),
        )
