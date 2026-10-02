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
"""Unit tests for the vllm (torchax) DFlash draft forward."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from jax.sharding import Mesh
from torchax.interop import jax_view

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.vllm import dflash_draft
from tpu_inference.models.vllm.vllm_model_wrapper import _VllmRunner
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context

HIDDEN = 16
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
VOCAB = 32
NUM_LAYERS = 2
BLOCK_SIZE = 3
MAX_LEN = 16


# ----- Minimal stand-ins for the vLLM DFlash draft model's modules -----
class _RMSNorm(torch.nn.Module):
    """Mimics vLLM's RMSNorm, including the fused residual-add variant."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(dim) + 0.5)
        self.variance_epsilon = eps

    def _norm(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.variance_epsilon) * self.weight

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        x = x + residual
        return self._norm(x), x


class _Linear(torch.nn.Module):
    """vLLM linear layers return an (output, bias) tuple."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x), None


class _Rope(torch.nn.Module):
    """Position-dependent stand-in for RoPE so position plumbing is checked."""

    def forward(self, positions, query, key):
        scale = (1.0 + 0.05 * positions.to(query.dtype)).unsqueeze(-1)
        return query * scale, key * scale


class _AttnLayer:

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.impl = None


class _DFlashAttention(torch.nn.Module):

    def __init__(self, layer_name: str, causal: bool = False):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HEAD_DIM
        self.q_size = NUM_HEADS * HEAD_DIM
        self.kv_size = NUM_KV_HEADS * HEAD_DIM
        self.scaling = HEAD_DIM**-0.5
        self.qkv_proj = _Linear(HIDDEN, self.q_size + 2 * self.kv_size)
        self.o_proj = _Linear(self.q_size, HIDDEN)
        self.q_norm = _RMSNorm(HEAD_DIM)
        self.k_norm = _RMSNorm(HEAD_DIM)
        self.rotary_emb = _Rope()
        self.attn = _AttnLayer(layer_name)
        self.causal = causal
        self.v_scale = None
        self.sliding_window = None
        self.attention_sink_bias = None


class _MLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(HIDDEN, 2 * HIDDEN, bias=False)
        self.down = torch.nn.Linear(2 * HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


class _Layer(torch.nn.Module):

    def __init__(self, idx: int):
        super().__init__()
        self.input_layernorm = _RMSNorm(HIDDEN)
        self.self_attn = _DFlashAttention(f"model.layers.{idx}.self_attn.attn")
        self.post_attention_layernorm = _RMSNorm(HIDDEN)
        self.mlp = _MLP()


class _Inner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList(
            [_Layer(i) for i in range(NUM_LAYERS)])
        self.hidden_norm = _RMSNorm(HIDDEN)
        self.norm = _RMSNorm(HIDDEN)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)


class _FakeDFlashModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = _Inner()


# ----- Contiguous-cache stand-in for the ragged paged attention kernel -----
def _fake_attention(kv_cache,
                    q,
                    k,
                    v,
                    attention_metadata,
                    mesh,
                    head_dim_original=None,
                    sm_scale=None,
                    k_scale=None,
                    v_scale=None,
                    update_kv_cache=True,
                    use_causal_mask=True):
    """Same contract as the RPA kernel, over a [req, pos, k/v, head, dim]
    cache: request i's new K/V occupy positions [seq_len - q_len, seq_len)
    and its queries attend to all cached positions < seq_len."""
    assert update_kv_cache
    md = attention_metadata
    qsl = np.asarray(md.query_start_loc)
    seq_lens = np.asarray(md.seq_lens)
    out = jnp.zeros(q.shape, dtype=jnp.float32)
    repeat = q.shape[1] // k.shape[1]
    for i in range(seq_lens.shape[0]):
        start, end = int(qsl[i]), int(qsl[i + 1])
        q_len, seq_len = end - start, int(seq_lens[i])
        if q_len == 0:
            continue
        first = seq_len - q_len
        kv_cache = kv_cache.at[i, first:seq_len, 0].set(k[start:end])
        kv_cache = kv_cache.at[i, first:seq_len, 1].set(v[start:end])
        keys = jnp.repeat(kv_cache[i, :seq_len, 0], repeat, axis=1)
        vals = jnp.repeat(kv_cache[i, :seq_len, 1], repeat, axis=1)
        logits = jnp.einsum("qnh,knh->nqk", q[start:end].astype(jnp.float32),
                            keys.astype(jnp.float32)) * sm_scale
        if use_causal_mask:
            q_pos = first + jnp.arange(q_len)[:, None]
            k_pos = jnp.arange(seq_len)[None, :]
            logits = jnp.where((k_pos <= q_pos)[None], logits, -1e30)
        probs = jax.nn.softmax(logits, axis=-1)
        out = out.at[start:end].set(
            jnp.einsum("nqk,knh->qnh", probs, vals.astype(jnp.float32)))
    return kv_cache, out.astype(q.dtype)


def _reference_forward(model, input_ids, positions, target_hidden,
                       target_positions, ctx_lens):
    """DFlash written out directly: every noise block attends (non-causally)
    to [its request's context K/V ; its own K/V]."""
    inner = model.model
    ctx = inner.hidden_norm(target_hidden)
    x = inner.embed_input_ids(input_ids)
    residual = None
    ctx_starts = np.concatenate([[0], np.cumsum(ctx_lens)])
    for layer in inner.layers:
        attn = layer.self_attn
        if residual is None:
            residual = x
            x = layer.input_layernorm(x)
        else:
            x, residual = layer.input_layernorm(x, residual)

        ctx_qkv, _ = attn.qkv_proj(ctx)
        _, k_ctx, v_ctx = ctx_qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        k_ctx = attn.k_norm(k_ctx.view(-1, NUM_KV_HEADS,
                                       HEAD_DIM)).view(-1, attn.kv_size)
        _, k_ctx = attn.rotary_emb(target_positions, k_ctx, k_ctx)

        qkv, _ = attn.qkv_proj(x)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = attn.q_norm(q.view(-1, NUM_HEADS, HEAD_DIM)).view(-1, attn.q_size)
        k = attn.k_norm(k.view(-1, NUM_KV_HEADS,
                               HEAD_DIM)).view(-1, attn.kv_size)
        q, k = attn.rotary_emb(positions, q, k)

        outs = []
        for i in range(len(ctx_lens)):
            cs, ce = ctx_starts[i], ctx_starts[i + 1]
            ns, ne = i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE
            keys = torch.cat([k_ctx[cs:ce],
                              k[ns:ne]]).view(-1, NUM_KV_HEADS, HEAD_DIM)
            vals = torch.cat([v_ctx[cs:ce],
                              v[ns:ne]]).view(-1, NUM_KV_HEADS, HEAD_DIM)
            keys = keys.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
            vals = vals.repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
            qi = q[ns:ne].view(-1, NUM_HEADS, HEAD_DIM)
            logits = torch.einsum("qnh,knh->nqk", qi, keys) * attn.scaling
            probs = torch.softmax(logits, dim=-1)
            outs.append(
                torch.einsum("nqk,knh->qnh", probs,
                             vals).reshape(-1, attn.q_size))
        x, _ = attn.o_proj(torch.cat(outs))
        x, residual = layer.post_attention_layernorm(x, residual)
        x = layer.mlp(x)
    x, _ = inner.norm(x, residual)
    return x


@pytest.fixture
def mesh():
    devices = np.array(jax.devices("cpu")[:1])
    return Mesh(devices.reshape((1, 1)), ("data", "model"))


def test_dflash_draft_forward_matches_reference(monkeypatch, mesh):
    monkeypatch.setattr(dflash_draft, "attention", _fake_attention)
    torch.manual_seed(0)
    model = _FakeDFlashModel().eval()

    # Two requests with 4 and 2 newly accepted context tokens and an empty
    # cache, so seq_lens equals the context lengths.
    ctx_lens = [4, 2]
    num_reqs = len(ctx_lens)
    seq_lens = np.array(ctx_lens, dtype=np.int32)
    target_query_start_loc = np.array([0, 4, 6], dtype=np.int32)
    target_positions = np.array([0, 1, 2, 3, 0, 1], dtype=np.int32)
    noise_positions = np.concatenate(
        [np.arange(BLOCK_SIZE) + s for s in seq_lens]).astype(np.int32)
    input_ids = np.array([5, 1, 1, 7, 1, 1], dtype=np.int32)
    target_hidden = np.random.default_rng(0).standard_normal(
        (sum(ctx_lens), HIDDEN)).astype(np.float32)

    md = AttentionMetadata(
        input_positions=jnp.asarray(noise_positions),
        block_tables=jnp.zeros((num_reqs, ), dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.arange(num_reqs + 1, dtype=jnp.int32) * BLOCK_SIZE,
        request_distribution=jnp.array([0, 0, num_reqs], dtype=jnp.int32),
    )
    target_hidden_states = (jnp.asarray(target_hidden),
                            jnp.asarray(target_query_start_loc),
                            jnp.asarray(target_positions))

    with torch.no_grad():
        expected = _reference_forward(model,
                                      torch.from_numpy(input_ids).long(),
                                      torch.from_numpy(noise_positions),
                                      torch.from_numpy(target_hidden),
                                      torch.from_numpy(target_positions),
                                      ctx_lens).numpy()

    # Full-precision matmuls so the TPU result is comparable to the CPU one.
    env = torchax.default_env()
    with env, torch.no_grad(), jax.default_matmul_precision("highest"):
        model.to("jax")
        kv_caches = [
            jnp.zeros((num_reqs, MAX_LEN, 2, NUM_KV_HEADS, HEAD_DIM),
                      dtype=jnp.float32) for _ in range(NUM_LAYERS)
        ]
        layer_map = {
            layer.self_attn.attn.layer_name: i
            for i, layer in enumerate(model.model.layers)
        }
        with set_vllm_model_wrapper_context(
                kv_caches=kv_caches,
                mesh=mesh,
                layer_name_to_kvcache_index=layer_map):
            kwargs = dflash_draft.dflash_draft_call_kwargs(
                jnp.asarray(input_ids), target_hidden_states, md)
            out = dflash_draft.dflash_draft_forward(model, **kwargs)
            actual = np.asarray(jax_view(out))
            new_caches = list(kv_caches)

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    # Every layer's cache holds context + noise K/V for each request.
    for cache in new_caches:
        for i, ctx_len in enumerate(ctx_lens):
            filled = np.asarray(cache[i, :ctx_len + BLOCK_SIZE])
            assert np.all(np.any(filled != 0, axis=(-1, -2, -3)))
            assert not np.any(np.asarray(cache[i, ctx_len + BLOCK_SIZE:]))


def test_dflash_draft_forward_honors_causal_layers(monkeypatch, mesh):
    seen = []

    def _recording_attention(*args, use_causal_mask, **kwargs):
        seen.append(use_causal_mask)
        return _fake_attention(*args,
                               use_causal_mask=use_causal_mask,
                               **kwargs)

    monkeypatch.setattr(dflash_draft, "attention", _recording_attention)
    model = _FakeDFlashModel().eval()
    model.model.layers[1].self_attn.causal = True

    md = AttentionMetadata(
        input_positions=jnp.arange(BLOCK_SIZE, dtype=jnp.int32) + 2,
        block_tables=jnp.zeros((1, ), dtype=jnp.int32),
        seq_lens=jnp.array([2], dtype=jnp.int32),
        query_start_loc=jnp.array([0, BLOCK_SIZE], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
    )
    target_hidden_states = (jnp.ones(
        (2, HIDDEN), dtype=jnp.float32), jnp.array([0, 2], dtype=jnp.int32),
                            jnp.array([0, 1], dtype=jnp.int32))
    with torchax.default_env(), torch.no_grad():
        model.to("jax")
        kv_caches = [
            jnp.zeros((1, MAX_LEN, 2, NUM_KV_HEADS, HEAD_DIM),
                      dtype=jnp.float32) for _ in range(NUM_LAYERS)
        ]
        layer_map = {
            layer.self_attn.attn.layer_name: i
            for i, layer in enumerate(model.model.layers)
        }
        with set_vllm_model_wrapper_context(
                kv_caches=kv_caches,
                mesh=mesh,
                layer_name_to_kvcache_index=layer_map):
            dflash_draft.dflash_draft_forward(
                model,
                **dflash_draft.dflash_draft_call_kwargs(
                    jnp.array([3, 1, 1], dtype=jnp.int32),
                    target_hidden_states, md))

    # (context insert, noise) per layer; context inserts are always causal.
    assert seen == [True, False, True, True]


def test_validate_dflash_draft_model_accepts_dflash_layout():
    dflash_draft.validate_dflash_draft_model(_FakeDFlashModel())


def test_validate_dflash_draft_model_rejects_other_models():
    with pytest.raises(NotImplementedError, match="not supported"):
        dflash_draft.validate_dflash_draft_model(torch.nn.Linear(2, 2))


@pytest.mark.parametrize("attr, value",
                         [("sliding_window", 4),
                          ("attention_sink_bias", torch.zeros(NUM_HEADS))])
def test_validate_dflash_draft_model_rejects_unsupported_attention(
        attr, value):
    model = _FakeDFlashModel()
    setattr(model.model.layers[0].self_attn, attr, value)
    with pytest.raises(NotImplementedError):
        dflash_draft.validate_dflash_draft_model(model)


def test_vllm_runner_call_fn_uses_functional_params():
    model = _FakeDFlashModel()
    runner = _VllmRunner(model)
    weight = torch.full((VOCAB, HIDDEN), 2.0)

    def _embed(vllm_model, input_ids):
        return vllm_model.model.embed_input_ids(input_ids)

    out = torch.func.functional_call(
        runner,
        {"vllm_model.model.embed_tokens.weight": weight},
        kwargs={
            "call_fn": _embed,
            "call_args": (torch.tensor([1, 2]), ),
        },
        tie_weights=False,
        strict=False,
    )
    assert torch.equal(out, torch.full((2, HIDDEN), 2.0))


def test_dflash_draft_call_kwargs_unpacks_target_hidden_states():
    md = AttentionMetadata(input_positions=jnp.arange(3, dtype=jnp.int32))
    target_hidden_states = (jnp.ones(
        (2, HIDDEN)), jnp.array([0, 2], dtype=jnp.int32),
                            jnp.array([4, 5], dtype=jnp.int32))
    with torchax.default_env():
        kwargs = dflash_draft.dflash_draft_call_kwargs(jnp.array([1, 2, 3]),
                                                       target_hidden_states,
                                                       md)
        assert kwargs["attention_metadata"] is md
        assert kwargs["target_query_start_loc"] is target_hidden_states[1]
        np.testing.assert_array_equal(
            np.asarray(jax_view(kwargs["target_positions"])), [4, 5])
        np.testing.assert_array_equal(
            np.asarray(jax_view(kwargs["positions"])), [0, 1, 2])
        assert isinstance(kwargs["input_ids"], torch.Tensor)
