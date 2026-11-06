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

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from transformers import AutoTokenizer, Olmo2Config
from transformers import Olmo2ForCausalLM as HFOlmo2ForCausalLM

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.olmo2 import Olmo2Attention, Olmo2ForCausalLM


@pytest.fixture
def tpu_only():
    if jax.default_backend() != 'tpu':
        pytest.skip('Requires TPU paged attention')


@pytest.fixture
def mesh():
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1, 1, 1),
        ('data', 'attn_dp', 'expert', 'model'))
    with jax.set_mesh(mesh):
        yield mesh


@pytest.fixture(autouse=True)
def pp_group(monkeypatch):
    group = SimpleNamespace(is_first_rank=True,
                            is_last_rank=True,
                            rank_in_group=0,
                            world_size=1)
    for module in ('tpu_inference.models.jax.olmo2',
                   'tpu_inference.layers.jax.pp_utils'):
        monkeypatch.setattr(module + '.get_pp_group', lambda: group)
    return group


def make_config(**kwargs):
    return Olmo2Config(vocab_size=128,
                       hidden_size=256,
                       intermediate_size=512,
                       num_hidden_layers=2,
                       num_attention_heads=2,
                       num_key_value_heads=2,
                       max_position_embeddings=128,
                       rms_norm_eps=1e-6,
                       eos_token_id=2,
                       rope_theta=500000.0,
                       **kwargs)


def make_model(config, mesh, dtype=jnp.float32):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config,
                                     dtype=dtype,
                                     hf_overrides={},
                                     get_vocab_size=lambda: config.vocab_size),
        cache_config=SimpleNamespace(cache_dtype='auto'),
        quant_config=None,
        speculative_config=None,
        parallel_config=None)
    return Olmo2ForCausalLM(vllm_config, jax.random.key(42), mesh)


def reference_model(config):
    torch.manual_seed(42)
    config._attn_implementation = 'eager'
    reference = HFOlmo2ForCausalLM(config).eval()
    # Nonuniform norm scales expose both incorrect axes and residual ordering.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if 'norm' in name:
                parameter.copy_(torch.linspace(0.5, 1.5, parameter.numel()))
    return reference


def metadata(start, count):
    return AttentionMetadata(input_positions=jnp.arange(start,
                                                        start + count,
                                                        dtype=jnp.int32),
                             block_tables=jnp.arange(4, dtype=jnp.int32),
                             seq_lens=jnp.array([start + count],
                                                dtype=jnp.int32),
                             query_start_loc=jnp.array([0, count],
                                                       dtype=jnp.int32),
                             request_distribution=jnp.array([0, 0, 1],
                                                            dtype=jnp.int32))


def dense_attention(cache, q, k, v, md, mesh, head_dim, **kwargs):
    """CPU reference for the paged kernel, including incremental decoding."""
    if cache is not None:
        k = jnp.concatenate((cache[0], k))
        v = jnp.concatenate((cache[1], v))
    new_cache = (k, v)
    repeats = q.shape[1] // k.shape[1]
    k = jnp.repeat(k, repeats, axis=1)
    v = jnp.repeat(v, repeats, axis=1)
    scores = jnp.einsum('tnh,snh->nts', q, k) / head_dim**0.5
    causal = jnp.arange(k.shape[0])[None, :] <= md.input_positions[:, None]
    scores = jnp.where(causal[None, :, :], scores, -jnp.inf)
    return new_cache, jnp.einsum('nts,snh->tnh', jax.nn.softmax(scores), v)


def split_forward(model, last_only=False):
    graphdef, state = nnx.split(model)

    def forward(state, caches, ids, md):
        model = nnx.merge(graphdef, state)
        caches, hidden, _, _ = model(list(caches), ids, md)
        if last_only:
            hidden = hidden[-1:]
        return caches, model.compute_logits(hidden)

    return state, forward


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('kv_heads', [1, 2])
@pytest.mark.parametrize('ndh', [False, True])
def test_transformers_parity(mesh, monkeypatch, tied, kv_heads, ndh):
    monkeypatch.setenv('LAYOUT_Q_PROJ_AS_NDH', str(int(ndh)))
    monkeypatch.setattr('tpu_inference.models.jax.qwen3.attention',
                        dense_attention)
    config = make_config(tie_word_embeddings=tied)
    config.num_key_value_heads = kv_heads
    reference = reference_model(config)
    model = make_model(config, mesh)
    loaded = model.load_weights(reference.state_dict().items())
    expected_names = set(reference.state_dict())
    if tied:
        expected_names.remove('lm_head.weight')
    assert loaded == expected_names

    state, forward = split_forward(model)

    tokens = jnp.array([4, 17, 9, 23, 8, 12, 31, 5], dtype=jnp.int32)
    for run in (forward, jax.jit(forward)):
        caches = [None] * config.num_hidden_layers
        for start, count in ((0, 6), (6, 1), (7, 1)):
            with jax.default_matmul_precision('highest'):
                caches, actual = run(state, caches,
                                     tokens[start:start + count],
                                     metadata(start, count))
            with torch.no_grad():
                expected = reference(torch.tensor(
                    np.array(tokens[:start + count]))[None],
                                     use_cache=False).logits[0,
                                                             start:].numpy()
            np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('kv_heads', [1, 2])
def test_paged_attention_parity(tpu_only, mesh, kv_heads):
    from tpu_inference.runner.kv_cache import create_kv_caches

    config = make_config()
    config.num_key_value_heads = kv_heads
    reference = reference_model(config)
    model = make_model(config, mesh, jnp.bfloat16)
    model.load_weights(reference.state_dict().items())
    caches = create_kv_caches(4, 32, kv_heads, 128, mesh,
                              ['layer'] * config.num_hidden_layers)

    state, forward = split_forward(model)
    forward = jax.jit(forward)

    tokens = jnp.array([4, 17, 9, 23, 8, 12, 31, 5], dtype=jnp.int32)
    for start, count in ((0, 6), (6, 1), (7, 1)):
        caches, actual = forward(state, caches, tokens[start:start + count],
                                 metadata(start, count))
        with torch.no_grad():
            expected = reference(torch.tensor(np.array(tokens[:start +
                                                              count]))[None],
                                 use_cache=False).logits[0, start:].numpy()
        np.testing.assert_allclose(actual.astype(jnp.float32),
                                   expected,
                                   atol=0.04,
                                   rtol=0.04)


@pytest.mark.parametrize('rank', [0, 1, 2])
def test_pipeline_partition(mesh, pp_group, rank):
    pp_group.rank_in_group = rank
    pp_group.world_size = 3
    pp_group.is_first_rank = rank == 0
    pp_group.is_last_rank = rank == 2
    config = make_config()
    config.num_hidden_layers = 3
    model = make_model(config, mesh)
    loaded = model.load_weights(reference_model(config).state_dict().items())
    assert model.model.start_layer == rank
    assert model.model.end_layer == rank + 1
    assert {
        name.split('.')[2]
        for name in loaded if name.startswith('model.layers.')
    } == {str(rank)}
    assert ('model.embed_tokens.weight' in loaded) == (rank == 0)
    assert ('lm_head.weight' in loaded) == (rank == 2)
    assert ('model.norm.weight' in loaded) == (rank == 2)


def test_reject_attention_bias(mesh):
    with pytest.raises(ValueError, match='attention biases'):
        Olmo2Attention(make_config(attention_bias=True), jnp.float32,
                       nnx.Rngs(42), mesh, 'auto', None)


def test_reject_padded_heads(mesh):
    config = make_config()
    config.hidden_size = 192
    with pytest.raises(ValueError, match='padded attention heads'):
        Olmo2Attention(config, jnp.float32, nnx.Rngs(42), mesh, 'auto', None)


def test_registered_architecture():
    from tpu_inference.models.common.model_loader import \
        _get_model_architecture

    config = make_config(architectures=['Olmo2ForCausalLM'])
    assert _get_model_architecture(config) is Olmo2ForCausalLM


def test_fp32_checkpoint_to_bfloat16(mesh):
    config = make_config()
    model = make_model(config, mesh, jnp.bfloat16)
    reference = reference_model(config)
    model.load_weights(reference.state_dict().items())
    for name, parameter in model.named_parameters():
        assert parameter.get_value().dtype == jnp.bfloat16, name
    hidden = model.model.embed_tokens(jnp.array([4, 17], dtype=jnp.int32))
    layer = model.model.layers[0]
    assert layer.self_attn.v_proj(hidden).dtype == jnp.bfloat16


def test_tensor_parallel_parity(monkeypatch):
    if len(jax.devices()) < 2:
        pytest.skip('Requires two JAX devices')
    monkeypatch.setattr('tpu_inference.models.jax.qwen3.attention',
                        dense_attention)
    mesh = Mesh(
        np.array(jax.devices()[:2]).reshape(1, 1, 1, 2),
        ('data', 'attn_dp', 'expert', 'model'))
    config = make_config()
    reference = reference_model(config)
    with jax.set_mesh(mesh), jax.default_matmul_precision('highest'):
        model = make_model(config, mesh)
        model.load_weights(reference.state_dict().items())

        state, forward = split_forward(model)
        forward = jax.jit(forward)

        tokens = jnp.array([4, 17, 9, 23, 8, 12], dtype=jnp.int32)
        _, actual = forward(state, [None] * config.num_hidden_layers, tokens,
                            metadata(0, tokens.shape[0]))
        with torch.no_grad():
            expected = reference(torch.tensor(np.array(tokens))[None],
                                 use_cache=False).logits[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.skipif(
    not os.environ.get('OLMO2_TEST_MODEL'),
    reason='Set OLMO2_TEST_MODEL to a checkpoint ID or local directory')
def test_checkpoint_prefill_decode(tpu_only, mesh):
    from tpu_inference.runner.kv_cache import create_kv_caches

    path = os.environ['OLMO2_TEST_MODEL']
    reference = HFOlmo2ForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation='eager').eval()
    config = reference.config
    model = make_model(config, mesh, jnp.bfloat16)
    loaded = model.load_weights(reference.state_dict().items())
    assert loaded == set(reference.state_dict())
    assert all(param.get_value().dtype == jnp.bfloat16
               for _, param in model.named_parameters())
    tokenizer = AutoTokenizer.from_pretrained(path)

    state, forward = split_forward(model, last_only=True)
    forward = jax.jit(forward)

    cases = []
    for prompt in ('The capital of France is',
                   'def fibonacci(n):\n    if n <= 1:\n        return'):
        tokens = tokenizer.encode(prompt)
        expected_steps = []
        for _ in range(3):
            with torch.no_grad():
                expected = reference(torch.tensor([tokens]),
                                     use_cache=False).logits[0, -1].numpy()
            expected_steps.append(expected.astype(np.float64))
            tokens.append(int(expected.argmax()))
        cases.append((prompt, expected_steps))
    reference.to(torch.bfloat16)

    for prompt, expected_steps in cases:
        tokens = tokenizer.encode(prompt)
        caches = create_kv_caches(
            4, 32, config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads, mesh,
            ['layer'] * config.num_hidden_layers)
        for step in range(3):
            count = len(tokens) if step == 0 else 1
            caches, actual = forward(
                state, caches, jnp.array(tokens[-count:], dtype=jnp.int32),
                metadata(len(tokens) - count, count))
            with torch.no_grad():
                bf16 = reference(torch.tensor([tokens]),
                                 use_cache=False).logits[0,
                                                         -1].float().numpy()
            actual = np.asarray(actual[0], dtype=np.float64)
            expected = expected_steps[step]
            bf16 = bf16.astype(np.float64)
            cosine = np.dot(actual, expected) / (np.linalg.norm(actual) *
                                                 np.linalg.norm(expected))
            relative_rmse = np.linalg.norm(actual -
                                           expected) / np.linalg.norm(expected)
            expected_logp = expected - np.logaddexp.reduce(expected)
            actual_logp = actual - np.logaddexp.reduce(actual)
            bf16_logp = bf16 - np.logaddexp.reduce(bf16)
            probabilities = np.exp(expected_logp)
            kl = np.sum(probabilities * (expected_logp - actual_logp))
            bf16_kl = np.sum(probabilities * (expected_logp - bf16_logp))
            print(f'{prompt!r} step={step}: cosine={cosine:.6f}, '
                  f'relative_rmse={relative_rmse:.6f}, '
                  f'kl={kl:.6f}, hf_bf16_kl={bf16_kl:.6f}')
            assert cosine > 0.999
            # Sampling ignores uniform logit shifts. Compare distributions,
            # allowing twice HF's BF16 error with a 1e-3-nat rounding floor.
            assert kl < 0.01
            assert kl < max(0.001, 2 * bf16_kl)
            assert actual.argmax() == expected.argmax()
            tokens.append(int(expected.argmax()))
