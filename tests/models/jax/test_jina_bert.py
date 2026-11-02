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
"""Tests for the JAX-native JinaBert embedding model.

Numerical parity is checked against the official ONNX export shipped in the
model repo (same weights, float32, no remote code) — cosine similarity of
per-token hidden states and mean-pooled embeddings must exceed 0.999.

NOTE: the HF `transformers` remote-code reference (jina-bert-implementation)
targets transformers 4.x and no longer loads under v5 (removed
transformers.onnx / pytorch_utils APIs, meta-device init), which is why the
ONNX export is used as the reference instead.
"""

import sys
import types
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig, set_current_vllm_config
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.jina_bert import (JinaBertForMaskedLM,
                                                get_alibi_slopes)

MODEL_ID = "jinaai/jina-embeddings-v2-small-en"


def _shim_transformers_onnx_module():
    """The jina remote-code *config* (configuration_bert.py) does
    `from transformers.onnx import OnnxConfig`; that module was removed in
    newer transformers. Provide an in-process stub (only referenced for ONNX
    export, never executed)."""
    import transformers
    try:
        import transformers.onnx  # noqa: F401
    except Exception:
        onnx_mod = types.ModuleType("transformers.onnx")

        class OnnxConfig:
            pass

        onnx_mod.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = onnx_mod
        transformers.onnx = onnx_mod


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the JinaBert model."""

    def __init__(self, model: str):
        self.model_config = ModelConfig(model,
                                        runner="pooling",
                                        trust_remote_code=True)
        self.model_config.dtype = jnp.float32
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.cache_config = MagicMock(cache_dtype="auto")
        self.quant_config = None
        self.parallel_config = None


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    m = Mesh(device_mesh, axis_names=('data', 'attn_dp', 'expert', 'model'))
    with jax.set_mesh(m):
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def mock_vllm_config():
    _shim_transformers_onnx_module()
    # Register the arch with vLLM's registry first (normally done by the
    # vllm.general_plugins entrypoint).
    from tpu_inference.models.vllm.experimental import register_models
    register_models()
    try:
        return MockVllmConfig(MODEL_ID)
    except Exception as e:  # e.g. no network to fetch the HF config
        pytest.skip(f"Could not build ModelConfig for {MODEL_ID}: {e}")


def _make_attention_metadata(seq_lens_list):
    total = int(sum(seq_lens_list))
    positions = jnp.concatenate(
        [jnp.arange(n, dtype=jnp.int32) for n in seq_lens_list])
    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)
    query_start_loc = jnp.array(np.cumsum([0] + list(seq_lens_list)),
                                dtype=jnp.int32)
    return AttentionMetadata(
        input_positions=positions,
        block_tables=jnp.zeros((len(seq_lens_list), ), dtype=jnp.int32),
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=jnp.array([0, 0, len(seq_lens_list)],
                                       dtype=jnp.int32),
    ), total


def test_alibi_slopes_reference():
    """Slopes must match the reference 2^(-8i/n) geometric sequence."""
    slopes = get_alibi_slopes(8)
    expected = [2**(-i) for i in range(1, 9)]
    np.testing.assert_allclose(slopes, expected, rtol=1e-12)
    # Non-power-of-2 head counts follow the interleaving rule.
    assert len(get_alibi_slopes(12)) == 12


class TestJinaBert:

    def test_model_init(self, mock_vllm_config, rng, mesh):
        with jax.set_mesh(mesh):
            model = JinaBertForMaskedLM(mock_vllm_config, rng, mesh)

        hf_config = mock_vllm_config.model_config.hf_config
        layers = model.model.encoder.layer
        assert len(layers) == hf_config.num_hidden_layers

        attn = getattr(layers[0].attention, "self")
        assert attn.num_heads == hf_config.num_attention_heads
        assert attn.head_dim == hf_config.hidden_size // \
            hf_config.num_attention_heads
        assert attn.query.weight.shape == (hf_config.hidden_size,
                                           attn.num_heads, attn.head_dim)
        assert attn.query.bias.shape == (attn.num_heads, attn.head_dim)

        mlp = layers[0].mlp
        assert mlp.gated_layers.weight.shape == (hf_config.hidden_size, 2 *
                                                 hf_config.intermediate_size)
        assert mlp.wo.weight.shape == (hf_config.intermediate_size,
                                       hf_config.hidden_size)
        emb = model.model.embeddings
        assert emb.word_embeddings.weight.shape == (hf_config.vocab_size,
                                                    hf_config.hidden_size)

    def test_forward_and_onnx_parity(self, mock_vllm_config, rng, mesh):
        ort = pytest.importorskip("onnxruntime")
        transformers = pytest.importorskip("transformers")

        # --- Reference: official ONNX export from the model repo (fp32) ---
        try:
            from huggingface_hub import hf_hub_download
            tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
            onnx_path = hf_hub_download(MODEL_ID, "model.onnx")
        except Exception as e:
            pytest.skip(f"Could not download tokenizer/ONNX reference: {e}")

        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        input_names = {i.name for i in sess.get_inputs()}

        sentences = [
            "How is the weather today?",
            "Jina embeddings run on TPU v6e with vLLM.",
        ]
        encoded = [tokenizer(s, return_tensors="np") for s in sentences]
        ref_hidden = []
        for e in encoded:
            feed = {
                name: e[name].astype(np.int64)
                for name in input_names if name in e
            }
            assert "input_ids" in feed, f"unexpected ONNX inputs {input_names}"
            # last_hidden_state: [1, T, hidden]
            ref_hidden.append(sess.run(None, feed)[0][0].astype(np.float32))

        # --- JAX model, same tokens as one concatenated ragged batch ---
        with jax.set_mesh(mesh):
            model = JinaBertForMaskedLM(mock_vllm_config, rng, mesh)
        with jax.set_mesh(mesh), set_current_vllm_config(mock_vllm_config):
            loader = get_model_loader(LoadConfig(load_format="hf"))
            loader.load_weights(model, mock_vllm_config.model_config)

        seq_lens = [int(e["input_ids"].shape[1]) for e in encoded]
        input_ids = jnp.array(np.concatenate(
            [e["input_ids"][0] for e in encoded]),
                              dtype=jnp.int32)
        metadata, total = _make_attention_metadata(seq_lens)

        kv_caches, hidden, aux, _ = model(kv_caches=[],
                                          input_ids=input_ids,
                                          attention_metadata=metadata)
        assert hidden.shape == (
            total, mock_vllm_config.model_config.hf_config.hidden_size)
        assert kv_caches == []
        assert aux == []

        hidden = np.asarray(hidden, dtype=np.float32)
        offset = 0
        for ref, n in zip(ref_hidden, seq_lens):
            got = hidden[offset:offset + n]
            offset += n
            # Per-token cosine similarity.
            cos = np.sum(ref * got, axis=-1) / (np.linalg.norm(ref, axis=-1) *
                                                np.linalg.norm(got, axis=-1))
            assert cos.min() > 0.999, f"per-token cosine too low: {cos.min()}"
            # Mean-pooled embedding cosine similarity.
            ref_emb = ref.mean(axis=0)
            got_emb = got.mean(axis=0)
            emb_cos = np.dot(ref_emb, got_emb) / (np.linalg.norm(ref_emb) *
                                                  np.linalg.norm(got_emb))
            assert emb_cos > 0.999, f"pooled cosine too low: {emb_cos}"
