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
"""JinaBert encoder megakernel vs the per-layer XLA path.

Loads jinaai/jina-embeddings-v2-small-en once per megakernel precision and
runs each step through `JinaBertForMaskedLM` twice, with the same weights and
inputs: once with the encoder megakernel and once with the per-layer XLA path
(what `USE_JINA_BERT_MEGAKERNEL=0` selects). Reports the max abs error and
the minimum per-token and mean-pooled cosine similarity over the real
(non-padding) tokens; run with `-s` to see the table.

TPU only: the megakernel is a Mosaic kernel, and so is the XLA path's
encoder flash attention.
"""

import contextlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tests.models.jax.test_jina_bert import (MODEL_ID, MockVllmConfig,
                                             _shim_transformers_onnx_module)
from tpu_inference.kernels.jina_bert_megakernel import (MAX_TOKENS,
                                                        mxu_dtype_for)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.jina_bert import (JinaBertForMaskedLM,
                                                _megakernel_unsupported_reason)

CLS_ID, SEP_ID = 101, 102
MAX_NUM_SEQS = 16
# Both paths feed bf16 operands to the MXU (the XLA path's matmuls run at
# lax.Precision.DEFAULT) at slightly different points, so they agree to bf16
# rounding rather than bit for bit.
MIN_TOKEN_COS = 0.999
MIN_POOLED_COS = 0.9995

# (name, seq_lens, num_tokens); num_tokens=None pads the step to the model
# runner's token bucket (the next power of two >= 16).
CASES = [
    *[(f"len{n}", [n], None)
      for n in (1, 127, 128, 129, 511, 512, 513, 1009, 1024, 2016, 2048)],
    ("len1009x2", [1009, 1009], None),
    ("len1009_len1000", [1009, 1000], None),
    ("short_seqs", [7, 33, 128, 1, 250, 64, 3, 19], None),
    ("trailing_padding", [300, 5, 150], 2048),
]

_REPORT = []


def _bucket(num_tokens: int) -> int:
    return max(16, 1 << (num_tokens - 1).bit_length())


def _make_step(seq_lens, num_tokens, vocab_size, seed=0):
    """Token ids + metadata for one step, laid out like the model runner:
    requests back to back from token 0, then padding up to `num_tokens`;
    `seq_lens` zero-padded to MAX_NUM_SEQS slots."""
    num_tokens = num_tokens or _bucket(sum(seq_lens))
    rng = np.random.default_rng(seed)
    ids = np.zeros(num_tokens, np.int32)
    positions = np.zeros(num_tokens, np.int32)
    off = 0
    for n in seq_lens:
        tokens = rng.integers(1000, vocab_size, size=n, dtype=np.int32)
        tokens[0] = CLS_ID
        if n > 1:
            tokens[-1] = SEP_ID
        ids[off:off + n] = tokens
        positions[off:off + n] = np.arange(n)
        off += n
    lens = np.zeros(max(MAX_NUM_SEQS, len(seq_lens)), np.int32)
    lens[:len(seq_lens)] = seq_lens
    metadata = AttentionMetadata(
        input_positions=jnp.asarray(positions),
        block_tables=jnp.zeros(lens.shape, jnp.int32),
        seq_lens=jnp.asarray(lens),
        query_start_loc=jnp.asarray(np.concatenate([[0], np.cumsum(lens)]),
                                    dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, len(seq_lens)], jnp.int32),
    )
    return jnp.asarray(ids), metadata


def _compare(got, want, seq_lens):
    """(max abs error, min per-token cosine, min mean-pooled cosine)."""
    total = sum(seq_lens)
    g = np.asarray(got, np.float64)[:total]
    w = np.asarray(want, np.float64)[:total]
    token_cos = np.sum(
        g * w, -1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(w, axis=-1))
    pooled_cos, off = [], 0
    for n in seq_lens:
        a, b = g[off:off + n].mean(0), w[off:off + n].mean(0)
        pooled_cos.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        off += n
    return (float(np.abs(g - w).max()), float(token_cos.min()),
            float(min(pooled_cos)))


def _hidden(model, input_ids, metadata):
    _, hidden, _, _ = model(kv_caches=[],
                            input_ids=input_ids,
                            attention_metadata=metadata)
    return np.asarray(hidden)


@contextlib.contextmanager
def _xla_path(model):
    """Runs the encoder layer by layer (the USE_JINA_BERT_MEGAKERNEL=0
    path) on the same model instance, i.e. the same weights."""
    encoder = model.model.encoder
    encoder.use_megakernel = False
    try:
        yield
    finally:
        encoder.use_megakernel = True


def _build_model(vllm_config, mesh, *, use_megakernel, precision="default"):
    with pytest.MonkeyPatch.context() as mp, jax.set_mesh(mesh):
        mp.setenv("USE_JINA_BERT_MEGAKERNEL", "1" if use_megakernel else "0")
        mp.setenv("JINA_BERT_MEGAKERNEL_PRECISION", precision)
        return JinaBertForMaskedLM(vllm_config, jax.random.PRNGKey(0), mesh)


@pytest.fixture(scope="module")
def jina_mesh():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Needs a TPU: both paths run Mosaic kernels.")
    devices = np.array(jax.local_devices()[:1]).reshape((1, 1, 1, 1))
    mesh = Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))
    with jax.set_mesh(mesh):
        yield mesh


@pytest.fixture(scope="module")
def jina_vllm_config():
    _shim_transformers_onnx_module()
    # Register the arch with vLLM's registry first (normally done by the
    # vllm.general_plugins entrypoint).
    from tpu_inference.models.vllm.experimental import register_models
    register_models()
    try:
        return MockVllmConfig(MODEL_ID)
    except Exception as e:  # e.g. no network to fetch the HF config
        pytest.skip(f"Could not build ModelConfig for {MODEL_ID}: {e}")


@pytest.fixture(scope="module", params=["default", "highest"])
def loaded_model(request, jina_vllm_config, jina_mesh):
    precision = request.param
    model = _build_model(jina_vllm_config,
                         jina_mesh,
                         use_megakernel=True,
                         precision=precision)
    encoder = model.model.encoder
    assert encoder.use_megakernel
    assert encoder.megakernel_precision == precision
    with jax.set_mesh(jina_mesh), set_current_vllm_config(jina_vllm_config):
        loader = get_model_loader(LoadConfig(load_format="hf"))
        loader.load_weights(model, jina_vllm_config.model_config)
    # load_weights also built the megakernel's packed layout.
    assert encoder.megakernel_wqkv.value.dtype == mxu_dtype_for(precision)
    return model


@pytest.fixture(scope="module", autouse=True)
def _print_report():
    yield
    if not _REPORT:
        return
    print("\nJinaBert encoder megakernel vs XLA path (real tokens only):")
    print(f"{'precision':<9} {'case':<17} {'T':>5} {'max abs err':>11} "
          f"{'min token cos':>13} {'min pooled cos':>14}")
    for precision, name, num_tokens, max_abs, token_cos, pooled_cos in (
            _REPORT):
        print(f"{precision:<9} {name:<17} {num_tokens:>5} {max_abs:>11.3e} "
              f"{token_cos:>13.7f} {pooled_cos:>14.8f}")


@pytest.mark.parametrize("name,seq_lens,num_tokens",
                         CASES,
                         ids=[case[0] for case in CASES])
def test_megakernel_matches_xla_path(loaded_model, name, seq_lens, num_tokens,
                                     record_property):
    hf_config = loaded_model.vllm_config.model_config.hf_config
    input_ids, metadata = _make_step(seq_lens, num_tokens,
                                     hf_config.vocab_size)
    got = _hidden(loaded_model, input_ids, metadata)
    with _xla_path(loaded_model):
        want = _hidden(loaded_model, input_ids, metadata)

    assert got.shape == want.shape == (input_ids.shape[0],
                                       hf_config.hidden_size)
    assert np.isfinite(got[:sum(seq_lens)]).all()
    max_abs, token_cos, pooled_cos = _compare(got, want, seq_lens)
    precision = loaded_model.model.encoder.megakernel_precision
    _REPORT.append((precision, name, int(input_ids.shape[0]), max_abs,
                    token_cos, pooled_cos))
    record_property("max_abs_err", max_abs)
    record_property("min_token_cos", token_cos)
    record_property("min_pooled_cos", pooled_cos)
    print(f"[{precision}] {name} (T={input_ids.shape[0]}): max abs err "
          f"{max_abs:.3e}, min token cos {token_cos:.7f}, min pooled cos "
          f"{pooled_cos:.8f}")
    assert token_cos > MIN_TOKEN_COS
    assert pooled_cos > MIN_POOLED_COS


def test_rejects_more_than_max_tokens(loaded_model):
    hf_config = loaded_model.vllm_config.model_config.hf_config
    input_ids, metadata = _make_step([MAX_TOKENS + 1], MAX_TOKENS + 1,
                                     hf_config.vocab_size)
    with pytest.raises(ValueError, match="at most 2048 tokens"):
        _hidden(loaded_model, input_ids, metadata)


def test_unpacked_weights_fallback(jina_vllm_config, jina_mesh):
    # Without JinaBertForMaskedLM.load_weights (e.g. --load-format dummy) the
    # megakernel packs the weights inside the step instead.
    model = _build_model(jina_vllm_config, jina_mesh, use_megakernel=True)
    assert getattr(model.model.encoder, "megakernel_wqkv", None) is None
    seq_lens = [100, 20]
    input_ids, metadata = _make_step(
        seq_lens, None, jina_vllm_config.model_config.hf_config.vocab_size)
    got = _hidden(model, input_ids, metadata)
    with _xla_path(model):
        want = _hidden(model, input_ids, metadata)
    _, token_cos, pooled_cos = _compare(got, want, seq_lens)
    assert token_cos > MIN_TOKEN_COS
    assert pooled_cos > MIN_POOLED_COS


def test_switch_off_selects_xla_path(jina_vllm_config, jina_mesh):
    model = _build_model(jina_vllm_config, jina_mesh, use_megakernel=False)
    encoder = model.model.encoder
    assert not encoder.use_megakernel
    encoder.pack_megakernel_weights()  # No-op when the megakernel is off.
    assert getattr(encoder, "megakernel_wqkv", None) is None


def test_unsupported_configs_fall_back(jina_vllm_config, jina_mesh):
    hf_config = jina_vllm_config.model_config.hf_config
    assert _megakernel_unsupported_reason(hf_config, jnp.float32, jina_mesh,
                                          "default") is None
    assert "single chip" in _megakernel_unsupported_reason(
        hf_config, jnp.float32, types.SimpleNamespace(size=4), "default")
    assert "float32" in _megakernel_unsupported_reason(hf_config, jnp.bfloat16,
                                                       jina_mesh, "default")
