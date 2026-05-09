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
"""PLE (Per-Layer Embedding) tests for Gemma-4 — kb_ple.md §3.1, §3.4.

Stage 2 commit-2 verification gate (per plan_stage2.md §4):
  - PLE-off parity: hidden_size_per_layer_input=0 -> compute_per_layer_inputs returns None
  - Shape: per_layer_inputs.shape == (T, L, P)
  - Multimodal mask: is_multimodal redirects to vocab slot 0
  - Out-of-vocab mask: token id >= vocab_size_per_layer_input -> slot 0
  - Numerical parity vs numpy oracle implementing kb_ple.md §3.1 verbatim

Synthetic tiny config — does not require an HF gated model. Constructs a
Gemma4TextConfig directly with toy dims so the test is fast.
"""
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma4TextConfig

from tpu_inference.models.jax.gemma4 import Gemma4Model

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, 1, 1, 1))
    with Mesh(device_mesh,
              axis_names=("data", "attn_dp", "expert", "model")) as m:
        yield m


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    mock_pp = MagicMock(is_first_rank=True,
                        is_last_rank=True,
                        rank_in_group=0,
                        world_size=1)
    with patch("tpu_inference.models.jax.gemma4.get_pp_group",
               return_value=mock_pp), patch(
                   "tpu_inference.layers.jax.pp_utils.get_pp_group",
                   return_value=mock_pp):
        yield


def _make_text_config(*,
                      hidden_size=32,
                      hidden_size_per_layer_input=8,
                      vocab_size=64,
                      vocab_size_per_layer_input=64,
                      num_hidden_layers=2):
    """Tiny Gemma4TextConfig for fast unit tests.

    PLE on iff hidden_size_per_layer_input > 0.
    """
    return Gemma4TextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        layer_types=["full_attention"] * num_hidden_layers,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        vocab_size_per_layer_input=vocab_size_per_layer_input,
        attention_bias=False,
    )


def _make_vllm_config(text_config):
    hf_config = MagicMock()
    hf_config.text_config = text_config
    hf_config.tie_word_embeddings = False

    model_config = MagicMock()
    model_config.hf_config = hf_config
    model_config.dtype = jnp.float32
    model_config.get_vocab_size = lambda: text_config.vocab_size

    vllm_config = MagicMock()
    vllm_config.model_config = model_config
    vllm_config.cache_config = MagicMock(cache_dtype="auto")
    vllm_config.quant_config = None
    return vllm_config


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_ple_off_returns_none(mesh, rng=jax.random.PRNGKey(0)):
    """When hidden_size_per_layer_input=0, compute_per_layer_inputs returns None."""
    text_config = _make_text_config(hidden_size_per_layer_input=0)
    vllm_config = _make_vllm_config(text_config)

    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    assert model.embed_tokens_per_layer is None
    assert model.per_layer_model_projection is None
    assert model.per_layer_projection_norm is None
    assert model.hidden_size_per_layer_input == 0

    T = 4
    input_ids = jnp.zeros((T, ), dtype=jnp.int32)
    inputs_embeds = jnp.zeros((T, text_config.hidden_size), dtype=jnp.float32)
    out = model.compute_per_layer_inputs(input_ids, inputs_embeds)
    assert out is None


def test_ple_on_shape_and_dtype(mesh, rng=jax.random.PRNGKey(0)):
    """PLE on: shape (T, L, P), dtype matches inputs_embeds."""
    P = 8
    L = 3
    H = 32
    V = 64
    text_config = _make_text_config(hidden_size=H,
                                    hidden_size_per_layer_input=P,
                                    num_hidden_layers=L,
                                    vocab_size=V,
                                    vocab_size_per_layer_input=V)
    vllm_config = _make_vllm_config(text_config)

    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    assert model.embed_tokens_per_layer is not None
    assert model.per_layer_model_projection is not None
    assert model.per_layer_projection_norm is not None

    T = 5
    input_ids = jnp.array([0, 1, 2, 3, V - 1], dtype=jnp.int32)
    inputs_embeds = jnp.ones((T, H), dtype=jnp.float32)
    with mesh:
        out = model.compute_per_layer_inputs(input_ids, inputs_embeds)

    assert out is not None
    assert out.shape == (T, L, P), f"got {out.shape}, expected ({T}, {L}, {P})"
    assert out.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(out))


def test_ple_multimodal_masking(mesh, rng=jax.random.PRNGKey(0)):
    """is_multimodal=True positions look up vocab slot 0 in embed_tokens_per_layer."""
    P = 4
    L = 2
    H = 16
    V = 32
    text_config = _make_text_config(hidden_size=H,
                                    hidden_size_per_layer_input=P,
                                    num_hidden_layers=L,
                                    vocab_size=V,
                                    vocab_size_per_layer_input=V)
    vllm_config = _make_vllm_config(text_config)

    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    # Two "tokens": one normal (id=5), one MM (id=10).
    # When MM mask is on for position 1, both should produce the same Track A
    # contribution (both look up slot 0 in embed_tokens_per_layer).
    T = 2
    input_ids = jnp.array([0, 10], dtype=jnp.int32)
    inputs_embeds = jnp.zeros((T, H), dtype=jnp.float32)

    with mesh:
        out_no_mask = model.compute_per_layer_inputs(input_ids,
                                                     inputs_embeds,
                                                     is_multimodal=None)
        # Mark token 1 as MM — it should now look up slot 0 (same as token 0).
        is_multimodal = jnp.array([False, True], dtype=jnp.bool_)
        out_with_mask = model.compute_per_layer_inputs(
            input_ids, inputs_embeds, is_multimodal=is_multimodal)

    # Without MM mask, the two tokens have different per_layer_inputs (different
    # vocab slots looked up). With MM mask, both look up slot 0.
    diff_unmasked = jnp.linalg.norm(out_no_mask[0] - out_no_mask[1])
    diff_masked = jnp.linalg.norm(out_with_mask[0] - out_with_mask[1])
    assert diff_unmasked > 1e-3, (
        "Sanity: tokens 0 and 10 should differ without MM mask")
    assert diff_masked < 1e-6, (
        f"With MM mask, both tokens should be identical. Got diff={diff_masked}"
    )


def test_ple_oov_masking(mesh, rng=jax.random.PRNGKey(0)):
    """token id >= vocab_size_per_layer_input gets masked to slot 0 (no NaN)."""
    P = 4
    L = 2
    H = 16
    V = 32
    V_ple = 16  # smaller than vocab_size
    text_config = _make_text_config(hidden_size=H,
                                    hidden_size_per_layer_input=P,
                                    num_hidden_layers=L,
                                    vocab_size=V,
                                    vocab_size_per_layer_input=V_ple)
    vllm_config = _make_vllm_config(text_config)

    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    # Tokens beyond V_ple-1 are OOV for embed_tokens_per_layer.
    # Without masking, would index out of bounds in lookup. Should NOT crash
    # and should NOT produce NaN.
    input_ids = jnp.array([0, V_ple - 1, V_ple, V - 1], dtype=jnp.int32)
    inputs_embeds = jnp.zeros((4, H), dtype=jnp.float32)

    with mesh:
        out = model.compute_per_layer_inputs(input_ids, inputs_embeds)

    assert out is not None
    assert jnp.all(jnp.isfinite(out)), \
        f"OOV tokens produced non-finite output:\n{out}"

    # Tokens 2 and 3 (both OOV) should be redirected to slot 0 in Track A.
    # Track B uses inputs_embeds (zeros here), so the only difference between
    # token-2 and token-3 outputs would come from track A — and both look up
    # slot 0. So they should match.
    np.testing.assert_allclose(np.asarray(out[2]),
                               np.asarray(out[3]),
                               atol=1e-5)


def test_ple_numpy_oracle_parity(mesh, rng=jax.random.PRNGKey(0)):
    """Numerical parity: production compute_per_layer_inputs vs numpy oracle.

    Implements kb_ple.md §3.1 verbatim in numpy and asserts the production
    JAX output matches at float32 precision.
    """
    P = 4
    L = 2
    H = 16
    V = 32
    text_config = _make_text_config(hidden_size=H,
                                    hidden_size_per_layer_input=P,
                                    num_hidden_layers=L,
                                    vocab_size=V,
                                    vocab_size_per_layer_input=V)
    vllm_config = _make_vllm_config(text_config)

    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    T = 4
    input_ids = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    inputs_embeds = jax.random.normal(jax.random.PRNGKey(7), (T, H),
                                      dtype=jnp.float32)
    with mesh:
        out = model.compute_per_layer_inputs(input_ids, inputs_embeds)

    # Numpy oracle (kb_ple.md §3.1 verbatim).
    et = np.asarray(model.embed_tokens_per_layer.weight[...])  # [V, L*P]
    pm = np.asarray(model.per_layer_model_projection.weight[...])  # [H, L*P]
    norm_w = np.asarray(model.per_layer_projection_norm.weight[...])  # [P]
    eps = 1e-6
    inputs_embeds_np = np.asarray(inputs_embeds)
    input_ids_np = np.asarray(input_ids)

    # Track A.
    embed_scale = float(P)**0.5
    per_layer_embeds = et[input_ids_np]  # [T, L*P]
    per_layer_embeds = per_layer_embeds * embed_scale
    per_layer_embeds = per_layer_embeds.reshape(T, L, P)

    # Track B.
    proj_scale = float(H)**-0.5
    per_layer_projection = inputs_embeds_np @ pm  # [T, L*P]
    per_layer_projection = per_layer_projection * proj_scale
    per_layer_projection = per_layer_projection.reshape(T, L, P)
    rms = np.sqrt(
        np.mean(per_layer_projection**2, axis=-1, keepdims=True) + eps)
    per_layer_projection = per_layer_projection / rms * norm_w

    # Combine.
    input_scale = 1.0 / (2.0**0.5)
    expected = (per_layer_projection + per_layer_embeds) * input_scale

    out_np = np.asarray(out)
    np.testing.assert_allclose(out_np, expected, atol=1e-4, rtol=1e-4)


def test_ple_per_layer_modules_present_in_decoder(mesh,
                                                  rng=jax.random.PRNGKey(0)):
    """Per-layer modules per kb_ple.md §4.2 are constructed when PLE is on."""
    P = 4
    text_config = _make_text_config(hidden_size_per_layer_input=P,
                                    num_hidden_layers=2)
    vllm_config = _make_vllm_config(text_config)
    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    for layer in model.layers:
        if layer is None:  # PPMissingLayer placeholder
            continue
        assert layer.per_layer_input_gate is not None
        assert layer.per_layer_projection is not None
        assert layer.post_per_layer_input_norm is not None
        assert layer.hidden_size_per_layer_input == P


def test_ple_off_decoder_modules_absent(mesh, rng=jax.random.PRNGKey(0)):
    """When PLE is off, decoder layer's PLE modules are None."""
    text_config = _make_text_config(hidden_size_per_layer_input=0,
                                    num_hidden_layers=2)
    vllm_config = _make_vllm_config(text_config)
    with mesh:
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    for layer in model.layers:
        if layer is None:
            continue
        assert layer.per_layer_input_gate is None
        assert layer.per_layer_projection is None
        assert layer.post_per_layer_input_norm is None
        assert layer.hidden_size_per_layer_input == 0
