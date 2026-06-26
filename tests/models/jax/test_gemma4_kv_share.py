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
"""KV-share structural tests for Gemma-4.

Covers:
  - is_kv_shared_layer correctly computed per layer index + num_kv_shared_layers.
  - kv_sharing_target_layer_name points at the last preceding layer of
    the same attention type (matches vllm-pytorch's `Gemma4Attention`
    target-layer derivation).
  - For shared layers, the source index is correct.
  - compute_kv_share_map() helper edge cases (no shared layers, missing
    layer_types, unmatched type → ValueError).

Forward-pass numerics through shared layers are exercised by the kernel
KV-share unit tests in tests/kernels/ragged_paged_attention_kernel_v3_test.py.
"""
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from transformers import Gemma4TextConfig

from tpu_inference.models.common.kv_share import compute_kv_share_map
from tpu_inference.models.jax.gemma4 import Gemma4Model


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
                      num_hidden_layers,
                      num_kv_shared_layers,
                      layer_types,
                      hidden_size_per_layer_input=0):
    """Synthetic Gemma4TextConfig with KV-share knobs."""
    assert len(layer_types) == num_hidden_layers
    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        layer_types=list(layer_types),
        num_kv_shared_layers=num_kv_shared_layers,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
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


def test_compute_kv_share_map_no_shared_layers():
    """num_kv_shared_layers=0 → empty dict (26B/31B path)."""
    cfg = _make_text_config(num_hidden_layers=4,
                            num_kv_shared_layers=0,
                            layer_types=["full_attention"] * 4)
    assert compute_kv_share_map(cfg) == {}


def test_compute_kv_share_map_e2b_alternating_pattern():
    """num_hidden=4, num_shared=2, alternating types → {2:0, 3:1}."""
    cfg = _make_text_config(num_hidden_layers=4,
                            num_kv_shared_layers=2,
                            layer_types=[
                                "sliding_attention",
                                "full_attention",
                                "sliding_attention",
                                "full_attention",
                            ])
    assert compute_kv_share_map(cfg) == {2: 0, 3: 1}


def test_compute_kv_share_map_picks_last_preceding_same_type():
    """When the source range has multiple same-type layers, picks the LAST
    preceding match (matches vllm-pytorch's
    `len(prev) - 1 - prev[::-1].index(current_type)` formula)."""
    cfg = _make_text_config(
        num_hidden_layers=5,
        num_kv_shared_layers=2,
        layer_types=[
            "sliding_attention",
            "sliding_attention",  # idx 1: last sliding before shared range
            "full_attention",
            "sliding_attention",  # idx 3: shared → src=1, not 0
            "full_attention",  # idx 4: shared → src=2
        ])
    assert compute_kv_share_map(cfg) == {3: 1, 4: 2}


def test_compute_kv_share_map_raises_on_unmatched_type():
    """A shared-layer type with no preceding same-type layer is a config bug
    — surface it loudly rather than silently mis-routing the cache."""
    cfg = _make_text_config(num_hidden_layers=3,
                            num_kv_shared_layers=1,
                            layer_types=[
                                "sliding_attention",
                                "sliding_attention",
                                "full_attention",
                            ])
    with pytest.raises(ValueError, match="full_attention"):
        compute_kv_share_map(cfg)


def test_compute_kv_share_map_missing_layer_types():
    """Defensive: if a (mocked) config lacks layer_types, return {} instead
    of crashing. kv_cache_manager.py needs this for tests with MagicMock."""
    cfg = MagicMock()
    cfg.num_kv_shared_layers = 4
    cfg.num_hidden_layers = 8
    cfg.layer_types = None  # explicit None, not MagicMock
    assert compute_kv_share_map(cfg) == {}


def test_no_assertion_when_kv_share_active(mesh, rng=jax.random.PRNGKey(0)):
    """Lifting verification: model instantiates with num_kv_shared_layers > 0."""
    text_config = _make_text_config(num_hidden_layers=4,
                                    num_kv_shared_layers=2,
                                    layer_types=["full_attention"] * 4)
    vllm_config = _make_vllm_config(text_config)

    with jax.set_mesh(mesh):
        # No AssertionError expected.
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    # Smoke: layers 0-1 non-shared, 2-3 shared.
    assert not model.layers[0].self_attn.is_kv_shared_layer
    assert not model.layers[1].self_attn.is_kv_shared_layer
    assert model.layers[2].self_attn.is_kv_shared_layer
    assert model.layers[3].self_attn.is_kv_shared_layer


def test_kv_share_target_layer_simple(mesh, rng=jax.random.PRNGKey(0)):
    """4 layers all full_attention, 2 shared. Layers 2,3 -> source layer 1."""
    text_config = _make_text_config(num_hidden_layers=4,
                                    num_kv_shared_layers=2,
                                    layer_types=["full_attention"] * 4)
    vllm_config = _make_vllm_config(text_config)
    with jax.set_mesh(mesh):
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    assert model.layers[2].self_attn.kv_sharing_target_layer_name == "layer.1"
    assert model.layers[3].self_attn.kv_sharing_target_layer_name == "layer.1"


def test_kv_share_target_alternating_attention(mesh,
                                               rng=jax.random.PRNGKey(0)):
    """Alternating layer_types: shared layers map to last preceding match.

    6-layer config; last 2 are shared:
      idx 0: sliding
      idx 1: full
      idx 2: sliding   (source for idx 4)
      idx 3: full      (source for idx 5)
      idx 4: sliding (shared) -> last preceding sliding = idx 2
      idx 5: full    (shared) -> last preceding full    = idx 3
    """
    layer_types = [
        "sliding_attention", "full_attention", "sliding_attention",
        "full_attention", "sliding_attention", "full_attention"
    ]
    text_config = _make_text_config(num_hidden_layers=6,
                                    num_kv_shared_layers=2,
                                    layer_types=layer_types)
    vllm_config = _make_vllm_config(text_config)
    with jax.set_mesh(mesh):
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    assert (model.layers[4].self_attn.kv_sharing_target_layer_name
            == "layer.2"), \
        f"got {model.layers[4].self_attn.kv_sharing_target_layer_name}"
    assert (model.layers[5].self_attn.kv_sharing_target_layer_name
            == "layer.3"), \
        f"got {model.layers[5].self_attn.kv_sharing_target_layer_name}"


def test_kv_share_off_when_zero(mesh, rng=jax.random.PRNGKey(0)):
    """num_kv_shared_layers=0: every layer is non-shared, target is None."""
    text_config = _make_text_config(num_hidden_layers=3,
                                    num_kv_shared_layers=0,
                                    layer_types=["full_attention"] * 3)
    vllm_config = _make_vllm_config(text_config)
    with jax.set_mesh(mesh):
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    for layer in model.layers:
        assert not layer.self_attn.is_kv_shared_layer
        assert layer.self_attn.kv_sharing_target_layer_name is None


def test_kv_share_with_ple_active(mesh, rng=jax.random.PRNGKey(0)):
    """Both KV-share AND PLE on: structural (model instantiates, attrs set)."""
    text_config = _make_text_config(num_hidden_layers=4,
                                    num_kv_shared_layers=2,
                                    layer_types=["full_attention"] * 4,
                                    hidden_size_per_layer_input=8)
    vllm_config = _make_vllm_config(text_config)
    with jax.set_mesh(mesh):
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    # KV-share state set on layers 2,3.
    assert model.layers[2].self_attn.is_kv_shared_layer
    assert model.layers[3].self_attn.is_kv_shared_layer
    # PLE state set on all layers.
    for layer in model.layers:
        assert layer.per_layer_input_gate is not None
        assert layer.per_layer_projection is not None
        assert layer.post_per_layer_input_norm is not None
    # Model-level PLE present.
    assert model.embed_tokens_per_layer is not None
    assert model.per_layer_model_projection is not None


def test_double_wide_mlp(mesh, rng=jax.random.PRNGKey(0)):
    """use_double_wide_mlp=True doubles intermediate_size on KV-shared layers."""
    intermediate_size = 64
    text_config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=intermediate_size,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        layer_types=["full_attention"] * 4,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        hidden_size_per_layer_input=0,
        attention_bias=False,
    )
    vllm_config = _make_vllm_config(text_config)
    with jax.set_mesh(mesh):
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    # Non-shared layers (0, 1) keep the regular size.
    # Shared layers (2, 3) get 2x.
    expected_regular = intermediate_size
    expected_double = intermediate_size * 2

    # gate_proj and up_proj are fused into a single gate_up_proj, so the fused
    # weight is [hidden_size, 2 * intermediate_size].
    assert model.layers[0].mlp.gate_up_proj.weight.shape == (32, 2 *
                                                             expected_regular)
    assert model.layers[1].mlp.gate_up_proj.weight.shape == (32, 2 *
                                                             expected_regular)
    assert model.layers[2].mlp.gate_up_proj.weight.shape == (32, 2 *
                                                             expected_double)
    assert model.layers[3].mlp.gate_up_proj.weight.shape == (32, 2 *
                                                             expected_double)


def test_double_wide_mlp_off(mesh, rng=jax.random.PRNGKey(0)):
    """use_double_wide_mlp=False (or absent): all layers use regular size."""
    intermediate_size = 64
    text_config = _make_text_config(num_hidden_layers=4,
                                    num_kv_shared_layers=2,
                                    layer_types=["full_attention"] * 4)
    vllm_config = _make_vllm_config(text_config)
    with jax.set_mesh(mesh):
        model = Gemma4Model(vllm_config, nnx.Rngs(rng), mesh)

    # gate_proj and up_proj are fused: gate_up_proj is [hidden, 2*intermediate].
    for layer in model.layers:
        assert layer.mlp.gate_up_proj.weight.shape == (32,
                                                       2 * intermediate_size)
