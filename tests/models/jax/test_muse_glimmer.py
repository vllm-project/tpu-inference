# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from transformers import PretrainedConfig

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.models.common.model_loader import _get_model_architecture
from tpu_inference.models.jax.muse_glimmer import (
    MuseGlimmerForConditionalGeneration, MuseGlimmerRMSNorm,
    map_muse_glimmer_weight_name, muse_glimmer_dflash_target_layer_ids,
    muse_glimmer_layer_uses_rope, muse_glimmer_query_prescale)


def test_registry_supports_both_checkpoint_architecture_names():
    for architecture in ("MuseGlimmerForConditionalGeneration",
                         "MuseGlimmerForCausalLM"):
        config = PretrainedConfig(architectures=[architecture])
        assert (_get_model_architecture(config)
                is MuseGlimmerForConditionalGeneration)


def test_muse_glimmer_dflash_target_layers_use_top_level_draft_config():
    draft_config = SimpleNamespace(target_layer_ids=[1, 13, 25, 37, 49])
    speculative_config = SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(hf_config=draft_config),
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)
    assert muse_glimmer_dflash_target_layer_ids(vllm_config) == (1, 13, 25, 37,
                                                                 49)


def test_query_prescale_normalizes_native_and_modular_configs():
    native = SimpleNamespace(head_dim=128, qk_scale_factor=43.7840518911)
    modular = SimpleNamespace(head_dim=128,
                              qk_scale_factor=43.7840518911 / np.sqrt(128))
    assert np.isclose(muse_glimmer_query_prescale(native),
                      muse_glimmer_query_prescale(modular))


def test_irope_layer_selection():
    config = SimpleNamespace(
        no_rope_layers=[0, 1],
        layer_types=["full_attention", "sliding_attention"])
    assert not muse_glimmer_layer_uses_rope(config, 0)
    assert muse_glimmer_layer_uses_rope(config, 1)


def test_rms_norm_uses_fp32_math_and_weight_offset():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model", ))
    with jax.set_mesh(mesh):
        norm = MuseGlimmerRMSNorm(2,
                                  epsilon=1e-5,
                                  dtype=jnp.bfloat16,
                                  rngs=nnx.Rngs(jax.random.key(0)),
                                  weight_offset=1.0)
    norm.weight.set_value(jnp.array([0.5, -0.25], dtype=jnp.bfloat16))
    inputs = jnp.array([[1.0, 2.0]], dtype=jnp.bfloat16)
    output = np.asarray(norm(inputs), dtype=np.float32)
    inputs_f32 = np.asarray(inputs, dtype=np.float32)
    expected = inputs_f32 / np.sqrt(
        np.mean(inputs_f32**2, axis=-1, keepdims=True) + 1e-5)
    expected *= np.array([1.5, 0.75], dtype=np.float32)
    np.testing.assert_allclose(output, expected, rtol=8e-3, atol=8e-3)


def test_weight_name_mapping_canonical_and_legacy():
    assert map_muse_glimmer_weight_name(
        "model.language_model.layers.0.self_attn.gate_proj.weight"
    ) == "model.layers.0.self_attn.output_gate_proj.weight"
    assert map_muse_glimmer_weight_name(
        "model.layers.0.post_attention_layernorm.weight"
    ) == "model.layers.0.pre_feedforward_layernorm.weight"
    assert map_muse_glimmer_weight_name(
        "model.layers.0.post_attn_norm.weight"
    ) == "model.layers.0.post_attention_layernorm.weight"
    assert map_muse_glimmer_weight_name(
        "model.layers.0.post_ffn_norm.weight"
    ) == "model.layers.0.post_feedforward_layernorm.weight"
    assert map_muse_glimmer_weight_name(
        "model.vision_tower.layers.0.weight") is None


def test_tiny_model_construction():
    text_config = SimpleNamespace(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="silu",
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        qk_scale_factor=2.0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        rope_theta=500000.0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 500000.0
        },
        output_multiplier=0.2,
        final_logit_softcapping=20.0,
    )
    model_config = MagicMock()
    model_config.hf_config = SimpleNamespace(text_config=text_config,
                                             image_token_id=62,
                                             video_token_id=63)
    model_config.dtype = jnp.bfloat16
    vllm_config = MagicMock()
    vllm_config.model_config = model_config
    vllm_config.quant_config = None
    vllm_config.cache_config.cache_dtype = "auto"

    devices = np.asarray(jax.devices()[:1]).reshape((1, 1, 1, 1))
    mesh = Mesh(devices, ("data", "attn_dp", "expert", "model"))
    init_pp_distributed_environment(ip="",
                                    rank=0,
                                    world_size=1,
                                    device=jax.devices()[0],
                                    need_pp=False)
    with jax.set_mesh(mesh):
        model = MuseGlimmerForConditionalGeneration(vllm_config,
                                                    jax.random.key(0), mesh)
    assert len(model.model.layers) == 2
    assert model.model.layers[0].self_attn.sliding_window == 16
    assert model.model.layers[1].self_attn.sliding_window is None
    assert model.lm_head.weight.get_value().shape == (32, 64)
    assert model.embed_input_ids(jnp.array([1, 2])).shape == (2, 32)
    multimodal = jnp.ones((1, 32), dtype=jnp.bfloat16)
    merged = model.embed_input_ids(jnp.array([1, 62]), multimodal)
    np.testing.assert_array_equal(np.asarray(merged[1]),
                                  np.ones((32, ), dtype=np.float32))
