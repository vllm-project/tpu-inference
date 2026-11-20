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
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import Glm4Config
from transformers import Glm4ForCausalLM as HfGlm4ForCausalLM
from vllm.config import ModelConfig, set_current_vllm_config
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.glm4 import Glm4ForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches


class MockVllmConfig:

    def __init__(self, model: str, kv_cache_dtype: str = "auto"):
        self.model_config = ModelConfig(model=model, skip_tokenizer_init=True)
        self.model_config.dtype = jnp.float32
        self.load_config = LoadConfig()
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None
        self.additional_config = {}
        self.parallel_config = None
        self.speculative_config = None


@pytest.fixture(scope="module")
def tiny_glm4_checkpoint(tmp_path_factory):
    """A tiny random-init GLM-4 dense checkpoint saved by HF transformers."""
    config = Glm4Config(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        partial_rotary_factor=0.5,
        attention_bias=True,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
    )
    torch.manual_seed(0)
    hf_model = HfGlm4ForCausalLM(config).eval()
    # Make the norms non-trivial so a missing/misplaced norm is detected.
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if "norm" in name or name.endswith("bias"):
                param.add_(0.1 * torch.randn_like(param))
    path = tmp_path_factory.mktemp("tiny_glm4")
    hf_model.save_pretrained(path)
    return str(path), hf_model


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.models.jax.glm4.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


@pytest.fixture(autouse=True)
def pp_env():
    init_pp_distributed_environment(
        ip="",
        rank=0,
        world_size=1,
        device=jax.devices()[0],
        need_pp=False,
    )


def _load(model_dir, rng, mesh):
    vllm_config = MockVllmConfig(model_dir)
    with jax.set_mesh(mesh):
        model = Glm4ForCausalLM(vllm_config, rng, mesh)
    with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
        loader = get_model_loader(LoadConfig(load_format="auto"))
        loader.load_weights(model, vllm_config.model_config)
    return vllm_config, model


class TestGlm4ForCausalLM:

    def test_init_and_load(self, tiny_glm4_checkpoint, rng, mesh):
        model_dir, hf_model = tiny_glm4_checkpoint
        vllm_config, model = _load(model_dir, rng, mesh)
        hf_config = vllm_config.model_config.hf_config

        layers = model.model.layers
        assert len(layers) == hf_config.num_hidden_layers
        layer = layers[0]
        attn = layer.self_attn
        hidden = hf_config.hidden_size
        assert attn.rotary_dim == hf_config.head_dim // 2
        assert attn.q_proj.weight.shape == (hidden, 4, 128)
        assert attn.k_proj.weight.shape == (hidden, 2, 128)
        assert attn.q_proj.bias.shape == (4, 128)
        assert attn.o_proj.weight.shape == (4, 128, hidden)
        assert layer.mlp.gate_up_proj.weight.shape == (
            hidden, 2 * hf_config.intermediate_size)
        for norm in ("input_layernorm", "post_self_attn_layernorm",
                     "post_attention_layernorm", "post_mlp_layernorm"):
            assert hasattr(layer, norm)

        # The fused checkpoint gate_up_proj must land unchanged in the
        # merged linear (gate first, then up).
        hf_layer = hf_model.model.layers[0]
        np.testing.assert_allclose(
            np.asarray(layer.mlp.gate_up_proj.weight.value),
            hf_layer.mlp.gate_up_proj.weight.detach().numpy().T,
            rtol=0,
            atol=0)
        np.testing.assert_allclose(
            np.asarray(layer.post_mlp_layernorm.weight.value),
            hf_layer.post_mlp_layernorm.weight.detach().numpy(),
            rtol=0,
            atol=0)

    def test_logits_match_hf(self, tiny_glm4_checkpoint, rng, mesh):
        """Prefill a single sequence and compare logits with HF."""
        model_dir, hf_model = tiny_glm4_checkpoint
        vllm_config, model = _load(model_dir, rng, mesh)
        hf_config = vllm_config.model_config.hf_config

        seq_len = 16
        token_ids = np.random.RandomState(0).randint(0, 512, size=seq_len)
        block_size = 16
        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_size=hf_config.head_dim,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            cache_dtype=jnp.float32)
        attention_metadata = AttentionMetadata(
            input_positions=jnp.arange(seq_len, dtype=jnp.int32),
            block_tables=jnp.arange(4, dtype=jnp.int32),
            seq_lens=jnp.array([seq_len], dtype=jnp.int32),
            query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
            request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        )
        with jax.set_mesh(mesh):
            _, hidden_states, aux, _ = model(
                kv_caches, jnp.asarray(token_ids, dtype=jnp.int32),
                attention_metadata)
            logits = np.asarray(model.compute_logits(hidden_states))
        assert aux == []
        assert logits.shape == (seq_len, hf_config.vocab_size)

        with torch.no_grad():
            hf_logits = hf_model(
                torch.tensor(token_ids)[None]).logits[0].float().numpy()
        np.testing.assert_allclose(logits, hf_logits, rtol=2e-3, atol=2e-3)
