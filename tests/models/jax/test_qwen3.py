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

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM
from tpu_inference.models.jax.utils.qwix.qwix_utils import \
    apply_qwix_quantization
from tpu_inference.runner.kv_cache import create_kv_caches


class MockVllmConfig:

    def __init__(self, model: str, kv_cache_dtype: str):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None
        self.additional_config = {}


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def mock_model_inputs():
    num_tokens = 8
    num_reqs = 1
    max_num_blocks_per_req = 4
    input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
    positions = jnp.ones((num_tokens, ), dtype=jnp.int32)
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.ones((num_reqs, ), dtype=jnp.int32)
    query_start_loc = jnp.ones((num_reqs + 1, ), dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 0], dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    indices_do_sample = jnp.ones((num_reqs, ), dtype=jnp.int32)

    return (input_ids, attention_metadata, indices_do_sample)


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestQwen3ForCausalLM:

    @pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B"])
    @pytest.mark.parametrize("kv_cache_type", ["auto", "fp8"])
    @pytest.mark.parametrize("qwix_rules", [
        None,
        [{
            "module_path": ".*",
            "weight_qtype": "float8_e4m3fn",
            "act_qtype": "float8_e4m3fn"
        }]
    ])
    def test_qwen3_600M(self, model_name, kv_cache_type, qwix_rules, rng, mesh,
                        mock_model_inputs):
        """Tests model init and model forward for the 0.6B model variant."""
        mock_vllm_config = MockVllmConfig(model_name, kv_cache_type)
        if qwix_rules:
            mock_vllm_config.additional_config["quanntization"] = dict(
                qwix=dict(rules=qwix_rules))

        # Test model init
        model = Qwen3ForCausalLM(mock_vllm_config, rng, mesh)

        model_config = mock_vllm_config.model_config
        hf_config = model_config.hf_config

        assert model.mesh.shape == {
            "data": 1,
            "attn_dp": 1,
            "expert": 1,
            "model": 1
        }

        layers = model.model.layers
        assert len(layers) == hf_config.num_hidden_layers

        attn = layers[0].self_attn
        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        rope_theta = hf_config.rope_theta
        original_head_dim = hf_config.head_dim
        head_dim = 128
        intermediate_size = hf_config.intermediate_size

        assert attn.hidden_size == hidden_size
        assert attn.num_heads == num_heads
        assert attn.num_kv_heads == num_kv_heads
        assert attn.rope_theta == rope_theta
        assert attn.head_dim_original == original_head_dim
        assert attn.head_dim == head_dim
        assert attn.q_proj.weight.shape == (hidden_size, num_heads, head_dim)
        assert attn.k_proj.weight.shape == (hidden_size, num_kv_heads,
                                            head_dim)
        assert attn.v_proj.weight.shape == (hidden_size, num_kv_heads,
                                            head_dim)
        assert attn.o_proj.weight.shape == (num_heads, head_dim, hidden_size)

        mlp = layers[0].mlp
        assert mlp.gate_proj.weight.shape == (hidden_size, intermediate_size)
        assert mlp.up_proj.weight.shape == (hidden_size, intermediate_size)
        assert mlp.down_proj.weight.shape == (intermediate_size, hidden_size)

        # Test model load
        model.load_weights(rng)

        # Apply qwix quantization, no-op if rules are not given.
        model = apply_qwix_quantization(mock_vllm_config,
                                        model,
                                        rng,
                                        mesh,
                                        apply_to_abstract_model=False)

        # Test model forward
        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=32,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            cache_dtype=jnp.float8_e4m3fn
            if mock_vllm_config.cache_config.cache_dtype == "fp8" else
            jnp.bfloat16)
        # 1 seq with 16 tokens
        input_ids, attention_metadata, indices_do_sample = mock_model_inputs
        kv_caches, hidden_states, aux_hidden_states = model(
            kv_caches, input_ids, attention_metadata)
        assert hidden_states.shape == (8, hidden_size)
        assert len(aux_hidden_states) == 0

        hidden_states = hidden_states[indices_do_sample]
        assert hidden_states.shape == (1, hidden_size)

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (1, hf_config.vocab_size)

    def test_loading(self, rng, mesh):
        import torch
        from vllm.model_executor.model_loader import (LoadConfig,
                                                      get_model_loader)
        model_name = "Qwen/Qwen3-0.6B"
        kv_cache_type = "auto"

        with jax.set_mesh(mesh):
            mock_vllm_config = MockVllmConfig(model_name, kv_cache_type)

            model_dim = mock_vllm_config.model_config.hf_config.hidden_size
            model_config = mock_vllm_config.model_config
            hf_config = model_config.hf_config

            # Test model init
            model = Qwen3ForCausalLM(mock_vllm_config, rng, mesh)

            layers = model.model.layers
            assert len(layers) == hf_config.num_hidden_layers

            # Test model load
            loader = get_model_loader(LoadConfig(load_format="hf"))
            loader.load_weights(model, model_config)

            # Compare each weight in 1st layer with HF model weights.
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                low_cpu_mem_usage=True,
            )
            hf_model = hf_model.eval()

            jax_layer_0 = model.model.layers[0]
            hf_layer_0 = hf_model.model.layers[0]

            # self_attn weights
            q_weight_hf = hf_layer_0.self_attn.q_proj.weight.data.float(
            ).numpy()
            q_weight_jax = jax_layer_0.self_attn.q_proj.weight
            np.testing.assert_allclose(q_weight_hf.T,
                                       q_weight_jax.reshape(
                                           q_weight_hf.T.shape),
                                       atol=1e-2,
                                       rtol=1e-2)
            input = [[0.1 * i for i in range(model_dim)]]
            input_hf = torch.tensor(input, dtype=torch.bfloat16)
            input_jax = jnp.array(input, dtype=jnp.bfloat16)
            after_q_proj_hf = hf_layer_0.self_attn.q_proj(
                input_hf).detach().float().numpy()
            after_q_proj_jax = jax_layer_0.self_attn.q_proj(input_jax)
            np.testing.assert_allclose(after_q_proj_hf,
                                       after_q_proj_jax,
                                       atol=1e-2,
                                       rtol=1e-2)

            k_weight_hf = hf_layer_0.self_attn.k_proj.weight.data.float(
            ).numpy()
            k_weight_jax = jax_layer_0.self_attn.k_proj.weight
            np.testing.assert_allclose(k_weight_hf.T,
                                       k_weight_jax.reshape(
                                           k_weight_hf.T.shape),
                                       atol=1e-2,
                                       rtol=1e-2)
            v_weight_hf = hf_layer_0.self_attn.v_proj.weight.data.float(
            ).numpy()
            v_weight_jax = jax_layer_0.self_attn.v_proj.weight
            np.testing.assert_allclose(v_weight_hf.T,
                                       v_weight_jax.reshape(
                                           v_weight_hf.T.shape),
                                       atol=1e-2,
                                       rtol=1e-2)
            o_weight_hf = hf_layer_0.self_attn.o_proj.weight.data.float(
            ).numpy()
            o_weight_jax = jax_layer_0.self_attn.o_proj.weight
            np.testing.assert_allclose(o_weight_hf.T,
                                       o_weight_jax.reshape(
                                           o_weight_hf.T.shape),
                                       atol=1e-2,
                                       rtol=1e-2)

            # mlp weights
            gate_proj_hf = hf_layer_0.mlp.gate_proj.weight.data.float().numpy()
            gate_proj_jax = jax_layer_0.mlp.gate_proj.weight
            np.testing.assert_allclose(gate_proj_hf.T,
                                       gate_proj_jax,
                                       atol=1e-2,
                                       rtol=1e-2)
            up_proj_hf = hf_layer_0.mlp.up_proj.weight.data.float().numpy()
            up_proj_jax = jax_layer_0.mlp.up_proj.weight
            np.testing.assert_allclose(up_proj_hf.T,
                                       up_proj_jax,
                                       atol=1e-2,
                                       rtol=1e-2)
            down_proj_hf = hf_layer_0.mlp.down_proj.weight.data.float().numpy()
            down_proj_jax = jax_layer_0.mlp.down_proj.weight
            np.testing.assert_allclose(down_proj_hf.T,
                                       down_proj_jax,
                                       atol=1e-2,
                                       rtol=1e-2)

            # layernorm weights
            input_layernorm_hf = hf_layer_0.input_layernorm.weight.data.float(
            ).numpy()
            input_layernorm_jax = jax_layer_0.input_layernorm.weight
            np.testing.assert_allclose(input_layernorm_hf,
                                       input_layernorm_jax,
                                       atol=1e-2,
                                       rtol=1e-2)
            post_attention_layernorm_hf = hf_layer_0.post_attention_layernorm.weight.data.float(
            ).numpy()
            post_attention_layernorm_jax = jax_layer_0.post_attention_layernorm.weight
            np.testing.assert_allclose(post_attention_layernorm_hf,
                                       post_attention_layernorm_jax,
                                       atol=1e-2,
                                       rtol=1e-2)
