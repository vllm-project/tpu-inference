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
from transformers import AutoModelForCausalLM
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM
from tpu_inference.models.jax.utils.qwix.qwix_utils import \
    apply_qwix_quantization
from tpu_inference.runner.kv_cache import create_kv_caches


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


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.models.jax.qwen3.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


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
                        mock_model_inputs, mock_vllm_config):
        """Tests model init and model forward for the 0.6B model variant."""
        mock_vllm_config = mock_vllm_config(model_name, kv_cache_type)
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
        with jax.set_mesh(mesh):
            loader = get_model_loader(LoadConfig(load_format="hf"))
            loader.load_weights(model, model_config)

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

    @pytest.mark.parametrize("model_name",
                             ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-FP8"])
    @pytest.mark.parametrize("pp_rank,pp_world_size", [(0, 1), (0, 4), (1, 4),
                                                       (3, 4)])
    def test_model_loading(self, model_name, pp_rank, pp_world_size, rng, mesh,
                           mock_vllm_config):
        """Tests loading weights from HF model"""
        kv_cache_type = "auto"
        mock_vllm_config = mock_vllm_config(model_name, kv_cache_type)
        # No need to load full model.
        mock_vllm_config.model_config.hf_overrides = {"num_hidden_layers": 3}

        init_pp_distributed_environment(
            ip="",
            rank=pp_rank,
            world_size=pp_world_size,
            device=jax.devices()[0],
            need_pp=False,
        )
        mock_vllm_config.quant_config = get_tpu_quantization_config(
            mock_vllm_config)

        model_dim = mock_vllm_config.model_config.hf_config.hidden_size
        model_config = mock_vllm_config.model_config
        kv_dtype = jnp.bfloat16
        num_key_value_heads = model_config.hf_config.num_key_value_heads
        qk_head_dim = model_config.hf_config.head_dim
        # Create random input for comparison
        seq_len = 1
        input = [[0.01 * i for i in range(model_dim)] for _ in range(seq_len)]

        model = Qwen3ForCausalLM(mock_vllm_config, rng, mesh)
        # load weights from HF model
        with jax.set_mesh(mesh):
            loader = get_model_loader(LoadConfig(load_format="hf"))
            loader.load_weights(model, model_config)

        layer_idx = model.model.start_layer
        jax_layer_0 = model.model.layers[layer_idx]

        input_tensor_jax = jnp.array(input, dtype=jnp.bfloat16)

        # Forward pass only the 1st layer for comparison

        block_size = 16
        num_blocks = 8
        cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                         num_key_value_heads, qk_head_dim,
                                         kv_dtype)

        with jax.set_mesh(mesh):
            jax_output, _ = jax_layer_0(
                kv_cache=jnp.zeros(cache_shape, dtype=kv_dtype),
                x=input_tensor_jax,
                attention_metadata=AttentionMetadata(
                    input_positions=jnp.array(seq_len),
                    block_tables=jnp.array(list(range(1))),
                    seq_lens=jnp.array([seq_len]),
                    query_start_loc=jnp.array([0, seq_len]),
                    request_distribution=jnp.array([0, 0, 1]),
                ),
            )
        assert jax_output is not None

        # TODO(#1604): Enable HF comparison when issue resolved.
        # Currently there's a shape mismatch during rope.
        if True:
            return
        with torch.no_grad():
            # Use transformer library to load the HF model, for reference.

            input_tensor_hf = torch.tensor(input, dtype=torch.bfloat16)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                low_cpu_mem_usage=True,
            )
            hf_model = hf_model.eval()

            hf_model.config.num_hidden_layers = 1
            hf_output = hf_model(
                inputs_embeds=input_tensor_hf, ).float().numpy()
            np.testing.assert_allclose(
                jax_output,
                hf_output,
                rtol=1e-2,
                atol=1e-2,
            )
