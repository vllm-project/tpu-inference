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

import jax
import pytest
from jax import numpy as jnp
from vllm.model_executor.model_loader import get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.qwen3_moe import Qwen3MoeForCausalLM


class TestQwen3MoeForCausalLM:

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ])
    @pytest.mark.parametrize("pp_rank,pp_world_size", [(0, 1), (0, 4), (1, 4),
                                                       (3, 4)])
    def test_model_loading(
            self,
            model_name,
            pp_rank,
            pp_world_size,
            # following are defined in conftest.py
            rng,
            mesh,
            mock_vllm_config):
        """Tests loading weights from HF model"""
        kv_cache_type = "auto"
        vllm_config = mock_vllm_config(model_name, kv_cache_type)
        # No need to load full model.
        vllm_config.model_config.hf_config.num_hidden_layers = 4
        vllm_config.load_config.load_format = "skip_layers_model_loader_for_test"
        vllm_config.load_config.num_layers_to_load_for_test = 4

        init_pp_distributed_environment(
            ip="",
            rank=pp_rank,
            world_size=pp_world_size,
            device=jax.devices()[0],
            need_pp=False,
        )
        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        model_dim = vllm_config.model_config.hf_config.hidden_size
        model_config = vllm_config.model_config
        kv_dtype = jnp.bfloat16
        num_key_value_heads = model_config.hf_config.num_key_value_heads
        qk_head_dim = model_config.hf_config.head_dim
        # As of writing the code, `fused_moe_func` requires (num_tokens * topk) % 16 == 0, so we set seq_len=2 for testing.
        seq_len = 2
        input = [[0.01 * i for i in range(model_dim)] for _ in range(seq_len)]

        model = Qwen3MoeForCausalLM(vllm_config, rng, mesh)
        # load weights from HF model
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            loader.load_weights(model, model_config)

        layer_idx = model.model.start_layer
        jax_layer_0 = model.model.layers[layer_idx]

        input_tensor_jax = jnp.array(input, dtype=jnp.bfloat16)

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
                    input_positions=jnp.arange(seq_len),
                    block_tables=jnp.array(list(range(1))),
                    seq_lens=jnp.array([seq_len]),
                    query_start_loc=jnp.array([0, seq_len]),
                    request_distribution=jnp.array([0, 0, 1]),
                ),
            )
        assert jax_output is not None
