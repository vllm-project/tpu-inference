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

from unittest.mock import MagicMock

import jax
import pytest
from jax import numpy as jnp
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.gemma4 import Gemma4DecoderLayer
from tpu_inference.models.jax.gemma4_mm import Gemma4ForConditionalGeneration


class TestGemma4ForConditionalGeneration:

    def _run_model_loading_test(self, model_name, pp_rank, pp_world_size,
                                load_format, truncate_layers, rng, mesh,
                                mock_vllm_config):
        """Construct Gemma4ForConditionalGeneration, load weights, run a
        single decoder-layer forward pass, and assert the output exists.

        Args:
          truncate_layers: int or None. When set, the test rewrites
            text_config.num_hidden_layers and vision_config.num_hidden_layers
            to this value so the per-test fixture can iterate quickly
            (used by the 31B / 26B-A4B cases). E-family configs (E2B/E4B)
            must pass None — their PLE weights are aggregated over L
            (`embed_tokens_per_layer.weight` shape `[V_ple, L*P]`), so
            truncating L breaks weight load with a shape mismatch.
        """
        kv_cache_type = "auto"
        vllm_config = mock_vllm_config(model_name, kv_cache_type)
        if truncate_layers is not None:
            # No need to load full model — 31B / 26B-A4B don't have PLE,
            # so truncating L is safe and keeps the per-test cost low.
            vllm_config.model_config.hf_config.text_config.num_hidden_layers = (
                truncate_layers)
            vllm_config.model_config.hf_config.vision_config.num_hidden_layers = (
                truncate_layers)
            vllm_config.load_config.num_layers_to_load_for_test = (
                truncate_layers)
        vllm_config.load_config.load_format = load_format
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.enable_expert_parallel = False

        init_pp_distributed_environment(
            ip="",
            rank=pp_rank,
            world_size=pp_world_size,
            device=jax.devices()[0],
            need_pp=False,
        )
        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        model_dim = vllm_config.model_config.hf_config.text_config.hidden_size
        model_config = vllm_config.model_config
        kv_dtype = jnp.bfloat16

        # As of writing the code, `fused_moe_func` requires (num_tokens * topk) % 16 == 0, so we set seq_len=2 for testing.
        seq_len = 2
        input = [[0.01 * i for i in range(model_dim)] for _ in range(seq_len)]

        with jax.set_mesh(mesh):
            model = Gemma4ForConditionalGeneration(vllm_config, rng, mesh)
            start_layer_idx = model.model.start_layer
            layer_0: Gemma4DecoderLayer = model.model.layers[start_layer_idx]
            num_key_value_heads = layer_0.self_attn.num_kv_heads
            qk_head_dim = layer_0.self_attn.head_dim_original

        # load weights from HF model, monitoring device memory
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            # We didn't tune RPA kernel for Gemma4, so the memory usage is expected
            # to be high.
            # TODO(#2282): Enable memory monitoring after hueristical RPA kernel implemented.
            # Monitor device memory during weight loading to catch
            # regressions.
            # with assert_weight_loading_memory_bounded(
            #         model,
            #         description=f"load_weights({model_name})",
            #         threshold_multiplier=0.3,
            # )
            with set_current_vllm_config(vllm_config):
                loader.load_weights(model, model_config)

        layer_idx_in_model = model.model.start_layer
        jax_layer_0 = model.model.layers[layer_idx_in_model]

        input_tensor_jax = jnp.array(input, dtype=jnp.bfloat16)

        block_size = 16
        num_blocks = 8
        cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                         num_key_value_heads, qk_head_dim,
                                         kv_dtype)

        with jax.set_mesh(mesh):
            _, jax_output, _ = jax_layer_0(
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

    # BVT: per-push CI sets BVT_ONLY=1, narrowing this file to @pytest.mark.bvt
    # cases (see tests/conftest.py). Only the single-host (0,1) pp combo is
    # marked; nightly runs the full pp-shard matrix.
    @pytest.mark.parametrize("model_name", [
        "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B-it",
    ])
    @pytest.mark.parametrize(
        "pp_rank,pp_world_size",
        [pytest.param(0, 1, marks=pytest.mark.bvt), (0, 4), (1, 4), (3, 4)])
    @pytest.mark.parametrize(
        "load_format", ["skip_layers_model_loader_for_test", "jax_dummy"])
    def test_model_loading(
            self,
            model_name,
            pp_rank,
            pp_world_size,
            load_format,
            # following are defined in conftest.py
            rng,
            mesh,
            mock_vllm_config,
            assert_weight_loading_memory_bounded):
        """Tests loading weights from HF model (31B, 26B-A4B). Truncates
        to 4 layers for fast iteration; E-family lives in
        `test_e_family_model_loading` instead."""
        self._run_model_loading_test(
            model_name=model_name,
            pp_rank=pp_rank,
            pp_world_size=pp_world_size,
            load_format=load_format,
            truncate_layers=4,
            rng=rng,
            mesh=mesh,
            mock_vllm_config=mock_vllm_config,
        )

    @pytest.mark.bvt
    @pytest.mark.parametrize("model_name", [
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
    ])
    def test_e_family_model_loading(
        self,
        model_name,
        # following are defined in conftest.py
        rng,
        mesh,
        mock_vllm_config,
    ):
        """E-family (E2B, E4B): PLE makes `embed_tokens_per_layer.weight`
        aggregate over L as shape `[V_ple, L*P]`, so the
        `num_hidden_layers=4` truncation used by `test_model_loading`
        shape-mismatches on weight load. Run at world_size=1 with the
        real L; `jax_dummy` skips real-weight download but still exercises
        the construct + load_weights path."""
        self._run_model_loading_test(
            model_name=model_name,
            pp_rank=0,
            pp_world_size=1,
            load_format="jax_dummy",
            truncate_layers=None,
            rng=rng,
            mesh=mesh,
            mock_vllm_config=mock_vllm_config,
        )
