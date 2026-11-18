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

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch
from flax import nnx
from jax import numpy as jnp
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.process_weights.moe_weights import \
    get_w13_padding_config
from tpu_inference.layers.jax.moe.moe import JaxRoutedExperts
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.unquantized import (
    MOE_WEIGHTS_STAGED_ON_HOST, UnquantizedConfig)
from tpu_inference.models.jax.gemma4 import Gemma4DecoderLayer, Gemma4MoE
from tpu_inference.models.jax.gemma4_mm import Gemma4ForConditionalGeneration


class TestGemma4MoE:

    @staticmethod
    def _mock_vllm_parallel_config():
        return SimpleNamespace(parallel_config=SimpleNamespace(
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            enable_expert_parallel=False,
        ))

    def test_load_weights_stages_fused_weights_until_postprocess(self, mesh):
        config = SimpleNamespace(num_experts=4,
                                 hidden_size=8,
                                 moe_intermediate_size=16,
                                 top_k_experts=2)
        vllm_cfg = self._mock_vllm_parallel_config()
        with jax.set_mesh(mesh), set_current_vllm_config(vllm_cfg):
            layer = Gemma4MoE(config=config,
                              dtype=jnp.bfloat16,
                              mesh=mesh,
                              rngs=nnx.Rngs(0),
                              quant_config=UnquantizedConfig({}),
                              prefix="layers.0.mlp")

        gate_up_proj = torch.zeros(
            (config.num_experts, 2 * config.moe_intermediate_size,
             config.hidden_size),
            dtype=torch.bfloat16)
        down_proj = torch.zeros(
            (config.num_experts, config.hidden_size,
             config.moe_intermediate_size),
            dtype=torch.bfloat16,
        )

        loaded = layer.load_weights([
            ("gate_up_proj", gate_up_proj),
            ("down_proj", down_proj),
        ])

        assert loaded == {
            "kernel_gating_EDF",
            "kernel_up_proj_EDF",
            "kernel_down_proj_EFD",
        }
        assert layer.kernel_gating_EDF.value.shape == (
            config.num_experts, config.hidden_size,
            config.moe_intermediate_size)
        assert layer.kernel_down_proj_EFD.value.shape == (
            config.num_experts, config.moe_intermediate_size,
            config.hidden_size)
        assert layer.kernel_gating_EDF._weights_to_load[0].shape == (
            config.num_experts, config.moe_intermediate_size,
            config.hidden_size)
        assert layer.kernel_up_proj_EDF._weights_to_load[0].shape == (
            config.num_experts, config.moe_intermediate_size,
            config.hidden_size)
        assert layer.kernel_down_proj_EFD._weights_to_load[0].shape == (
            config.num_experts, config.hidden_size,
            config.moe_intermediate_size)
        assert layer.kernel_gating_EDF.get_metadata(
        )[MOE_WEIGHTS_STAGED_ON_HOST]
        assert layer.kernel_up_proj_EDF.get_metadata(
        )[MOE_WEIGHTS_STAGED_ON_HOST]
        assert layer.kernel_down_proj_EFD.get_metadata(
        )[MOE_WEIGHTS_STAGED_ON_HOST]

        with jax.set_mesh(mesh):
            assert layer.quant_method.process_weights_after_loading(layer)

        assert not hasattr(layer, "kernel_gating_EDF")
        assert not hasattr(layer, "kernel_up_proj_EDF")
        assert hasattr(layer, "kernel_gating_upproj_EDF")
        assert hasattr(layer, "kernel_down_proj_EFD")
        padding_config = get_w13_padding_config(
            config.moe_intermediate_size,
            reorder_size=mesh.shape["model"],
            align=128,
        )
        assert layer.kernel_gating_upproj_EDF.value.shape == (
            config.num_experts,
            config.hidden_size,
            2 * padding_config.padded_intermediate_size,
        )
        assert layer.kernel_down_proj_EFD.value.shape == (
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
        )

    def test_load_weights_rejects_bad_staged_weight_shape(self, mesh):
        config = SimpleNamespace(num_experts=4,
                                 hidden_size=8,
                                 moe_intermediate_size=16,
                                 top_k_experts=2)
        vllm_cfg = self._mock_vllm_parallel_config()
        with jax.set_mesh(mesh), set_current_vllm_config(vllm_cfg):
            layer = Gemma4MoE(config=config,
                              dtype=jnp.bfloat16,
                              mesh=mesh,
                              rngs=nnx.Rngs(0),
                              quant_config=UnquantizedConfig({}),
                              prefix="layers.0.mlp")

        bad_gate_up_proj = torch.zeros(
            (config.num_experts, 2 * config.moe_intermediate_size,
             config.hidden_size + 1),
            dtype=torch.bfloat16)

        with pytest.raises(ValueError,
                           match="Unexpected Gemma4 MoE weight shape"):
            layer.load_weights([("gate_up_proj", bad_gate_up_proj)])

    def test_load_weights_fused_matches_per_expert_weights(self, mesh):
        config = SimpleNamespace(num_experts=4,
                                 hidden_size=8,
                                 moe_intermediate_size=16,
                                 top_k_experts=2)
        vllm_cfg = self._mock_vllm_parallel_config()
        with jax.set_mesh(mesh), set_current_vllm_config(vllm_cfg):
            fused_layer = Gemma4MoE(config=config,
                                    dtype=jnp.bfloat16,
                                    mesh=mesh,
                                    rngs=nnx.Rngs(0),
                                    quant_config=UnquantizedConfig({}),
                                    prefix="layers.0.mlp")
            unfused_layer = Gemma4MoE(config=config,
                                      dtype=jnp.bfloat16,
                                      mesh=mesh,
                                      rngs=nnx.Rngs(0),
                                      quant_config=UnquantizedConfig({}),
                                      prefix="layers.0.mlp")

        torch.manual_seed(0)
        gate_up_proj = torch.randn(
            (config.num_experts, 2 * config.moe_intermediate_size,
             config.hidden_size),
            dtype=torch.bfloat16)
        down_proj = torch.randn((config.num_experts, config.hidden_size,
                                 config.moe_intermediate_size),
                                dtype=torch.bfloat16)

        per_expert_weights = []
        f_dim = config.moe_intermediate_size
        for i in range(config.num_experts):
            per_expert_weights.append(
                (f"experts.{i}.gate_proj.weight", gate_up_proj[i, :f_dim, :]))
            per_expert_weights.append(
                (f"experts.{i}.up_proj.weight", gate_up_proj[i, f_dim:, :]))
            per_expert_weights.append(
                (f"experts.{i}.down_proj.weight", down_proj[i]))

        with jax.set_mesh(mesh):
            assert fused_layer.load_weights([
                ("gate_up_proj", gate_up_proj),
                ("down_proj", down_proj),
            ]) == {
                "kernel_gating_EDF",
                "kernel_up_proj_EDF",
                "kernel_down_proj_EFD",
            }
            assert unfused_layer.load_weights(per_expert_weights) == {
                "kernel_gating_EDF",
                "kernel_up_proj_EDF",
                "kernel_down_proj_EFD",
            }
            assert fused_layer.quant_method.process_weights_after_loading(
                fused_layer)
            assert unfused_layer.quant_method.process_weights_after_loading(
                unfused_layer)

        np.testing.assert_array_equal(
            np.asarray(fused_layer.kernel_gating_upproj_EDF.value),
            np.asarray(unfused_layer.kernel_gating_upproj_EDF.value))
        np.testing.assert_array_equal(
            np.asarray(fused_layer.kernel_down_proj_EFD.value),
            np.asarray(unfused_layer.kernel_down_proj_EFD.value))


class TestGemma4ForConditionalGeneration:

    @pytest.mark.parametrize("enabled", [False, True])
    def test_moe_respects_return_routed_experts_config(self, enabled):
        text_config = MagicMock()
        text_config.num_experts = 128
        text_config.hidden_size = 2816
        text_config.moe_intermediate_size = 1408
        text_config.top_k_experts = 8

        with patch.object(JaxRoutedExperts, "__init__",
                          return_value=None) as routed_experts_init:
            Gemma4MoE(
                config=text_config,
                dtype=jnp.bfloat16,
                mesh=MagicMock(),
                rngs=MagicMock(),
                quant_config=MagicMock(),
                enable_return_routed_experts=enabled,
            )

        assert (routed_experts_init.call_args.
                kwargs["enable_return_routed_experts"] is enabled)

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
        gc.collect()
        jax.clear_caches()
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
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
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

        with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
            model = Gemma4ForConditionalGeneration(vllm_config, rng, mesh)
            start_layer_idx = model.model.language_model.start_layer
            layer_0: Gemma4DecoderLayer = model.model.language_model.layers[
                start_layer_idx]
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

        layer_idx_in_model = model.model.language_model.start_layer
        jax_layer_0 = model.model.language_model.layers[layer_idx_in_model]

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
        del model
        gc.collect()
        jax.clear_caches()

    @pytest.mark.parametrize("model_name", [
        "google/gemma-4-31B-it",
        "google/gemma-4-26B-A4B-it",
        "RedHatAI/gemma-4-31B-it-FP8-Dynamic",
        "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic",
    ])
    @pytest.mark.parametrize("pp_rank,pp_world_size", [(0, 1), (0, 4), (1, 4),
                                                       (3, 4)])
    @pytest.mark.parametrize("load_format",
                             ["skip_layers_model_loader_for_test"])
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
