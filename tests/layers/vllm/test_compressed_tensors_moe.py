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

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchax
from jax.sharding import PartitionSpec
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import FusedMoE

# yapf: disable
from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe import \
    VllmCompressedTensorsW8A8Fp8MoEMethod

# yapf: enable

P = PartitionSpec

MODEL = 'BCCard/Qwen3-30B-A3B-FP8-Dynamic'


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.distributed.jax_parallel_state.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # RowParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )

    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)


def initialize_layer_weights(layer: torch.nn.Module):
    torch.manual_seed(42)
    assert isinstance(layer, FusedMoE)

    e = layer.num_experts
    h = layer.hidden_size
    i = layer.intermediate_size_per_partition

    # 1. Initialize w13 (gate and up projections) -> Shape: (E, 2*I, H)
    w13_bf16 = torch.rand((e, 2 * i, h), dtype=torch.bfloat16) / 10
    w13_q, w13_s = test_utils.ref_quantize_fp8(w13_bf16,
                                               torch.float8_e4m3fn,
                                               axis=2)

    assert layer.w13_weight.data.shape == w13_q.shape
    assert layer.w13_weight_scale.data.shape == w13_s.shape

    layer.w13_weight.data = w13_q
    layer.w13_weight_scale.data = w13_s

    # 2. Initialize w2 (down_proj) -> Shape: (E, H, I)
    w2_bf16 = torch.rand((e, h, i), dtype=torch.bfloat16) / 10
    w2_q, w2_s = test_utils.ref_quantize_fp8(w2_bf16,
                                             torch.float8_e4m3fn,
                                             axis=2)

    assert layer.w2_weight.data.shape == w2_q.shape
    assert layer.w2_weight_scale.data.shape == w2_s.shape

    layer.w2_weight.data = w2_q
    layer.w2_weight_scale.data = w2_s

    # Handle optional MoE biases
    if hasattr(layer, 'w13_bias') and layer.w13_bias is not None:
        layer.w13_bias.data = torch.rand_like(layer.w13_bias.data)
    if hasattr(layer, 'w2_bias') and layer.w2_bias is not None:
        layer.w2_bias.data = torch.rand_like(layer.w2_bias.data)


@pytest.mark.parametrize(
    "mesh", [test_utils.get_spmd_mesh(1),
             test_utils.get_spmd_mesh(2)])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("use_ep", [True, False])
def test_fused_moe_method(mesh, num_tokens, intermediate_size, hidden_size,
                          num_experts, topk, use_ep):
    engine_args = EngineArgs(
        model=MODEL,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = False

    # Call tpu_inference code
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(num_experts=num_experts,
                         top_k=topk,
                         hidden_size=hidden_size,
                         intermediate_size=intermediate_size)
    weight_quant = quant_config.target_scheme_map['Linear']['weights']
    input_quant = quant_config.target_scheme_map['Linear']['input_activations']
    moe = quant_config.get_moe_config(layer)
    method = VllmCompressedTensorsW8A8Fp8MoEMethod(weight_quant, input_quant,
                                                   moe, mesh)
    method.create_weights(layer,
                          num_experts,
                          hidden_size,
                          intermediate_size,
                          params_dtype=torch.float8_e4m3fn)

    initialize_layer_weights(layer)
    method.process_weights_after_loading(layer)

    def unquantize_weight_for_ref(weight, scale):
        return (weight.to(torch.float32) * scale.squeeze(1)).transpose(
            1, 2).cpu()

    seqlen = num_tokens
    with torchax.default_env():
        x = torch.ones((seqlen, hidden_size), dtype=torch.bfloat16).to('jax')
        router_logits = torch.randn((seqlen, num_experts),
                                    dtype=torch.bfloat16).to('jax')
        result = method.apply_monolithic(layer, x, router_logits)
        expected = test_utils.ref_moe(
            x.to(torch.float32).cpu(),
            router_logits.to(torch.float32).cpu(),
            unquantize_weight_for_ref(layer.w13_weight,
                                      layer.w13_weight_scale),
            unquantize_weight_for_ref(layer.w2_weight, layer.w2_weight_scale),
            w1_bias=None,
            w2_bias=None,
            top_k=topk,
            renormalize=True,
            activation="silu")
        assert np.allclose(result, expected, atol=0.05, rtol=0.05)
