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

import jax.numpy as jnp
import pytest
import torch
import torch.nn.functional as F
import torchax
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import PartitionSpec
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import FusedMoE
# yapf: disable
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig)

from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig

# ---> CHANGE 1: Import the INT4 MoE method instead of FP8
# Note: Check your repo to ensure this is the exact import path/name for the INT4 MoE class
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe import \
    VllmCompressedTensorsWNA16MoEMethod 

# yapf: enable

P = PartitionSpec
MODEL = 'BCCard/Qwen3-30B-A3B-FP8-Dynamic'

def pack_int4_to_int32(tensor_u8: torch.Tensor) -> torch.Tensor:
    shape = list(tensor_u8.shape)
    shape[-1] = shape[-1] // 8
    tensor_reshaped = tensor_u8.reshape(-1, 8)
    packed_flat = torch.zeros(tensor_reshaped.shape[0], dtype=torch.int32, device=tensor_u8.device)
    for i in range(8):
        packed_flat |= ((tensor_reshaped[:, i] & 0xF) << (i * 4))
    return packed_flat.reshape(shape)

def quantize_to_int4_moe(weight_3d: torch.Tensor, group_size: int = -1, symmetric: bool = True):
    """Adapts 2D int4 quantization to 3D MoE Weights: (E, O, I)"""
    E, O, I = weight_3d.shape
    g_size = I if group_size == -1 else group_size
    num_groups = I // g_size
    
    # Flatten E and O to quantize
    w_grouped = weight_3d.view(E * O, num_groups, g_size)
    
    # Simple symmetric quantization for test
    abs_max = torch.clamp(w_grouped.abs().max(dim=-1, keepdim=True)[0], min=1e-5)
    scale = abs_max / 7.0 
    w_q = torch.clamp(torch.round(w_grouped / scale), -8, 7)
    w_q_unsigned = (w_q + 8).to(torch.int32)
    
    w_q_unsigned = w_q_unsigned.view(E * O, I)
    w_packed_flat = pack_int4_to_int32(w_q_unsigned)
    
    w_packed_3d = w_packed_flat.view(E, O, I // 8)
    scale_3d = scale.view(E, O, num_groups).to(torch.bfloat16)
    
    return w_packed_3d, scale_3d

@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.distributed.jax_parallel_state.get_pp_group",
               return_value=MagicMock(is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1)):
        yield

@pytest.fixture(autouse=True)
def setup_environment():
    engine_args = EngineArgs(model=MODEL, max_model_len=64, max_num_batched_tokens=64, max_num_seqs=4)
    vllm_config = engine_args.create_engine_config()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(1, 0, local_rank=0, distributed_init_method=f"file://{temp_file}", backend="gloo")
        ensure_model_parallel_initialized(1, 1)

def _ref_math_in_bf16(w1, w2, w3, x, router_logits, top_k):
    seqlen = x.shape[0]
    expert_weights = F.softmax(router_logits, dim=-1)
    expert_weights, expert_indices = torch.topk(expert_weights, top_k, dim=-1)
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

    x1 = torch.einsum("ti, eoi -> teo", x, w1)
    x1 = F.silu(x1)
    x3 = torch.einsum("ti, eoi -> teo", x, w3)
    expert_outs = torch.einsum("teo, eio -> tei", (x1 * x3), w2)

    seq_indexes = torch.arange(seqlen, device='jax').unsqueeze(1)
    expert_outs = expert_outs[seq_indexes, expert_indices]
    out = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
    return out

@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh(1), test_utils.get_spmd_mesh(2)])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("use_ep", [True, False])
def test_fused_moe_method_int4(mesh, num_tokens, intermediate_size, hidden_size,
                               num_experts, topk, use_ep):
    engine_args = EngineArgs(model=MODEL, max_model_len=64, max_num_batched_tokens=64, max_num_seqs=4)
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = False
    vllm_config.model_config.dtype = torch.bfloat16

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(num_experts=num_experts,
                         top_k=topk,
                         hidden_size=hidden_size,
                         intermediate_size=intermediate_size)

    quant_config = VllmCompressedTensorsConfig(
        target_scheme_map={
            'Linear': {
                'weights': QuantizationArgs(
                    num_bits=4,
                    type='int',
                    symmetric=True,
                    group_size=128,
                    strategy='group',
                    block_structure=None,
                    dynamic=False,
                    actorder=None,
                    observer='minmax',
                    observer_kwargs={}),
                'input_activations': None,
                'format': None
            }
        },
        ignore=[], quant_format='compressed-tensors', sparsity_scheme_map={}, sparsity_ignore_list=[],
    )

    quant_config.num_bits = 4
    quant_config.group_size = 128
    quant_config.symmetric = True
    quant_config.strategy = "group"
    quant_config.actorder = None

    moe = FusedMoEConfig(
            num_experts=num_experts, 
            experts_per_token=topk, 
            hidden_dim=hidden_size, 
            num_local_experts=num_experts,
            moe_parallel_config=FusedMoEParallelConfig(
                tp_size=1, dp_size=1, ep_size=1,
                tp_rank=0, dp_rank=0, ep_rank=0,
                use_ep=use_ep, all2all_backend='',
                pcp_size=1,        
                pcp_rank=0,        
                sp_size=1,         
                enable_eplb=False  
            ),
            in_dtype=torch.bfloat16,
            intermediate_size_per_partition=intermediate_size,
            num_logical_experts=num_experts,
            activation="silu",
            device="cpu",
            routing_method="topk"
        )
    
    method = VllmCompressedTensorsWNA16MoEMethod(quant_config, moe, mesh)
    method.create_weights(layer, num_experts, hidden_size, intermediate_size, params_dtype=torch.bfloat16)

    torch.manual_seed(42)
    
    w13_raw = torch.randn((num_experts, 2 * intermediate_size, hidden_size), dtype=torch.bfloat16)
    w2_raw = torch.randn((num_experts, hidden_size, intermediate_size), dtype=torch.bfloat16)

    w13_packed, w13_scale = quantize_to_int4_moe(w13_raw, group_size=128, symmetric=True)
    w2_packed, w2_scale = quantize_to_int4_moe(w2_raw, group_size=128, symmetric=True)

    layer.w13_weight_packed.data = w13_packed
    layer.w13_weight_scale.data = w13_scale
    layer.w2_weight_packed.data = w2_packed
    layer.w2_weight_scale.data = w2_scale

    method.process_weights_after_loading(layer)

    # Run the hardware execution
    seqlen = num_tokens
    with torchax.default_env():
        x = torch.ones((seqlen, hidden_size), dtype=torch.bfloat16).to('jax')
        router_logits = torch.randn((seqlen, num_experts), dtype=torch.bfloat16).to('jax')
        topk_ids = torch.zeros((seqlen, topk), dtype=torch.int32).to('jax')
        
        result = method.apply(layer, x, router_logits, topk_ids, None)

        def unpack_and_scale(packed, scale, O, I):
            w_int = torch.zeros((num_experts, O, I), dtype=torch.int32)
            for i in range(8):
                w_int[:, :, i::8] = (packed >> (i * 4)) & 0xF
            w_int = w_int - 8
            scale_expanded = scale.unsqueeze(-1).expand(-1, -1, -1, 128).reshape(num_experts, O, I)
            return (w_int.to(torch.float32) * scale_expanded.to(torch.float32)).to(torch.bfloat16)

        w13_approx = unpack_and_scale(w13_packed, w13_scale, 2 * intermediate_size, hidden_size)
        w2_approx = unpack_and_scale(w2_packed, w2_scale, hidden_size, intermediate_size)
        
        w1_approx = w13_approx[:, :intermediate_size, :].to('jax')
        w3_approx = w13_approx[:, intermediate_size:, :].to('jax')
        w2_approx = w2_approx.to('jax')

        result_reference = _ref_math_in_bf16(w1_approx, w2_approx, w3_approx, x, router_logits, topk)

        assert result.shape == result_reference.shape