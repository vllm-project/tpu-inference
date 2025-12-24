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

import os
import tempfile

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

from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe import \
    VllmCompressedTensorsW8A8Fp8MoEMethod

from . import utils as test_utils

# yapf: enable

P = PartitionSpec

os.environ['VLLM_DISABLE_SHARED_EXPERTS_STREAM'] = '1'

MODEL = 'BCCard/Qwen3-30B-A3B-FP8-Dynamic'


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


def _ref_math_in_bf16(w1, w2, w3, x, router_logits, top_k):
    seqlen = x.shape[0]
    expert_weights = F.softmax(router_logits, dim=-1)
    expert_weights, expert_indices = torch.topk(expert_weights, top_k, dim=-1)
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

    # cond ffn
    # e = total num of exp = 160
    # t = seqlen
    # o = config.imtermediate size
    # i = config.dim
    x1 = torch.einsum("ti, eoi -> teo", x, w1)
    x1 = F.silu(x1)
    x3 = torch.einsum("ti, eoi -> teo", x, w3)
    expert_outs = torch.einsum("teo, eio -> tei", (x1 * x3), w2)

    seq_indexes = torch.arange(seqlen, device='jax').unsqueeze(1)
    expert_outs = expert_outs[seq_indexes, expert_indices]
    out = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
    return out


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
    quant_config = VllmCompressedTensorsConfig(
        target_scheme_map={
            'Linear': {
                'weights':
                QuantizationArgs(num_bits=8,
                                 type='float',
                                 symmetric=True,
                                 group_size=None,
                                 strategy='channel',
                                 block_structure=None,
                                 dynamic=False,
                                 actorder=None,
                                 observer='minmax',
                                 observer_kwargs={}),
                'input_activations':
                QuantizationArgs(num_bits=8,
                                 type='float',
                                 symmetric=True,
                                 group_size=None,
                                 strategy='token',
                                 block_structure=None,
                                 dynamic=True,
                                 actorder=None,
                                 observer=None,
                                 observer_kwargs={}),
                'format':
                None
            }
        },
        ignore=[],
        quant_format='compressed-tensors',
        sparsity_scheme_map={},
        sparsity_ignore_list=[],
    )
    moe = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        num_local_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            dp_rank=0,
            ep_rank=0,
            use_ep=use_ep,
            all2all_backend='',
        ),
        in_dtype=torch.bfloat16,
    )
    method = VllmCompressedTensorsW8A8Fp8MoEMethod(quant_config, moe, mesh)
    method.create_weights(layer,
                          num_experts,
                          hidden_size,
                          intermediate_size,
                          params_dtype=torch.float8_e4m3fn)
    method.process_weights_after_loading(layer)

    seqlen = num_tokens
    with torchax.default_env():
        x = torch.ones((seqlen, hidden_size), dtype=torch.bfloat16).to('jax')
        router_logits = torch.randn((seqlen, num_experts),
                                    dtype=torch.bfloat16).to('jax')
        result = method.apply(layer,
                              x,
                              router_logits,
                              top_k=topk,
                              renormalize=True)

        result_reference = _ref_math_in_bf16(
            layer.w13_weight.to(torch.bfloat16) * layer.w13_weight_scale,
            layer.w2_weight.to(torch.bfloat16) * layer.w2_weight_scale,
            layer.w3_weight.to(torch.bfloat16) * layer.w3_weight_scale, x,
            router_logits, topk)

        assert jnp.allclose(result.jax(), result_reference.jax())
