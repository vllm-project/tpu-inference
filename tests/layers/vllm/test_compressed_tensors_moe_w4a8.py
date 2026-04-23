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

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchax
from compressed_tensors.quantization import QuantizationArgs
from jax.sharding import PartitionSpec
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, quantize_weights)
from vllm.scalar_type import scalar_types

# yapf: disable
from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8 import \
    VllmCompressedTensorsW4A8MoEMethod

# yapf: enable

P = PartitionSpec

MODEL = 'nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16'


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


def initialize_int4_layer_weights(layer: torch.nn.Module,
                                  weight_quant: QuantizationArgs,
                                  hidden_size: int):
    torch.manual_seed(42)
    assert isinstance(layer, FusedMoE)

    group_size = weight_quant.group_size
    experts = layer.global_num_experts
    intermediate_size = layer.intermediate_size_per_partition

    def generate_moe_expert_weights(
        expert_shape: tuple[int, int], ) -> tuple[torch.Tensor, torch.Tensor]:
        """
      Generates, quantizes, and packs a weight.
      """
        w_ref_per_expert = []
        q_per_expert = []
        s_per_expert = []

        for _ in range(experts):
            w_block = (torch.rand(expert_shape, dtype=torch.bfloat16) -
                       0.5) / 10

            # Transpose to force quantize_weights to group along input_size (dim 0 of weight.T)
            weight_ref_t, weight_q_t, weight_scale_t, _ = quantize_weights(
                w_block.T, scalar_types.int4, group_size=group_size)

            w_ref_per_expert.append(weight_ref_t.T)
            # Offset to uint4 range [0, 15] for storage packing simulation
            q_per_expert.append(weight_q_t.T + 8)
            s_per_expert.append(weight_scale_t.T)

        # Pack the quantized uint4 bits into int32 containers
        q_packed = pack_quantized_values_into_int32(torch.stack(q_per_expert),
                                                    scalar_types.uint4,
                                                    packed_dim=2)

        return torch.stack(w_ref_per_expert), q_packed, torch.stack(
            s_per_expert)

    # 1. Initialize w13 (gate and up projections) -> Shape: (E, 2*I, H)
    w13_ref, w13_q_packed, w13_s = generate_moe_expert_weights(
        expert_shape=(2 * intermediate_size, hidden_size))
    assert layer.w13_weight_packed.data.shape == w13_q_packed.shape
    assert layer.w13_weight_scale.data.shape == w13_s.shape

    layer.w13_weight_packed.data = w13_q_packed
    layer.w13_weight_scale.data = w13_s

    # 2. Initialize w2 (down_proj) -> Shape: (E, H, I)
    w2_ref, w2_q_packed, w2_s = generate_moe_expert_weights(
        expert_shape=(hidden_size, intermediate_size))
    assert layer.w2_weight_packed.data.shape == w2_q_packed.shape
    assert layer.w2_weight_scale.data.shape == w2_s.shape

    layer.w2_weight_packed.data = w2_q_packed
    layer.w2_weight_scale.data = w2_s

    # Handle optional MoE biases
    if hasattr(layer, 'w13_bias') and layer.w13_bias is not None:
        layer.w13_bias.data = torch.rand_like(layer.w13_bias.data)
    if hasattr(layer, 'w2_bias') and layer.w2_bias is not None:
        layer.w2_bias.data = torch.rand_like(layer.w2_bias.data)

    return w13_ref, w2_ref


@pytest.mark.parametrize(
    "mesh", [test_utils.get_spmd_mesh(1),
             test_utils.get_spmd_mesh(2)])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("use_ep", [True, False])
def test_fused_moe_method_w4(mesh, num_tokens, intermediate_size, hidden_size,
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
    weight_quant = quant_config.target_scheme_map['Linear']['weights']
    input_quant = quant_config.target_scheme_map['Linear']['input_activations']

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(num_experts=num_experts,
                         top_k=topk,
                         hidden_size=hidden_size,
                         intermediate_size=intermediate_size)
        moe = quant_config.get_moe_config(layer)
        method = VllmCompressedTensorsW4A8MoEMethod(weight_quant, input_quant,
                                                    moe, mesh)
    method.create_weights(layer,
                          num_experts,
                          hidden_size,
                          intermediate_size,
                          params_dtype=torch.bfloat16)

    w13_weight_ref, w2_weight_ref = initialize_int4_layer_weights(
        layer, weight_quant, hidden_size=hidden_size)
    method.process_weights_after_loading(layer)

    seqlen = num_tokens
    with torchax.default_env():
        x = torch.ones((seqlen, hidden_size), dtype=torch.bfloat16).to('jax')
        router_logits = torch.randn((seqlen, num_experts),
                                    dtype=torch.bfloat16).to('jax')
        result = method.apply_monolithic(layer, x, router_logits)
        expected = test_utils.ref_moe(x.to(torch.float32).cpu(),
                                      router_logits.to(torch.float32).cpu(),
                                      w13_weight_ref.to(torch.float32).cpu(),
                                      w2_weight_ref.to(torch.float32).cpu(),
                                      w1_bias=None,
                                      w2_bias=None,
                                      top_k=topk,
                                      renormalize=True,
                                      activation="silu")
        assert np.allclose(result, expected, atol=0.2, rtol=0.05)
