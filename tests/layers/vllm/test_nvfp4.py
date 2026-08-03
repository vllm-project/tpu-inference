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
"""Tests for NVFP4 (ModelOpt FP4) quantization on TPU."""

import tempfile

import numpy as np
import pytest
import torch
import torchax
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

from tests.layers.common import utils as test_utils
from tests.layers.vllm.nvfp4_utils import (NVFP4_GROUP_SIZE, quantize_to_nvfp4,
                                           ref_dequant_nvfp4)
from tpu_inference.layers.common.quant_methods import NVFP4
from tpu_inference.layers.vllm.quantization.nvfp4 import (VllmNvfp4Config,
                                                          VllmNvfp4MoEMethod)

P = PartitionSpec

# NVFP4 uses FP4 MoE kernels only available on v7+.
if not jtu.is_device_tpu_at_least(version=7):
    pytest.skip(allow_module_level=True, reason="Expected TPUv7+")


def create_nvfp4_config(mesh):
    """Create a VllmNvfp4Config with the given mesh."""
    config = VllmNvfp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=NVFP4_GROUP_SIZE,
    )
    config.mesh = mesh
    return config


@pytest.fixture(autouse=True)
def setup_environment():
    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
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


# ----------------------------------------------------------------
# Test 1: Config override detection
# ----------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh", [test_utils.get_spmd_mesh(1),
             test_utils.get_spmd_mesh(2)])
def test_quant_override(mesh):
    """VllmNvfp4Config.override_quantization_method detects NVFP4."""
    # 1. New style: quant_library + quant_algo (ModelOpt)
    result = VllmNvfp4Config.override_quantization_method(
        {
            "quant_library": "MODELOPT",
            "quant_algo": "NVFP4"
        },
        user_quant=None,
    )
    assert result == NVFP4

    # 2. Case insensitivity check
    result = VllmNvfp4Config.override_quantization_method(
        {
            "quant_library": "modelopt",
            "quant_algo": "nvfp4"
        },
        user_quant=None,
    )
    assert result == NVFP4

    # 3. Old style: quant_method (fallback to super)
    result = VllmNvfp4Config.override_quantization_method(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4"
        },
        user_quant=None,
    )
    assert result == NVFP4

    # 4. Non-NVFP4 should return None (or whatever super returns).
    result = VllmNvfp4Config.override_quantization_method(
        {
            "quant_library": "MODELOPT",
            "quant_algo": "FP8"
        },
        user_quant=None,
    )
    assert result is None

    # 5. hf_quant_cfg=None should return None (fallback to super).
    result = VllmNvfp4Config.override_quantization_method(
        None,
        user_quant=None,
    )
    assert result is None


# ----------------------------------------------------------------
# Test 2: Dequantize round-trip
# ----------------------------------------------------------------
def test_dequantize_nvfp4_weights():
    """Test that NVFP4 pack → dequant produces values close to original."""
    torch.manual_seed(42)
    weight = torch.randn(128, 256, dtype=torch.float32) / 10

    packed, scale, global_scale = quantize_to_nvfp4(weight)
    deq = ref_dequant_nvfp4(packed, scale, global_scale, NVFP4_GROUP_SIZE)

    # FP4 quantization has limited precision; allow generous tolerance.
    np.testing.assert_allclose(deq.numpy(),
                               weight.numpy(),
                               atol=0.15,
                               rtol=0.3)


# ----------------------------------------------------------------
# Test 3: Linear layers (various parallel types)
# ----------------------------------------------------------------
def _create_linear_and_run(layer_cls, mesh, batch_size, bias, **layer_kwargs):
    """Helper: create NVFP4 linear layer, load fake weights, run forward."""
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    config = create_nvfp4_config(mesh)
    config.vllm_config = vllm_config

    with set_current_vllm_config(vllm_config):
        layer = layer_cls(
            **layer_kwargs,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=config,
        )

    # Generate reference weight and quantize to NVFP4.
    torch.manual_seed(42)
    w_ref = torch.randn(
        layer.output_size, layer.input_size, dtype=torch.float32) / 10
    packed, scale, global_scale = quantize_to_nvfp4(w_ref)

    # Load weights into layer.
    layer.weight.data = packed
    layer.weight_scale.data = scale
    layer.weight_scale_2.data.fill_(global_scale.item())
    layer.input_scale.data.fill_(1.0)

    if bias:
        layer.bias.data = torch.rand_like(layer.bias.data) / 10

    # Reference: dequant → matmul.
    deq = ref_dequant_nvfp4(packed, scale, global_scale, NVFP4_GROUP_SIZE)
    x = torch.randn(batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    ref_out = torch.einsum('bd,fd->bf', x.float(), deq.float())
    if bias:
        ref_out = ref_out + layer.bias.data.float()
    ref_out = ref_out.to(dtype)

    # Actual: process_weights → forward.
    with torchax.default_env():
        layer.quant_method.process_weights_after_loading(layer)
        jax_x = torch_view(t2j(x, use_dlpack=False))
        actual = layer(jax_x)
        actual = j2t(actual.to(torch.float32)).to(dtype)

    return ref_out, actual


@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("num_devices", [1])
def test_row_parallel_linear(bias, num_devices):
    mesh = test_utils.get_spmd_mesh(num_devices)
    ref_out, actual = _create_linear_and_run(
        RowParallelLinear,
        mesh,
        batch_size=8,
        bias=bias,
        input_size=256,
        output_size=512,
    )
    torch.testing.assert_close(ref_out, actual, atol=0.15, rtol=0.1)


@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("num_devices", [1])
def test_column_parallel_linear(bias, num_devices):
    mesh = test_utils.get_spmd_mesh(num_devices)
    ref_out, actual = _create_linear_and_run(
        ColumnParallelLinear,
        mesh,
        batch_size=8,
        bias=bias,
        input_size=256,
        output_size=512,
    )
    torch.testing.assert_close(ref_out, actual, atol=0.15, rtol=0.1)


# ----------------------------------------------------------------
# Test 4: MoE layer
# ----------------------------------------------------------------
@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("activation",
                         [MoEActivation.SILU, MoEActivation.SWIGLUOAI])
def test_fused_moe(num_devices, num_tokens, intermediate_size, hidden_size,
                   num_experts, topk, has_bias, activation):
    mesh = test_utils.get_spmd_mesh(num_devices)
    torch.manual_seed(42)
    dtype = torch.bfloat16

    # Reference weights (unquantized).
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), dtype=dtype) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size), dtype=dtype) / 10

    a = torch.randn((num_tokens, hidden_size), dtype=dtype) / 10
    score = torch.randn((num_tokens, num_experts), dtype=dtype)

    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=mesh.devices.size)

    config = create_nvfp4_config(mesh)
    config.vllm_config = vllm_config

    with set_current_vllm_config(vllm_config):
        moe_layer = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=False,
            tp_size=1,
            dp_size=1,
            quant_config=config,
            has_bias=has_bias,
            activation=activation.value,
        )

    # Quantize w1 and w2 per-expert to NVFP4.
    group_size = NVFP4_GROUP_SIZE
    for e in range(num_experts):
        # w13 (gate+up fused).
        w1_packed, w1_scale, w1_gs = quantize_to_nvfp4(w1[e].float(),
                                                       group_size)
        moe_layer.routed_experts.w13_weight.data[e] = w1_packed
        moe_layer.routed_experts.w13_weight_scale.data[e] = w1_scale
        moe_layer.routed_experts.w13_weight_scale_2.data[e] = w1_gs.item()

        # w2 (down).
        w2_packed, w2_scale, w2_gs = quantize_to_nvfp4(w2[e].float(),
                                                       group_size)
        moe_layer.routed_experts.w2_weight.data[e] = w2_packed
        moe_layer.routed_experts.w2_weight_scale.data[e] = w2_scale
        moe_layer.routed_experts.w2_weight_scale_2.data[e] = w2_gs.item()

    if has_bias:
        moe_layer.routed_experts.w13_bias.data = torch.randn(
            (num_experts, 2 * intermediate_size), dtype=dtype) / 10
        moe_layer.routed_experts.w2_bias.data = torch.randn(
            (num_experts, hidden_size), dtype=dtype) / 10

    w1_bias = moe_layer.routed_experts.w13_bias.data if has_bias else None
    w2_bias = moe_layer.routed_experts.w2_bias.data if has_bias else None

    # Reference MoE computation.
    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias, topk,
                                  moe_layer.routed_experts.renormalize,
                                  moe_layer.activation.value)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(moe_layer.routed_experts.quant_method,
                          VllmNvfp4MoEMethod)
        jax_a = a.to('jax')
        jax_score = score.to('jax')
        moe_layer.routed_experts.quant_method.process_weights_after_loading(
            moe_layer.routed_experts)
        actual = moe_layer(jax_a, jax_score)

        # FP4 quantization + requantization adds noise; use generous tolerance.
        torch.testing.assert_close(expected,
                                   actual,
                                   check_device=False,
                                   atol=0.3,
                                   rtol=0.1)
