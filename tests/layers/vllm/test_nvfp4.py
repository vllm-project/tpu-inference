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

import jax
import jax.numpy as jnp
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
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)

from tests.layers.common import utils as test_utils
from tpu_inference.layers.common.quant_methods import NVFP4
from tpu_inference.layers.vllm.quantization.nvfp4 import (VllmNvfp4Config,
                                                          VllmNvfp4MoEMethod)

P = PartitionSpec

# NVFP4 uses FP4 MoE kernels only available on v7+.
if not jtu.is_device_tpu_at_least(version=7):
    pytest.skip(allow_module_level=True, reason="Expected TPUv7+")

# NVFP4 block size (elements per scale group).
NVFP4_GROUP_SIZE = 16


def quantize_to_nvfp4(weight: torch.Tensor,
                      group_size: int = NVFP4_GROUP_SIZE):
    """Quantize a float32 weight to NVFP4 format (packed uint8 + scales).

    Returns:
        weight_packed: uint8 [out, in//2]
        weight_scale: float8_e4m3fn [out, in//group_size]
        weight_global_scale: float32 scalar
    """
    assert weight.ndim == 2
    out_size, in_size = weight.shape
    assert in_size % group_size == 0

    # Use JAX for FP4 quantization since torch doesn't have native FP4.
    w_jax = t2j(weight.float())

    # Compute per-block scales.
    num_blocks = in_size // group_size
    w_blocked = w_jax.reshape(out_size, num_blocks, group_size)
    block_abs_max = jnp.max(jnp.abs(w_blocked), axis=2, keepdims=True)

    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    block_scale = block_abs_max / fp4_max  # [out, num_blocks, 1]
    block_scale = jnp.where(block_scale == 0, 1.0, block_scale)

    # Compute global scale: max of all block scales.
    global_scale = jnp.max(block_scale)

    # Effective block scale = block_scale / global_scale (stored as FP8).
    effective_scale = (block_scale / global_scale).astype(jnp.float8_e4m3fn)

    # Quantize to FP4.
    scale_inv = jnp.where(block_scale == 0, 0.0, 1.0 / block_scale)
    w_q = jnp.clip(w_blocked * scale_inv, -fp4_max,
                   fp4_max).astype(jnp.float4_e2m1fn)
    w_q = w_q.reshape(out_size, in_size)

    # Pack FP4 into uint8 (2 values per byte).
    w_packed = w_q.reshape(out_size, in_size // 2, 2)
    w_packed = jax.lax.bitcast_convert_type(w_packed, jnp.uint8)

    effective_scale = effective_scale.reshape(out_size, num_blocks)

    # Convert via numpy to avoid j2t FP8 dtype issues.
    w_packed_t = torch.from_numpy(np.asarray(w_packed))
    scale_t = torch.from_numpy(np.asarray(effective_scale).view(
        np.uint8)).view(torch.float8_e4m3fn)
    global_t = torch.tensor(float(global_scale), dtype=torch.float32)
    return (w_packed_t, scale_t, global_t)


def ref_dequant_nvfp4(weight_packed, weight_scale, global_scale, group_size):
    """Reference dequantization: unpack → float32 using block_scale * global."""
    w_jax = jnp.array(weight_packed.numpy())
    # FP8 scale: go through uint8 view to avoid dtype conversion issues.
    s_np = weight_scale.view(torch.uint8).numpy()
    s_jax = jax.lax.bitcast_convert_type(jnp.array(s_np), jnp.float8_e4m3fn)
    g_jax = jnp.float32(global_scale.item())

    # Unpack uint8 → float4_e2m1fn.
    e2m1 = jax.lax.bitcast_convert_type(w_jax, jnp.float4_e2m1fn)
    fp4 = jnp.reshape(e2m1, e2m1.shape[:-2] + (-1, ))

    # Fold scales.
    eff_scale = s_jax.astype(jnp.float32) * g_jax
    out_size = fp4.shape[0]
    in_size = fp4.shape[1]
    num_blocks = in_size // group_size

    fp4_blocked = fp4.reshape(out_size, num_blocks, group_size)
    scale_expanded = eff_scale.reshape(out_size, num_blocks, 1)
    deq = (fp4_blocked.astype(jnp.float32) * scale_expanded).reshape(
        out_size, in_size)
    return torch.from_numpy(np.asarray(deq))


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
    result = VllmNvfp4Config.override_quantization_method(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4"
        },
        user_quant=None,
    )
    assert result == NVFP4

    # Non-NVFP4 should return None.
    result = VllmNvfp4Config.override_quantization_method(
        {
            "quant_method": "modelopt",
            "quant_algo": "FP8"
        },
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
@pytest.mark.parametrize("num_devices", [1])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [4])
@pytest.mark.parametrize("topk", [2])
def test_fused_moe(num_devices, num_tokens, intermediate_size, hidden_size,
                   num_experts, topk):
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
            has_bias=False,
        )

    # Quantize w1 and w2 per-expert to NVFP4.
    group_size = NVFP4_GROUP_SIZE
    for e in range(num_experts):
        # w13 (gate+up fused).
        w1_packed, w1_scale, w1_gs = quantize_to_nvfp4(w1[e].float(),
                                                       group_size)
        moe_layer.w13_weight.data[e] = w1_packed
        moe_layer.w13_weight_scale.data[e] = w1_scale
        moe_layer.w13_weight_scale_2.data[e] = w1_gs.item()

        # w2 (down).
        w2_packed, w2_scale, w2_gs = quantize_to_nvfp4(w2[e].float(),
                                                       group_size)
        moe_layer.w2_weight.data[e] = w2_packed
        moe_layer.w2_weight_scale.data[e] = w2_scale
        moe_layer.w2_weight_scale_2.data[e] = w2_gs.item()

    # Reference MoE computation.
    expected = test_utils.ref_moe(a, score, w1, w2, None, None, topk,
                                  moe_layer.renormalize,
                                  moe_layer.activation.value)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(moe_layer.quant_method, VllmNvfp4MoEMethod)
        jax_a = a.to('jax')
        jax_score = score.to('jax')
        moe_layer.quant_method.process_weights_after_loading(moe_layer)
        actual = moe_layer(jax_a, jax_score)

        # FP4 quantization + requantization adds noise; use generous tolerance.
        torch.testing.assert_close(expected,
                                   actual,
                                   check_device=False,
                                   atol=0.3,
                                   rtol=0.1)
