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

import jax
import jax.numpy as jnp
import pytest
import torch
import torchax
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec
from torchax.interop import jax_view, torch_view

from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from compressed_tensors.quantization import ActivationOrdering

from tests.layers.common import utils as test_utils
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import VllmCompressedTensorsWNA16

P = PartitionSpec
MODELS = ["openai/gpt-oss-20b"]

def pack_int4_to_int32(tensor_u8: torch.Tensor) -> torch.Tensor:
    shape = list(tensor_u8.shape)
    shape[-1] = shape[-1] // 8
    
    tensor_reshaped = tensor_u8.reshape(-1, 8)
    packed_flat = torch.zeros(tensor_reshaped.shape[0], dtype=torch.int32, device=tensor_u8.device)
    
    for i in range(8):
        shift_val = (tensor_reshaped[:, i] & 0xF) << (i * 4)
        packed_flat |= shift_val
        
    return packed_flat.reshape(shape)

def quantize_to_int4(weight: torch.Tensor, group_size: int = -1, symmetric: bool = True):
    out_features, in_features = weight.shape
    g_size = in_features if group_size == -1 else group_size
    num_groups = in_features // g_size
    w_grouped = weight.reshape(out_features, num_groups, g_size)
    
    if symmetric:
        abs_max = w_grouped.abs().max(dim=-1, keepdim=True)[0]
        abs_max = torch.clamp(abs_max, min=1e-5)
        scale = abs_max / 7.0 
        w_q = torch.clamp(torch.round(w_grouped / scale), -8, 7)
        w_q_unsigned = (w_q + 8).to(torch.int32)
        zp_packed = None
    else:
        w_min = w_grouped.min(dim=-1, keepdim=True)[0]
        w_max = w_grouped.max(dim=-1, keepdim=True)[0]
        scale = torch.clamp((w_max - w_min) / 15.0, min=1e-5)
        zp = torch.clamp(torch.round(-w_min / scale), 0, 15).to(torch.int32)
        w_q = torch.round(w_grouped / scale + zp)
        w_q_unsigned = torch.clamp(w_q, 0, 15).to(torch.int32)
        
        zp_reshaped = zp.squeeze(-1)
        zp_packed = pack_int4_to_int32(zp_reshaped.T).T 
    
    w_q_unsigned = w_q_unsigned.reshape(out_features, in_features)
    weight_packed = pack_int4_to_int32(w_q_unsigned)
    scale = scale.squeeze(-1).to(torch.bfloat16) 
    if group_size == -1:
        scale = scale.squeeze(-1)
        
    return weight_packed, scale, zp_packed

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
    engine_args = EngineArgs(
        model=MODELS[0], max_model_len=64, max_num_batched_tokens=64,
        max_num_seqs=4, load_format='dummy',
    )
    vllm_config = engine_args.create_engine_config()
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(1, 0, local_rank=0,
                                     distributed_init_method=f"file://{temp_file}",
                                     backend="gloo")
        ensure_model_parallel_initialized(1, 1)

class MockLinearLayer(torch.nn.Module):
    """Dummy module to mimic a vLLM layer for testing."""
    def __init__(self):
        super().__init__()
        self.skip_bias_add = False
    def _get_name(self):
        return "MockLinear"

@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("in_features", [256])
@pytest.mark.parametrize("out_features", [512])
@pytest.mark.parametrize("group_size", [-1, 64])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("fuse_matmuls", [True, False])
def test_wna16_linear(num_devices, batch_size, in_features, out_features, 
                      group_size, symmetric, fuse_matmuls):
    
    mesh = test_utils.get_spmd_mesh(num_devices)
    torch.manual_seed(42)
    dtype = torch.bfloat16

    # 1. Generate Input Data
    x = torch.randn((batch_size, in_features), dtype=dtype) / 10
    w = torch.randn((out_features, in_features), dtype=dtype) / 10
    bias = torch.randn((out_features,), dtype=dtype) / 10

    # 2. Get Oracle Quantized Weights
    weight_packed, weight_scale, zp_packed = quantize_to_int4(w, group_size, symmetric)

    g_size = in_features if group_size == -1 else group_size
    num_groups = in_features // g_size
    weight_scale = weight_scale.reshape(out_features, num_groups).contiguous()

    # 3. Calculate Expected Output using unquantized math
    expected = torch.nn.functional.linear(x, w, bias)

    # 4. Setup vLLM & Layer Configs
    linear_config = QuantLinearConfig(enable_sp=False, output_sizes=[out_features])
    linear_config.mesh = mesh
    linear_config.fuse_matmuls = fuse_matmuls
    
    quant_method = VllmCompressedTensorsWNA16(
        strategy="group" if group_size != -1 else "channel",
        num_bits=4,
        linear_config=linear_config,
        group_size=group_size,
        symmetric=symmetric,
        actorder=None # Skipping actorder for this basic test
    )

    layer = MockLinearLayer()

    # 5. Execute weight lifecycle
    quant_method.create_weights(
        layer=layer,
        output_partition_sizes=[out_features],
        input_size_per_partition=in_features,
        output_size=out_features,
        input_size=in_features,
        params_dtype=dtype,
        weight_loader=lambda *args, **kwargs: None
    )

    # 6. Inject the Oracle Data
    layer.weight_packed.data = weight_packed
    layer.weight_scale.data.copy_(weight_scale)
    if not symmetric and zp_packed is not None:
        layer.weight_zero_point.data = zp_packed
    layer.bias = torch.nn.Parameter(bias)

    # 7. Process & Forward Pass
    # jax_view() requires torchax tensors, not plain torch.Tensor (see torchax/interop.py).
    env = torchax.default_env()
    with env:
        quant_method.process_weights_after_loading(layer)
        x_in, bias_in = env.to_xla((x, layer.bias))
        actual = quant_method.apply_weights(layer, x_in, bias_in)
        # assert_close uses aten._to_copy outside the torchax env; materialize CPU torch tensors.
        actual_torch = actual.torch()

    # 8. Verify
    torch.testing.assert_close(
        actual_torch,
        expected,
        check_device=False,
        atol=2e-1,
        rtol=2e-1,
    )

def quantize_to_int4_actorder(weight: torch.Tensor, group_size: int, symmetric: bool = True):
    """Oracle for Actorder: Weights are grouped by a random g_idx permutation."""
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    
    # Generate a valid g_idx (e.g., shuffling the assignment of features to groups)
    g_idx = torch.arange(in_features) // group_size
    g_idx = g_idx[torch.randperm(in_features)].to(torch.int32)
    
    scales = torch.zeros((out_features, num_groups), dtype=torch.float32, device=weight.device)
    zps = torch.zeros((out_features, num_groups), dtype=torch.int32, device=weight.device)
    w_q_unsigned = torch.zeros_like(weight, dtype=torch.int32)
    
    for g in range(num_groups):
        mask = (g_idx == g)
        w_g = weight[:, mask] # Shape: (out_features, features_in_group)
        
        if symmetric:
            abs_max = torch.clamp(w_g.abs().max(dim=-1, keepdim=True)[0], min=1e-5)
            scale = abs_max / 7.0
            scales[:, g:g+1] = scale
            w_q_g = torch.clamp(torch.round(w_g / scale), -8, 7)
            w_q_unsigned[:, mask] = (w_q_g + 8).to(torch.int32)
        else:
            w_min = w_g.min(dim=-1, keepdim=True)[0]
            w_max = w_g.max(dim=-1, keepdim=True)[0]
            scale = torch.clamp((w_max - w_min) / 15.0, min=1e-5)
            zp = torch.clamp(torch.round(-w_min / scale), 0, 15).to(torch.int32)
            
            scales[:, g:g+1] = scale
            zps[:, g:g+1] = zp
            w_q_unsigned[:, mask] = torch.clamp(torch.round(w_g / scale + zp), 0, 15).to(torch.int32)
            
    weight_packed = pack_int4_to_int32(w_q_unsigned)
    scales = scales.to(torch.bfloat16)
    zp_packed = pack_int4_to_int32(zps.T).T if not symmetric else None
        
    return weight_packed, scales, zp_packed, g_idx

@pytest.mark.parametrize("symmetric", [True, False])
def test_wna16_actorder(symmetric):
    """Validates the g_idx fast-path where permutations are bypassed."""
    batch_size, in_features, out_features, group_size = 8, 256, 512, 64
    torch.manual_seed(42)
    dtype = torch.bfloat16

    x = torch.randn((batch_size, in_features), dtype=dtype) / 10
    w = torch.randn((out_features, in_features), dtype=dtype) / 10
    bias = torch.randn((out_features,), dtype=dtype) / 10

    # 1. Quantize with g_idx
    w_packed, w_scale, zp_packed, g_idx = quantize_to_int4_actorder(w, group_size, symmetric)

    # 2. Reconstruct expected weights (The "Strict" Math Check)
    scale_expanded = w_scale[:, g_idx].to(torch.float32)

    # Unpack weights safely for the expected math
    w_int = torch.zeros_like(w, dtype=torch.int32)
    for i in range(8):
        w_int[:, i::8] = (w_packed >> (i * 4)) & 0xF

    if symmetric:
        w_approx = (w_int - 8).to(torch.float32) * scale_expanded
    else:
        num_groups = in_features // group_size
        zp_packed_t = zp_packed.T
        zps_unpacked_t = torch.zeros((num_groups, out_features), dtype=torch.int32)
        for i in range(8):
            zps_unpacked_t[:, i::8] = (zp_packed_t >> (i * 4)) & 0xF
        zps_unpacked = zps_unpacked_t.T # Transpose back to (out_features, num_groups)
        
        zp_expanded = zps_unpacked[:, g_idx].to(torch.float32)
        w_approx = (w_int.to(torch.float32) - zp_expanded) * scale_expanded

    # Calculate expected using the RECONSTRUCTED weights, not the original floats.
    # This removes quantization noise from the test.
    expected = torch.nn.functional.linear(x, w_approx.to(dtype), bias)

    # 3. Setup Class
    linear_config = QuantLinearConfig(enable_sp=False, output_sizes=[out_features])
    linear_config.mesh = test_utils.get_spmd_mesh(1)
    
    quant_method = VllmCompressedTensorsWNA16(
        strategy="group", num_bits=4, linear_config=linear_config,
        group_size=group_size, symmetric=symmetric, actorder=ActivationOrdering.GROUP
    )

    layer = MockLinearLayer()
    quant_method.create_weights(layer, [out_features], in_features, out_features, in_features, dtype, lambda *args: None)

    # Inject
    layer.weight_packed.data = w_packed
    layer.weight_scale.data.copy_(w_scale)
    if not symmetric:
        layer.weight_zero_point.data = zp_packed
    layer.weight_g_idx.data = g_idx
    layer.bias = torch.nn.Parameter(bias)

    # Apply
    env = torchax.default_env()
    with env:
        quant_method.process_weights_after_loading(layer)
        x_in, bias_in = env.to_xla((x, layer.bias))
        actual_torch = quant_method.apply_weights(layer, x_in, bias_in).torch()

    torch.testing.assert_close(actual_torch, expected, check_device=False, atol=2e-1, rtol=2e-1)


def test_wna16_outliers():
    """Tests that implementation matches strict math even when outliers degrade INT4 accuracy."""
    batch_size, in_features, out_features, group_size = 4, 256, 128, 64
    torch.manual_seed(99)
    dtype = torch.bfloat16

    # 1. Create weights and inject massive outliers
    x = torch.randn((batch_size, in_features), dtype=dtype)
    w = torch.randn((out_features, in_features), dtype=dtype)
    
    # Inject 2% massive outliers
    outlier_mask = torch.rand_like(w) < 0.02
    w[outlier_mask] *= 50.0  

    bias = torch.zeros((out_features,), dtype=dtype)

    w_packed, w_scale, _ = quantize_to_int4(w, group_size, symmetric=True)
    w_scale = w_scale.reshape(out_features, in_features // group_size).contiguous()

    # 2. Strict Reconstructed Math
    w_int = torch.zeros_like(w, dtype=torch.int32)
    for i in range(8):
        w_int[:, i::8] = (w_packed >> (i * 4)) & 0xF
    
    # w_int is currently [0, 15]. Shift to [-8, 7]
    w_int = w_int - 8
    
    # Broadcast scale: (out_features, num_groups) -> (out_features, num_groups, group_size)
    scale_expanded = w_scale.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, in_features)
    
    # The pure reconstructed float matrix
    w_approx = (w_int.to(torch.float32) * scale_expanded.to(torch.float32)).to(dtype)
    
    # Expected is now based on the DEGRADED weights, not original
    expected = torch.nn.functional.linear(x, w_approx, bias)

    # 3. Setup Class
    linear_config = QuantLinearConfig(enable_sp=False, output_sizes=[out_features])
    linear_config.mesh = test_utils.get_spmd_mesh(1)
    quant_method = VllmCompressedTensorsWNA16(
        strategy="group", num_bits=4, linear_config=linear_config, group_size=group_size, symmetric=True, actorder=None
    )
    layer = MockLinearLayer()
    quant_method.create_weights(layer, [out_features], in_features, out_features, in_features, dtype, lambda *args: None)

    layer.weight_packed.data = w_packed
    layer.weight_scale.data.copy_(w_scale)
    layer.bias = torch.nn.Parameter(bias)

    # Apply
    env = torchax.default_env()
    with env:
        quant_method.process_weights_after_loading(layer)
        x_in, bias_in = env.to_xla((x, layer.bias))
        actual_torch = quant_method.apply_weights(layer, x_in, bias_in).torch()

    torch.testing.assert_close(actual_torch, expected, check_device=False, atol=1e-3, rtol=1e-3)