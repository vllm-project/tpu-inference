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

import jax
import pytest
import torch
import torchax
from jax.sharding import PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.fp8 import (VllmFp8Config,
                                                        VllmFp8LinearMethod)

from . import utils as test_utils

P = PartitionSpec
MODELS = [
    "Qwen/Qwen3-0.6B-FP8",
    "MiniMaxAI/MiniMax-M2",
]


def ref_quantize_fp8(x: torch.Tensor,
                     dtype: torch.dtype,
                     per_tensor: bool = False):
    dtype_info = torch.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    dim = () if per_tensor else 1
    x_abs_max = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    if per_tensor:
        x_abs_max = torch.squeeze(x_abs_max, dim=-1)
    x_s = x_abs_max / dtype_max
    x_q = torch.clip(x / x_s, dtype_min, dtype_max).to(dtype)
    return x_q, x_s.to(torch.float32)


def ref_fp8_activation(
        x: torch.Tensor,
        dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    x_q, x_scale = ref_quantize_fp8(x, dtype=dtype, per_tensor=False)
    return x_q.to(torch.float32) * x_scale


def ref_dequantize_fp8_block_2d(w_q: torch.Tensor, scale_blocks: torch.Tensor,
                                block_m: int, block_n: int) -> torch.Tensor:
    out, inn = w_q.shape
    assert scale_blocks.shape == (out // block_m, inn // block_n)
    scale_e = scale_blocks[:, None, :, None].repeat(1, block_m, 1, block_n)
    w_deq = (w_q.to(torch.float32).view(out // block_m, block_m,
                                        inn // block_n, block_n) * scale_e)
    return w_deq.reshape(out, inn)


def return_ref_and_layer_output(layer: torch.nn.Module, batch_size: int = 16):
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, VllmFp8LinearMethod)

    input_tensor = torch.rand(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    x_deq = ref_fp8_activation(input_tensor, torch.float8_e4m3fn)

    assert hasattr(layer.quant_method, "weight_block_size")
    wbs = layer.quant_method.weight_block_size
    block_m, block_n = int(wbs[0]), int(wbs[1])

    w_deq = ref_dequantize_fp8_block_2d(layer.weight.data,
                                        layer.weight_scale_inv.data, block_m,
                                        block_n)

    ref_output = torch.einsum('bd,fd->bf', x_deq.to(torch.float32),
                              w_deq.to(torch.float32))

    if layer.bias is not None:
        ref_output = ref_output + layer.bias.data

    ref_output = ref_output.to(input_tensor.dtype)

    with torchax.default_env():
        quant_method.process_weights_after_loading(layer)

        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        layer_output = layer(jax_input_tensor)
        layer_output = j2t(layer_output.to(torch.float32)).to(torch.bfloat16)

    return ref_output, layer_output


def ref_quantize_fp8_block_2d(w: torch.Tensor, block_m: int, block_n: int,
                              dtype: torch.dtype):
    dtype_info = torch.finfo(dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    out, inn = w.shape
    assert out % block_m == 0 and inn % block_n == 0
    w_view = w.view(out // block_m, block_m, inn // block_n, block_n)

    abs_max = torch.amax(torch.abs(w_view), dim=(1, 3), keepdim=True)
    scale = abs_max / dtype_max
    w_q = torch.clamp(w_view / scale, dtype_min, dtype_max).to(dtype)

    w_q = w_q.reshape(out, inn)
    scale_blocks = scale.squeeze(1).squeeze(-1).to(torch.float32)
    return w_q, scale_blocks


def initialize_layer_weights(layer: torch.nn.Module):
    assert isinstance(layer, LinearBase)
    assert isinstance(layer.quant_method, VllmFp8LinearMethod)

    assert hasattr(layer.quant_method, "weight_block_size")
    block_m, block_n = layer.quant_method.weight_block_size

    w_f32 = (
        torch.rand(layer.output_size, layer.input_size, dtype=torch.float32) /
        10)
    w_q, w_scale_blocks = ref_quantize_fp8_block_2d(w_f32, block_m, block_n,
                                                    torch.float8_e4m3fn)

    layer.weight.data = w_q
    assert hasattr(layer, "weight_scale_inv")

    layer.weight_scale_inv.data = w_scale_blocks
    assert layer.weight_scale_inv.data.shape == w_scale_blocks.shape

    if layer.bias is not None:
        layer.bias.data = torch.rand_like(layer.bias.data) / 10.0


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # RowParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(model=MODELS[0],
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4,
                             trust_remote_code=True)

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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
def test_quant_override(model, mesh):

    engine_args = EngineArgs(model=model,
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4,
                             trust_remote_code=True)
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    assert isinstance(quant_config, VllmFp8Config)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_row_parallel_linear(model, bias, num_devices, enable_sp,
                             enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)
    dtype = torch.bfloat16

    engine_args = EngineArgs(model=model,
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4,
                             trust_remote_code=True)
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    initialize_layer_weights(linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_column_parallel_linear(model, bias, num_devices, enable_sp,
                                enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)
    dtype = torch.bfloat16

    engine_args = EngineArgs(model=model,
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4,
                             trust_remote_code=True)
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    initialize_layer_weights(linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_qkv_parallel_linear(model, bias, num_devices, enable_sp, fuse_matmuls,
                             enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)
    dtype = torch.bfloat16

    engine_args = EngineArgs(model=model,
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4,
                             trust_remote_code=True)
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = QKVParallelLinear(
            hidden_size=4096,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.quant_method.fuse_matmuls = fuse_matmuls

    initialize_layer_weights(linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_merged_column_parallel_linear(model, bias, num_devices, fuse_matmuls,
                                       enable_sp, enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)
    dtype = torch.bfloat16

    engine_args = EngineArgs(model=model,
                             max_model_len=64,
                             max_num_batched_tokens=64,
                             max_num_seqs=4,
                             trust_remote_code=True)
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[14336] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.quant_method.fuse_matmuls = fuse_matmuls

    initialize_layer_weights(linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output)
