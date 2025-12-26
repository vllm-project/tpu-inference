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
from typing import Optional

import jax
import pytest
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec
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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.model_loader import get_model as vllm_get_model

from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import \
    VllmCompressedTensorsW8A8Int8

from . import utils as test_utils

P = PartitionSpec
MODELS = ["RedHatAI/Qwen2.5-1.5B-quantized.w8a8"]


def ref_quantize_int8(x: torch.Tensor):
    x_abs_max = torch.amax(torch.abs(x), dim=1, keepdim=True)
    x_s = x_abs_max / 127
    x_q = torch.round(x / x_s).to(torch.int8)
    return x_q, x_s.to(torch.float32)


def ref_w8a8_int8(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                  b: Optional[torch.Tensor]):
    x_q, x_s = ref_quantize_int8(x)
    out = torch.einsum('bd,fd->bf', x_q.to(torch.float32),
                       w_q.to(torch.float32))
    out = (out * x_s) * w_s.T
    if b is not None:
        out += b
    return out.to(x.dtype)


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # RowParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model=MODELS[0],
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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
def test_quant_override(model, mesh):

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    assert isinstance(quant_config, VllmCompressedTensorsConfig)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
def test_loading_model(model, mesh):
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config, mesh)
    vllm_config.device_config.device = "cpu"

    vllm_model = vllm_get_model(vllm_config=vllm_config)
    layers = test_utils.find_all_layer_type(vllm_model, LinearBase)
    for layer in layers:
        assert isinstance(layer.quant_config, VllmCompressedTensorsConfig)
        assert isinstance(layer.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(layer.scheme, VllmCompressedTensorsW8A8Int8)


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

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    # Call tpu_inference code
    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    weight_data_float = torch.rand(
        (jax_row_linear.output_size, jax_row_linear.input_size),
        dtype=dtype) / 10
    weight_data, weight_scale_data = ref_quantize_int8(weight_data_float)
    if bias:
        bias_data = torch.rand_like(jax_row_linear.bias.data)

    jax_row_linear.weight.data = weight_data
    jax_row_linear.weight_scale.data = weight_scale_data
    if bias:
        jax_row_linear.bias.data = bias_data

    input_tensor = torch.rand(10, jax_row_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_row_linear.quant_method,
                          CompressedTensorsLinearMethod)
        assert isinstance(jax_row_linear.scheme, VllmCompressedTensorsW8A8Int8)
        jax_row_linear.quant_method.process_weights_after_loading(
            jax_row_linear)
        jax_output = jax_row_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    # Call reference w8a8 int8
    output = ref_w8a8_int8(
        input_tensor,
        weight_data,
        weight_scale_data,
        bias_data if bias else None,
    )

    torch.testing.assert_close(output, jax_output)


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

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    # Call tpu_inference code
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_column_linear = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    weight_data_float = torch.rand(
        (jax_column_linear.output_size, jax_column_linear.input_size),
        dtype=dtype) / 10
    weight_data, weight_scale_data = ref_quantize_int8(weight_data_float)
    if bias:
        bias_data = torch.rand_like(jax_column_linear.bias.data)

    jax_column_linear.weight.data = weight_data
    jax_column_linear.weight_scale.data = weight_scale_data
    if bias:
        jax_column_linear.bias.data = bias_data

    input_tensor = torch.rand(10, jax_column_linear.input_size,
                              dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_column_linear.quant_method,
                          CompressedTensorsLinearMethod)
        assert isinstance(jax_column_linear.scheme,
                          VllmCompressedTensorsW8A8Int8)
        jax_column_linear.quant_method.process_weights_after_loading(
            jax_column_linear)
        jax_output = jax_column_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    # Call reference w8a8 int8
    output = ref_w8a8_int8(
        input_tensor,
        weight_data,
        weight_scale_data,
        bias_data if bias else None,
    )

    torch.testing.assert_close(output, jax_output)


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

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    # Call tpu_inference code
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_qkv_linear = QKVParallelLinear(
            hidden_size=4096,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        jax_qkv_linear.quant_method.fuse_matmuls = fuse_matmuls

    weight_data_float = torch.rand(
        (jax_qkv_linear.output_size, jax_qkv_linear.input_size),
        dtype=dtype) / 10
    weight_data, weight_scale_data = ref_quantize_int8(weight_data_float)
    if bias:
        bias_data = torch.rand_like(jax_qkv_linear.bias.data)

    jax_qkv_linear.weight.data = weight_data
    jax_qkv_linear.weight_scale.data = weight_scale_data
    if bias:
        jax_qkv_linear.bias.data = bias_data

    input_tensor = torch.rand(10, jax_qkv_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_qkv_linear.quant_method,
                          CompressedTensorsLinearMethod)
        assert isinstance(jax_qkv_linear.scheme, VllmCompressedTensorsW8A8Int8)
        jax_qkv_linear.quant_method.process_weights_after_loading(
            jax_qkv_linear)
        jax_output = jax_qkv_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    # Call reference w8a8 int8
    output = ref_w8a8_int8(
        input_tensor,
        weight_data,
        weight_scale_data,
        bias_data if bias else None,
    )

    torch.testing.assert_close(output, jax_output)


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

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    # Call tpu_inference code
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_merged_column_linear = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[14336] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        jax_merged_column_linear.quant_method.fuse_matmuls = fuse_matmuls

    weight_data_float = torch.rand((jax_merged_column_linear.output_size,
                                    jax_merged_column_linear.input_size),
                                   dtype=dtype) / 10
    weight_data, weight_scale_data = ref_quantize_int8(weight_data_float)
    if bias:
        bias_data = torch.rand_like(jax_merged_column_linear.bias.data)

    jax_merged_column_linear.weight.data = weight_data
    jax_merged_column_linear.weight_scale.data = weight_scale_data
    if bias:
        jax_merged_column_linear.bias.data = bias_data

    input_tensor = torch.rand(
        10, jax_merged_column_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_merged_column_linear.quant_method,
                          CompressedTensorsLinearMethod)
        assert isinstance(jax_merged_column_linear.scheme,
                          VllmCompressedTensorsW8A8Int8)
        jax_merged_column_linear.quant_method.process_weights_after_loading(
            jax_merged_column_linear)
        jax_output = jax_merged_column_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    # Call reference w8a8 int8
    output = ref_w8a8_int8(
        input_tensor,
        weight_data,
        weight_scale_data,
        bias_data if bias else None,
    )

    torch.testing.assert_close(output, jax_output)
