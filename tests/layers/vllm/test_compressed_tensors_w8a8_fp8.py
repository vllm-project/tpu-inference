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
import jax.numpy as jnp
import pytest
import torch
import torchax
from compressed_tensors.quantization import QuantizationStrategy
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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.model_loader import get_model as vllm_get_model

from tpu_inference.layers.common.quantization import (dequantize_tensor,
                                                      quantize_tensor)
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import \
    VllmCompressedTensorsW8A8Fp8
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

from . import utils as test_utils

P = PartitionSpec
MODELS = [
    "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic",
    "RedHatAI/Llama-3.2-1B-Instruct-FP8"
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


def ref_w8a8_fp8_dynamic(x: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                         b: Optional[torch.Tensor]):
    x_q, x_s = ref_quantize_fp8(x, w_q.dtype)
    out = torch.einsum('bd,fd->bf', x_q.to(torch.float32),
                       w_q.to(torch.float32))
    out = (out * x_s) * w_s.T
    if b is not None:
        out += b
    return out.to(x.dtype)


def ref_w8a8_fp8_static(x: torch.Tensor, x_s: torch.Tensor, w_q: torch.Tensor,
                        w_s: torch.Tensor, b: Optional[torch.Tensor]):
    dtype_info = torch.finfo(w_q.dtype)
    dtype_max = float(dtype_info.max)
    dtype_min = float(dtype_info.min)

    x_q = torch.clamp(x / x_s, dtype_min, dtype_max).to(w_q.dtype)
    out = torch.einsum('bd,fd->bf', x_q.to(torch.float32),
                       w_q.to(torch.float32))
    out = (out * x_s) * w_s.T
    if b is not None:
        out += b
    return out.to(x.dtype)


def return_ref_and_layer_output(layer: torch.nn.Module, batch_size: int = 16):
    assert isinstance(layer, LinearBase)
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsW8A8Fp8)
    quant_config = scheme.linear_config
    assert isinstance(quant_config, VllmQuantLinearConfig)
    quant_method = layer.quant_method
    assert isinstance(quant_method, CompressedTensorsLinearMethod)
    per_tensor = scheme.strategy == QuantizationStrategy.TENSOR
    is_static_input_scheme = scheme.is_static_input_scheme

    input_tensor = torch.rand(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    weight_scale, weight = layer.weight_scale, layer.weight
    input_scale = getattr(layer, 'input_scale', None)
    # For per_tensor with merged layers, vLLM requenzites them so all merged
    # layers shared the same scale values.
    if per_tensor:
        dtype = weight.dtype

        weight = t2j(weight)
        weight_scale = t2j(weight_scale)
        weights = []
        start = 0
        # Multiple weights may have been concatenated. Loop through
        # each weight and perform dequantization.
        for i, output_size in enumerate(quant_config.output_sizes):
            end = start + output_size
            weights.append(
                dequantize_tensor(weight[start:end], weight_scale[i]))
            start = end
        weight = jnp.concat(weights, axis=0)
        weight, weight_scale = quantize_tensor(
            jnp.float8_e4m3fn,
            weight,
            None,
        )
        weight = j2t(weight.astype(jnp.float32)).to(dtype)
        weight_scale = j2t(weight_scale)
        if input_scale is not None:
            input_scale = input_scale.max()

    # Run reference implementation
    if is_static_input_scheme:
        ref_output = ref_w8a8_fp8_static(
            input_tensor,
            input_scale,
            weight,
            weight_scale,
            layer.bias,
        )
    else:
        ref_output = ref_w8a8_fp8_dynamic(
            input_tensor,
            weight,
            weight_scale,
            layer.bias,
        )

    # Run torchax/jax function
    with torchax.default_env():
        quant_method.process_weights_after_loading(layer)

        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        layer_output = layer(jax_input_tensor)
        layer_output = j2t(layer_output.to(torch.float32)).to(torch.bfloat16)

    return ref_output, layer_output


def initialize_layer_weights(layer: torch.nn.Module):
    assert isinstance(layer, LinearBase)
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsW8A8Fp8)
    quant_config = scheme.linear_config
    assert isinstance(quant_config, VllmQuantLinearConfig)
    per_tensor = scheme.strategy == QuantizationStrategy.TENSOR

    weight_list = []
    weight_scale_list = []
    for output_size in quant_config.output_sizes:
        weight = torch.rand(
            (output_size, layer.input_size), dtype=torch.bfloat16) / 10
        weight_, weight_scale_ = ref_quantize_fp8(weight, torch.float8_e4m3fn,
                                                  per_tensor)
        weight_list.append(weight_)
        weight_scale_list.append(weight_scale_)

    weight = torch.concatenate(weight_list)
    weight_scale = torch.concatenate(weight_scale_list)

    assert layer.weight.data.shape == weight.shape
    assert layer.weight_scale.data.shape == weight_scale.shape

    layer.weight.data = weight
    layer.weight_scale.data = weight_scale

    if layer.bias is not None:
        layer.bias.data = torch.rand_like(layer.bias.data)


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
        assert isinstance(layer.scheme, VllmCompressedTensorsW8A8Fp8)


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
