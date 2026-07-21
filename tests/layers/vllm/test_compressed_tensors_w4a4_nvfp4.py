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

import os
import tempfile
from unittest.mock import MagicMock, patch

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
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod

from tests.layers.common import utils as test_utils
from tests.layers.vllm.nvfp4_utils import (NVFP4_GROUP_SIZE, quantize_to_nvfp4,
                                           ref_dequant_nvfp4)
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    VllmCompressedTensorsW4A4Fp4, W4A4ActivationType)
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

P = PartitionSpec

torch.manual_seed(42)

MODELS = [
    "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16",
]


def override_activation_quant_config(vllm_config):
    vllm_config.model_config.hf_config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "dynamic": True,
                }
            }
        }
    }
    vllm_config.model_config.hf_text_config.quantization_config = vllm_config.model_config.hf_config.quantization_config


def initialize_layer_weights(
    layer: torch.nn.Module,
    activation_type: W4A4ActivationType = W4A4ActivationType.BF16
) -> torch.Tensor:
    assert isinstance(layer, LinearBase)
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsW4A4Fp4)
    quant_config = scheme.linear_config
    assert isinstance(quant_config, VllmQuantLinearConfig)

    group_size = getattr(scheme, "group_size", NVFP4_GROUP_SIZE)

    weight_list = []
    weight_ref_list = []
    weight_scale_list = []
    weight_global_scale_list = []

    for output_size in quant_config.output_sizes:
        weight = torch.randn(
            (output_size, layer.input_size), dtype=torch.float32) / 10

        packed_weight, scale_t, global_scale = quantize_to_nvfp4(
            weight, group_size=group_size)

        weight_list.append(packed_weight)
        weight_scale_list.append(scale_t)
        # Note: the scheme expects 1/global_scale since it computes 1/layer.weight_global_scale.max()
        weight_global_scale_list.append(
            torch.tensor([1.0 / global_scale], dtype=torch.float32))

        weight_ref = ref_dequant_nvfp4(packed_weight, scale_t, global_scale,
                                       group_size)
        weight_ref_list.append(weight_ref)

    weight_packed = torch.cat(weight_list)
    weight_ref = torch.cat(weight_ref_list)
    weight_scale = torch.cat(weight_scale_list)
    weight_global_scale = torch.cat(weight_global_scale_list)

    layer.weight_packed = torch.nn.Parameter(weight_packed,
                                             requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(weight_global_scale,
                                                   requires_grad=False)

    if activation_type == W4A4ActivationType.NVFP4:
        layer.input_global_scale = torch.nn.Parameter(torch.tensor(
            [1.0], dtype=torch.float32),
                                                      requires_grad=False)

    if layer.bias is not None:
        layer.bias.data = torch.rand_like(layer.bias.data)
    return weight_ref


def return_ref_and_layer_output(
        layer: torch.nn.Module,
        activation_type: W4A4ActivationType = W4A4ActivationType.BF16,
        batch_size: int = 16):

    weight_ref = initialize_layer_weights(layer,
                                          activation_type=activation_type)
    assert isinstance(layer, LinearBase)
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsW4A4Fp4)
    quant_config = scheme.linear_config
    assert isinstance(quant_config, VllmQuantLinearConfig)
    quant_method = layer.quant_method
    assert isinstance(quant_method, CompressedTensorsLinearMethod)

    input_tensor = torch.randn(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    # Run reference implementation
    if activation_type == W4A4ActivationType.NVFP4:
        # Quantize activation to FP4
        # Since activation_type=NVFP4 tests FP4xFP4, we need to quantize the activation
        # However, TPU does not natively support FP4xFP4 and raises an error
        # So we skip actual activation quantization in reference for now
        pass

    # Just do a bfloat16 matmul with dequantized weights for reference
    ref_output = torch.einsum('bd,fd->bf', input_tensor.to(torch.float32),
                              weight_ref)
    if layer.bias is not None:
        ref_output += layer.bias.float()
    ref_output = ref_output.to(torch.bfloat16)

    # Run torchax/jax function
    with torchax.default_env():
        quant_method.process_weights_after_loading(layer)

        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        layer_output = layer(jax_input_tensor)
        layer_output = j2t(layer_output.to(torch.float32)).to(torch.bfloat16)

    return ref_output, layer_output


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


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("activation_type", [W4A4ActivationType.BF16])
@pytest.mark.parametrize("requantize_block_size", [None, 32])
def test_row_parallel_linear(model, bias, num_devices, enable_sp,
                             enable_attn_dp, activation_type,
                             requantize_block_size):
    if requantize_block_size is not None:
        os.environ["REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE"] = str(
            requantize_block_size)
    else:
        os.environ.pop("REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE", None)

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

    override_activation_quant_config(vllm_config)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = RowParallelLinear(
            input_size=2048,
            output_size=4096,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.scheme.activation_type = activation_type

    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, activation_type=activation_type)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.1, atol=0.35)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("activation_type", [W4A4ActivationType.BF16])
@pytest.mark.parametrize("requantize_block_size", [None, 32])
def test_column_parallel_linear(model, bias, num_devices, enable_sp,
                                enable_attn_dp, activation_type,
                                requantize_block_size):
    if requantize_block_size is not None:
        os.environ["REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE"] = str(
            requantize_block_size)
    else:
        os.environ.pop("REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE", None)

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

    override_activation_quant_config(vllm_config)

    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = ColumnParallelLinear(
            input_size=2048,
            output_size=4096,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.scheme.activation_type = activation_type

    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, activation_type=activation_type)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.1, atol=0.35)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("activation_type", [W4A4ActivationType.BF16])
@pytest.mark.parametrize("requantize_block_size", [None, 32])
def test_qkv_parallel_linear(model, bias, num_devices, enable_sp, fuse_matmuls,
                             enable_attn_dp, activation_type,
                             requantize_block_size):
    if requantize_block_size is not None:
        os.environ["REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE"] = str(
            requantize_block_size)
    else:
        os.environ.pop("REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE", None)

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

    override_activation_quant_config(vllm_config)

    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = QKVParallelLinear(
            hidden_size=2048,
            head_size=128,
            total_num_heads=16,
            total_num_kv_heads=4,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.quant_method.fuse_matmuls = fuse_matmuls
        linear_layer.scheme.activation_type = activation_type

    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, activation_type=activation_type)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.1, atol=0.35)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("activation_type", [W4A4ActivationType.BF16])
@pytest.mark.parametrize("requantize_block_size", [None, 32])
def test_merged_column_parallel_linear(model, bias, num_devices, fuse_matmuls,
                                       enable_sp, enable_attn_dp,
                                       activation_type, requantize_block_size):
    if requantize_block_size is not None:
        os.environ["REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE"] = str(
            requantize_block_size)
    else:
        os.environ.pop("REQUANTIZE_COMPRESSED_TENSOR_NVFP4_BLOCK_SIZE", None)

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

    override_activation_quant_config(vllm_config)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = MergedColumnParallelLinear(
            input_size=2048,
            output_sizes=[7168] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.quant_method.fuse_matmuls = fuse_matmuls
        linear_layer.scheme.activation_type = activation_type

    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, activation_type=activation_type)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.1, atol=0.35)
