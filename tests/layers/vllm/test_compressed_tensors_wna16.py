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
from typing import Optional
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
                                               LinearBase, QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, quantize_weights)
from vllm.scalar_type import scalar_types

from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_wna16 import \
    VllmCompressedTensorsWNA16

P = PartitionSpec

torch.manual_seed(42)

MODELS = [
    "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16",
]


def ref_w4a16(x: torch.Tensor, w_float: torch.Tensor,
              b: Optional[torch.Tensor]):
    out = torch.einsum('bd,fd->bf', x, w_float)
    if b is not None:
        out += b
    return out.to(x.dtype)


def override_asymmetric_quant_config(vllm_config, group_size=32):
    weights = vllm_config.model_config.hf_config.quantization_config[
        "config_groups"]["group_0"]["weights"]
    weights["symmetric"] = False
    weights["group_size"] = group_size
    vllm_config.model_config.hf_text_config.quantization_config = \
        vllm_config.model_config.hf_config.quantization_config


def initialize_layer_weights(layer: torch.nn.Module) -> torch.Tensor:
    assert isinstance(layer, LinearBase)
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsWNA16)
    quant_config = scheme.linear_config

    group_size = scheme.group_size
    symmetric = scheme.symmetric

    weight_list = []
    weight_ref_list = []
    weight_scale_list = []
    weight_zp_list = []
    for output_size in quant_config.output_sizes:
        # Shift the distribution so asymmetric quantization has non-trivial
        # zero points.
        weight = (torch.rand(
            (output_size, layer.input_size), dtype=torch.bfloat16) - 0.3) / 10

        # Transpose to force quantize_weights to group along input_size
        # (dim 0 of weight.T)
        if symmetric:
            weight_ref_t, weight_q_t, weight_scale_t, _ = quantize_weights(
                weight.T, scalar_types.int4, group_size=group_size)
            # Offset to uint4 range [0, 15] for storage packing simulation
            weight_uint4 = weight_q_t.T + 8
        else:
            weight_ref_t, weight_q_t, weight_scale_t, weight_zp_t = \
                quantize_weights(weight.T,
                                 scalar_types.uint4,
                                 group_size=group_size,
                                 zero_points=True)
            weight_uint4 = weight_q_t.T
            # Zero points are packed along the output dim.
            weight_zp_list.append(
                pack_quantized_values_into_int32(weight_zp_t.T.to(torch.int32),
                                                 scalar_types.uint4,
                                                 packed_dim=0))

        packed_weight_ = pack_quantized_values_into_int32(weight_uint4.to(
            torch.int32),
                                                          scalar_types.uint4,
                                                          packed_dim=1)

        weight_list.append(packed_weight_)
        weight_ref_list.append(weight_ref_t.T)
        weight_scale_list.append(weight_scale_t.T)

    weight_packed = torch.concatenate(weight_list)
    weight_ref = torch.concatenate(weight_ref_list)
    weight_scale = torch.concatenate(weight_scale_list)

    assert layer.weight_packed.data.shape == weight_packed.shape
    assert layer.weight_scale.data.shape == weight_scale.shape

    layer.weight_packed.data = weight_packed
    layer.weight_scale.data = weight_scale.to(layer.weight_scale.dtype)

    if not symmetric:
        weight_zp = torch.concatenate(weight_zp_list)
        assert layer.weight_zero_point.data.shape == weight_zp.shape
        layer.weight_zero_point.data = weight_zp

    if layer.bias is not None:
        layer.bias.data = torch.rand_like(layer.bias.data)
    return weight_ref


def return_ref_and_layer_output(layer: torch.nn.Module, batch_size: int = 16):
    weight_ref = initialize_layer_weights(layer)
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, CompressedTensorsLinearMethod)

    input_tensor = torch.rand(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    # Run reference implementation
    ref_output = ref_w4a16(input_tensor, weight_ref.to(torch.bfloat16),
                           layer.bias)

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


def make_vllm_and_quant_config(model, mesh, symmetric):
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = False
    vllm_config.model_config.dtype = torch.bfloat16
    if not symmetric:
        override_asymmetric_quant_config(vllm_config)
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    return vllm_config, quant_config


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("model", MODELS)
def test_row_parallel_linear(model, bias, symmetric, num_devices):
    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    vllm_config, quant_config = make_vllm_and_quant_config(
        model, mesh, symmetric)
    with set_current_vllm_config(vllm_config):
        linear_layer = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
    assert isinstance(linear_layer.scheme, VllmCompressedTensorsWNA16)
    assert linear_layer.scheme.symmetric == symmetric

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("model", MODELS)
def test_column_parallel_linear(model, bias, symmetric, num_devices):
    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    vllm_config, quant_config = make_vllm_and_quant_config(
        model, mesh, symmetric)
    with set_current_vllm_config(vllm_config):
        linear_layer = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
    assert isinstance(linear_layer.scheme, VllmCompressedTensorsWNA16)

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("model", MODELS)
def test_qkv_parallel_linear(model, bias, symmetric, fuse_matmuls,
                             num_devices):
    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    vllm_config, quant_config = make_vllm_and_quant_config(
        model, mesh, symmetric)
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
    assert isinstance(linear_layer.scheme, VllmCompressedTensorsWNA16)

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)
