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
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    pack_quantized_values_into_int32
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.scalar_type import scalar_types

from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.awq import (VllmAWQConfig,
                                                        VllmAWQLinearMethod)
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

from . import utils as test_utils

P = PartitionSpec
MODELS = ["Qwen/Qwen2.5-1.5B-Instruct-AWQ"]


def ref_quantize_uint4(x: torch.Tensor, group_size: int):
    uint4_max = 15

    # For group quantization, we reshape so that x[0], x[1], ... x[i] are
    # quantized with different scale values.
    x = torch.reshape(x, (-1, group_size) + (x.shape[1:]))

    # Equation for asymmetric quantization is x_q = (x + x_z) / scale where
    # x_z is calculated to ensure x + x_z does not contain any negative values.
    offset = torch.clamp(-torch.amin(x, dim=1, keepdim=True), min=0)
    x += offset
    # After adding offset, x will not contain any negative values.
    assert x.min() >= 0

    x_abs_max = torch.amax(x, dim=1, keepdim=True)
    x_s = x_abs_max / uint4_max
    # torch does not support uint4, therefore, we cast to int32 instead.
    x_q = torch.clip(x / x_s, 0, uint4_max).to(torch.int32)
    x_z = torch.clip(offset / x_s, 0, uint4_max).to(torch.int32)
    return x_q, x_z, x_s.to(torch.float32)


def ref_w4a16(x: torch.Tensor, w_q: torch.Tensor, w_z: torch.Tensor,
              w_s: torch.Tensor, b: Optional[torch.Tensor]):
    # Dequantize asymetric quantized weight.
    w = (w_q.to(torch.float32) - w_z.to(torch.float32)) * w_s
    w = w.reshape((-1, w.shape[-1]))
    out = torch.einsum('bd,df->bf', x.to(torch.float32), w)
    if b is not None:
        out += b
    return out.to(x.dtype)


def pack_awq_weight_into_int32(weight: torch.Tensor):
    # AWQ packs 8 uint4 into 32-bits in this order.
    awq_order = (0, 2, 4, 6, 1, 3, 5, 7)

    orig_shape = weight.shape
    weight = weight.reshape(orig_shape[:-1] + (-1, 8))
    weight = weight[..., awq_order].reshape(orig_shape)

    return pack_quantized_values_into_int32(weight, scalar_types.uint4, 1)


def return_ref_and_layer_output(
    layer: torch.nn.Module,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    batch_size: int = 16,
):
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, VllmAWQLinearMethod)
    quant_config = quant_method.quant_config
    assert isinstance(quant_config, VllmAWQConfig)
    jax_config = quant_method.linear_config
    assert isinstance(jax_config, VllmQuantLinearConfig)

    input_tensor = torch.rand(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    ref_output = ref_w4a16(
        input_tensor,
        qweight,
        qzeros,
        scales,
        layer.bias,
    )

    # Run torchax/jax function
    quant_method.process_weights_after_loading(layer)
    with torchax.default_env():
        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        layer_output = layer(jax_input_tensor)
        layer_output = j2t(layer_output.to(torch.float32)).to(torch.bfloat16)

    return ref_output, layer_output


def initialize_and_return_layer_weights(layer: torch.nn.Module):
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, VllmAWQLinearMethod)
    quant_config = quant_method.quant_config
    assert isinstance(quant_config, VllmAWQConfig)
    jax_config = quant_method.linear_config
    assert isinstance(jax_config, VllmQuantLinearConfig)

    # torch.rand returns value in the range of [0, 1). We subtract by 0.2 to
    # simulate asymmetry
    weight = torch.rand((layer.input_size, layer.output_size)) - 0.2
    qweight, qzeros, scales = ref_quantize_uint4(weight,
                                                 quant_config.group_size)

    # We modify uint4 quantized weights into AWQ format.
    layer_qweight = qweight.reshape((-1, layer.output_size))
    layer_qzeros = qzeros.reshape((-1, layer.output_size))
    layer_scales = scales.reshape((-1, layer.output_size))

    layer_qweight = pack_awq_weight_into_int32(layer_qweight)
    layer_qzeros = pack_awq_weight_into_int32(layer_qzeros)

    assert layer.qweight.data.shape == layer_qweight.shape
    assert layer.qzeros.data.shape == layer_qzeros.shape
    assert layer.scales.data.shape == layer_scales.shape

    layer.qweight.data = layer_qweight
    layer.qzeros.data = layer_qzeros
    layer.scales.data = layer_scales

    bias = None
    if layer.bias is not None:
        bias = torch.rand_like(layer.bias.data)
        layer.bias.data = bias

    return qweight, qzeros, scales, bias


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
    assert isinstance(quant_config, VllmAWQConfig)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "mesh",
    [
        test_utils.get_spmd_mesh(1),
        # We limit device count by 2 instead of using all devices (like 8) since
        # AWQ requires n_groups to be divisible by number of shards. Qwen uses
        # group size of 128 and one of the layer has input size of 1536, meaning
        # n_groups = 1536//128 = 12 - which is not divisible by 8.
        test_utils.get_spmd_mesh(min(jax.local_device_count(), 2))
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
        assert isinstance(layer.quant_config, VllmAWQConfig)
        assert isinstance(layer.quant_method, VllmAWQLinearMethod)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_row_parallel_linear(model, bias, mesh, enable_sp):
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

    qweight, qzeros, scales, _ = initialize_and_return_layer_weights(
        linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, qweight, qzeros, scales)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_column_parallel_linear(model, bias, mesh, enable_sp):
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

    qweight, qzeros, scales, _ = initialize_and_return_layer_weights(
        linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, qweight, qzeros, scales)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
def test_qkv_parallel_linear(model, bias, mesh, enable_sp, fuse_matmuls):
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

    qweight, qzeros, scales, _ = initialize_and_return_layer_weights(
        linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, qweight, qzeros, scales)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_merged_column_parallel_linear(model, bias, mesh, fuse_matmuls,
                                       enable_sp):
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

    qweight, qzeros, scales, _ = initialize_and_return_layer_weights(
        linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, qweight, qzeros, scales)
    torch.testing.assert_close(ref_output, layer_output)
