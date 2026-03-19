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
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    pack_quantized_values_into_int32
from vllm.scalar_type import scalar_types

from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.gptq import (VllmGPTQConfig,
                                                         VllmGPTQLinearMethod)

P = PartitionSpec
MODELS = ["RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"]


def ref_quantize_gptq(weight: torch.Tensor, group_size: int):
    """Quantize a weight tensor in GPTQ v1 format (asymmetric, uint4).

    Args:
        weight: shape (input_dim, output_dim), float32
        group_size: number of input channels per quantization group

    Returns:
        x_q: quantized weight, shape (num_groups, group_size, output_dim), int32
        x_z: zero points in GPTQ v1 format (true_zero - 1),
              shape (num_groups, output_dim), int32
        x_s: scales, shape (num_groups, output_dim), float32
    """
    uint4_max = 15

    # Reshape for group quantization: (input, output) -> (num_groups, group_size, output)
    weight = weight.reshape(-1, group_size, weight.shape[-1])

    offset = torch.clamp(-torch.amin(weight, dim=1, keepdim=True), min=0)
    weight = weight + offset

    abs_max = torch.amax(weight, dim=1, keepdim=True)
    x_s = abs_max / uint4_max

    x_q = torch.clip(weight / x_s, 0, uint4_max).to(torch.int32)
    x_z = torch.clip(offset / x_s, 0, uint4_max).to(torch.int32)

    # GPTQ v1: stored zero = true_zero - 1 (AutoGPTQ does `zeros -= 1`)
    x_z = x_z.squeeze(1) - 1

    return x_q, x_z, x_s.squeeze(1).to(torch.float32)


def pack_gptq_weight_into_int32(qweight: torch.Tensor):
    """Pack uint4 qweight into int32 along dim 0 (GPTQ format).

    Args:
        qweight: shape (input_dim, output_dim), int32 with values 0-15
    Returns:
        packed: shape (input_dim/8, output_dim), int32
    """
    return pack_quantized_values_into_int32(qweight, scalar_types.uint4, 0)


def pack_gptq_zeros_into_int32(qzeros: torch.Tensor):
    """Pack uint4 qzeros into int32 along dim 1 (GPTQ format).

    Args:
        qzeros: shape (num_groups, output_dim), int32 with values 0-15
    Returns:
        packed: shape (num_groups, output_dim/8), int32
    """
    return pack_quantized_values_into_int32(qzeros, scalar_types.uint4, 1)


def ref_w4a16_gptq(x: torch.Tensor,
                   w_q: torch.Tensor,
                   w_z: torch.Tensor,
                   w_s: torch.Tensor,
                   b: Optional[torch.Tensor],
                   g_idx: Optional[torch.Tensor] = None):
    """Reference GPTQ v1 W4A16 dequantization and matmul.

    Args:
        x: input activations, (batch, input_dim)
        w_q: quantized weights, (num_groups, group_size, output_dim), int values 0-15
        w_z: zero points in GPTQ v1 format (true_zero - 1), (num_groups, output_dim)
        w_s: scales, (num_groups, output_dim)
        b: optional bias
        g_idx: optional group index tensor, (input_dim,). If provided, uses
            g_idx gather for dequantization (desc_act=True). Otherwise uses
            sequential group layout.
    """
    # GPTQ v1: true_zero = stored_zero + 1 (AutoGPTQ stores true_zero - 1)
    w_z_true = w_z.to(torch.float32) + 1.0

    if g_idx is not None:
        # desc_act=True: use g_idx to gather per-row scales and zeros
        w_flat = w_q.reshape((-1, w_q.shape[-1]))  # (input, output)
        scales_per_row = w_s[g_idx]  # (input, output)
        zeros_per_row = w_z_true[g_idx]  # (input, output)
        w = (w_flat.to(torch.float32) - zeros_per_row) * scales_per_row
    else:
        # desc_act=False: sequential groups
        w = (w_q.to(torch.float32) - w_z_true.unsqueeze(1)) * w_s.unsqueeze(1)
        w = w.reshape((-1, w.shape[-1]))

    out = torch.einsum('bd,df->bf', x.to(torch.float32), w)
    if b is not None:
        out += b
    return out.to(x.dtype)


def return_ref_and_layer_output(
    layer: torch.nn.Module,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    batch_size: int = 16,
    g_idx: Optional[torch.Tensor] = None,
):
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, VllmGPTQLinearMethod)

    input_tensor = torch.rand(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    ref_output = ref_w4a16_gptq(
        input_tensor,
        qweight,
        qzeros,
        scales,
        layer.bias,
        g_idx=g_idx,
    )

    # Run torchax/jax function
    quant_method.process_weights_after_loading(layer)
    with torchax.default_env():
        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        layer_output = layer(jax_input_tensor)
        layer_output = j2t(layer_output.to(torch.float32)).to(torch.bfloat16)

    return ref_output, layer_output


def initialize_and_return_layer_weights(layer: torch.nn.Module,
                                        desc_act: bool = False):
    """Create synthetic GPTQ weights and assign to layer.

    Args:
        layer: The linear layer to initialize.
        desc_act: If True, shuffle g_idx to simulate descending activation
            order (non-contiguous group assignments), matching real GPTQ
            checkpoints with desc_act=True.
    """
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, VllmGPTQLinearMethod)
    quant_config = quant_method.quant_config
    assert isinstance(quant_config, VllmGPTQConfig)

    # Generate random weight and quantize in GPTQ format
    weight = torch.rand((layer.input_size, layer.output_size)) - 0.2
    qweight, qzeros, scales = ref_quantize_gptq(weight,
                                                quant_config.group_size)

    # Pack into int32 for layer assignment
    # qweight: (num_groups, group_size, output) -> flatten to (input, output) -> pack dim 0
    layer_qweight = qweight.reshape((-1, layer.output_size))
    layer_qweight = pack_gptq_weight_into_int32(layer_qweight)

    # qzeros: (num_groups, output) -> pack dim 1
    layer_qzeros = pack_gptq_zeros_into_int32(qzeros)

    # Create g_idx
    g_idx = torch.tensor(
        [i // quant_config.group_size for i in range(layer.input_size)],
        dtype=torch.int32)

    if desc_act:
        # Shuffle g_idx to simulate descending activation order.
        # Use a fixed seed for reproducibility.
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(layer.input_size, generator=rng)
        g_idx = g_idx[perm]

    assert layer.qweight.data.shape == layer_qweight.shape, \
        f"qweight shape mismatch: {layer.qweight.data.shape} vs {layer_qweight.shape}"
    assert layer.qzeros.data.shape == layer_qzeros.shape, \
        f"qzeros shape mismatch: {layer.qzeros.data.shape} vs {layer_qzeros.shape}"
    assert layer.scales.data.shape == scales.shape, \
        f"scales shape mismatch: {layer.scales.data.shape} vs {scales.shape}"

    layer.qweight.data = layer_qweight
    layer.qzeros.data = layer_qzeros
    layer.scales.data = scales
    layer.g_idx.data = g_idx

    bias = None
    if layer.bias is not None:
        bias = torch.rand_like(layer.bias.data)
        layer.bias.data = bias

    return qweight, qzeros, scales, bias, g_idx


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
        model=MODELS[0],
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        dtype='bfloat16',
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
        dtype='bfloat16',
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    assert isinstance(quant_config, VllmGPTQConfig)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


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
        dtype='bfloat16',
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

    qweight, qzeros, scales, _, _ = initialize_and_return_layer_weights(
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
        dtype='bfloat16',
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

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

    qweight, qzeros, scales, _, _ = initialize_and_return_layer_weights(
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
        dtype='bfloat16',
    )
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

    qweight, qzeros, scales, _, _ = initialize_and_return_layer_weights(
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
        dtype='bfloat16',
    )
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

    qweight, qzeros, scales, _, _ = initialize_and_return_layer_weights(
        linear_layer)
    ref_output, layer_output = return_ref_and_layer_output(
        linear_layer, qweight, qzeros, scales)
    torch.testing.assert_close(ref_output, layer_output)


@pytest.mark.parametrize("model", MODELS)
def test_desc_act(model):
    """Test dequantization with shuffled g_idx (desc_act=True).

    Real GPTQ checkpoints with desc_act=True have non-contiguous group
    assignments. This test verifies the g_idx gather logic handles that
    correctly.
    """
    dtype = torch.bfloat16
    mesh = test_utils.get_spmd_mesh(1)

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        dtype='bfloat16',
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    qweight, qzeros, scales, _, g_idx = initialize_and_return_layer_weights(
        linear_layer, desc_act=True)
    ref_output, layer_output = return_ref_and_layer_output(linear_layer,
                                                           qweight,
                                                           qzeros,
                                                           scales,
                                                           g_idx=g_idx)
    torch.testing.assert_close(ref_output, layer_output)
