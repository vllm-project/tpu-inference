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
from unittest import mock
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch
import torchax
from jax._src import test_util as jtu
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.model_loader import get_model as vllm_get_model

from tests.layers.common import utils as test_utils
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.vllm.custom_ops.fused_moe import _all_reduce_over_tp
from tpu_inference.layers.vllm.interface.moe import FusedMoEFactory
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmQuantizedBf16LinearMethod, VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod, VllmUnquantizedLinearMethod,
    _host_numpy_view, _load_weight_for_layer, _load_weight_on_host,
    should_quantize_bf16_linear)

P = PartitionSpec
MODELS = ["Qwen/Qwen2-1.5B-Instruct"]

# Qwen3.5's packed_modules_mapping, inlined so these tests don't depend on the
# model definition.
QWEN3_5_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
# Every attention and gated-delta-net projection, named the way the checkpoint
# names them rather than the way vLLM fuses them.
QWEN3_5_ATTN_PATTERNS = (
    r"re:.*self_attn\..*,"
    r"re:.*linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$")


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
    assert isinstance(quant_config, VllmUnquantizedConfig)
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

    with set_current_vllm_config(vllm_config):
        vllm_model = vllm_get_model(vllm_config=vllm_config)
    layers = test_utils.find_all_layer_type(vllm_model, LinearBase)
    for layer in layers:
        assert isinstance(layer.quant_config, VllmUnquantizedConfig)
        assert isinstance(layer.quant_method, VllmUnquantizedLinearMethod)


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

    with set_current_vllm_config(vllm_config):
        row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = torch.rand(10, row_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    weight_data = torch.rand_like(row_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(row_linear.bias.data)

    row_linear.weight.data = weight_data
    if bias:
        row_linear.bias.data = bias_data
    row_linear = row_linear.to('cpu')
    row_linear.quant_method.process_weights_after_loading(row_linear)
    output = row_linear(input_tensor).to(dtype)

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

    jax_row_linear.weight.data = weight_data
    if bias:
        jax_row_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_row_linear.quant_method,
                          VllmUnquantizedLinearMethod)
        jax_row_linear.quant_method.process_weights_after_loading(
            jax_row_linear)
        jax_output = jax_row_linear(jax_input_tensor)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
def test_row_parallel_linear_defer_all_reduce(model, num_devices):
    """reduce_results=False routes through sharded_matmul: the layer returns
    per-shard partial sums (the psum is actually skipped, which a plain einsum
    under GSPMD cannot do) and the caller's single deferred all-reduce
    (VllmMoERunner._all_reduce_over_tp) reconstructs the full result.

    No bias: vLLM's RowParallelLinear rejects reduce_results=False with an
    in-layer bias."""
    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()

    # Reference: fully reduced output.
    with set_current_vllm_config(vllm_config):
        row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = torch.rand(10, row_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    weight_data = torch.rand_like(row_linear.weight.data) / 10
    row_linear.weight.data = weight_data
    row_linear = row_linear.to('cpu')
    row_linear.quant_method.process_weights_after_loading(row_linear)
    expected = row_linear(input_tensor).to(dtype)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=False,
            params_dtype=dtype,
            reduce_results=False,
            return_bias=False,
            quant_config=quant_config,
        )
    # What VllmRowParallelLinear.__init__ derives from reduce_results=False.
    jax_row_linear.quant_method.linear_config.defer_all_reduce = True

    jax_row_linear.weight.data = weight_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_row_linear.quant_method,
                          VllmUnquantizedLinearMethod)
        jax_row_linear.quant_method.process_weights_after_loading(
            jax_row_linear)
        deferred = jax_row_linear(jax_input_tensor)

        if num_devices > 1:
            # The psum was actually skipped: pre-reduction output is partial,
            # not the full matmul.
            partial = j2t(deferred.to(torch.float32)).to(dtype)
            assert not torch.allclose(partial, expected)

        reduced = _all_reduce_over_tp(deferred, mesh)
        jax_output = j2t(reduced.to(torch.float32)).to(dtype)

    torch.testing.assert_close(expected, jax_output)


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

    with set_current_vllm_config(vllm_config):
        column_linear = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = torch.rand(10, column_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    weight_data = torch.rand_like(column_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(column_linear.bias.data)

    column_linear.weight.data = weight_data
    if bias:
        column_linear.bias.data = bias_data
    column_linear = column_linear.to('cpu')
    column_linear.quant_method.process_weights_after_loading(column_linear)
    output = column_linear(input_tensor).to(dtype)

    vllm_config.model_config.dtype = dtype
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

    jax_column_linear.weight.data = weight_data
    if bias:
        jax_column_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_column_linear.quant_method,
                          VllmUnquantizedLinearMethod)
        jax_column_linear.quant_method.process_weights_after_loading(
            jax_column_linear)
        jax_output = jax_column_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

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

    with set_current_vllm_config(vllm_config):
        qkv_linear = QKVParallelLinear(
            hidden_size=4096,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = torch.rand(10, qkv_linear.input_size, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    weight_data = torch.rand_like(qkv_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(qkv_linear.bias.data)

    qkv_linear.weight.data = weight_data
    if bias:
        qkv_linear.bias.data = bias_data
    qkv_linear = qkv_linear.to('cpu')
    qkv_linear.quant_method.process_weights_after_loading(qkv_linear)
    output = qkv_linear(input_tensor).to(dtype)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    vllm_config.quant_config = quant_config
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

    jax_qkv_linear.weight.data = weight_data
    if bias:
        jax_qkv_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_qkv_linear.quant_method,
                          VllmUnquantizedLinearMethod)
        jax_qkv_linear.quant_method.process_weights_after_loading(
            jax_qkv_linear)
        jax_output = jax_qkv_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

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

    # Call vLLM code
    with set_current_vllm_config(vllm_config):
        merged_column_linear = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[14336] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = torch.rand(10, merged_column_linear.input_size,
                              dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    weight_data = torch.rand_like(merged_column_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(merged_column_linear.bias.data)

    merged_column_linear.weight.data = weight_data
    if bias:
        merged_column_linear.bias.data = bias_data
    merged_column_linear = merged_column_linear.to('cpu')
    merged_column_linear.quant_method.process_weights_after_loading(
        merged_column_linear)
    output = merged_column_linear(input_tensor).to(dtype)

    # Call tpu_inference code
    vllm_config.model_config.dtype = dtype
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
        assert isinstance(jax_merged_column_linear.quant_method.linear_config,
                          QuantLinearConfig)
        jax_merged_column_linear.quant_method.linear_config.fuse_matmuls = fuse_matmuls

    jax_merged_column_linear.weight.data = weight_data
    if bias:
        jax_merged_column_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_merged_column_linear.quant_method,
                          VllmUnquantizedLinearMethod)
        jax_merged_column_linear.quant_method.process_weights_after_loading(
            jax_merged_column_linear)
        jax_output = jax_merged_column_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.parametrize("use_ep", [True, False])
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize("hidden_size", [128, 512])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("activation", ["silu", "swigluoai"])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_fused_moe(use_ep, num_devices, num_tokens, intermediate_size,
                   hidden_size, num_experts, topk, has_bias, activation,
                   enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)

    torch.manual_seed(42)
    dtype = torch.bfloat16

    a = torch.randn((num_tokens, hidden_size), dtype=dtype) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), dtype=dtype) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size), dtype=dtype) / 10
    score = torch.randn((num_tokens, num_experts), dtype=dtype)

    w1_bias = w2_bias = None
    if has_bias:
        w1_bias = torch.randn(
            (num_experts, 2 * intermediate_size), dtype=dtype) / 10
        w2_bias = torch.randn((num_experts, hidden_size), dtype=dtype) / 10

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=mesh.devices.size, enable_expert_parallel=use_ep)

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        vllm_fused_moe = FusedMoEFactory(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=False,
            tp_size=1,
            dp_size=1,
            quant_config=quant_config,
            has_bias=has_bias,
            activation=activation,
        )
        vllm_fused_moe.moe_config.moe_parallel_config.use_ep = use_ep
    vllm_fused_moe.routed_experts.w13_weight.data = w1
    vllm_fused_moe.routed_experts.w2_weight.data = w2
    if has_bias:
        vllm_fused_moe.routed_experts.w13_bias.data = w1_bias
        vllm_fused_moe.routed_experts.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.routed_experts.top_k,
                                  vllm_fused_moe.routed_experts.renormalize,
                                  vllm_fused_moe.activation.value)

    with torchax.default_env(), set_forward_context(
            None, vllm_config), jax.set_mesh(mesh):
        assert isinstance(vllm_fused_moe.routed_experts.quant_method,
                          VllmUnquantizedFusedMoEMethod)
        if use_ep:
            assert vllm_fused_moe.routed_experts.quant_method.moe_backend == MoEBackend.GMM_EP
        else:
            assert vllm_fused_moe.routed_experts.quant_method.moe_backend == MoEBackend.GMM_TP

        jax_a = a.to('jax')
        score = score.to('jax')

        vllm_fused_moe.routed_experts.quant_method.process_weights_after_loading(
            vllm_fused_moe.routed_experts)
        actual = vllm_fused_moe(jax_a, score)

        torch.testing.assert_close(expected,
                                   actual,
                                   check_device=False,
                                   atol=1e-1,
                                   rtol=1e-1)


@pytest.mark.parametrize("num_devices", [jax.local_device_count()])
@pytest.mark.parametrize("num_tokens", [128, 512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
@mock.patch("os.environ", {"USE_MOE_EP_KERNEL": "1"})
def test_fused_moe_use_kernel(num_devices, num_tokens, intermediate_size,
                              hidden_size, num_experts, topk, has_bias,
                              enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    # Skip attn_dp tests for fused_moe_use_kernel since the kernel only supports 2D mesh
    if enable_attn_dp:
        pytest.skip(
            "fused_moe kernel does not support attn_dp (requires 2D mesh)")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)

    # TODO(Qiliang Cui): Remove when issue is resolved.
    if not jtu.is_device_tpu_at_least(version=7):
        pytest.skip(allow_module_level=True, reason="Expected TPUv7+")

    torch.manual_seed(42)
    dtype = torch.bfloat16

    a = torch.randn((num_tokens, hidden_size), dtype=dtype) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), dtype=dtype) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size), dtype=dtype) / 10

    w1_bias = w2_bias = None
    if has_bias:
        w1_bias = torch.randn(
            (num_experts, 2 * intermediate_size), dtype=dtype) / 10
        w2_bias = torch.randn((num_experts, hidden_size), dtype=dtype) / 10

    # Use deterministic gating_output generation (same logic as fused_moe_v1_test.py)
    # Generate base gating scores with deterministic pattern
    score = (
        torch.randn((num_tokens, num_experts), dtype=torch.float32) +
        torch.arange(num_tokens * num_experts, dtype=torch.float32).reshape(
            num_tokens, num_experts) / 100)

    # Generate unique top-k indices
    generator = torch.Generator()
    generator.manual_seed(42)
    top_k_indices = torch.randint(0,
                                  num_experts - 1, (num_tokens, topk),
                                  dtype=torch.int32,
                                  generator=generator)

    # Add one-hot encoding weighted by 10 to ensure selected experts have highest scores
    one_hot = torch.nn.functional.one_hot(top_k_indices.long(),
                                          num_classes=num_experts).float()
    one_hot = one_hot.sum(dim=1) * 10
    score = (score + one_hot).to(dtype)

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    vllm_config.parallel_config = ParallelConfig(
        tensor_parallel_size=mesh.devices.size, enable_expert_parallel=True)

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        vllm_fused_moe = FusedMoEFactory(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=False,
            tp_size=mesh.devices.size,
            dp_size=1,
            quant_config=quant_config,
            has_bias=has_bias,
        )
        vllm_fused_moe.moe_config.moe_parallel_config.use_ep = True

    vllm_fused_moe.routed_experts.w13_weight.data = w1
    vllm_fused_moe.routed_experts.w2_weight.data = w2
    if has_bias:
        vllm_fused_moe.routed_experts.w13_bias.data = w1_bias
        vllm_fused_moe.routed_experts.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.routed_experts.top_k,
                                  vllm_fused_moe.routed_experts.renormalize,
                                  vllm_fused_moe.activation.value)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.routed_experts.quant_method,
                          VllmUnquantizedFusedMoEMethod)
        assert vllm_fused_moe.routed_experts.quant_method.moe_backend == MoEBackend.FUSED_MOE

        jax_a = a.to('jax')
        score = score.to('jax')

        vllm_fused_moe.routed_experts.quant_method.process_weights_after_loading(
            vllm_fused_moe.routed_experts)
        vllm_fused_moe.routed_experts.quant_method.extra_backend_kwargs.update(
            {
                "bt": 32,
                "bf": 512,
                "bd1": 512,
                "bd2": 512,
                "btc": 32,
                "bfc": 256,
                "bd1c": 256,
                "bd2c": 256,
            })
        actual = vllm_fused_moe(jax_a, score)

        torch.testing.assert_close(
            expected,
            actual,
            check_device=False,
            atol=1e-2,
            rtol=1e-2,
        )


# --- _load_weight_for_layer tests ---


def _make_layer_with_weight(shape, dtype):
    """Create a simple torch.nn.Module with a 'weight' attribute."""
    layer = torch.nn.Module()
    # `.to(dtype)` rather than `torch.randn(dtype=...)` so the fp8 formats,
    # which randn cannot generate directly, work here too.
    layer.weight = torch.randn(shape).to(dtype)
    return layer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", False)
def test_load_weight_for_layer_non_pathways(dtype):
    """_load_weight_for_layer falls back to t2j when not using Pathways."""
    layer = _make_layer_with_weight((4, 8), dtype)
    mesh = test_utils.get_spmd_mesh(1)
    sharding = NamedSharding(mesh, P(None, None))
    result = _load_weight_for_layer(layer, "weight", sharding)
    expected = t2j(layer.weight, use_dlpack=False)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@patch(
    "tpu_inference.layers.vllm.quantization.unquantized.is_pathways_dummy_load",
    return_value=False)
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", True)
def test_load_weight_for_layer_pathways_real_weights(_, dtype):
    """_load_weight_for_layer converts via numpy + device_put under Pathways (real weights)."""
    layer = _make_layer_with_weight((4, 8), dtype)
    mesh = test_utils.get_spmd_mesh(1)
    sharding = NamedSharding(mesh, P(None, None))
    result = _load_weight_for_layer(layer, "weight", sharding)
    # Check shape and values are preserved (converted through float32 intermediate)
    assert result.shape == (4, 8)
    # Under Pathways path the dtype is converted via to_jax_dtype
    from tpu_inference.utils import to_jax_dtype
    assert result.dtype == to_jax_dtype(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@patch(
    "tpu_inference.layers.vllm.quantization.unquantized.is_pathways_dummy_load",
    return_value=True)
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", True)
def test_load_weight_for_layer_pathways_dummy(_, dtype):
    """_load_weight_for_layer creates dummy weights on TPU when in pathways dummy mode."""
    layer = _make_layer_with_weight((4, 8), dtype)
    mesh = test_utils.get_spmd_mesh(1)
    sharding = NamedSharding(mesh, P(None, None))
    result = _load_weight_for_layer(layer, "weight", sharding)
    assert result.shape == (4, 8)
    from tpu_inference.utils import to_jax_dtype
    assert result.dtype == to_jax_dtype(dtype)
    # The original tensor's storage should have been freed
    assert layer.weight.untyped_storage().size() == 0


# --- host staging tests ---

# The dtypes weights actually arrive in: one numpy can represent natively and
# three it cannot, which travel as uint8 and are reinterpreted.
STAGED_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

# CI runs these on a single- or dual-chip host, so nothing below may assume a
# mesh of a particular width: shapes are sized off the mesh so every spec here
# splits evenly, and the one case that needs a sharding to *not* split says so.
NUM_DEVICES = jax.local_device_count()
STAGED_SHAPE = (8 * NUM_DEVICES, 16 * NUM_DEVICES)


@pytest.mark.parametrize("dtype", STAGED_DTYPES)
def test_host_numpy_view_preserves_shape_dtype_and_values(dtype):
    """The numpy view sees the same weight t2j would produce."""
    tensor = torch.randn(4, 8).to(dtype)
    view = _host_numpy_view(tensor)
    expected = t2j(tensor, use_dlpack=False)
    assert view.shape == (4, 8)
    assert view.dtype == expected.dtype
    np.testing.assert_array_equal(view, np.asarray(expected))


def test_host_numpy_view_is_zero_copy():
    """The whole point is not to copy the weight on the way out."""
    tensor = torch.randn(4, 8)
    assert np.shares_memory(_host_numpy_view(tensor), tensor.numpy())


def test_host_numpy_view_rejects_non_cpu_tensor():
    """A tensor with no host storage to view has to fall back."""
    assert _host_numpy_view(torch.empty(4, 8, device="meta")) is None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_host_numpy_view_rejects_unbitcastable_layouts(dtype):
    """torch can only reinterpret as uint8 on contiguous, non-scalar data."""
    assert _host_numpy_view(torch.randn(4, 8).to(dtype).t()) is None
    assert _host_numpy_view(torch.tensor(1.0, dtype=dtype)) is None
    # A dtype numpy represents natively needs neither, so it still goes.
    assert _host_numpy_view(torch.randn(4, 8).t()) is not None


@pytest.mark.parametrize("dtype", STAGED_DTYPES)
@pytest.mark.parametrize("spec", [P(None, None), P(None, "model"), P("model")])
def test_load_weight_on_host_matches_device_staging(dtype, spec):
    """Host staging returns the same array the t2j-then-put path would."""
    tensor = torch.randn(*STAGED_SHAPE).to(dtype)
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, spec)

    result = _load_weight_on_host(tensor, sharding)

    expected = jax.device_put(t2j(tensor, use_dlpack=False), sharding)
    assert result is not None
    assert result.dtype == expected.dtype
    assert result.sharding == sharding
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@patch("jax.block_until_ready")
def test_load_weight_on_host_blocks_before_returning(mock_block):
    """The staged copy is forced here, not left to whoever consumes the array.

    The numpy view aliases torch storage that callers free the moment this
    returns, so the transfer has to have landed before it does. Nothing
    observable distinguishes that from a deferred copy: `device_put` from a
    host buffer is synchronous on the CPU backend CI runs, so reading the
    array back -- or even scribbling over the source buffer first -- passes
    either way. Hence the assertion on the barrier itself.
    """
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)

    result = _load_weight_on_host(torch.randn(*STAGED_SHAPE),
                                  NamedSharding(mesh, P(None, "model")))

    mock_block.assert_called_once_with(result)


@pytest.mark.skipif(NUM_DEVICES < 2,
                    reason="a one-device mesh divides every shape")
def test_load_weight_on_host_falls_back_when_sharding_does_not_divide():
    """Block scales aren't always divisible along the axes their weight is."""
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, P("model", None))
    # One row more than the mesh can hand out evenly.
    rows = NUM_DEVICES + 1
    assert _load_weight_on_host(torch.randn(rows, 8), sharding) is None


@patch("tpu_inference.layers.vllm.quantization.unquantized.general_device_put",
       side_effect=ValueError("cannot shard"))
def test_load_weight_on_host_falls_back_when_the_put_is_refused(_):
    """Whatever the put objects to, staging declines instead of raising.

    Same contract as the test above, reached by making the put fail outright
    rather than by handing it an indivisible shape, so single-chip runners --
    where every shape divides -- still cover the fallback.
    """
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, P("model", None))
    assert _load_weight_on_host(torch.randn(*STAGED_SHAPE), sharding) is None


@pytest.mark.parametrize("dtype", STAGED_DTYPES)
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", False)
def test_load_weight_for_layer_stage_on_host(dtype):
    """stage_on_host produces the same weight, already at `sharding`."""
    layer = _make_layer_with_weight(STAGED_SHAPE, dtype)
    expected = t2j(layer.weight, use_dlpack=False)
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, P(None, "model"))

    result = _load_weight_for_layer(layer,
                                    "weight",
                                    sharding,
                                    stage_on_host=True)

    assert result.dtype == expected.dtype
    assert result.sharding == sharding
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@patch(
    "tpu_inference.layers.vllm.quantization.unquantized._load_weight_on_host",
    return_value=None)
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", False)
def test_load_weight_for_layer_stage_on_host_falls_back_to_t2j(mock_on_host):
    """An unstageable weight still loads, just through the slower path."""
    layer = _make_layer_with_weight((3, 8), torch.float32)
    expected = t2j(layer.weight, use_dlpack=False)
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, P("model", None))

    result = _load_weight_for_layer(layer,
                                    "weight",
                                    sharding,
                                    stage_on_host=True)

    mock_on_host.assert_called_once()
    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


@patch(
    "tpu_inference.layers.vllm.quantization.unquantized._load_weight_on_host")
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", False)
def test_load_weight_for_layer_does_not_stage_on_host_by_default(mock_on_host):
    """Callers that don't ask for host staging keep the old behaviour."""
    layer = _make_layer_with_weight(STAGED_SHAPE, torch.bfloat16)
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, P(None, "model"))

    _load_weight_for_layer(layer, "weight", sharding)

    mock_on_host.assert_not_called()


@patch(
    "tpu_inference.layers.vllm.quantization.unquantized._load_weight_on_host")
@patch(
    "tpu_inference.layers.vllm.quantization.unquantized.is_pathways_dummy_load",
    return_value=False)
@patch("vllm.envs.VLLM_TPU_USING_PATHWAYS", True)
def test_load_weight_for_layer_stage_on_host_ignored_under_pathways(
        _, mock_on_host):
    """Pathways does its own transfer; host staging must not intercept it."""
    layer = _make_layer_with_weight(STAGED_SHAPE, torch.bfloat16)
    mesh = test_utils.get_spmd_mesh(NUM_DEVICES)
    sharding = NamedSharding(mesh, P(None, "model"))

    result = _load_weight_for_layer(layer,
                                    "weight",
                                    sharding,
                                    stage_on_host=True)

    mock_on_host.assert_not_called()
    assert result.shape == STAGED_SHAPE


@pytest.mark.parametrize(("prefix", "expected"), [
    ("model.layers.3.self_attn.qkv_proj", True),
    ("model.layers.3.self_attn.o_proj", True),
    ("model.layers.0.linear_attn.in_proj_qkvz", True),
    ("model.layers.0.linear_attn.in_proj_ba", True),
    ("model.layers.0.linear_attn.out_proj", True),
    ("model.layers.0.linear_attn.conv1d", False),
    ("model.layers.0.mlp.gate", False),
    ("model.layers.0.mlp.shared_expert.gate_up_proj", False),
    ("model.layers.0.mlp.shared_expert_gate", False),
    ("lm_head", False),
])
def test_quantize_bf16_linear_pattern_match(monkeypatch, prefix, expected):
    """Checkpoint-level names have to reach the modules vLLM actually builds.

    Nothing in the checkpoint is called `qkv_proj`, `in_proj_qkvz` or
    `in_proj_ba`, so selecting those depends on expanding them back into the
    shards they fuse."""
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS", QWEN3_5_ATTN_PATTERNS)
    assert should_quantize_bf16_linear(prefix,
                                       QWEN3_5_FUSED_MAPPING) is expected


def test_quantize_bf16_linear_no_patterns_selects_nothing(monkeypatch):
    monkeypatch.delenv("QUANTIZE_BF16_LINEAR_PATTERNS", raising=False)
    assert not should_quantize_bf16_linear("model.layers.3.self_attn.qkv_proj",
                                           QWEN3_5_FUSED_MAPPING)


def test_quantize_bf16_linear_bare_pattern_is_the_whole_name(monkeypatch):
    """Bare patterns are exact layer names, per `is_equal_or_regex_match`.

    A suffix has to be spelled as a regex, so the bare form cannot quietly
    select more layers than it names."""
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS",
                       "model.layers.3.self_attn.o_proj,o_proj")
    assert should_quantize_bf16_linear("model.layers.3.self_attn.o_proj",
                                       QWEN3_5_FUSED_MAPPING)
    assert not should_quantize_bf16_linear("model.layers.4.self_attn.o_proj",
                                           QWEN3_5_FUSED_MAPPING)


def test_quantize_bf16_linear_regex_pattern_matches_suffix(monkeypatch):
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS",
                       r"re:.*\.o_proj$,re:.*\.down_proj$")
    assert should_quantize_bf16_linear("model.layers.3.self_attn.o_proj",
                                       QWEN3_5_FUSED_MAPPING)
    assert not should_quantize_bf16_linear("model.layers.3.self_attn.qkv_proj",
                                           QWEN3_5_FUSED_MAPPING)


def test_quantize_bf16_linear_partial_fused_shard_raises(monkeypatch):
    """One fused weight cannot be half fp8, so a half-selection is an error
    rather than a silent choice either way.

    The `$` is load-bearing: `re:` patterns are anchored at the start only, so
    an unanchored `.*\\.in_proj_b` would match `in_proj_ba` itself and select the
    fused weight outright instead of half of it."""
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS", r"re:.*\.in_proj_b$")
    with pytest.raises(ValueError, match="some but not all shards"):
        should_quantize_bf16_linear("model.layers.0.linear_attn.in_proj_ba",
                                    QWEN3_5_FUSED_MAPPING)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("w8a8", [False, True])
@pytest.mark.parametrize("block_size", [None, 128])
def test_quantized_bf16_merged_column_parallel_linear(monkeypatch, model,
                                                      num_devices,
                                                      fuse_matmuls, w8a8,
                                                      block_size):
    """A bf16 checkpoint weight is quantized on its way to the device.

    The layer keeps loading and sharding exactly as the unquantized one does;
    what changes is that the parameter that lands on the device is fp8 with a
    per-output-channel scale (or a blockwise one), and the result still tracks
    the bf16 matmul."""
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS", QWEN3_5_ATTN_PATTERNS)
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_W8A8", "1" if w8a8 else "0")
    if block_size is not None:
        monkeypatch.setenv("QUANTIZE_BF16_LINEAR_BLOCK_SIZE", str(block_size))

    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16
    prefix = "model.layers.0.linear_attn.in_proj_qkvz"
    output_sizes = [512, 512]

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        ref_linear = MergedColumnParallelLinear(
            input_size=1024,
            output_sizes=output_sizes,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = (torch.rand(10, ref_linear.input_size, dtype=dtype) /
                    10).to('cpu')
    weight_data = torch.rand_like(ref_linear.weight.data) / 10
    ref_linear.weight.data = weight_data
    ref_linear = ref_linear.to('cpu')
    ref_linear.quant_method.process_weights_after_loading(ref_linear)
    expected = ref_linear(input_tensor).to(torch.float32)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    quant_config.packed_modules_mapping = QWEN3_5_FUSED_MAPPING
    with set_current_vllm_config(vllm_config):
        jax_linear = MergedColumnParallelLinear(
            input_size=1024,
            output_sizes=output_sizes,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        assert isinstance(jax_linear.quant_method,
                          VllmQuantizedBf16LinearMethod)
        jax_linear.quant_method.linear_config.fuse_matmuls = fuse_matmuls

    jax_linear.weight.data = weight_data
    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        jax_linear.quant_method.process_weights_after_loading(jax_linear)

        weights = ([jax_linear.weight]
                   if fuse_matmuls else list(jax_linear.weight))
        scales = ([jax_linear.weight_scale]
                  if fuse_matmuls else list(jax_linear.weight_scale))
        for weight, scale in zip(weights, scales):
            assert weight.dtype == torch.float8_e4m3fn
            if block_size is None:
                # One scale per output feature, not per tensor.
                assert scale.shape == (weight.shape[-1], )
            else:
                # The kernel layout: [1, n_blocks, 1, out].
                n_blocks = jax_linear.input_size // block_size
                assert scale.shape == (1, n_blocks, 1, weight.shape[-1])
        assert sum(w.shape[-1] for w in weights) == sum(output_sizes)

        jax_output = j2t(jax_linear(jax_input_tensor).to(torch.float32))

    torch.testing.assert_close(expected, jax_output, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
@pytest.mark.parametrize("block_size", [None, 128])
def test_quantized_bf16_row_parallel_linear(monkeypatch, model, num_devices,
                                            block_size):
    """Row-parallel shards the contracting axis, so the per-output-channel
    scale is replicated and the psum still sums like-scaled partial products.
    A blockwise scale shards along that axis with the weight instead."""
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS", QWEN3_5_ATTN_PATTERNS)
    if block_size is not None:
        monkeypatch.setenv("QUANTIZE_BF16_LINEAR_BLOCK_SIZE", str(block_size))

    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        ref_linear = RowParallelLinear(
            input_size=1024,
            output_size=512,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
        )

    input_tensor = (torch.rand(10, ref_linear.input_size, dtype=dtype) /
                    10).to('cpu')
    weight_data = torch.rand_like(ref_linear.weight.data) / 10
    ref_linear.weight.data = weight_data
    ref_linear = ref_linear.to('cpu')
    ref_linear.quant_method.process_weights_after_loading(ref_linear)
    expected = ref_linear(input_tensor).to(torch.float32)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    quant_config.packed_modules_mapping = QWEN3_5_FUSED_MAPPING
    with set_current_vllm_config(vllm_config):
        jax_linear = RowParallelLinear(
            input_size=1024,
            output_size=512,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix="model.layers.0.linear_attn.out_proj",
        )
        assert isinstance(jax_linear.quant_method,
                          VllmQuantizedBf16LinearMethod)

    jax_linear.weight.data = weight_data
    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        jax_linear.quant_method.process_weights_after_loading(jax_linear)
        assert jax_linear.weight.dtype == torch.float8_e4m3fn
        if block_size is None:
            assert jax_linear.weight_scale.shape == (512, )
        else:
            assert jax_linear.weight_scale.shape == (1, 1024 // block_size, 1,
                                                     512)
        jax_output = j2t(jax_linear(jax_input_tensor).to(torch.float32))

    torch.testing.assert_close(expected, jax_output, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "block_size, input_size, match",
    [
        # 1024 input features do not split into blocks of 384.
        (384, 1024, "does not divide"),
        # 1024 / 512 = 2 blocks, which 8 shards of the contracting axis cannot
        # divide.
        (512, 1024, "cannot shard"),
    ])
def test_quantized_bf16_block_size_rejected(monkeypatch, model, block_size,
                                            input_size, match):
    """A block size the shape or the sharding cannot support is refused at load
    time, naming the env var rather than failing deep inside the jit."""
    if jax.local_device_count() < 8:
        pytest.skip("needs 8 devices to shard the contracting axis 8 ways")
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS", QWEN3_5_ATTN_PATTERNS)
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_BLOCK_SIZE", str(block_size))

    mesh = test_utils.get_spmd_mesh(8)
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    quant_config.packed_modules_mapping = QWEN3_5_FUSED_MAPPING

    with set_current_vllm_config(vllm_config):
        jax_linear = RowParallelLinear(
            input_size=input_size,
            output_size=512,
            bias=False,
            params_dtype=torch.bfloat16,
            return_bias=False,
            quant_config=quant_config,
            prefix="model.layers.0.linear_attn.out_proj",
        )
    jax_linear.weight.data = torch.rand_like(jax_linear.weight.data) / 10
    with torchax.default_env(), pytest.raises(ValueError, match=match):
        jax_linear.quant_method.process_weights_after_loading(jax_linear)


@pytest.mark.parametrize("model", MODELS)
def test_unselected_linear_stays_unquantized(monkeypatch, model):
    monkeypatch.setenv("QUANTIZE_BF16_LINEAR_PATTERNS", QWEN3_5_ATTN_PATTERNS)
    mesh = test_utils.get_spmd_mesh(1)

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    quant_config.packed_modules_mapping = QWEN3_5_FUSED_MAPPING

    with set_current_vllm_config(vllm_config):
        layer = MergedColumnParallelLinear(
            input_size=1024,
            output_sizes=[512, 512],
            bias=False,
            params_dtype=torch.bfloat16,
            return_bias=False,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.shared_expert.gate_up_proj",
        )
    assert isinstance(layer.quant_method, VllmUnquantizedLinearMethod)
    assert not isinstance(layer.quant_method, VllmQuantizedBf16LinearMethod)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_unquantized_linear_stores_no_scale(monkeypatch, model, fuse_matmuls,
                                            bias):
    """The shared store step hangs a scale off the layer only when the build
    step produced one, so an unquantized layer keeps a bare bf16 weight."""
    monkeypatch.delenv("QUANTIZE_BF16_LINEAR_PATTERNS", raising=False)
    mesh = test_utils.get_spmd_mesh(1)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)

    with set_current_vllm_config(vllm_config):
        layer = MergedColumnParallelLinear(
            input_size=1024,
            output_sizes=[512, 512],
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.shared_expert.gate_up_proj",
        )
        assert isinstance(layer.quant_method, VllmUnquantizedLinearMethod)
        layer.quant_method.linear_config.fuse_matmuls = fuse_matmuls

    layer.weight.data = torch.rand_like(layer.weight.data) / 10
    if bias:
        layer.bias.data = torch.rand_like(layer.bias.data) / 10
    with torchax.default_env():
        layer.quant_method.process_weights_after_loading(layer)

    weights = [layer.weight] if fuse_matmuls else list(layer.weight)
    assert all(w.dtype == dtype for w in weights)
    assert getattr(layer, "weight_scale", None) is None
    if bias:
        biases = [layer.bias] if fuse_matmuls else list(layer.bias)
        assert all(b.dtype == dtype for b in biases)
    else:
        assert layer.bias is None


def test_quantized_bf16_only_overrides_the_build_step():
    """Quantizing changes what the weight is turned into, not how it is loaded
    or stored -- keep those three steps from drifting back into a copy."""
    for name in ("process_weights_after_loading", "_load_linear_weights",
                 "_store_linear_weights"):
        inherited = getattr(VllmUnquantizedLinearMethod, name)
        assert getattr(VllmQuantizedBf16LinearMethod, name) is inherited, (
            f"{name} should be inherited, not reimplemented")
    assert (VllmQuantizedBf16LinearMethod._build_linear_weights
            is not VllmUnquantizedLinearMethod._build_linear_weights)
