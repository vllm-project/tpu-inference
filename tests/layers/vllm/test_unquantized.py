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
from jax._src import test_util as jtu
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.model_loader import get_model as vllm_get_model

from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedConfig, VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod)

from . import utils as test_utils

P = PartitionSpec
MODELS = ["Qwen/Qwen2-1.5B-Instruct"]


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

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    with set_current_vllm_config(vllm_config):
        row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

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

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    with set_current_vllm_config(vllm_config):
        column_linear = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

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

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

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

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    # Call vLLM code
    with set_current_vllm_config(vllm_config):
        merged_column_linear = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[14336] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

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
        jax_merged_column_linear.quant_method.fuse_matmuls = fuse_matmuls

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

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        vllm_fused_moe = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            renormalize=False,
            tp_size=1,
            dp_size=1,
            quant_config=quant_config,
            has_bias=has_bias,
            activation=activation,
        )
        vllm_fused_moe.moe_parallel_config.use_ep = use_ep
    vllm_fused_moe.w13_weight.data = w1
    vllm_fused_moe.w2_weight.data = w2
    if has_bias:
        vllm_fused_moe.w13_bias.data = w1_bias
        vllm_fused_moe.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.top_k,
                                  vllm_fused_moe.renormalize,
                                  vllm_fused_moe.activation)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.quant_method,
                          VllmUnquantizedFusedMoEMethod)

        jax_a = a.to('jax')
        score = score.to('jax')

        vllm_fused_moe.quant_method.process_weights_after_loading(
            vllm_fused_moe)
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
        tensor_parallel_size=mesh.devices.size)

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        vllm_fused_moe = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            renormalize=False,
            tp_size=mesh.devices.size,
            dp_size=1,
            quant_config=quant_config,
            has_bias=has_bias,
        )
        vllm_fused_moe.moe_parallel_config.use_ep = True
        vllm_fused_moe.quant_method.use_kernel = True

    vllm_fused_moe.w13_weight.data = w1
    vllm_fused_moe.w2_weight.data = w2
    if has_bias:
        vllm_fused_moe.w13_bias.data = w1_bias
        vllm_fused_moe.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.top_k,
                                  vllm_fused_moe.renormalize,
                                  vllm_fused_moe.activation)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.quant_method,
                          VllmUnquantizedFusedMoEMethod)
        jax_a = a.to('jax')
        score = score.to('jax')

        vllm_fused_moe.quant_method.process_weights_after_loading(
            vllm_fused_moe)
        vllm_fused_moe.quant_method.block_size = {
            "bt": 32,
            "bf": 512,
            "bd1": 512,
            "bd2": 512,
            "btc": 32,
            "bfc": 256,
            "bd1c": 256,
            "bd2c": 256,
        }
        actual = vllm_fused_moe(jax_a, score)

        torch.testing.assert_close(
            expected,
            actual,
            check_device=False,
            atol=1e-2,
            rtol=1e-2,
        )
