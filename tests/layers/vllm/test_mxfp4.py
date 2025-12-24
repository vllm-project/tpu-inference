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
import jax.numpy as jnp
import pytest
import torch
import torchax
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec
from torchax.ops.mappings import j2t, t2j
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.mxfp4 import (VllmMxfp4Config,
                                                          VllmMxfp4MoEMethod)

from . import utils as test_utils

P = PartitionSpec
MODELS = ["openai/gpt-oss-20b"]
MXFP4_BLOCK_SIZE = 32

if not jtu.is_device_tpu_at_least(version=7):
    pytest.skip(allow_module_level=True, reason="Expected TPUv7+")


def quantize_to_mxfp4(weight: torch.tensor):
    # Utilize JAX because native support for e2m1 makes it easier to work with.
    weight = t2j(weight)
    e2m1_finfo = jnp.finfo(jnp.float4_e2m1fn)
    dtype_min = float(e2m1_finfo.min)
    dtype_max = float(e2m1_finfo.max)

    # Do a subchannel quantization where block size is 32.
    weight_shape = weight.shape
    weight_block = weight.reshape(weight_shape[:-1] + (-1, MXFP4_BLOCK_SIZE))
    abs_max = jnp.max(jnp.abs(weight_block), axis=-1, keepdims=True)
    scale = abs_max / dtype_max

    weight_q = jnp.clip(weight_block / scale, dtype_min, dtype_max)
    weight_q = weight_q.astype(jnp.float4_e2m1fn).reshape(weight_shape[:-1] +
                                                          (-1, 2))
    weight_packed = jax.lax.bitcast_convert_type(weight_q, jnp.uint8)

    # We convert scale into e8m0 manually because there is no hardware support.
    e8m0_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    _, scale_exp = jnp.frexp(scale.squeeze(axis=-1))
    # Subtract by one sinced e8m0 has no decimal
    scale_exp -= 1
    scale_exp = (scale_exp - e8m0_finfo.minexp).astype(jnp.uint8)

    return j2t(weight_packed), j2t(scale_exp)


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # RowParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model=MODELS[0],
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        load_format='dummy',
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
        load_format='dummy',
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    assert isinstance(quant_config, VllmMxfp4Config)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("use_ep", [True, False])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_mxfp4_fused_moe(num_devices, num_tokens, intermediate_size,
                         hidden_size, num_experts, topk, use_ep,
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
    w1_weight, w1_weight_scale = quantize_to_mxfp4(w1)
    w2_weight, w2_weight_scale = quantize_to_mxfp4(w2)

    w1_bias = torch.randn(
        (num_experts, 2 * intermediate_size), dtype=dtype) / 10
    w2_bias = torch.randn((num_experts, hidden_size), dtype=dtype) / 10
    score = torch.randn((num_tokens, num_experts), dtype=dtype)

    engine_args = EngineArgs(
        model=MODELS[0],
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        load_format='dummy',
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
            has_bias=True,
        )
        vllm_fused_moe.moe_parallel_config.use_ep = use_ep
    vllm_fused_moe.w13_weight.data = w1_weight
    vllm_fused_moe.w2_weight.data = w2_weight
    vllm_fused_moe.w13_weight_scale.data = w1_weight_scale
    vllm_fused_moe.w2_weight_scale.data = w2_weight_scale
    vllm_fused_moe.w13_bias.data = w1_bias
    vllm_fused_moe.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.top_k,
                                  vllm_fused_moe.renormalize,
                                  vllm_fused_moe.activation)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.quant_method, VllmMxfp4MoEMethod)

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


@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
def test_mxfp4_fused_moe_use_kernel(num_devices, num_tokens, intermediate_size,
                                    hidden_size, num_experts, topk,
                                    enable_attn_dp):
    # Skip if enable_attn_dp is True but we don't have enough devices
    if enable_attn_dp and num_devices < 2:
        pytest.skip("enable_attn_dp requires at least 2 devices")

    # Skip attn_dp tests for fused_moe_use_kernel since the kernel only supports 2D mesh
    if enable_attn_dp:
        pytest.skip(
            "fused_moe kernel does not support attn_dp (requires 2D mesh)")

    mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)

    torch.manual_seed(42)
    dtype = torch.bfloat16

    a = torch.randn((num_tokens, hidden_size), dtype=dtype) / 10
    w1 = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size), dtype=dtype) / 10
    w2 = torch.randn(
        (num_experts, hidden_size, intermediate_size), dtype=dtype) / 10
    w1_weight, w1_weight_scale = quantize_to_mxfp4(w1)
    w2_weight, w2_weight_scale = quantize_to_mxfp4(w2)

    w1_bias = torch.randn(
        (num_experts, 2 * intermediate_size), dtype=dtype) / 10
    w2_bias = torch.randn((num_experts, hidden_size), dtype=dtype) / 10
    score = torch.randn((num_tokens, num_experts), dtype=dtype)

    engine_args = EngineArgs(
        model=MODELS[0],
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        load_format='dummy',
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
            reduce_results=False,
            renormalize=False,
            tp_size=1,
            dp_size=1,
            quant_config=quant_config,
            has_bias=True,
        )
        vllm_fused_moe.moe_parallel_config.use_ep = True

    vllm_fused_moe.w13_weight.data = w1_weight
    vllm_fused_moe.w2_weight.data = w2_weight
    vllm_fused_moe.w13_weight_scale.data = w1_weight_scale
    vllm_fused_moe.w2_weight_scale.data = w2_weight_scale
    vllm_fused_moe.w13_bias.data = w1_bias
    vllm_fused_moe.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.top_k,
                                  vllm_fused_moe.renormalize,
                                  vllm_fused_moe.activation)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.quant_method, VllmMxfp4MoEMethod)

        jax_a = a.to('jax')
        score = score.to('jax')

        vllm_fused_moe.quant_method.use_kernel = True
        vllm_fused_moe.quant_method.process_weights_after_loading(
            vllm_fused_moe)
        vllm_fused_moe.quant_method.block_size = {
            "bt": 32,
            "bf": 512,
            "bd1": 1024,
            "bd2": 1024,
            "btc": 32,
            "bfc": 512,
            "bd1c": 1024,
            "bd2c": 1024,
        }

        actual = vllm_fused_moe(jax_a, score)

        torch.testing.assert_close(expected,
                                   actual,
                                   check_device=False,
                                   atol=2e-1,
                                   rtol=2e-1)
