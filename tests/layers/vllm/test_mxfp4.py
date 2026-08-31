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

import importlib.util
import sys
import tempfile
from unittest import mock
from unittest.mock import MagicMock, patch

if sys.modules.get("tokamax") is None and importlib.util.find_spec(
        "tokamax") is None:
    sys.modules["tokamax"] = MagicMock()
    sys.modules["tokamax._src"] = MagicMock()
    sys.modules["tokamax._src.ops"] = MagicMock()
    sys.modules["tokamax._src.ops.experimental"] = MagicMock()
    sys.modules["tokamax._src.ops.experimental.gmm_v2"] = MagicMock()
    sys.modules["tokamax._src.ops.experimental.gmm_v2.gmm_v2"] = MagicMock()

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

from tests.layers.common import utils as test_utils
from tpu_inference import envs
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.vllm.interface.moe import (FusedMoEFactory,
                                                     RoutedExperts)
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.base import _free_torch_storage
from tpu_inference.layers.vllm.quantization.mxfp4 import (VllmMxfp4Config,
                                                          VllmMxfp4MoEMethod)

P = PartitionSpec
MODELS = ["openai/gpt-oss-20b"]
MXFP4_BLOCK_SIZE = 32


def _require_tpu7():
    if not jtu.is_device_tpu_at_least(version=7):
        pytest.skip("Expected TPUv7+")


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
def mock_get_pp_group():
    with patch("tpu_inference.distributed.jax_parallel_state.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


@pytest.fixture(autouse=True)
def setup_environment(request):
    if not (request.node.name.startswith("test_quant_override")
            or request.node.name.startswith("test_mxfp4_fused_moe")):
        yield
        return

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
    _require_tpu7()

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
    _require_tpu7()
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
            has_bias=True,
        )
        vllm_fused_moe.moe_config.moe_parallel_config.use_ep = use_ep
    vllm_fused_moe.routed_experts.w13_weight.data = w1_weight
    vllm_fused_moe.routed_experts.w2_weight.data = w2_weight
    vllm_fused_moe.routed_experts.w13_weight_scale.data = w1_weight_scale
    vllm_fused_moe.routed_experts.w2_weight_scale.data = w2_weight_scale
    vllm_fused_moe.routed_experts.w13_bias.data = w1_bias
    vllm_fused_moe.routed_experts.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.routed_experts.top_k,
                                  vllm_fused_moe.routed_experts.renormalize,
                                  vllm_fused_moe.activation.value)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.routed_experts.quant_method,
                          VllmMxfp4MoEMethod)
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


@pytest.mark.parametrize("num_devices", [1, 2])
@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("enable_attn_dp", [False, True])
@mock.patch("os.environ", {"USE_MOE_EP_KERNEL": "1"})
def test_mxfp4_fused_moe_use_kernel(num_devices, num_tokens, intermediate_size,
                                    hidden_size, num_experts, topk,
                                    enable_attn_dp):
    _require_tpu7()
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
        tensor_parallel_size=mesh.devices.size, enable_expert_parallel=True)

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
            has_bias=True,
        )
        vllm_fused_moe.moe_config.moe_parallel_config.use_ep = True

    vllm_fused_moe.routed_experts.w13_weight.data = w1_weight
    vllm_fused_moe.routed_experts.w2_weight.data = w2_weight
    vllm_fused_moe.routed_experts.w13_weight_scale.data = w1_weight_scale
    vllm_fused_moe.routed_experts.w2_weight_scale.data = w2_weight_scale
    vllm_fused_moe.routed_experts.w13_bias.data = w1_bias
    vllm_fused_moe.routed_experts.w2_bias.data = w2_bias

    expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
                                  vllm_fused_moe.routed_experts.top_k,
                                  vllm_fused_moe.routed_experts.renormalize,
                                  vllm_fused_moe.activation.value)

    with torchax.default_env(), set_forward_context(None, vllm_config):
        assert isinstance(vllm_fused_moe.routed_experts.quant_method,
                          VllmMxfp4MoEMethod)
        assert vllm_fused_moe.routed_experts.quant_method.moe_backend == MoEBackend.FUSED_MOE

        jax_a = a.to('jax')
        score = score.to('jax')

        vllm_fused_moe.routed_experts.quant_method.process_weights_after_loading(
            vllm_fused_moe.routed_experts)
        vllm_fused_moe.routed_experts.quant_method.extra_backend_kwargs.update(
            {
                "bt": 32,
                "bf": 512,
                "bd1": 1024,
                "bd2": 1024,
                "btc": 32,
                "bfc": 512,
                "bd1c": 1024,
                "bd2c": 1024,
            })

        actual = vllm_fused_moe(jax_a, score)

        torch.testing.assert_close(expected,
                                   actual,
                                   check_device=False,
                                   atol=2e-1,
                                   rtol=2e-1)


def test_mxfp4_moe_incremental_loading_trigger_no_bias():
    """Test that VllmMxfp4MoEMethod triggers sharding on exact last shard (6E shards) when has_bias=False."""
    mesh = test_utils.get_spmd_mesh(1)
    mock_moe_config = MagicMock()
    mock_moe_config.num_experts = 4
    mock_moe_config.moe_backend = "auto"
    mock_moe_config.has_bias = False

    layer = MagicMock()
    layer.moe_config = mock_moe_config
    layer.global_num_experts = 4
    layer._loaded_weights = set()

    with patch(
            "tpu_inference.layers.vllm.quantization.mxfp4.select_moe_backend_from_fused_moe_config"
    ):
        method = VllmMxfp4MoEMethod(mock_moe_config, mesh)

    with patch.object(method, "process_weights_after_loading") as mock_process:
        with patch.object(envs, "VLLM_INCREMENTAL_MXFP4_LOADING", True):
            # Load 23 shards (1 short of expected 6 * 4 = 24)
            for expert_id in range(4):
                for shard_id in range(5):
                    method.maybe_process_weights(layer, "w13_weight", (), {
                        "expert_id": expert_id,
                        "shard_id": shard_id
                    })
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 0,
                "shard_id": 5
            })
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 1,
                "shard_id": 5
            })
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 2,
                "shard_id": 5
            })

            assert mock_process.call_count == 0

            # 24th shard triggers process_weights_after_loading
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 3,
                "shard_id": 5
            })
            assert mock_process.call_count == 1


def test_mxfp4_moe_incremental_loading_trigger_with_bias():
    """Test that VllmMxfp4MoEMethod triggers sharding on exact last shard (9E shards) when has_bias=True."""
    mesh = test_utils.get_spmd_mesh(1)
    mock_moe_config = MagicMock()
    mock_moe_config.num_experts = 4
    mock_moe_config.moe_backend = "auto"
    mock_moe_config.has_bias = True

    layer = MagicMock()
    layer.moe_config = mock_moe_config
    layer.global_num_experts = 4
    layer._loaded_weights = set()

    with patch(
            "tpu_inference.layers.vllm.quantization.mxfp4.select_moe_backend_from_fused_moe_config"
    ):
        method = VllmMxfp4MoEMethod(mock_moe_config, mesh)

    with patch.object(method, "process_weights_after_loading") as mock_process:
        with patch.object(envs, "VLLM_INCREMENTAL_MXFP4_LOADING", True):
            # Load 35 shards (1 short of expected (6 + 3) * 4 = 36)
            for expert_id in range(4):
                for shard_id in range(8):
                    method.maybe_process_weights(layer, "w13_weight", (), {
                        "expert_id": expert_id,
                        "shard_id": shard_id
                    })
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 0,
                "shard_id": 8
            })
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 1,
                "shard_id": 8
            })
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 2,
                "shard_id": 8
            })

            assert mock_process.call_count == 0

            # 36th shard triggers process_weights_after_loading
            method.maybe_process_weights(layer, "w2_weight", (), {
                "expert_id": 3,
                "shard_id": 8
            })
            assert mock_process.call_count == 1


@pytest.mark.parametrize("has_bias", [False, True])
def test_mxfp4_moe_incremental_loading_monolithic_fallback_gpt_oss(has_bias):
    """Test that a loader call with expert_id=None (e.g. GPT-OSS) falls back without crashing and triggers after all parameters arrive."""
    mesh = test_utils.get_spmd_mesh(1)
    mock_moe_config = MagicMock()
    mock_moe_config.num_experts = 4
    mock_moe_config.moe_backend = "auto"
    mock_moe_config.has_bias = has_bias

    layer = MagicMock()
    layer.moe_config = mock_moe_config
    layer.global_num_experts = 4
    layer._loaded_weights = set()

    with patch(
            "tpu_inference.layers.vllm.quantization.mxfp4.select_moe_backend_from_fused_moe_config"
    ):
        method = VllmMxfp4MoEMethod(mock_moe_config, mesh)

    params = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]
    if has_bias:
        params.extend(["w13_bias", "w2_bias"])

    layer.named_parameters.return_value = [(p, MagicMock()) for p in params]

    with patch.object(method, "process_weights_after_loading") as mock_process:
        with patch.object(envs, "VLLM_INCREMENTAL_MXFP4_LOADING", True):
            # Load all but the last parameter with expert_id=None, shard_id=None
            for param_name in params[:-1]:
                method.maybe_process_weights(layer, param_name, (), {
                    "expert_id": None,
                    "shard_id": None
                })
                assert mock_process.call_count == 0
                assert param_name in layer._loaded_weights

            # Last parameter triggers processing
            method.maybe_process_weights(layer, params[-1], (), {
                "expert_id": None,
                "shard_id": None
            })
            assert mock_process.call_count == 1


def test_mxfp4_free_torch_storage_frees_memory():
    """Test that _free_torch_storage frees host storage (untyped_storage().size() == 0)."""
    t = torch.randn(16, 32, dtype=torch.float32)
    assert t.untyped_storage().size() > 0
    _free_torch_storage(t)
    assert t.untyped_storage().size() == 0


@pytest.mark.parametrize("has_bias", [False, True])
def test_mxfp4_moe_process_weights_frees_host_storage(has_bias):
    """Test that process_weights_after_loading frees the original CPU storage for all layer parameters."""
    mesh = test_utils.get_spmd_mesh(1)
    mock_moe_config = MagicMock()
    mock_moe_config.num_experts = 4
    mock_moe_config.moe_backend = "auto"
    mock_moe_config.has_bias = has_bias

    with patch(
            "tpu_inference.layers.vllm.quantization.mxfp4.select_moe_backend_from_fused_moe_config"
    ):
        method = VllmMxfp4MoEMethod(mock_moe_config, mesh)

    layer = MagicMock(spec=RoutedExperts)
    layer.moe_config = mock_moe_config
    layer.activation = MagicMock(value="silu")

    # Real CPU tensors
    p_w13_weight = torch.zeros((4, 32, 16), dtype=torch.uint8)
    p_w13_scale = torch.zeros((4, 32, 2), dtype=torch.uint8)
    p_w2_weight = torch.zeros((4, 16, 16), dtype=torch.uint8)
    p_w2_scale = torch.zeros((4, 16, 2), dtype=torch.uint8)

    layer.w13_weight = p_w13_weight
    layer.w13_weight_scale = p_w13_scale
    layer.w2_weight = p_w2_weight
    layer.w2_weight_scale = p_w2_scale

    if has_bias:
        p_w13_bias = torch.zeros((4, 32), dtype=torch.bfloat16)
        p_w2_bias = torch.zeros((4, 16), dtype=torch.bfloat16)
        layer.w13_bias = p_w13_bias
        layer.w2_bias = p_w2_bias

    # Verify initial storage is allocated
    assert p_w13_weight.untyped_storage().size() > 0
    assert p_w13_scale.untyped_storage().size() > 0
    assert p_w2_weight.untyped_storage().size() > 0
    assert p_w2_scale.untyped_storage().size() > 0
    if has_bias:
        assert p_w13_bias.untyped_storage().size() > 0
        assert p_w2_bias.untyped_storage().size() > 0

    dummy_fused = MagicMock()
    dummy_sharded = MagicMock()
    dummy_tv = MagicMock()
    dummy_tv.w13_weight = torch.empty((1, ))
    dummy_tv.w13_weight_scale = torch.empty((1, ))
    dummy_tv.w2_weight = torch.empty((1, ))
    dummy_tv.w2_weight_scale = torch.empty((1, ))
    if has_bias:
        dummy_tv.w13_bias = torch.empty((1, ))
        dummy_tv.w2_bias = torch.empty((1, ))

    with patch("tpu_inference.layers.vllm.quantization.mxfp4._load_weight_for_layer", return_value=MagicMock()), \
         patch("tpu_inference.layers.vllm.quantization.mxfp4.shard_moe_weights", return_value=dummy_sharded), \
         patch("tpu_inference.layers.vllm.quantization.mxfp4.torch_view", return_value=dummy_tv), \
         patch("tpu_inference.layers.vllm.quantization.mxfp4._release_host_memory"), \
         patch.object(method, "moe_backend", MagicMock()):
        with patch("jax.jit",
                   side_effect=lambda fn: lambda *args, **kwargs: dummy_fused):
            method.process_weights_after_loading(layer)

    # Verify that CPU storages are completely freed
    assert p_w13_weight.untyped_storage().size() == 0
    assert p_w13_scale.untyped_storage().size() == 0
    assert p_w2_weight.untyped_storage().size() == 0
    assert p_w2_scale.untyped_storage().size() == 0
    if has_bias:
        assert p_w13_bias.untyped_storage().size() == 0
        assert p_w2_bias.untyped_storage().size() == 0
