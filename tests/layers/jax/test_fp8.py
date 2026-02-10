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

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig, VllmConfig

from tests.layers.common import utils as test_utils
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)
from tpu_inference.layers.jax.moe.moe import JaxMoE
# yapf: disable
from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3Router

# yapf: enable


def quantize_to_fp8_block_3d(weight: jax.Array,
                             block_m: int,
                             block_n: int,
                             dtype=jnp.float8_e4m3fn):
    dtype_info = jnp.finfo(dtype)
    dtype_max = jnp.array(dtype_info.max, dtype=jnp.float32)
    dtype_min = jnp.array(dtype_info.min, dtype=jnp.float32)

    num_experts, out_dim, in_dim = weight.shape

    assert out_dim % block_m == 0
    assert in_dim % block_n == 0

    weight_view = weight.reshape(num_experts, out_dim // block_m, block_m,
                                 in_dim // block_n, block_n)

    abs_max = jnp.max(jnp.abs(weight_view), axis=(2, 4),
                      keepdims=True).astype(jnp.float32)
    scale = abs_max / dtype_max

    scaled_weight = weight_view.astype(jnp.float32) / scale

    w_q = jnp.clip(scaled_weight, dtype_min, dtype_max).astype(dtype)

    w_q = w_q.reshape(num_experts, out_dim, in_dim)
    scale_blocks = jnp.squeeze(scale, axis=(2, 4)).astype(jnp.float32)

    return w_q, scale_blocks


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh, axis_names=MESH_AXIS_NAMES) as m:
        yield m


@pytest.fixture
def rng():
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestFp8FusedMoE:

    @pytest.mark.parametrize("use_ep", [True, False])
    @pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
    @pytest.mark.parametrize("num_tokens", [8])
    @pytest.mark.parametrize("intermediate_size", [1024, 2048])
    @pytest.mark.parametrize("hidden_size", [128, 512])
    @pytest.mark.parametrize("num_experts", [8])
    @pytest.mark.parametrize("topk", [2])
    @pytest.mark.parametrize("enable_attn_dp", [False, True])
    def test_fused_moe(self, use_ep, num_devices, num_tokens,
                       intermediate_size, hidden_size, num_experts, topk,
                       enable_attn_dp, rng):
        # Skip if enable_attn_dp is True but we don't have enough devices
        if enable_attn_dp and num_devices < 2:
            pytest.skip("enable_attn_dp requires at least 2 devices")

        mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)

        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B-FP8", quantization="fp8"))

        # TODO (jacobplatin): don't mock this out once support for
        # FP8 lands officialy
        # quant_config = get_tpu_quantization_config(vllm_config)
        quant_config = Fp8Config(
            vllm_config.model_config.hf_config.quantization_config)

        edf_sharding = (None, ShardingAxisNameBase.MODEL_1,
                        ShardingAxisNameBase.MODEL_2)
        expert_axis_name = edf_sharding[0]
        moe_backend = MoEBackend.GMM_EP if use_ep else MoEBackend.GMM_TP

        dtype = jnp.bfloat16

        # This won't be used in reality since we are patching
        # the router_logits
        router = DeepSeekV3Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=topk,
            n_groups=8,
            topk_groups=4,
            norm_topk_prob=True,
            rngs=nnx.Rngs(0),
            routed_scaling_factor=2.5,
            dtype=dtype,
            moe_backend=moe_backend,
            activation_ffw_td=(ShardingAxisNameBase.MLP_DATA, None),
            ed_sharding=(None, None),
            e_sharding=(None, ))

        layer = JaxMoE(dtype=jnp.float8_e4m3fn,
                       num_local_experts=num_experts,
                       apply_expert_weight_before_computation=False,
                       expert_axis_name=expert_axis_name,
                       num_expert_parallelism=2 if use_ep else 1,
                       hidden_size=hidden_size,
                       intermediate_size_moe=intermediate_size,
                       num_experts_per_tok=topk,
                       mesh=mesh,
                       hidden_act="silu",
                       rngs=nnx.Rngs(0),
                       quant_config=quant_config,
                       activation_ffw_td=(ShardingAxisNameBase.MLP_DATA,
                                          ShardingAxisNameBase.MODEL_1),
                       activation_ffw_ted=(ShardingAxisNameBase.MLP_DATA, None,
                                           ShardingAxisNameBase.MODEL_1),
                       edf_sharding=(None, ShardingAxisNameBase.MODEL_1,
                                     ShardingAxisNameBase.MODEL_2),
                       efd_sharding=(None, ShardingAxisNameBase.MODEL_2,
                                     ShardingAxisNameBase.MODEL_1),
                       moe_backend=moe_backend,
                       renormalize=False,
                       router=router)

        assert layer.use_ep == use_ep

        k1, k2, k3, k4 = jax.random.split(rng, 4)

        a = jax.random.normal(k1,
                              (num_tokens, hidden_size), dtype=dtype) / 10.0
        score = jax.random.normal(k2, (num_tokens, num_experts), dtype=dtype)

        w13_shape = (num_experts, 2 * intermediate_size, hidden_size)
        w2_shape = (num_experts, hidden_size, intermediate_size)
        w13 = jax.random.normal(k3, w13_shape, dtype=dtype) / 10.0
        w2 = jax.random.normal(k4, w2_shape, dtype=dtype) / 10.0

        expected = test_utils.ref_moe_jax(a, score, w13, w2, None, None,
                                          layer.top_k, layer.renormalize,
                                          layer.activation)

        if use_ep:
            assert layer.moe_backend == MoEBackend.GMM_EP
        else:
            assert layer.moe_backend == MoEBackend.GMM_TP

        block_m, block_n = quant_config.weight_block_size
        w1_weight, w1_weight_scale = quantize_to_fp8_block_3d(
            w13, block_m, block_n, jnp.float8_e4m3fn)
        w2_weight, w2_weight_scale = quantize_to_fp8_block_3d(
            w2, block_m, block_n, jnp.float8_e4m3fn)

        layer.quant_method.create_weights_jax(layer)

        scale_suffix = layer.quant_method.weight_scale_name

        getattr(
            layer,
            f"kernel_gating_upproj_EDF_{scale_suffix}").value = w1_weight_scale
        getattr(layer,
                f"kernel_down_proj_EFD_{scale_suffix}").value = w2_weight_scale

        w_gate_fp8, w_up_fp8 = jnp.split(jnp.transpose(w1_weight, (0, 2, 1)),
                                         2,
                                         axis=2)

        # Overwrite the layer's parameters with our FP8 data
        layer.kernel_gating_EDF.value = w_gate_fp8
        layer.kernel_up_proj_EDF.value = w_up_fp8

        layer.kernel_down_proj_EFD.value = w2_weight

        layer.quant_method.process_weights_after_loading(layer)

        # Patch the router since we don't want to use the
        # real router
        with patch.object(layer, 'router', return_value=score):
            # Run the actual forward pass and up-cast
            # to avoid promote error
            actual = layer(a).astype(expected.dtype)

        assert jnp.allclose(expected, actual, atol=2.5e-2, rtol=1e-1)
