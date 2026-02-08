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
import torch
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig, VllmConfig

from tests.layers.common import utils as test_utils
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.moe.moe import JaxMoE
# yapf: disable
from tpu_inference.layers.jax.quantization.fp8 import (
    Fp8Config, Fp8TensorwiseLinearMethod)
from tpu_inference.layers.jax.quantization.unquantized import \
    UnquantizedLinearMethod
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3Router


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
def rngs():
    return nnx.Rngs(42)


class TestFp8BlockwiseJaxLinear:

    @pytest.mark.parametrize("in_features,out_features", [(128, 64),
                                                          (256, 128)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_linear_forward_correctness(self, in_features, out_features,
                                        use_bias, batch_size, rngs):
        hf_quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
        quant_config = Fp8Config(hf_quant_config)

        # Initialize quantized layer
        layer = JaxLinear(
            input_size=in_features,
            output_size=out_features,
            rngs=rngs,
            use_bias=use_bias,
            quant_config=quant_config,
        )

        # Use a dummy mesh for testing
        devices = jax.devices()
        mesh = jax.sharding.Mesh(np.array(devices), ('device', ))
        with jax.set_mesh(mesh):
            # Process weights in mesh context
            layer.quant_method.process_weights_after_loading(layer)

            # Prepare input
            x = jax.random.normal(rngs.params(), (batch_size, in_features))

            # Forward pass
            output = layer(x)

        assert output.shape == (batch_size, out_features)

    @pytest.mark.parametrize("kernel_shape", [(128, 8, 16), (256, 32, 32)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_einsum_forward_correctness(self, kernel_shape, use_bias,
                                        batch_size, rngs):
        hf_quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [8, 16],
        }
        quant_config = Fp8Config(hf_quant_config)

        layer = JaxEinsum(
            einsum_str='TD,DNH->TNH',
            kernel_shape=kernel_shape,
            rngs=rngs,
            bias_shape=kernel_shape[1:] if use_bias else None,
            quant_config=quant_config,
        )

        # Use a dummy mesh for testing
        devices = jax.devices()
        mesh = jax.sharding.Mesh(np.array(devices), ('device', ))
        with jax.set_mesh(mesh):
            # Process weights in mesh context
            layer.quant_method.process_weights_after_loading(layer)

            # Prepare input (B, D)
            x = jax.random.normal(rngs.params(), (batch_size, kernel_shape[0]))

            # Forward pass
            output = layer(x)

        # Output shape should be (B, N, H)
        expected_shape = (batch_size, ) + kernel_shape[1:]
        assert output.shape == expected_shape


class TestFp8TensorwiseJaxLinear:

    def test_fp8_linear_method_create_weights(self, rngs):
        layer = JaxEinsum("ab,bc->ac", (32, 16), rngs, bias_shape=None)
        config = QuantLinearConfig(enable_sp=False, output_sizes=[16])
        method = Fp8TensorwiseLinearMethod(layer, config)
        method.create_weights_jax(layer, rngs=rngs)

        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'weight_scale')
        assert layer.weight.value.dtype == jnp.float8_e4m3fn
        assert layer.weight_scale.value.dtype == jnp.float32
        assert layer.weight.value.shape == (16, 32)
        assert layer.weight_scale.value.shape == (16, )
        assert hasattr(layer.weight, 'weight_loader')

    def test_fp8_loader_prevents_upcast(self, rngs):
        layer = JaxEinsum("ab,bc->ac", (4, 2), rngs, bias_shape=None)
        config = QuantLinearConfig(enable_sp=False, output_sizes=[2])
        method = Fp8TensorwiseLinearMethod(layer, config)
        method.create_weights_jax(layer, rngs=rngs)

        torch_fp8 = torch.zeros((2, 4), dtype=torch.float8_e4m3fn)
        layer.weight.weight_loader(layer.weight, torch_fp8)

        assert layer.weight.value.dtype == jnp.float8_e4m3fn

    @pytest.mark.parametrize("in_features,out_features", [(128, 64),
                                                          (256, 128)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_linear_forward_correctness(self, in_features, out_features,
                                        use_bias, batch_size, rngs):
        hf_quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        }
        quant_config = Fp8Config(hf_quant_config)

        layer = JaxLinear(
            input_size=in_features,
            output_size=out_features,
            rngs=rngs,
            use_bias=use_bias,
            quant_config=quant_config,
        )

        devices = jax.devices()
        mesh = jax.sharding.Mesh(np.array(devices), ('device', ))
        with jax.set_mesh(mesh):
            x = jax.random.normal(rngs.params(), (batch_size, in_features))
            output = layer(x)

        assert output.shape == (batch_size, out_features)


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
                       enable_attn_dp, rngs):
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
            rngs=rngs,
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
                       rngs=rngs,
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

        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)

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


class TestFp8Config:

    def test_skip_layers(self, rngs):
        """Test that if quantization_config has ignored layers, those layers are skipped from quantization."""

        class MLP(nnx.Module):

            def __init__(self,
                         in_features,
                         out_features,
                         rngs,
                         quant_config,
                         prefix=''):
                self.proj1 = JaxLinear(in_features,
                                       out_features,
                                       rngs=rngs,
                                       quant_config=quant_config,
                                       prefix="proj1")
                self.proj2 = JaxLinear(in_features,
                                       out_features,
                                       rngs=rngs,
                                       quant_config=quant_config,
                                       prefix="proj2")

            def __call__(self, x):
                return self.proj2(self.proj1(x))

        hf_quant_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "ignored_layers": ["mlp.proj1"]
        }
        quant_config = Fp8Config(hf_quant_config)

        mlp = MLP(16, 16, rngs, quant_config, prefix="mlp")

        # Check that proj1 is NOT quantized (UnquantizedLinearMethod)
        assert isinstance(mlp.proj1.quant_method, UnquantizedLinearMethod)
        # Check that proj2 IS quantized (Fp8TensorwiseLinearMethod)
        assert isinstance(mlp.proj2.quant_method, Fp8TensorwiseLinearMethod)
