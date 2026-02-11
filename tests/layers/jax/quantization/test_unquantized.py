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

from tests.layers.common import utils as test_utils
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import ShardingAxisNameBase
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedConfig, UnquantizedFusedMoEMethod)
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3Router


@pytest.fixture
def rngs():
    return nnx.Rngs(42)


class TestUnquantizedJaxLinear:

    @pytest.mark.parametrize("in_features,out_features", [(4, 6), (8, 16)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_linear_forward_correctness(self, in_features, out_features,
                                        use_bias, batch_size, rngs):
        # Create input data with rngs
        x = jax.random.uniform(rngs.params(), (batch_size, in_features))

        jax_linear = JaxLinear(in_features,
                               out_features,
                               rngs,
                               use_bias=use_bias)
        y_from_layer = jax_linear(x)

        method = UnquantizedConfig({}).get_quant_method(jax_linear, prefix='')
        assert isinstance(method, QuantizeMethodBase)
        y_from_method = method.apply_jax(jax_linear, x)

        # compare outputs
        np.testing.assert_allclose(y_from_layer,
                                   y_from_method,
                                   rtol=1e-5,
                                   atol=1e-5)

    @pytest.mark.parametrize("kernel_shape", [(128, 8, 32), (512, 4, 16)])
    @pytest.mark.parametrize("use_bias", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_einsum_forward_correctness(self, kernel_shape, use_bias,
                                        batch_size, rngs):
        # Create input data with rngs
        x = jax.random.uniform(rngs.params(), (batch_size, kernel_shape[0]))

        jax_einsum = JaxEinsum(
            'TD,DNH->TNH',
            kernel_shape,
            rngs,
            bias_shape=kernel_shape[1:] if use_bias else None)
        y_from_layer = jax_einsum(x)

        method = UnquantizedConfig({}).get_quant_method(jax_einsum, prefix='')
        assert isinstance(method, QuantizeMethodBase)
        y_from_method = method.apply_jax(jax_einsum, x)

        # compare outputs
        np.testing.assert_allclose(y_from_layer,
                                   y_from_method,
                                   rtol=1e-5,
                                   atol=1e-5)


class TestUnquantizedFusedMoE:

    @pytest.mark.parametrize("use_ep", [True, False])
    @pytest.mark.parametrize("num_devices", [1, jax.local_device_count()])
    @pytest.mark.parametrize("num_tokens", [8])
    @pytest.mark.parametrize("intermediate_size", [1024, 2048])
    @pytest.mark.parametrize("hidden_size", [128, 512])
    @pytest.mark.parametrize("num_experts", [8])
    @pytest.mark.parametrize("topk", [2])
    # TODO: support bias
    @pytest.mark.parametrize("has_bias", [False, True])
    @pytest.mark.parametrize("activation", ["silu"])  # TODO: swigluoai
    @pytest.mark.parametrize("enable_attn_dp", [False, True])
    def test_fused_moe(self, use_ep, num_devices, num_tokens,
                       intermediate_size, hidden_size, num_experts, topk,
                       has_bias, activation, enable_attn_dp, rngs):
        # Skip if enable_attn_dp is True but we don't have enough devices
        if enable_attn_dp and num_devices < 2:
            pytest.skip("enable_attn_dp requires at least 2 devices")

        mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)

        quant_config = UnquantizedConfig({})

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

        layer = JaxMoE(dtype=jnp.bfloat16,
                       num_local_experts=num_experts,
                       apply_expert_weight_before_computation=False,
                       expert_axis_name=expert_axis_name,
                       num_expert_parallelism=2 if use_ep else 1,
                       hidden_size=hidden_size,
                       intermediate_size_moe=intermediate_size,
                       num_experts_per_tok=topk,
                       mesh=mesh,
                       hidden_act=activation,
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

        layer.quant_method.create_weights_jax(layer)

        w_gate, w_up = jnp.split(w13, 2, axis=1)

        # Overwrite the layer's parameters
        layer.kernel_gating_EDF.value = w_gate
        layer.kernel_up_proj_EDF.value = w_up

        layer.kernel_down_proj_EFD.value = w2

        layer.quant_method.process_weights_after_loading(layer)

        assert not hasattr(layer, 'kernel_gating_EDF')
        assert not hasattr(layer, 'kernel_up_proj_EDF')
        assert hasattr(layer, 'kernel_down_proj_EFD')
        assert hasattr(layer, 'kernel_gating_upproj_EDF')

        # TODO: why is this needed
        layer.kernel_gating_upproj_EDF.value = jnp.transpose(
            layer.kernel_gating_upproj_EDF.value, (0, 2, 1))
        layer.kernel_down_proj_EFD.value = jnp.transpose(
            layer.kernel_down_proj_EFD.value, (0, 2, 1))

        assert isinstance(layer.quant_method, UnquantizedFusedMoEMethod)

        # Patch the router since we don't want to use the
        # real router
        with patch.object(layer, 'router', return_value=score):
            # Run the actual forward pass and up-cast
            # to avoid promote error
            actual = layer(a).astype(expected.dtype)

        assert jnp.allclose(expected, actual, atol=1e-1, rtol=1e-1)

    # @pytest.mark.parametrize("num_devices", [jax.local_device_count()])
    # @pytest.mark.parametrize("num_tokens", [128, 512])
    # @pytest.mark.parametrize("intermediate_size", [512])
    # @pytest.mark.parametrize("hidden_size", [512])
    # @pytest.mark.parametrize("num_experts", [32])
    # @pytest.mark.parametrize("topk", [8])
    # @pytest.mark.parametrize("has_bias", [False, True])
    # @pytest.mark.parametrize("enable_attn_dp", [False, True])
    # @mock.patch("os.environ", {"USE_MOE_EP_KERNEL": "1"})
    # def test_fused_moe_use_kernel(num_devices, num_tokens, intermediate_size,
    #                             hidden_size, num_experts, topk, has_bias,
    #                             enable_attn_dp):
    #     # Skip if enable_attn_dp is True but we don't have enough devices
    #     if enable_attn_dp and num_devices < 2:
    #         pytest.skip("enable_attn_dp requires at least 2 devices")

    #     # Skip attn_dp tests for fused_moe_use_kernel since the kernel only supports 2D mesh
    #     if enable_attn_dp:
    #         pytest.skip(
    #             "fused_moe kernel does not support attn_dp (requires 2D mesh)")

    #     mesh = test_utils.get_spmd_mesh(num_devices, enable_attn_dp)

    #     # TODO(Qiliang Cui): Remove when issue is resolved.
    #     if not jtu.is_device_tpu_at_least(version=7):
    #         pytest.skip(allow_module_level=True, reason="Expected TPUv7+")

    #     torch.manual_seed(42)
    #     dtype = torch.bfloat16

    #     a = torch.randn((num_tokens, hidden_size), dtype=dtype) / 10
    #     w1 = torch.randn(
    #         (num_experts, 2 * intermediate_size, hidden_size), dtype=dtype) / 10
    #     w2 = torch.randn(
    #         (num_experts, hidden_size, intermediate_size), dtype=dtype) / 10

    #     w1_bias = w2_bias = None
    #     if has_bias:
    #         w1_bias = torch.randn(
    #             (num_experts, 2 * intermediate_size), dtype=dtype) / 10
    #         w2_bias = torch.randn((num_experts, hidden_size), dtype=dtype) / 10

    #     # Use deterministic gating_output generation (same logic as fused_moe_v1_test.py)
    #     # Generate base gating scores with deterministic pattern
    #     score = (
    #         torch.randn((num_tokens, num_experts), dtype=torch.float32) +
    #         torch.arange(num_tokens * num_experts, dtype=torch.float32).reshape(
    #             num_tokens, num_experts) / 100)

    #     # Generate unique top-k indices
    #     generator = torch.Generator()
    #     generator.manual_seed(42)
    #     top_k_indices = torch.randint(0,
    #                                 num_experts - 1, (num_tokens, topk),
    #                                 dtype=torch.int32,
    #                                 generator=generator)

    #     # Add one-hot encoding weighted by 10 to ensure selected experts have highest scores
    #     one_hot = torch.nn.functional.one_hot(top_k_indices.long(),
    #                                         num_classes=num_experts).float()
    #     one_hot = one_hot.sum(dim=1) * 10
    #     score = (score + one_hot).to(dtype)

    #     engine_args = EngineArgs(
    #         model="Qwen/Qwen2-1.5B-Instruct",
    #         max_model_len=64,
    #         max_num_batched_tokens=64,
    #         max_num_seqs=4,
    #     )
    #     vllm_config = engine_args.create_engine_config()
    #     vllm_config.model_config.dtype = dtype
    #     vllm_config.parallel_config = ParallelConfig(
    #         tensor_parallel_size=mesh.devices.size, enable_expert_parallel=True)

    #     quant_config = get_tpu_quantization_config(vllm_config, mesh)
    #     with set_current_vllm_config(vllm_config):
    #         vllm_fused_moe = FusedMoE(
    #             num_experts=num_experts,
    #             top_k=topk,
    #             hidden_size=hidden_size,
    #             intermediate_size=intermediate_size,
    #             reduce_results=True,
    #             renormalize=False,
    #             tp_size=mesh.devices.size,
    #             dp_size=1,
    #             quant_config=quant_config,
    #             has_bias=has_bias,
    #         )
    #         vllm_fused_moe.moe_parallel_config.use_ep = True

    #     vllm_fused_moe.w13_weight.data = w1
    #     vllm_fused_moe.w2_weight.data = w2
    #     if has_bias:
    #         vllm_fused_moe.w13_bias.data = w1_bias
    #         vllm_fused_moe.w2_bias.data = w2_bias

    #     expected = test_utils.ref_moe(a, score, w1, w2, w1_bias, w2_bias,
    #                                 vllm_fused_moe.top_k,
    #                                 vllm_fused_moe.renormalize,
    #                                 vllm_fused_moe.activation)

    #     with torchax.default_env(), set_forward_context(None, vllm_config):
    #         assert isinstance(vllm_fused_moe.quant_method,
    #                         VllmUnquantizedFusedMoEMethod)
    #         assert vllm_fused_moe.quant_method.moe_backend == MoEBackend.FUSED_MOE

    #         jax_a = a.to('jax')
    #         score = score.to('jax')

    #         vllm_fused_moe.quant_method.process_weights_after_loading(
    #             vllm_fused_moe)
    #         vllm_fused_moe.quant_method.extra_backend_kwargs.update({
    #             "bt": 32,
    #             "bf": 512,
    #             "bd1": 512,
    #             "bd2": 512,
    #             "btc": 32,
    #             "bfc": 256,
    #             "bd1c": 256,
    #             "bd2c": 256,
    #         })
    #         actual = vllm_fused_moe(jax_a, score)

    #         torch.testing.assert_close(
    #             expected,
    #             actual,
    #             check_device=False,
    #             atol=1e-2,
    #             rtol=1e-2,
    #         )
