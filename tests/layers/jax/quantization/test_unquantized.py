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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedConfig,
    UnquantizedFusedMoEMethod,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(42)


@pytest.fixture
def mesh():
    devices = np.array(jax.local_devices())
    return Mesh(devices.reshape((len(devices), 1, 1)),
                axis_names=('data', 'attn_dp', 'model'))


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


class TestUnquantizedJaxMoE:

    def test_process_weights_after_loading_matches_common_moe_contract(
        self, mesh
    ):
        num_experts = 2
        hidden_size = 4
        intermediate_size = 3
        gate = jnp.arange(num_experts * hidden_size * intermediate_size,
                          dtype=jnp.bfloat16).reshape(num_experts,
                                                      hidden_size,
                                                      intermediate_size)
        up = gate + 100
        down = jnp.arange(num_experts * intermediate_size * hidden_size,
                          dtype=jnp.bfloat16).reshape(num_experts,
                                                      intermediate_size,
                                                      hidden_size)

        layer = SimpleNamespace(
            moe_backend=MoEBackend.GMM_TP,
            kernel_gating_EDF=nnx.Param(gate, _weights_to_load=[object()]),
            kernel_up_proj_EDF=nnx.Param(up, _weights_to_load=[object()]),
            kernel_down_proj_EFD=nnx.Param(down, _weights_to_load=[object()]),
            edf_sharding=(None, None, None),
            efd_sharding=(None, None, None),
            mesh=mesh,
            activation="silu",
        )
        method = UnquantizedFusedMoEMethod()

        captured = {}

        def fake_process_unquantized_moe_weights(**kwargs):
            captured["w13_weight"] = kwargs["w13_weight"]
            captured["w2_weight"] = kwargs["w2_weight"]
            return FusedMoEWeights(
                w13_weight=kwargs["w13_weight"],
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=kwargs["w2_weight"],
                w2_weight_scale=None,
                w2_bias=None,
            )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "tpu_inference.layers.jax.quantization.unquantized.shard_put",
                lambda x, *args, **kwargs: x,
            )
            mp.setattr(
                "tpu_inference.layers.jax.quantization.unquantized.jax_common.process_unquantized_moe_weights",
                fake_process_unquantized_moe_weights,
            )
            assert method.process_weights_after_loading(layer)

        np.testing.assert_array_equal(
            np.array(captured["w13_weight"], dtype=np.float32),
            np.array(
                jnp.swapaxes(jnp.concatenate([gate, up], axis=2), 1, 2),
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            np.array(captured["w2_weight"], dtype=np.float32),
            np.array(jnp.swapaxes(down, 1, 2), dtype=np.float32),
        )
        assert layer.kernel_gating_upproj_EDF.value.shape == (
            num_experts,
            2 * intermediate_size,
            hidden_size,
        )
        assert layer.kernel_down_proj_EFD.value.shape == (
            num_experts,
            hidden_size,
            intermediate_size,
        )

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

    @pytest.mark.parametrize(
        "einsum_str,kernel_shape,input_shape",
        [
            ("TNH,ANH->TNA", (512, 8, 128), (4, 8, 128)),
            ("TNA,ANH->TNH", (512, 8, 128), (4, 8, 512)),
        ],
    )
    def test_batched_einsum_output_sizes(self, einsum_str, kernel_shape,
                                         input_shape, rngs):
        """Verify UnquantizedConfig computes correct output_sizes for 3D
        batched einsums where the last output dim is not kernel_shape[-1].

        Before the fix, UnquantizedConfig.get_quant_method always used
        kernel_shape[-1] as the output size. For 'TNH,ANH->TNA' with
        kernel_shape (A=512, N=8, H=128), this set output_sizes=[128]
        instead of [512], causing _apply_fused to truncate the output.
        """
        x = jax.random.uniform(rngs.params(), input_shape)

        jax_einsum = JaxEinsum(einsum_str, kernel_shape, rngs)
        y_from_layer = jax_einsum(x)

        method = UnquantizedConfig({}).get_quant_method(jax_einsum, prefix='')
        assert isinstance(method, QuantizeMethodBase)
        y_from_method = method.apply_jax(jax_einsum, x)

        assert y_from_layer.shape == y_from_method.shape, (
            f"Shape mismatch: layer produced {y_from_layer.shape} but "
            f"quant method produced {y_from_method.shape}")
        np.testing.assert_allclose(y_from_layer,
                                   y_from_method,
                                   rtol=1e-5,
                                   atol=1e-5)
