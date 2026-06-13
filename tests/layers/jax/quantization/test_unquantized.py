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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import \
    process_unquantized_moe_weights
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.linear import (JaxEinsum, JaxLinear,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedConfig, UnquantizedMergedLinearMethod)


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


class TestJaxMergedColumnParallelLinear:
    """Tests for the fused gate_up_proj path: a single kernel holding several
    logical projections, loaded from separate per-projection checkpoint tensors
    via ``UnquantizedMergedLinearMethod``.
    """

    @staticmethod
    def _build_layer(in_size, output_sizes, rngs):
        return JaxMergedColumnParallelLinear(input_size=in_size,
                                             output_sizes=output_sizes,
                                             rngs=rngs,
                                             use_bias=False,
                                             quant_config=UnquantizedConfig(
                                                 {}),
                                             prefix="mlp.gate_up_proj")

    def test_get_quant_method_returns_merged_method(self, rngs):
        """JaxMergedColumnParallelLinear dispatches to the merged method, which
        must see each projection's size (so forward/load can interleave by
        shard)."""
        layer = self._build_layer(4, [6, 6], rngs)
        assert isinstance(layer.quant_method, UnquantizedMergedLinearMethod)
        cfg = layer.quant_method.linear_config
        assert cfg.output_sizes == [6, 6]
        # n_shards == number of fused projections; required for the
        # interleave (load) / de-interleave (forward) to be inverses.
        assert cfg.n_shards == 2
        assert cfg.fuse_matmuls
        # The weight_loader is attached at create_weights_jax time and the
        # per-projection accumulation slots start empty.
        assert layer.weight.get_metadata("_weights_to_load") == [None, None]

    @pytest.mark.parametrize("in_size,gate_out,up_out,batch", [
        (4, 6, 6, 2),
        (8, 16, 16, 4),
        (8, 16, 8, 1),
    ])
    def test_load_then_forward_matches_split_matmuls(self, in_size, gate_out,
                                                     up_out, batch, rngs):
        """End-to-end: load gate/up from separate checkpoint tensors into the
        fused kernel, then verify the fused forward equals running the two
        projections separately and concatenating — i.e. the load-time interleave
        is the exact inverse of the forward-time de-interleave."""
        layer = self._build_layer(in_size, [gate_out, up_out], rngs)

        # Checkpoint tensors in HF layout (out_i, in_size).
        g = torch.randn(gate_out, in_size, dtype=torch.float32)
        u = torch.randn(up_out, in_size, dtype=torch.float32)

        # Drive the real loader contract (matches JaxAutoWeightsLoader): the
        # fused param's weight_loader is called once per projection with its
        # shard_id; the fuse only completes once both have arrived.
        weight_loader = layer.weight.weight_loader
        weight_loader(layer.weight, g, 0)  # gate -> shard_id 0
        weight_loader(layer.weight, u, 1)  # up   -> shard_id 1
        # Assembly is deferred to process_weights_after_loading (called by the
        # weight-loading infrastructure after all shards have been streamed).
        layer.quant_method.process_weights_after_loading(layer)
        assert layer.weight.value.shape == (in_size, gate_out + up_out)

        x = jax.random.uniform(rngs.params(), (batch, in_size))

        # TPU defaults to bf16 matmuls; force highest precision so the fused
        # and reference paths are bit-comparable.
        with jax.default_matmul_precision("highest"):
            fused_out = layer(x)
            g_jax = jnp.asarray(g.numpy()).T  # (in_size, gate_out)
            u_jax = jnp.asarray(u.numpy()).T  # (in_size, up_out)
            ref = jnp.concatenate([x @ g_jax, x @ u_jax], axis=-1)

        assert fused_out.shape == ref.shape == (batch, gate_out + up_out)
        np.testing.assert_allclose(fused_out, ref, rtol=1e-5, atol=1e-5)

    def test_load_is_deferred_until_all_projections_arrive(self, rngs):
        """A single projection must not trigger the fuse (projections may span
        multiple checkpoint files)."""
        layer = self._build_layer(4, [6, 6], rngs)
        weight_loader = layer.weight.weight_loader

        weight_loader(layer.weight, torch.randn(6, 4), 0)
        shards = layer.weight.get_metadata("_weights_to_load")
        assert shards[0] is not None and shards[1] is None


class TestUnquantizedJaxMoe:

    @pytest.fixture
    def mesh(self):
        devices = jax.devices()
        return Mesh(
            np.array([devices[0]]).reshape(1), (ShardingAxisName.MLP_TENSOR, ))

    @staticmethod
    def _make_inputs(num_experts: int = 4,
                     hidden_size: int = 128,
                     intermediate_size: int = 256):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        w13 = jax.random.normal(
            k1, (num_experts, 2 * intermediate_size, hidden_size),
            dtype=jnp.bfloat16)
        w2 = jax.random.normal(k2,
                               (num_experts, hidden_size, intermediate_size),
                               dtype=jnp.bfloat16)
        return w13, w2

    @pytest.mark.parametrize("requantize_dtype,expect_w_dtype,expect_scale", [
        ("", jnp.bfloat16, None),
        ("fp8_e4m3", jnp.float8_e4m3fn, jnp.float32),
    ])
    def test_no_requantize_when_env_unset(self, mesh, monkeypatch,
                                          requantize_dtype, expect_w_dtype,
                                          expect_scale):
        """No moe weights requantization -> weights stay bf16 and scales stay None."""
        monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", requantize_dtype)
        monkeypatch.delenv("MOE_REQUANTIZE_BLOCK_SIZE", raising=False)
        # The function is jax.jit'ed and reads the env at trace time, so a
        # cached trace from a prior test would otherwise mask the env change.
        jax.clear_caches()

        w13, w2 = self._make_inputs()
        weights = process_unquantized_moe_weights(
            mesh=mesh,
            moe_backend=MoEBackend.GMM_EP,
            activation=MoEActivation.SILU,
            w13_weight=w13,
            w13_bias=None,
            w2_weight=w2,
            w2_bias=None,
        )

        assert weights.w13_weight.dtype == expect_w_dtype
        assert weights.w2_weight.dtype == expect_w_dtype
        if expect_scale is None:
            assert weights.w13_weight_scale is None
            assert weights.w2_weight_scale is None
        else:
            assert weights.w13_weight_scale.dtype == expect_scale
            assert weights.w2_weight_scale.dtype == expect_scale
