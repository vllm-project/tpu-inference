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

import dataclasses
from unittest import mock

import jax
import numpy as np
import pytest
import qwix
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig, VllmConfig

from tpu_inference.layers.jax import norm as norm_lib
from tpu_inference.layers.jax.norm import (JaxLayerNorm, JaxRmsNorm,
                                           sharded_fused_rmsnorm_fp8_quant)
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config

_FP8_RULE = qwix.QuantizationRule(
    module_path=".*",
    weight_qtype=jax.numpy.float8_e4m3fn,
    act_qtype=jax.numpy.float8_e4m3fn,
)
_NUM_FEATURES = 256


def _make_rms_norm(prefix: str = "", fuse_fp8_quant: bool = False):
    return JaxRmsNorm(
        _NUM_FEATURES,
        epsilon=1e-6,
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nnx.Rngs(0),
        prefix=prefix,
        fuse_fp8_quant=fuse_fp8_quant,
    )


class _NormLinearBlock(nnx.Module):
    """RMSNorm feeding a linear projection, like Qwen's input_layernorm."""

    def __init__(self, fuse_fp8_quant: bool):
        self.input_layernorm = _make_rms_norm("layers.0.input_layernorm",
                                              fuse_fp8_quant)
        self.proj = nnx.Linear(_NUM_FEATURES,
                               64,
                               use_bias=False,
                               param_dtype=jax.numpy.bfloat16,
                               rngs=nnx.Rngs(1))

    def __call__(self, x):
        return self.proj(self.input_layernorm(x))


class _SandwichNormBlock(nnx.Module):
    """Gemma-style post-attention norm whose output is added to the residual."""

    def __init__(self):
        self.o_proj = nnx.Linear(_NUM_FEATURES,
                                 _NUM_FEATURES,
                                 use_bias=False,
                                 param_dtype=jax.numpy.bfloat16,
                                 rngs=nnx.Rngs(1))
        self.post_attention_layernorm = _make_rms_norm(
            "layers.0.post_attention_layernorm")

    def __call__(self, x):
        return x + self.post_attention_layernorm(self.o_proj(x))


def _input(shape=(16, _NUM_FEATURES)):
    return jax.random.normal(jax.random.PRNGKey(0),
                             shape,
                             dtype=jax.numpy.bfloat16)


def _rel_err(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


class TestJaxRmsNorm:
    """Test for JaxRmsNorm layer."""

    @pytest.mark.parametrize(
        "rng_key",
        [jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    @pytest.mark.parametrize("num_features", [16, 32, 64])
    @pytest.mark.parametrize(
        "dtype", [jax.numpy.float32, jax.numpy.float16, jax.numpy.bfloat16])
    def test_numerical_correctness_against_flax_unquantized(
            self, rng_key, num_features, dtype):
        """Run the same input through JaxRmsNorm vs. flax RMSNorm and compare outputs.
        """
        mesh = Mesh(jax.devices('cpu')[:1], ("model", ))
        unquantize_config = get_tpu_quantization_config(
            VllmConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B")),
            mesh)

        kwargs = {
            "num_features": num_features,
            "epsilon": 1e-6,
            "dtype": dtype,
            "param_dtype": dtype,
            "rngs": nnx.Rngs(0)
        }
        layer = JaxRmsNorm(quant_config=unquantize_config, **kwargs)
        layer.weight.value = jax.random.uniform(rng_key, (num_features, ),
                                                dtype=dtype)

        flax_layer = nnx.RMSNorm(**kwargs)
        flax_layer.scale.value = layer.weight.value

        x = jax.random.uniform(rng_key, (2, 8, num_features), dtype=dtype)

        jax_output = layer(x)
        flax_output = flax_layer(x)

        assert jax.numpy.allclose(jax_output,
                                  flax_output,
                                  rtol=1e-5,
                                  atol=1e-5)

    @pytest.mark.parametrize("fuse_fp8_quant", [False, True])
    def test_fused_rmsnorm_fp8_quant_is_opt_in(self, fuse_fp8_quant):
        """Only opted-in norms feeding qwix FP8 projections emit QArrays."""
        x = _input()
        q_block = qwix.quantize_model(_NormLinearBlock(fuse_fp8_quant),
                                      qwix.PtqProvider([_FP8_RULE]), x)
        with mock.patch.object(
                norm_lib,
                "sharded_fused_rmsnorm_fp8_quant",
                wraps=norm_lib.sharded_fused_rmsnorm_fp8_quant) as mock_fused:
            out = q_block(x)
        assert mock_fused.call_count == int(fuse_fp8_quant)

        unfused = qwix.quantize_model(_NormLinearBlock(False),
                                      qwix.PtqProvider([_FP8_RULE]), x)
        unfused_out = unfused(x)
        bf16_out = _NormLinearBlock(False)(x)
        # Fusion only changes the FP8 rounding of individual activations, so
        # the W8A8 error w.r.t. the unquantized output stays the same.
        assert _rel_err(out, unfused_out) < 0.03
        assert _rel_err(out, bf16_out) < 1.1 * _rel_err(unfused_out, bf16_out)

    def test_fused_rmsnorm_fp8_quant_without_qwix_rule(self):
        """Without an active qwix rule an opted-in norm stays a plain RMSNorm."""
        layer = _make_rms_norm(fuse_fp8_quant=True)
        flax_layer = nnx.RMSNorm(_NUM_FEATURES,
                                 epsilon=1e-6,
                                 param_dtype=jax.numpy.bfloat16,
                                 rngs=nnx.Rngs(0))
        x = _input()
        out = layer(x)
        assert isinstance(out, jax.Array)
        assert jax.numpy.array_equal(out, flax_layer(x))

    def test_sandwich_post_attention_norm_is_not_fused(self, monkeypatch):
        """A post-attention norm feeding the residual add never emits a QArray."""
        monkeypatch.setenv("USE_FUSED_RMSNORM_QUANT", "1")
        x = _input()
        q_block = qwix.quantize_model(_SandwichNormBlock(),
                                      qwix.PtqProvider([_FP8_RULE]), x)
        with mock.patch.object(
                norm_lib,
                "sharded_fused_rmsnorm_fp8_quant",
                wraps=norm_lib.sharded_fused_rmsnorm_fp8_quant) as mock_fused:
            out = q_block(x)
        mock_fused.assert_not_called()
        assert out.dtype == jax.numpy.bfloat16

    @pytest.mark.parametrize("rule_kwargs", [
        {
            "act_qtype": None
        },
        {
            "weight_qtype": jax.numpy.int8,
            "act_qtype": jax.numpy.int8
        },
        {
            "tile_size": 128
        },
        {
            "act_calibration_method": "absmax,0.8"
        },
    ])
    def test_fused_rmsnorm_fp8_quant_skips_incompatible_rules(
            self, rule_kwargs):
        """Fusion only applies to dynamic per-token absmax FP8 activations."""
        x = _input()
        rule = dataclasses.replace(_FP8_RULE, **rule_kwargs)
        q_block = qwix.quantize_model(_NormLinearBlock(True),
                                      qwix.PtqProvider([rule]), x)
        with mock.patch.object(
                norm_lib,
                "sharded_fused_rmsnorm_fp8_quant",
                wraps=norm_lib.sharded_fused_rmsnorm_fp8_quant) as mock_fused:
            q_block(x)
        mock_fused.assert_not_called()

    def test_sharded_fused_rmsnorm_fp8_quant_residual(self):
        """Verifies the fused residual add and sub-channel scales."""
        mesh = Mesh(jax.devices()[:1], ("model", ))
        x = _input()
        res = jax.random.normal(jax.random.PRNGKey(1), (16, _NUM_FEATURES),
                                dtype=jax.numpy.bfloat16)
        weight = jax.numpy.ones((_NUM_FEATURES, ), dtype=jax.numpy.bfloat16)
        q_res, updated_res = sharded_fused_rmsnorm_fp8_quant(
            x,
            weight,
            epsilon=1e-6,
            residual=res,
            return_residual=True,
            block_k=128,
            mesh=mesh,
        )
        assert isinstance(q_res, qwix.QArray)
        assert q_res.qvalue.dtype == jax.numpy.float8_e4m3fn
        assert q_res.scale.dtype == jax.numpy.bfloat16
        assert q_res.scale.shape == (16, 2)
        assert jax.numpy.array_equal(updated_res,
                                     (x + res).astype(jax.numpy.bfloat16))

    def test_sharded_fused_rmsnorm_fp8_quant_jit_context_mesh(self):
        """Verifies shard_map wrapping inside @jax.jit without an explicit mesh."""
        mesh = Mesh(jax.devices()[:1], ("model", ))
        x = _input()
        weight = jax.numpy.ones((_NUM_FEATURES, ), dtype=jax.numpy.bfloat16)

        with mock.patch("jax.shard_map", wraps=jax.shard_map) as mock_smap:
            with jax.set_mesh(mesh):
                q_out = jax.jit(sharded_fused_rmsnorm_fp8_quant)(x, weight)
            assert mock_smap.call_count == 1
        assert isinstance(q_out, qwix.QArray)
        assert q_out.qvalue.shape == (16, _NUM_FEATURES)
        assert q_out.scale.shape == (16, 1)

        sharded_x = jax.device_put(
            x,
            jax.sharding.NamedSharding(mesh,
                                       jax.sharding.PartitionSpec(None, None)))
        with mock.patch("jax.shard_map", wraps=jax.shard_map) as mock_smap:
            q_sharded = jax.jit(sharded_fused_rmsnorm_fp8_quant)(sharded_x,
                                                                 weight)
            assert mock_smap.call_count == 1
        assert jax.numpy.array_equal(q_sharded.scale, q_out.scale)


class TestJaxLayerNorm:
    """Test for JaxLayerNorm layer."""

    @pytest.mark.parametrize(
        "rng_key",
        [jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
    @pytest.mark.parametrize("num_features", [16, 32, 64])
    @pytest.mark.parametrize(
        "dtype", [jax.numpy.float32, jax.numpy.float16, jax.numpy.bfloat16])
    def test_numerical_correctness_against_flax(self, rng_key, num_features,
                                                dtype):
        """Run the same input through JaxLayerNorm vs. flax LayerNorm and compare outputs.
        """
        kwargs = {
            "num_features": num_features,
            "epsilon": 1e-6,
            "dtype": dtype,
            "param_dtype": dtype,
            "rngs": nnx.Rngs(0)
        }
        layer = JaxLayerNorm(**kwargs)
        layer.weight[...] = jax.random.uniform(rng_key, (num_features, ),
                                               dtype=dtype)
        layer.bias[...] = jax.random.uniform(rng_key, (num_features, ),
                                             dtype=dtype)
        flax_layer = nnx.LayerNorm(**kwargs)
        flax_layer.scale[...] = layer.weight[...]
        flax_layer.bias[...] = layer.bias[...]
        x = jax.random.uniform(rng_key, (2, 8, num_features), dtype=dtype)
        jax_output = layer(x)
        flax_output = flax_layer(x)
        assert jax.numpy.allclose(jax_output,
                                  flax_output,
                                  rtol=1e-5,
                                  atol=1e-5)

    def test_parameter_aliasing(self):
        """Verifies weight aliasing, scale deletion, and named_parameters matching."""
        rngs = nnx.Rngs(0)
        jax_ln = JaxLayerNorm(
            num_features=16,
            rngs=rngs,
        )
        assert "scale" not in jax_ln.__dict__
        assert "weight" in jax_ln.__dict__
        assert isinstance(jax_ln.weight, nnx.Param)
        assert jax_ln.scale is jax_ln.weight

        named_params = dict(jax_ln.named_parameters())

        assert "weight" in named_params
        assert "scale" not in named_params
