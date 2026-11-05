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

from unittest import mock

import jax
import pytest
import qwix
from flax import nnx
from jax.sharding import Mesh
from vllm.config import ModelConfig, VllmConfig

from tpu_inference.layers.jax.norm import (JaxLayerNorm, JaxRmsNorm,
                                           sharded_fused_rmsnorm_fp8_quant)
from tpu_inference.layers.jax.quantization.unquantized import UnquantizedConfig
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config


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

    def test_fused_rmsnorm_fp8_quant(self, monkeypatch):
        """Verifies JaxRmsNorm fused FP8 quant and Qwix QArray passthrough."""
        mesh = Mesh(jax.devices()[:1], ("model", ))
        unquantize_config = get_tpu_quantization_config(
            VllmConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B")),
            mesh)
        num_features = 256
        rng_key = jax.random.PRNGKey(0)
        layer = JaxRmsNorm(
            num_features=num_features,
            epsilon=1e-6,
            dtype=jax.numpy.bfloat16,
            param_dtype=jax.numpy.bfloat16,
            rngs=nnx.Rngs(0),
            quant_config=unquantize_config,
            fuse_fp8_quant=True,
        )
        x = jax.random.normal(rng_key, (16, num_features),
                              dtype=jax.numpy.bfloat16)
        q_out = layer(x)
        assert isinstance(q_out, qwix.QArray)
        assert q_out.qvalue.dtype == jax.numpy.float8_e4m3fn
        assert q_out.scale.dtype == jax.numpy.bfloat16
        assert q_out.qvalue.shape == (16, num_features)
        assert q_out.scale.shape == (16, 1)

        res = jax.random.normal(rng_key, (16, num_features),
                                dtype=jax.numpy.bfloat16)
        q_res, updated_res = sharded_fused_rmsnorm_fp8_quant(
            x,
            layer.weight.value,
            epsilon=1e-6,
            residual=res,
            return_residual=True,
            block_k=128,
            mesh=mesh,
        )
        assert isinstance(q_res, qwix.QArray)
        assert q_res.scale.shape == (16, 2)
        assert jax.numpy.array_equal(updated_res,
                                     (x + res).astype(jax.numpy.bfloat16))

        class _NormLinearBlock(nnx.Module):

            def __init__(self):
                self.input_layernorm = JaxRmsNorm(
                    num_features,
                    epsilon=1e-6,
                    dtype=jax.numpy.bfloat16,
                    param_dtype=jax.numpy.bfloat16,
                    rngs=nnx.Rngs(0),
                    quant_config=unquantize_config,
                    prefix="layers.0.input_layernorm",
                )
                self.proj = nnx.Linear(
                    num_features,
                    64,
                    use_bias=False,
                    param_dtype=jax.numpy.bfloat16,
                    rngs=nnx.Rngs(1),
                )

            def __call__(self, inp):
                return self.proj(self.input_layernorm(inp))

        rules = [
            qwix.QuantizationRule(
                module_path=".*",
                weight_qtype=jax.numpy.float8_e4m3fn,
                act_qtype=jax.numpy.float8_e4m3fn,
            )
        ]
        q_block = qwix.quantize_model(_NormLinearBlock(),
                                      qwix.PtqProvider(rules), x)
        monkeypatch.setenv("USE_FUSED_RMSNORM_QUANT", "0")
        out_unfused = q_block(x)
        monkeypatch.setenv("USE_FUSED_RMSNORM_QUANT", "1")
        out_fused = q_block(x)
        assert jax.numpy.allclose(out_fused, out_unfused, rtol=2e-2, atol=2e-2)

    def test_fused_rmsnorm_fp8_jit_context_mesh(self):
        """Verifies shard_map wrapping inside @jax.jit when layer.mesh is None."""
        mesh = Mesh(jax.devices()[:1], ("model", ))
        num_features = 256
        layer = JaxRmsNorm(
            num_features=num_features,
            epsilon=1e-6,
            dtype=jax.numpy.bfloat16,
            param_dtype=jax.numpy.bfloat16,
            rngs=nnx.Rngs(0),
            quant_config=UnquantizedConfig({}),
            fuse_fp8_quant=True,
        )
        assert layer.mesh is None
        x = jax.random.normal(jax.random.PRNGKey(0), (16, num_features),
                              dtype=jax.numpy.bfloat16)

        with mock.patch("jax.shard_map", wraps=jax.shard_map) as mock_smap:
            with jax.set_mesh(mesh):
                q_out = jax.jit(lambda inp: layer(inp))(x)
            assert mock_smap.call_count == 1
        assert isinstance(q_out, qwix.QArray)
        assert q_out.qvalue.shape == (16, num_features)
        assert q_out.scale.shape == (16, 1)

        sharded_x = jax.device_put(
            x,
            jax.sharding.NamedSharding(mesh,
                                       jax.sharding.PartitionSpec(None, None)))
        with mock.patch("jax.shard_map", wraps=jax.shard_map) as mock_smap:
            q_sharded = jax.jit(lambda inp: layer(inp))(sharded_x)
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
