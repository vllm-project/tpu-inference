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
"""Tests for the JAX-native int8 W8A8 compressed-tensors linear method."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig
from tpu_inference.layers.jax.quantization.int8 import \
    Int8ChannelwiseLinearMethod
from tpu_inference.layers.jax.quantization.unquantized import \
    UnquantizedLinearMethod


# A compressed-tensors `quantization_config` modeled on llm-compressor
# int8 W8A8 output (e.g. RedHatAI *-quantized.w8a8 checkpoints): int8
# channelwise static symmetric weights + dynamic per-token int8 activations.
def _int8_channel_config(ignore=None):
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
        "ignore": ignore or [],
    }


# Static (non-dynamic) activations are not supported by the JAX int8 method;
# the dispatch must not silently mis-route this scheme.
def _int8_static_act_config():
    cfg = _int8_channel_config()
    cfg["config_groups"]["group_0"]["input_activations"] = {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
        "dynamic": False,
    }
    return cfg


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((1, ) * len(MESH_AXIS_NAMES))
    with Mesh(device_mesh, axis_names=MESH_AXIS_NAMES) as m:
        yield m


@pytest.fixture
def rngs():
    return nnx.Rngs(42)


def _make_linear(rngs, quant_config, in_features=64, out_features=32):
    return JaxLinear(in_features,
                     out_features,
                     rngs=rngs,
                     quant_config=quant_config,
                     kernel_init=nnx.initializers.uniform(),
                     prefix="mlp.proj1")


class TestInt8ChannelwiseDispatch:

    def test_int8_routes_to_channelwise_method(self, rngs, mesh):
        """int8 channel weights + dynamic token acts -> Int8 method."""
        config = CompressedTensorsConfig(_int8_channel_config())
        with jax.set_mesh(mesh):
            layer = _make_linear(rngs, config)
        assert isinstance(layer.quant_method, Int8ChannelwiseLinearMethod)

    def test_ignored_layer_is_skipped(self, rngs, mesh):
        """A layer matched by the ignore regex -> UnquantizedLinearMethod."""
        config = CompressedTensorsConfig(
            _int8_channel_config(ignore=["re:.*proj1"]))
        with jax.set_mesh(mesh):
            layer = _make_linear(rngs, config)
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)

    def test_static_act_scheme_is_rejected(self, rngs, mesh):
        """Static-tensor int8 activations are unsupported -> loud failure."""
        config = CompressedTensorsConfig(_int8_static_act_config())
        with jax.set_mesh(mesh):
            with pytest.raises(NotImplementedError):
                _make_linear(rngs, config)

    def test_int8_weight_params(self, rngs, mesh):
        """Weight is int8 [in, out]; scale is fp32 [out] named weight_scale."""
        config = CompressedTensorsConfig(_int8_channel_config())
        with jax.set_mesh(mesh):
            layer = _make_linear(rngs, config, 64, 32)
        assert layer.weight.get_value().dtype == jnp.int8
        assert layer.weight.get_value().shape == (64, 32)
        assert hasattr(layer, "weight_scale")
        assert layer.weight_scale.get_value().shape == (32, )
        assert layer.weight_scale.get_value().dtype == jnp.float32


class TestInt8ChannelwiseNumerics:

    def test_apply_matches_dequant_reference(self, rngs, mesh):
        """int8 W8A8 forward tracks the fp32 dequantized reference."""
        in_features, out_features, tokens = 64, 32, 8
        config = CompressedTensorsConfig(_int8_channel_config())
        with jax.set_mesh(mesh):
            layer = _make_linear(rngs, config, in_features, out_features)

            rng = np.random.default_rng(0)
            w_f = rng.standard_normal(
                (in_features, out_features)).astype(np.float32) * 0.05
            w_scale = np.abs(w_f).max(axis=0) / 127.0
            w_q = np.round(w_f / w_scale).astype(np.int8)
            layer.weight.set_value(jnp.asarray(w_q))
            layer.weight_scale.set_value(jnp.asarray(w_scale, jnp.float32))

            x = (rng.standard_normal(
                (tokens, in_features)) * 0.5).astype(np.float32)
            ref = x @ (w_q.astype(np.float32) * w_scale)
            out = np.asarray(layer(jnp.asarray(x, jnp.bfloat16)))

        rel_err = np.abs(out.astype(np.float32) -
                         ref).max() / np.abs(ref).max()
        # Error budget: dynamic int8 activation quantization (~1/127)
        # plus the bf16 input cast.
        assert rel_err < 0.03, rel_err
