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
"""Tests for the JAX-native compressed-tensors wNa16 (W4A16) linear method.

Dispatch tests mirror ``test_compressed_tensors.py``. The load tests pack a
random int4 weight exactly as compressed-tensors ``pack-quantized`` does, feed
the three checkpoint tensors through the layer's weight loaders, and compare
the forward pass with ``x @ dequant(W).T``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.linear import (JaxLinear,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.quantization import wna16
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig
from tpu_inference.layers.jax.quantization.unquantized import \
    UnquantizedLinearMethod
from tpu_inference.layers.jax.quantization.wna16 import (
    WNA16LinearMethod, WNA16MergedLinearMethod)

GROUP_SIZE = 32


# Modeled on google/gemma-4-31B-it-qat-w4a16-ct's quantization_config.
def _w4a16_config(group_size=GROUP_SIZE, strategy="group", ignore=None):
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": strategy,
                    "group_size": group_size,
                    "dynamic": False,
                    "actorder": None,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ignore or [],
    }


def _pack(q: np.ndarray) -> np.ndarray:
    """Pack int4 values [out, in] as compressed-tensors does: +8 bias, eight
    per int32 word along the input dim, lowest nibble first."""
    out, n_in = q.shape
    biased = (q.astype(np.int64) + 8).reshape(out, n_in // 8, 8)
    words = np.zeros((out, n_in // 8), np.int64)
    for k in range(8):
        words |= biased[:, :, k] << (4 * k)
    return words.astype(np.uint32).view(np.int32)


def _random_w4a16(rng, out, n_in, group_size=GROUP_SIZE):
    q = rng.integers(-8, 8, size=(out, n_in), dtype=np.int8)
    scale = rng.uniform(0.001, 0.02,
                        size=(out, n_in // group_size)).astype(np.float32)
    scale = torch.tensor(scale).to(torch.bfloat16)
    # The reference uses the bf16-rounded scale the layer will see.
    dequant = q.astype(np.float32) * np.repeat(
        scale.float().numpy(), group_size, axis=1)
    tensors = {
        "weight_packed": torch.from_numpy(_pack(q)),
        "weight_scale": scale,
        "weight_shape": torch.tensor([out, n_in], dtype=torch.int64),
    }
    return tensors, dequant


def _load(layer, tensors, shard_id=None):
    for name, tensor in tensors.items():
        param = getattr(layer, name)
        if shard_id is None:
            param.weight_loader(param, tensor)
        else:
            param.weight_loader(param, tensor, shard_id)


def _rel_err(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


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


class TestWNA16Dispatch:

    def test_linear_routes_to_wna16(self, rngs, mesh):
        config = CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(mesh):
            layer = JaxLinear(64,
                              32,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="mlp.down_proj")
        assert isinstance(layer.quant_method, WNA16LinearMethod)
        assert not hasattr(layer, "weight")
        assert layer.weight_packed.shape == (32, 64 // 8)
        assert layer.weight_scale.shape == (32, 64 // GROUP_SIZE)

    def test_merged_routes_to_merged_wna16(self, rngs, mesh):
        config = CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(mesh):
            layer = JaxMergedColumnParallelLinear(64, [32, 32],
                                                  rngs,
                                                  use_bias=False,
                                                  quant_config=config,
                                                  prefix="mlp.gate_proj")
        assert isinstance(layer.quant_method, WNA16MergedLinearMethod)
        assert layer.weight_packed.shape == (64, 64 // 8)

    def test_ignored_layer_is_unquantized(self, rngs, mesh):
        config = CompressedTensorsConfig(
            _w4a16_config(ignore=["re:.*down_proj"]))
        with jax.set_mesh(mesh):
            layer = JaxLinear(64,
                              32,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="mlp.down_proj")
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)

    def test_asymmetric_is_rejected(self, rngs, mesh):
        cfg = _w4a16_config()
        cfg["config_groups"]["group_0"]["weights"]["symmetric"] = False
        config = CompressedTensorsConfig(cfg)
        with jax.set_mesh(mesh), pytest.raises(NotImplementedError):
            JaxLinear(64,
                      32,
                      rngs,
                      use_bias=False,
                      quant_config=config,
                      prefix="mlp.down_proj")


class TestWNA16Load:

    @pytest.mark.parametrize("strategy,group_size", [("group", GROUP_SIZE),
                                                     ("channel", None)])
    def test_linear_forward_matches_dequant(self, rngs, mesh, strategy,
                                            group_size):
        n_in, out = 256, 128
        config = CompressedTensorsConfig(
            _w4a16_config(group_size=group_size, strategy=strategy))
        rng = np.random.default_rng(0)
        tensors, dequant = _random_w4a16(rng, out, n_in, group_size or n_in)
        with jax.set_mesh(mesh):
            layer = JaxLinear(n_in,
                              out,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="mlp.down_proj")
            _load(layer, tensors)
            assert layer.quant_method.process_weights_after_loading(layer)
            assert layer.weight.dtype == jnp.int4
            assert layer.weight.shape == (n_in, out)
            # Grouped weights take the gmm_v2 kernel (3D scale), channelwise
            # ones the XLA path (2D scale).
            n_groups = n_in // (group_size or n_in)
            assert layer.weight_scale.shape == ((n_groups, 1,
                                                 out) if group_size else
                                                (n_groups, out))
            x = jnp.asarray(rng.standard_normal((8, n_in)), dtype=jnp.bfloat16)
            y = layer(x)
        expected = np.asarray(x, np.float32) @ dequant.T
        assert y.shape == (8, out)
        assert _rel_err(y, expected) < 1e-2

    def test_process_waits_for_all_tensors(self, rngs, mesh):
        config = CompressedTensorsConfig(_w4a16_config())
        tensors, _ = _random_w4a16(np.random.default_rng(1), 32, 64)
        with jax.set_mesh(mesh):
            layer = JaxLinear(64,
                              32,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="mlp.down_proj")
            _load(layer, {
                k: v
                for k, v in tensors.items() if k != "weight_shape"
            })
            assert not layer.quant_method.process_weights_after_loading(layer)

    def test_weight_shape_mismatch_raises(self, rngs, mesh):
        config = CompressedTensorsConfig(_w4a16_config())
        tensors, _ = _random_w4a16(np.random.default_rng(2), 32, 64)
        tensors["weight_shape"] = torch.tensor([32, 128])
        with jax.set_mesh(mesh):
            layer = JaxLinear(64,
                              32,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="mlp.down_proj")
            with pytest.raises(ValueError):
                _load(layer, tensors)

    def test_merged_forward_matches_dequant(self, rngs, mesh):
        n_in, sizes = 128, [64, 96]
        config = CompressedTensorsConfig(_w4a16_config())
        rng = np.random.default_rng(3)
        shards = [_random_w4a16(rng, size, n_in) for size in sizes]
        with jax.set_mesh(mesh):
            layer = JaxMergedColumnParallelLinear(n_in,
                                                  sizes,
                                                  rngs,
                                                  use_bias=False,
                                                  quant_config=config,
                                                  prefix="mlp.gate_proj")
            # The loader routes each projection with its shard_id, and not
            # necessarily in order.
            for shard_id in (1, 0):
                _load(layer, shards[shard_id][0], shard_id=shard_id)
            assert layer.quant_method.process_weights_after_loading(layer)
            x = jnp.asarray(rng.standard_normal((4, n_in)), dtype=jnp.bfloat16)
            y = layer(x)
        dequant = np.concatenate([d for _, d in shards], axis=0)
        expected = np.asarray(x, np.float32) @ dequant.T
        assert y.shape == (4, sum(sizes))
        assert _rel_err(y, expected) < 1e-2


class TestWNA16KernelSelection:
    """The kernel is used exactly when gmm_v2 would dequantize in VMEM."""

    @pytest.mark.parametrize("mxu,group_size,expected", [
        (256, 32, True),
        (256, 128, True),
        (256, 256, False),
        (128, 32, True),
        (128, 128, False),
    ])
    def test_cutoff_follows_the_mxu(self, mxu, group_size, expected):
        with patch.object(wna16, "_mxu_column_size", return_value=mxu):
            assert wna16._uses_kernel(group_size) is expected

    @staticmethod
    def _method(group_size, in_sharding):
        layer = SimpleNamespace(prefix="mlp.down_proj", einsum_str="mn,np->mp")
        cfg = SimpleNamespace(batch_features=(),
                              in_features=(256, ),
                              out_features=(128, ),
                              output_sizes=[128],
                              in_features_sharding=(in_sharding, ))
        with patch.object(wna16, "_mxu_column_size", return_value=256):
            return WNA16LinearMethod(layer, cfg, group_size)

    def test_channelwise_on_sharded_input_is_rejected(self):
        with pytest.raises(NotImplementedError, match="input dim is sharded"):
            self._method(None, "model")

    def test_grouped_on_sharded_input_uses_the_kernel(self):
        assert self._method(GROUP_SIZE, "model").use_kernel

    def test_channelwise_on_unsharded_input_uses_xla(self):
        assert not self._method(None, None).use_kernel


def test_quantized_moe_is_rejected_not_served_dense():
    """A W4A16 MoE must fail at load rather than read packed int4 as dense."""
    config = CompressedTensorsConfig(_w4a16_config())
    with pytest.raises(NotImplementedError, match="MoE layer"):
        config.get_quant_method(MagicMock(spec=JaxMoE),
                                prefix="layers.0.experts")
