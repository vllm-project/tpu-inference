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
from tpu_inference.models.jax.utils.weight_utils import JaxDummyModelLoader

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

    def test_row_parallel_scales_split_per_shard(self, rngs, mesh):
        """A row-parallel linear whose rows do not split into whole groups
        (26B-A4B's dense down_proj: 2112 over 4 shards) gets per-shard scale
        groups, and still computes x @ dequant(W).T."""
        n_in, out = 192, 64  # 192 / 4 = 48 rows per shard: 1.5 groups
        config = CompressedTensorsConfig(_w4a16_config())
        rng = np.random.default_rng(8)
        tensors, dequant = _random_w4a16(rng, out, n_in)
        with jax.set_mesh(mesh):
            layer = JaxLinear(n_in,
                              out,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="mlp.down_proj",
                              kernel_init=nnx.with_partitioning(
                                  nnx.initializers.uniform(), ("model", None)))
            assert layer.quant_method.linear_config.in_features_sharding == (
                "model", )
            _load(layer, tensors)
            with patch.object(wna16, "get_mesh_shape_product", return_value=4):
                assert layer.quant_method.process_weights_after_loading(layer)
            # gcd(32, 48) = 16: 12 groups of 16, 3 per shard.
            assert layer.weight_scale.shape == (12, 1, out)
            x = jnp.asarray(rng.standard_normal((8, n_in)), dtype=jnp.bfloat16)
            y = layer(x)
        assert _rel_err(y, np.asarray(x, np.float32) @ dequant.T) < 1e-2

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
    def _method(group_size, in_sharding, in_shards=1):
        layer = SimpleNamespace(prefix="mlp.down_proj", einsum_str="mn,np->mp")
        cfg = SimpleNamespace(batch_features=(),
                              in_features=(256, ),
                              out_features=(128, ),
                              output_sizes=[128],
                              in_features_sharding=(in_sharding, ),
                              mesh=None)
        with patch.object(wna16, "_mxu_column_size", return_value=256), \
                patch.object(wna16, "get_mesh_shape_product",
                             return_value=in_shards):
            return WNA16LinearMethod(layer, cfg, group_size)

    def test_channelwise_on_sharded_input_is_rejected(self):
        with pytest.raises(NotImplementedError, match="input dim is sharded"):
            self._method(None, "model", in_shards=4)

    def test_channelwise_on_a_size_one_axis_uses_xla(self):
        # TP=1: the input dim names a mesh axis, but that axis has one device.
        assert not self._method(None, "model", in_shards=1).use_kernel

    def test_grouped_on_sharded_input_uses_the_kernel(self):
        assert self._method(GROUP_SIZE, "model").use_kernel

    def test_channelwise_on_unsharded_input_uses_xla(self):
        assert not self._method(None, None).use_kernel


def test_unsupported_quantized_moe_is_rejected_not_served_dense():
    """A quantized MoE with no JAX method must fail at load rather than read
    packed weights as dense ones."""
    cfg = _w4a16_config()
    cfg["config_groups"]["group_0"]["weights"]["symmetric"] = False
    config = CompressedTensorsConfig(cfg)
    with pytest.raises(NotImplementedError, match="MoE layer"):
        config.get_quant_method(MagicMock(spec=JaxMoE),
                                prefix="layers.0.experts")


# ---- routed experts -------------------------------------------------------


def _w4a16_moe_expert_tensors(rng, num_experts, hidden, inter):
    """Per-expert checkpoint tensors named as compressed-tensors exports them
    (cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit), plus the dequantized references in
    HF [out, in] layout."""
    tensors, ref = [], {"gate": [], "up": [], "down": []}
    for e in range(num_experts):
        for proj, role, (out, n_in) in (("gate_proj", "gate", (inter, hidden)),
                                        ("up_proj", "up", (inter, hidden)),
                                        ("down_proj", "down", (hidden,
                                                               inter))):
            parts, dequant = _random_w4a16(rng, out, n_in)
            ref[role].append(dequant)
            for name, t in parts.items():
                tensors.append((f"{e}.{proj}.{name}", t))
    return tensors, {k: np.stack(v) for k, v in ref.items()}


def _moe_layer(moe_backend=None, num_experts=4, hidden=128, inter=64):
    from tpu_inference.layers.common.moe import MoEBackend
    return SimpleNamespace(
        dtype=jnp.bfloat16,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_moe=inter,
        activation="silu",
        moe_backend=moe_backend or MoEBackend.GMM_TP,
        mesh=Mesh(
            np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("data", "model")),
        prefix="model.language_model.layers.0.experts",
        kernel_gating_EDF=nnx.Param(jnp.zeros((num_experts, hidden, inter))),
        kernel_up_proj_EDF=nnx.Param(jnp.zeros((num_experts, hidden, inter))),
        kernel_down_proj_EFD=nnx.Param(jnp.zeros(
            (num_experts, inter, hidden))),
    )


def _moe_method_and_layer(**kw):
    layer = _moe_layer(**kw)
    method = wna16.WNA16FusedMoEMethod(GROUP_SIZE)
    method.create_weights_jax(layer, rngs=nnx.Rngs(0))
    return method, layer


def _load_moe(method, layer, tensors):
    return method.load_weights(layer=layer,
                               original_load_weights_fn=None,
                               weights=iter(tensors))


class TestWNA16MoEDispatch:

    def test_w4a16_experts_route_to_the_moe_method(self):
        config = CompressedTensorsConfig(_w4a16_config())
        method = config.get_quant_method(MagicMock(spec=JaxMoE),
                                         prefix="layers.0.experts")
        assert isinstance(method, wna16.WNA16FusedMoEMethod)
        assert method.group_size == GROUP_SIZE

    def test_channelwise_experts_are_rejected(self):
        # Channelwise would quantize the activation inside gmm_v2 (W4A8).
        config = CompressedTensorsConfig(
            _w4a16_config(group_size=None, strategy="channel"))
        with pytest.raises(NotImplementedError, match="narrower than the MXU"):
            config.get_quant_method(MagicMock(spec=JaxMoE),
                                    prefix="layers.0.experts")

    def test_groups_as_wide_as_the_mxu_are_rejected(self):
        config = CompressedTensorsConfig(_w4a16_config(group_size=128))
        with patch.object(wna16, "_mxu_column_size", return_value=128):
            with pytest.raises(NotImplementedError,
                               match="narrower than the MXU"):
                config.get_quant_method(MagicMock(spec=JaxMoE),
                                        prefix="layers.0.experts")


class TestWNA16MoELifecycle:

    def test_create_replaces_the_dense_placeholders(self):
        _, layer = _moe_method_and_layer()
        for name in ("kernel_gating_EDF", "kernel_up_proj_EDF",
                     "kernel_down_proj_EFD"):
            assert not hasattr(layer, name)
        # [E, cols, out]: the dummy loaders swap the last two dims.
        assert layer.w4a16_gate_packed.shape == (4, 128 // 8, 64)
        assert layer.w4a16_down_scale.shape == (4, 64 // GROUP_SIZE, 128)

    def test_process_waits_for_every_expert(self):
        method, layer = _moe_method_and_layer()
        tensors, _ = _w4a16_moe_expert_tensors(np.random.default_rng(0), 4,
                                               128, 64)
        first = [t for t in tensors if not t[0].startswith("3.")]
        rest = [t for t in tensors if t[0].startswith("3.")]
        assert _load_moe(method, layer, first) == set()
        assert method.process_weights_after_loading(layer) is False
        assert len(_load_moe(method, layer, rest)) == 6

    def test_prefixed_names_load_too(self):
        method, layer = _moe_method_and_layer()
        tensors, _ = _w4a16_moe_expert_tensors(np.random.default_rng(1), 4,
                                               128, 64)
        prefixed = [(f"{layer.prefix}.{n}", t) for n, t in tensors]
        assert len(_load_moe(method, layer, prefixed)) == 6

    def test_weight_shape_mismatch_raises(self):
        method, layer = _moe_method_and_layer()
        bad = [("0.gate_proj.weight_shape", torch.tensor([64, 256]))]
        with pytest.raises(ValueError, match="weight_shape"):
            _load_moe(method, layer, bad)

    def test_float_packed_words_are_rejected(self):
        method, layer = _moe_method_and_layer()
        bad = [("0.up_proj.weight_packed", torch.zeros((64, 16)))]
        with pytest.raises(TypeError, match="integer dtype"):
            _load_moe(method, layer, bad)

    def test_unknown_tensor_is_rejected(self):
        method, layer = _moe_method_and_layer()
        with pytest.raises(ValueError, match="unexpected"):
            _load_moe(method, layer,
                      [("0.gate_proj.weight_zero_point", torch.zeros(1))])

    def test_dummy_weights_load_and_process(self):
        """--load-format dummy fills the staged params and they process into
        the same kernel shapes as a real checkpoint."""
        E, D, F = 4, 128, 64
        method, layer = _moe_method_and_layer(num_experts=E, hidden=D, inter=F)
        model = SimpleNamespace(named_parameters=lambda: [(
            name, getattr(layer, name)) for name in method._staged_names()])
        loader = SimpleNamespace(_process_weights_after_loading=lambda m: None)
        JaxDummyModelLoader.load_weights(loader, model, None)
        for name in method._staged_names():
            role, kind = name.split("_")[1:]
            assert all(w.shape == (1, ) +
                       method._expert_shape(layer, role, kind)
                       for w in getattr(layer, name)._weights_to_load), name
        assert method.process_weights_after_loading(layer) is True
        real_method, real_layer = _moe_method_and_layer(num_experts=E,
                                                        hidden=D,
                                                        inter=F)
        tensors, _ = _w4a16_moe_expert_tensors(np.random.default_rng(8), E, D,
                                               F)
        _load_moe(real_method, real_layer, tensors)
        assert real_method.process_weights_after_loading(real_layer) is True
        for name in ("kernel_gating_upproj_EDF",
                     "kernel_gating_upproj_EDF_weight_scale",
                     "kernel_down_proj_EFD",
                     "kernel_down_proj_EFD_weight_scale"):
            assert (getattr(layer,
                            name)[...].shape == getattr(real_layer,
                                                        name)[...].shape), name

    def test_split_groups_keeps_aligned_shards(self):
        scale = jnp.ones((2, 5, 22), jnp.float32)
        out, group = wna16._split_groups_for_shards(scale, 32, 704, 2)
        assert group == 32 and out.shape == scale.shape

    def test_split_groups_refines_straddling_groups(self):
        # 704 rows over 4 shards is 176 per shard: 5.5 groups of 32.
        scale = jnp.asarray(
            np.random.default_rng(5).uniform(0.001, 0.02, (2, 3, 22)),
            jnp.float32)
        out, group = wna16._split_groups_for_shards(scale, 32, 704, 4)
        assert group == 16 and out.shape == (2, 3, 44)
        assert (704 // 4) % group == 0 and out.shape[-1] % 4 == 0
        np.testing.assert_array_equal(np.repeat(np.asarray(out), 16, -1),
                                      np.repeat(np.asarray(scale), 32, -1))

    def test_split_groups_along_axis_0_for_row_parallel_linears(self):
        # Gemma 4 26B-A4B's dense MLP down_proj: 2112 rows over 4 shards.
        scale = jnp.asarray(
            np.random.default_rng(7).uniform(0.001, 0.02, (66, 5)),
            jnp.float32)
        out, group = wna16._split_groups_for_shards(scale, 32, 2112, 4, axis=0)
        assert group == 16 and out.shape == (132, 5) and 132 % 4 == 0
        np.testing.assert_array_equal(np.repeat(np.asarray(out), 16, 0),
                                      np.repeat(np.asarray(scale), 32, 0))

    def test_split_groups_rejects_uneven_shards(self):
        with pytest.raises(ValueError, match="does not split"):
            wna16._split_groups_for_shards(jnp.ones((1, 1, 3)), 32, 96, 5)

    def test_gmm_tp_down_scales_split_per_shard(self):
        """With 4 shards, w2's scales come out per 16 rows and still
        dequantize exactly to the checkpoint."""
        E, D, F = 4, 128, 64  # 64 / 4 = 16 rows per shard, half a group
        method, layer = _moe_method_and_layer(num_experts=E, hidden=D, inter=F)
        tensors, ref = _w4a16_moe_expert_tensors(np.random.default_rng(6), E,
                                                 D, F)
        _load_moe(method, layer, tensors)
        with patch.object(wna16, "get_mesh_shape_product", return_value=4):
            assert method.process_weights_after_loading(layer) is True
        w2 = np.asarray(layer.kernel_down_proj_EFD[...], np.float32)
        s2 = np.asarray(layer.kernel_down_proj_EFD_weight_scale[...])
        assert s2.shape == (E, F // 16, 1, D)
        deq2 = w2 * np.repeat(s2[:, :, 0, :], 16, axis=1)
        np.testing.assert_allclose(deq2,
                                   ref["down"].transpose(0, 2, 1),
                                   rtol=0,
                                   atol=0)

    def test_processed_weights_dequantize_to_the_checkpoint(self):
        """Gate, up and down land where GMM expects them, with their scales."""
        E, D, F = 4, 128, 64
        method, layer = _moe_method_and_layer(num_experts=E, hidden=D, inter=F)
        tensors, ref = _w4a16_moe_expert_tensors(np.random.default_rng(2), E,
                                                 D, F)
        _load_moe(method, layer, tensors)
        assert method.process_weights_after_loading(layer) is True
        w13 = np.asarray(layer.kernel_gating_upproj_EDF[...], np.float32)
        s13 = np.asarray(layer.kernel_gating_upproj_EDF_weight_scale[...])
        w2 = np.asarray(layer.kernel_down_proj_EFD[...], np.float32)
        s2 = np.asarray(layer.kernel_down_proj_EFD_weight_scale[...])
        assert layer.kernel_gating_upproj_EDF[...].dtype == jnp.int4
        assert layer.kernel_down_proj_EFD[...].dtype == jnp.int4
        # No requantization: one scale per 32-wide input group survives.
        assert s13.shape[1:3] == (D // GROUP_SIZE, 1)
        assert s2.shape == (E, F // GROUP_SIZE, 1, D)
        # w13 is [E, D, 2 * F_padded], gate then up, each padded to 128.
        f_pad = w13.shape[2] // 2
        deq13 = w13 * np.repeat(s13[:, :, 0, :], GROUP_SIZE, axis=1)
        np.testing.assert_allclose(deq13[:, :, :F],
                                   ref["gate"].transpose(0, 2, 1),
                                   rtol=0,
                                   atol=0)
        np.testing.assert_allclose(deq13[:, :, f_pad:f_pad + F],
                                   ref["up"].transpose(0, 2, 1),
                                   rtol=0,
                                   atol=0)
        assert not deq13[:, :, F:f_pad].any()
        deq2 = w2 * np.repeat(s2[:, :, 0, :], GROUP_SIZE, axis=1)
        np.testing.assert_allclose(deq2,
                                   ref["down"].transpose(0, 2, 1),
                                   rtol=0,
                                   atol=0)
        for name in method._staged_names():
            assert not hasattr(layer, name)


def _moe_reference(x, logits, ref, top_k):
    """softmax -> top-k -> renormalize -> sum_k w_k * down(silu(gate) * up)."""
    x = np.asarray(x, np.float32)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        top = np.argsort(-p[t])[:top_k]
        w = p[t, top] / p[t, top].sum()
        for e, wk in zip(top, w):
            g = ref["gate"][e] @ x[t]
            h = (g / (1 + np.exp(-g))) * (ref["up"][e] @ x[t])
            out[t] += wk * (ref["down"][e] @ h)
    return out


def _check_moe_forward(mesh, num_experts, hidden, inter, top_k):
    from tpu_inference.layers.jax.moe.moe import JaxRoutedExperts
    prefix = "model.language_model.layers.0.experts"
    rng = np.random.default_rng(3)
    tensors, ref = _w4a16_moe_expert_tensors(rng, num_experts, hidden, inter)
    # Serving builds, loads and runs the layer under the TPU mesh; so does
    # this test.
    with jax.set_mesh(mesh), patch.object(JaxRoutedExperts,
                                          "_compute_use_ep",
                                          return_value=False):
        layer = JaxRoutedExperts(dtype=jnp.bfloat16,
                                 num_local_experts=num_experts,
                                 hidden_size=hidden,
                                 intermediate_size_moe=inter,
                                 hidden_act="silu",
                                 rngs=nnx.Rngs(0),
                                 mesh=mesh,
                                 top_k=top_k,
                                 quant_config=CompressedTensorsConfig(
                                     _w4a16_config()),
                                 prefix=prefix)
        assert isinstance(layer.quant_method, wna16.WNA16FusedMoEMethod)
        layer.load_weights([(f"{prefix}.{n}", t) for n, t in tensors])
        assert layer.quant_method.process_weights_after_loading(layer)
        assert layer.kernel_gating_upproj_EDF[...].dtype == jnp.int4

        tokens = 16
        x = (rng.standard_normal((tokens, hidden)) * 0.5).astype(np.float32)
        logits = rng.standard_normal((tokens, num_experts)).astype(np.float32)
        x_bf16 = jnp.asarray(x, jnp.bfloat16)
        y, _ = layer(x_bf16, jnp.asarray(logits))
    expected = _moe_reference(np.asarray(x_bf16, np.float32), logits, ref,
                              top_k)
    assert _rel_err(y, expected) < 1e-2


@pytest.mark.skipif(jax.default_backend() != "tpu",
                    reason="gmm_v2 is a TPU kernel")
@pytest.mark.parametrize(
    "num_experts,hidden,inter,top_k",
    [
        # Intermediate not a multiple of 128, like the 26B's 704.
        (8, 256, 96, 2),
        # gemma-4-26B-A4B's expert shape, fewer experts.
        (8, 2816, 704, 4),
    ])
def test_moe_forward_matches_dequant_on_tpu(mesh, num_experts, hidden, inter,
                                            top_k):
    _check_moe_forward(mesh, num_experts, hidden, inter, top_k)


@pytest.mark.skipif(jax.default_backend() != "tpu"
                    or len(jax.local_devices()) < 4,
                    reason="needs 4 TPU chips")
def test_moe_forward_at_tensor_parallel_4_on_tpu():
    """GMM_TP over 4 chips: the 26B's 704 intermediate is 5.5 groups of 32 per
    shard, which only loads with the per-shard scale split."""
    shape = [1] * len(MESH_AXIS_NAMES)
    shape[MESH_AXIS_NAMES.index("model")] = 4
    devices = np.array(jax.local_devices()[:4]).reshape(shape)
    with Mesh(devices, axis_names=MESH_AXIS_NAMES) as mesh4:
        _check_moe_forward(mesh4, 8, 2816, 704, 4)
