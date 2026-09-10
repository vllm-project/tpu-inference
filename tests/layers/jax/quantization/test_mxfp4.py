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
"""Unit tests for the JAX-native GPT-OSS MXFP4 quantization method (#2952).

`TestMxfp4Config` exercises the dispatch logic (`get_quant_method`), like
`test_compressed_tensors.py` does: does each layer type get routed to the
right quant method? `TestMxfp4FusedMoEMethod` drives the method's public
lifecycle (`create_weights_jax` -> `load_weights` ->
`process_weights_after_loading` -> `apply_jax`) on synthetic checkpoint
tensors laid out like gpt-oss-20b's expert tensors, scaled down.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh

import tpu_inference.layers.jax.quantization.mxfp4 as mxfp4
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import \
    FusedMoEWeights
from tpu_inference.layers.common.quant_methods import MXFP4
from tpu_inference.layers.common.quantization import \
    MXFP4_REQUANTIZED_BLOCK_SIZE
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.mxfp4 import (Mxfp4Config,
                                                         Mxfp4FusedMoEMethod)


def _single_device_mesh():
    return Mesh(
        np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("data", "model"))


class _FakeJaxMoE(JaxMoE):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _make_layer(moe_backend=MoEBackend.GMM_TP, activation="swigluoai"):
    return SimpleNamespace(
        dtype=jnp.float32,
        num_local_experts=2,
        hidden_size=64,
        intermediate_size_moe=32,
        activation=activation,
        moe_backend=moe_backend,
        mesh=_single_device_mesh(),
        prefix="model.layers.0.mlp.experts",
        kernel_gating_EDF=nnx.Param(jnp.zeros((2, 64, 32), dtype=jnp.float32)),
        kernel_up_proj_EDF=nnx.Param(jnp.zeros((2, 64, 32),
                                               dtype=jnp.float32)),
        kernel_down_proj_EFD=nnx.Param(
            jnp.zeros((2, 32, 64), dtype=jnp.float32)),
    )


def _checkpoint_weights():
    return [
        ("model.layers.0.mlp.experts.gate_up_proj_blocks",
         torch.zeros((2, 64, 32), dtype=torch.uint8)),
        ("model.layers.0.mlp.experts.gate_up_proj_scales",
         torch.ones((2, 64, 2), dtype=torch.uint8)),
        ("model.layers.0.mlp.experts.gate_up_proj_bias",
         torch.full((2, 64), 2.0, dtype=torch.float32)),
        ("model.layers.0.mlp.experts.down_proj_blocks",
         torch.full((2, 64, 16), 3, dtype=torch.uint8)),
        ("model.layers.0.mlp.experts.down_proj_scales",
         torch.full((2, 64, 1), 4, dtype=torch.uint8)),
        ("model.layers.0.mlp.experts.down_proj_bias",
         torch.full((2, 64), 5.0, dtype=torch.float32)),
    ]


def _checkpoint_weights_with_4d_blocks():
    w13_blocks = torch.arange(2 * 64 * 2 * 16).to(torch.uint8).reshape(
        2, 64, 2, 16)
    w2_blocks = torch.arange(2 * 64 * 1 * 16).to(torch.uint8).reshape(
        2, 64, 1, 16)
    return [
        ("model.layers.0.mlp.experts.gate_up_proj_blocks", w13_blocks),
        ("model.layers.0.mlp.experts.gate_up_proj_scales",
         torch.ones((2, 64, 2), dtype=torch.uint8)),
        ("model.layers.0.mlp.experts.gate_up_proj_bias",
         torch.full((2, 64), 2.0, dtype=torch.float32)),
        ("model.layers.0.mlp.experts.down_proj_blocks", w2_blocks),
        ("model.layers.0.mlp.experts.down_proj_scales",
         torch.full((2, 64, 1), 4, dtype=torch.uint8)),
        ("model.layers.0.mlp.experts.down_proj_bias",
         torch.full((2, 64), 5.0, dtype=torch.float32)),
    ]


def _flatten_blocks_weights(weights):
    flattened = []
    for name, weight in weights:
        if name.endswith("_blocks"):
            weight = weight.reshape(weight.shape[0], weight.shape[1], -1)
        flattened.append((name, weight))
    return flattened


def _create_and_load(layer, weights=None):
    method = Mxfp4FusedMoEMethod()
    method.create_weights_jax(layer, rngs=nnx.Rngs(0))
    loaded = method.load_weights(
        layer=layer,
        original_load_weights_fn=lambda: NotImplementedError(
            "JaxRoutedExperts does not implement _load_weights"),
        weights=_checkpoint_weights() if weights is None else weights)
    return method, loaded


class TestMxfp4Config:

    def test_registry_returns_mxfp4_config(self):
        """`gpt_oss_mxfp4` selects Mxfp4Config in the quant config registry."""
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(
            quantization=MXFP4,
            hf_config=SimpleNamespace(
                quantization_config={"quant_method": MXFP4}),
        ))

        quant_config = get_tpu_quantization_config(vllm_config)

        assert isinstance(quant_config, Mxfp4Config)

    def test_selects_routed_experts_method_only(self):
        """Only JaxRoutedExperts gets the method; other layers stay bf16."""
        quant_config = Mxfp4Config({"quant_method": MXFP4})
        # __new__ skips __init__, which needs a vLLM config and TPU devices
        # unavailable on CPU-only hosts; dispatch only checks isinstance.
        routed_experts = JaxRoutedExperts.__new__(JaxRoutedExperts)
        legacy_moe = JaxMoE.__new__(JaxMoE)
        linear_layer = JaxEinsum.__new__(JaxEinsum)

        routed_method = quant_config.get_quant_method(
            routed_experts, prefix="model.layers.0.mlp.experts")
        legacy_method = quant_config.get_quant_method(
            legacy_moe, prefix="model.layers.0.mlp")
        linear_method = quant_config.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.q_proj")

        assert isinstance(routed_method, Mxfp4FusedMoEMethod)
        assert legacy_method is None
        assert linear_method is None

    def test_ignored_layers_do_not_skip_experts(self):
        """GPT-OSS MXFP4 has no ignored-layers support (vLLM parity)."""
        quant_config = Mxfp4Config({
            "quant_method":
            MXFP4,
            "ignored_layers": ["model.layers.0.mlp.experts"],
        })
        routed_experts = JaxRoutedExperts.__new__(JaxRoutedExperts)

        method = quant_config.get_quant_method(
            routed_experts, prefix="model.layers.0.mlp.experts")

        assert isinstance(method, Mxfp4FusedMoEMethod)


class TestMxfp4FusedMoEMethod:

    def test_create_weights_replaces_placeholders(self):
        """Placeholder kernels are dropped for six CPU-staged params."""
        layer = _make_layer()
        method = Mxfp4FusedMoEMethod()

        method.create_weights_jax(layer, rngs=nnx.Rngs(0))

        assert not hasattr(layer, "kernel_gating_EDF")
        assert not hasattr(layer, "kernel_up_proj_EDF")
        assert not hasattr(layer, "kernel_down_proj_EFD")
        assert tuple(layer.w13_blocks.shape) == (2, 64, 32)
        assert tuple(layer.w13_scales.shape) == (2, 64, 2)
        assert tuple(layer.w13_bias.shape) == (2, 64)
        assert tuple(layer.w2_blocks.shape) == (2, 64, 16)
        assert tuple(layer.w2_scales.shape) == (2, 64, 1)
        assert tuple(layer.w2_bias.shape) == (2, 64)
        for name in ("w13_blocks", "w13_scales", "w13_bias", "w2_blocks",
                     "w2_scales", "w2_bias"):
            param = getattr(layer, name)
            assert isinstance(param, nnx.Param)
            assert param.get_metadata("mesh") is not None
            assert not param.get_metadata("_is_loaded", False)

    def test_load_weights_stages_expert_tensors(self):
        """All six GPT-OSS expert tensors stage with checkpoint dtypes."""
        layer = _make_layer()
        _, loaded = _create_and_load(layer)

        assert loaded == {
            "w13_blocks", "w13_scales", "w13_bias", "w2_blocks", "w2_scales",
            "w2_bias"
        }
        assert tuple(layer.w13_blocks.shape) == (2, 64, 32)
        assert layer.w13_blocks.dtype == jnp.uint8
        assert tuple(layer.w13_scales.shape) == (2, 64, 2)
        assert layer.w13_scales.dtype == jnp.uint8
        assert tuple(layer.w13_bias.shape) == (2, 64)
        assert layer.w13_bias.dtype == jnp.float32
        assert tuple(layer.w2_blocks.shape) == (2, 64, 16)
        assert layer.w2_blocks.dtype == jnp.uint8
        assert tuple(layer.w2_scales.shape) == (2, 64, 1)
        assert layer.w2_scales.dtype == jnp.uint8
        assert tuple(layer.w2_bias.shape) == (2, 64)
        assert layer.w2_bias.dtype == jnp.float32
        for name in loaded:
            assert getattr(layer, name).get_metadata("_is_loaded", False)

    def test_load_weights_flattens_4d_blocks(self):
        """Checkpoint-shaped 4-D blocks stage identically to 3-D ones."""
        weights_4d = _checkpoint_weights_with_4d_blocks()
        layer_4d = _make_layer()
        method_4d, loaded_4d = _create_and_load(layer_4d, weights_4d)

        layer_3d = _make_layer()
        method_3d, loaded_3d = _create_and_load(
            layer_3d, _flatten_blocks_weights(weights_4d))

        assert loaded_4d == loaded_3d
        assert tuple(layer_4d.w13_blocks.shape) == (2, 64, 32)
        assert tuple(layer_4d.w2_blocks.shape) == (2, 64, 16)
        for name in loaded_4d:
            assert jnp.array_equal(
                getattr(layer_4d, name)[...],
                getattr(layer_3d, name)[...])

        assert method_4d.process_weights_after_loading(layer_4d) is True
        assert method_3d.process_weights_after_loading(layer_3d) is True
        for name in ("kernel_gating_upproj_EDF",
                     "kernel_gating_upproj_EDF_weight_scale",
                     "kernel_gating_upproj_EDF_bias", "kernel_down_proj_EFD",
                     "kernel_down_proj_EFD_weight_scale",
                     "kernel_down_proj_EFD_bias"):
            assert jnp.array_equal(
                getattr(layer_4d, name)[...].astype(jnp.float32),
                getattr(layer_3d, name)[...].astype(jnp.float32))

    def test_process_weights_end_to_end(self):
        """Unmocked lifecycle: staged tensors become fp4 runtime attrs."""
        layer = _make_layer()
        method, _ = _create_and_load(layer)

        assert method.process_weights_after_loading(layer) is True

        for name in ("kernel_gating_upproj_EDF", "kernel_down_proj_EFD"):
            kernel = getattr(layer, name)[...]
            assert kernel.dtype == jnp.float4_e2m1fn
            assert kernel.shape[0] == 2
        for name in ("kernel_gating_upproj_EDF_weight_scale",
                     "kernel_down_proj_EFD_weight_scale"):
            assert getattr(layer, name)[...].dtype == jnp.float32
        # The checkpoint biases (2.0 and 5.0 per element) survive the
        # interleave reorder and zero-padding, so their sums are preserved.
        assert jnp.asarray(layer.kernel_gating_upproj_EDF_bias[...],
                           jnp.float32).sum() == 2.0 * 2 * 64
        assert jnp.asarray(layer.kernel_down_proj_EFD_bias[...],
                           jnp.float32).sum() == 5.0 * 2 * 64
        for name in ("w13_blocks", "w13_scales", "w13_bias", "w2_blocks",
                     "w2_scales", "w2_bias"):
            assert not hasattr(layer, name)

    def test_process_passes_reorder_size_for_gmm_ep(self, monkeypatch):
        """GMM_EP computes w13_reorder_size and disables interleave for silu."""
        layer = _make_layer(moe_backend=MoEBackend.GMM_EP, activation="silu")
        method, _ = _create_and_load(layer)
        quantized_weights = object()
        processed_weights = FusedMoEWeights(
            w13_weight=jnp.ones((2, 64, 64), dtype=jnp.float32),
            w13_weight_scale=jnp.ones((2, 1, 1, 64), dtype=jnp.float32),
            w13_bias=jnp.ones((2, 1, 64), dtype=jnp.float32),
            w2_weight=jnp.ones((2, 64, 32), dtype=jnp.float32),
            w2_weight_scale=jnp.ones((2, 1, 1, 64), dtype=jnp.float32),
            w2_bias=jnp.ones((2, 1, 64), dtype=jnp.float32),
        )

        def fake_quantize(weights, dtype, block_size, w13_interleave):
            assert dtype == jnp.float4_e2m1fn
            assert block_size == MXFP4_REQUANTIZED_BLOCK_SIZE
            assert w13_interleave is False
            return quantized_weights

        def fake_process(weights, moe_backend, w13_reorder_size,
                         w13_interleave):
            assert weights is quantized_weights
            assert moe_backend == MoEBackend.GMM_EP
            assert w13_reorder_size == mxfp4.get_mesh_shape_product(
                layer.mesh, mxfp4.ShardingAxisName.MLP_TENSOR)
            assert w13_interleave is False
            return processed_weights

        monkeypatch.setattr(
            mxfp4, "dequantize_tensor_from_mxfp4_packed",
            lambda blocks, scales, axis, dtype: jnp.ones(
                (2, 64, 64) if blocks.shape == (2, 64, 32) else (2, 64, 32),
                dtype=jnp.float32))
        monkeypatch.setattr(mxfp4, "quantize_moe_weights", fake_quantize)
        monkeypatch.setattr(mxfp4, "process_moe_weights", fake_process)
        monkeypatch.setattr(mxfp4, "shard_moe_weights",
                            lambda weights, *args, **kwargs: weights)

        assert method.process_weights_after_loading(layer) is True

    def test_process_returns_false_until_all_tensors_loaded(self, monkeypatch):
        """Partial staging returns False and installs no runtime attrs."""
        layer = _make_layer()
        partial_weights = _checkpoint_weights()[:2]
        method, loaded = _create_and_load(layer, partial_weights)

        def fail_if_called(*args, **kwargs):
            raise AssertionError(
                "processing should wait for all required tensors")

        monkeypatch.setattr(mxfp4, "dequantize_tensor_from_mxfp4_packed",
                            fail_if_called)
        monkeypatch.setattr(mxfp4, "quantize_moe_weights", fail_if_called)
        monkeypatch.setattr(mxfp4, "process_moe_weights", fail_if_called)

        assert loaded == {"w13_blocks", "w13_scales"}
        assert method.process_weights_after_loading(layer) is False
        assert layer.w13_blocks.get_metadata("_is_loaded", False)
        assert layer.w13_scales.get_metadata("_is_loaded", False)
        for name in ("w13_bias", "w2_blocks", "w2_scales", "w2_bias"):
            assert not getattr(layer, name).get_metadata("_is_loaded", False)
        assert not hasattr(layer, "kernel_gating_upproj_EDF")
        assert not hasattr(layer, "kernel_gating_upproj_EDF_weight_scale")
        assert not hasattr(layer, "kernel_down_proj_EFD")
        assert not hasattr(layer, "kernel_down_proj_EFD_weight_scale")

    def test_apply_jax_passes_processed_weights_and_biases(self, monkeypatch):
        """apply_jax forwards the runtime attrs to moe_apply unchanged."""
        with _single_device_mesh() as mesh:
            w13_weight = jnp.ones((2, 8, 4), dtype=jnp.float4_e2m1fn)
            w13_scale = jnp.ones((2, 1, 1, 8), dtype=jnp.float32)
            w13_bias = jnp.ones((2, 1, 8), dtype=jnp.float32)
            w2_weight = jnp.ones((2, 4, 8), dtype=jnp.float4_e2m1fn)
            w2_scale = jnp.ones((2, 1, 1, 4), dtype=jnp.float32)
            w2_bias = jnp.ones((2, 1, 4), dtype=jnp.float32)
            x = jnp.ones((3, 4), dtype=jnp.float32)
            router_logits = jnp.ones((3, 2), dtype=jnp.float32)
            expected_output = jnp.full((3, 4), 7, dtype=jnp.bfloat16)
            calls = []

            layer = _FakeJaxMoE(
                dtype=jnp.bfloat16,
                mesh=mesh,
                activation_ffw_td=("data", None),
                moe_backend=MoEBackend.GMM_TP,
                kernel_gating_upproj_EDF=nnx.Param(w13_weight),
                kernel_gating_upproj_EDF_weight_scale=nnx.Param(w13_scale),
                kernel_gating_upproj_EDF_bias=nnx.Param(w13_bias),
                kernel_down_proj_EFD=nnx.Param(w2_weight),
                kernel_down_proj_EFD_weight_scale=nnx.Param(w2_scale),
                kernel_down_proj_EFD_bias=nnx.Param(w2_bias),
            )
            method = Mxfp4FusedMoEMethod()

            def fake_moe_apply(layer_arg, x_arg, router_logits_arg, weights,
                               moe_backend, mesh_arg, extra_backend_kwargs):
                calls.append((layer_arg, x_arg, router_logits_arg, weights,
                              moe_backend, mesh_arg, extra_backend_kwargs))
                assert layer_arg is layer
                assert x_arg.shape == x.shape
                assert x_arg.dtype == jnp.bfloat16
                assert router_logits_arg is router_logits
                assert moe_backend == MoEBackend.GMM_TP
                assert mesh_arg is mesh
                assert extra_backend_kwargs is method.extra_backend_kwargs
                assert isinstance(weights, FusedMoEWeights)
                assert jnp.array_equal(weights.w13_weight, w13_weight)
                assert jnp.array_equal(weights.w13_weight_scale, w13_scale)
                assert jnp.array_equal(weights.w13_bias, w13_bias)
                assert jnp.array_equal(weights.w2_weight, w2_weight)
                assert jnp.array_equal(weights.w2_weight_scale, w2_scale)
                assert jnp.array_equal(weights.w2_bias, w2_bias)
                return expected_output

            monkeypatch.setattr(mxfp4, "moe_apply", fake_moe_apply)

            output = method.apply_jax(layer, x, router_logits=router_logits)

            assert output is expected_output
            assert len(calls) == 1

    def test_apply_jax_rejects_unsupported_backend(self):
        """Backends outside GMM_EP/GMM_TP raise instead of computing."""
        with _single_device_mesh() as mesh:
            layer = _FakeJaxMoE(
                dtype=jnp.bfloat16,
                mesh=mesh,
                activation_ffw_td=("data", None),
                moe_backend=MoEBackend.FUSED_MOE,
            )
            method = Mxfp4FusedMoEMethod()

            with pytest.raises(NotImplementedError,
                               match="Unsupported moe backend"):
                method.apply_jax(layer,
                                 jnp.ones((3, 4), dtype=jnp.float32),
                                 router_logits=jnp.ones((3, 2),
                                                        dtype=jnp.float32))
