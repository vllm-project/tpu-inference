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

import os

if "--xla_force_host_platform_device_count" not in os.environ.get(
        "XLA_FLAGS", ""):
    # Multi-device coverage for the sharded-decode tests on CPU-only hosts.
    # A no-op on TPU (the flag only sizes the CPU backend) and when jax was
    # already initialized by an earlier test module.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               " --xla_force_host_platform_device_count=8")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

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
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                quantization=MXFP4,
                model="openai/gpt-oss-120b",
                hf_config=SimpleNamespace(
                    quantization_config={"quant_method": MXFP4}),
            ),
            # Configs are handed where the weights live so they can consult
            # the checkpoint itself; `Mxfp4Config` ignores it, but the
            # registry reads both off the vLLM config.
            load_config=SimpleNamespace(download_dir=None),
        )

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


class TestCompressedTensorsMxfp4MultiHostPlacement:
    """Placing the decoded MXFP4 experts must work when the mesh spans hosts.

    Under the Ray multi-host backend each process addresses only its own
    devices, so `jax.device_put(x, NamedSharding(full_mesh, ...))` is rejected
    ("must be a Device or a Sharding which represents addressable devices").

    `process_weights_after_loading` has two decode paths and each reaches that
    constraint its own way, so both are covered here:

    - the host decode (`MXFP4_SHARD_THEN_DECODE=0`) produces process-local
      arrays on CPU and must hand them to `general_device_put` (via
      `shard_put`) rather than to a plain `device_put`;
    - the shard decode (the default) needs no multi-host branch at all,
      because it only ever builds one array per *addressable* device and
      assembles the global array from those.

    Everything below runs in one process, so the path and the multi-host
    branch are both selected by patching env vars -- the same way
    `tests/layers/common/test_utils.py` covers `general_device_put`.
    """

    E, D, F, GS = 2, 64, 32, 32

    def _mesh(self, num_devices):
        devices = jax.devices()[:num_devices]
        if len(devices) < num_devices:
            pytest.skip(f"needs {num_devices} devices, have {len(devices)}")
        return Mesh(
            np.array(devices).reshape(1, num_devices), ("data", "model"))

    def _layer(self, mesh, sharding=(None, None, "model")):
        return SimpleNamespace(
            dtype=jnp.float32,
            num_local_experts=self.E,
            hidden_size=self.D,
            intermediate_size_moe=self.F,
            moe_backend=MoEBackend.MEGABLX_GMM,
            mesh=mesh,
            prefix="model.layers.1.block_sparse_moe.experts",
            edf_sharding=sharding,
            efd_sharding=sharding,
            kernel_gating_EDF=nnx.Param(jnp.zeros((self.E, self.D, self.F))),
            kernel_up_proj_EDF=nnx.Param(jnp.zeros((self.E, self.D, self.F))),
            kernel_down_proj_EFD=nnx.Param(jnp.zeros(
                (self.E, self.F, self.D))),
        )

    def _staged_checkpoint(self):
        """Per-expert `weight_packed`/`weight_scale`, checkpoint-oriented."""
        rng = np.random.RandomState(0)
        out = []
        for e in range(self.E):
            for proj, (o, i) in (("w1", (self.F, self.D)),
                                 ("w3", (self.F, self.D)), ("w2", (self.D,
                                                                   self.F))):
                base = f"model.layers.1.block_sparse_moe.experts.{e}.{proj}"
                out.append((f"{base}.weight_packed",
                            torch.from_numpy(
                                rng.randint(0,
                                            256, (o, i // 2),
                                            dtype=np.uint8))))
                # E8M0 exponents around 127 so the decoded scales are ~1.
                out.append((f"{base}.weight_scale",
                            torch.from_numpy(
                                rng.randint(120,
                                            132, (o, i // self.GS),
                                            dtype=np.uint8))))
        return out

    def _loaded_method(self, mesh):
        layer = self._layer(mesh)
        method = mxfp4.CompressedTensorsMxfp4MoEMethod(layer)
        method.create_weights_jax(layer, rngs=nnx.Rngs(0))
        method.load_weights(layer=layer,
                            original_load_weights_fn=None,
                            weights=self._staged_checkpoint())
        return layer, method

    def _decode(self, mesh, backend):
        """Run the real load + host decode with the given multi-host backend.

        Pins `MXFP4_SHARD_THEN_DECODE=0`: the `shard_put` contract these tests
        assert belongs to the host decode. The default path is covered by
        `test_shard_decode_places_only_process_local_shards` below.
        """
        from unittest import mock

        from tpu_inference import envs
        layer, method = self._loaded_method(mesh)
        real = jax.make_array_from_callback
        calls = []

        def spy(shape, sharding, cb):
            calls.append((tuple(shape), sharding))
            return real(shape, sharding, cb)

        with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", backend), \
                mock.patch.object(envs, "MXFP4_SHARD_THEN_DECODE", False), \
                mock.patch("jax.make_array_from_callback", side_effect=spy):
            assert method.process_weights_after_loading(layer)
        return layer, calls

    def _shard_decode(self, mesh, backend):
        """Run the real load + shard decode, recording how it assembles each
        global array and out of which devices' shards."""
        from unittest import mock

        from tpu_inference import envs
        layer, method = self._loaded_method(mesh)
        real = jax.make_array_from_single_device_arrays
        real_device_put = jax.device_put
        calls, put_targets = [], []

        def spy(shape, sharding, arrays):
            calls.append((tuple(shape), sharding,
                          {d
                           for a in arrays
                           for d in a.devices()}))
            return real(shape, sharding, arrays)

        def device_put_spy(x, device=None, **kwargs):
            put_targets.append(device)
            return real_device_put(x, device, **kwargs)

        with mock.patch.object(envs, "TPU_MULTIHOST_BACKEND", backend), \
                mock.patch.object(envs, "MXFP4_SHARD_THEN_DECODE", True), \
                mock.patch("jax.device_put", side_effect=device_put_spy), \
                mock.patch("jax.make_array_from_single_device_arrays",
                           side_effect=spy):
            assert method.process_weights_after_loading(layer)
        return layer, calls, put_targets

    def test_shard_decode_places_only_process_local_shards(self):
        """The shard decode is multi-host-safe by construction rather than by
        branching: every array it builds is assembled from exactly the shards
        of the devices this process addresses, so no placement is ever asked
        for a device the process cannot reach. Asserted under both backends,
        because unlike the host decode this path does not read the env var.

        One process addresses the whole mesh here, so the set of devices the
        shards land on cannot by itself separate the addressable enumeration
        from the global one. What it can check is the thing that actually
        breaks across hosts: every placement names a single `Device`, never a
        `Sharding` spanning the mesh.

        That the shards themselves decode to the same bytes as the host decode
        is `tests/models/jax/test_kimi_k3_mxfp4_shard_decode.py`.
        """
        mesh = self._mesh(2)
        for backend in ("ray", ""):
            layer, calls, put_targets = self._shard_decode(mesh, backend)
            # 3 projections x (values, scale).
            assert len(calls) == 6, (
                f"backend={backend!r}: expected 6 assembled arrays, saw "
                f"{len(calls)}: {calls}")
            for shape, sharding, devices in calls:
                assert sharding.mesh is mesh
                addressable = set(
                    sharding.addressable_devices_indices_map(shape))
                assert devices == addressable, (
                    f"backend={backend!r}: {shape} assembled from {devices}, "
                    f"expected the addressable devices {addressable}")
            assert put_targets, f"backend={backend!r}: nothing was placed"
            for target in put_targets:
                assert isinstance(target, jax.Device), (
                    f"backend={backend!r}: a shard was placed with target "
                    f"{target!r}; a process that addresses only part of the "
                    "mesh can name a device but not a mesh-wide Sharding.")
            assert layer.kernel_gating_EDF.value.sharding.spec == P(
                None, None, "model")

    def test_multihost_placement_uses_the_process_local_api(self):
        """The fix, stated as the assertion that fails without it: the
        multi-host branch must build each parameter with
        `make_array_from_callback`. A raw `device_put` never calls it."""
        mesh = self._mesh(2)
        _layer, calls = self._decode(mesh, "ray")
        # 3 projections x (values, scale).
        assert len(calls) == 6, (
            f"expected 6 process-local placements, saw {len(calls)}: {calls}")
        for _shape, sharding in calls:
            assert sharding.mesh is mesh

    def test_single_host_placement_does_not_take_that_branch(self):
        """Anti-vacuity for the test above: with the backend unset the count
        is 0, so a passing multi-host assertion is really about the branch."""
        _layer, calls = self._decode(self._mesh(2), "")
        assert calls == []

    def test_multihost_decode_equals_single_host_decode(self):
        """The shard math, not just the API: assembling the global array from
        process-local shards must produce the same weights, and the same
        sharding, as the single-process path."""
        mesh = self._mesh(2)
        single, _ = self._decode(mesh, "")
        multi, _ = self._decode(mesh, "ray")
        checked = 0
        for attr in ("kernel_gating_EDF", "kernel_up_proj_EDF",
                     "kernel_down_proj_EFD"):
            for name in (attr, f"{attr}_weight_scale"):
                a, b = getattr(single, name).value, getattr(multi, name).value
                assert a.shape == b.shape, name
                assert a.sharding == b.sharding, (
                    f"{name}: {a.sharding} vs {b.sharding}")
                np.testing.assert_array_equal(
                    np.asarray(a.astype(jnp.float32)),
                    np.asarray(b.astype(jnp.float32)),
                    err_msg=f"{name} differs between the two placements")
                checked += 1
        assert checked == 6

    def test_placement_shards_the_expert_kernels(self):
        """The sharding is actually applied -- otherwise the comparison above
        would hold trivially for two replicated arrays."""
        mesh = self._mesh(2)
        layer, _ = self._decode(mesh, "ray")
        values = layer.kernel_gating_EDF.value
        assert values.sharding.spec == P(None, None, "model")
        # Sharded 2 ways on the last axis.
        assert values.addressable_shards[0].data.shape == (self.E, self.D,
                                                           self.F // 2)


# Every fp4 (E2M1) code point by nibble value: sign bit 3, two exponent bits
# (bias 1), one mantissa bit. Written out so the reference below shares
# nothing with the implementation under test.
_E2M1_VALUES = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
    -4.0, -6.0
],
                        dtype=np.float32)


def _ref_unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    """Numpy reference unpack: the LOW nibble of each byte comes first."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    pairs = np.stack([_E2M1_VALUES[low], _E2M1_VALUES[high]], axis=-1)
    return pairs.reshape(packed.shape[:-1] + (-1, ))


def _ref_e8m0_to_fp32(u8: np.ndarray) -> np.ndarray:
    """Numpy reference E8M0 expansion: exact powers of two, bias 127."""
    return np.float32(2.0)**(u8.astype(np.int32) - 127)


def _assert_bit_equal(actual: jax.Array, expected: np.ndarray, name: str):
    """fp4/fp32 equality that distinguishes -0.0 from +0.0."""
    actual_f32 = np.asarray(actual.astype(jnp.float32))
    np.testing.assert_array_equal(actual_f32, expected, err_msg=name)
    np.testing.assert_array_equal(np.signbit(actual_f32),
                                  np.signbit(expected),
                                  err_msg=f"{name}: -0.0 vs +0.0 mismatch")


class TestCompressedTensorsMxfp4GoldenDecode:
    """Hand-computed reference for the lossless decode.

    Pins the three properties the checkpoint bytes must survive: the nibble
    order of the fp4 unpack (low nibble of each byte first), the E8M0 scale
    expansion (exact powers of two, exponent bias 127), and the final
    checkpoint-orientation `[E, out, in]` -> kernel `[E, in, out]` transpose.
    """

    def _mesh(self):
        return Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model"))

    def test_golden_bytes_decode_to_hand_computed_values(self):
        # One expert, out=2 rows of in=4 values (2 packed bytes per row), one
        # E8M0 scale byte per row.
        packed = np.array([[[0x2E, 0x71], [0x10, 0x9F]]], dtype=np.uint8)
        scales = np.array([[[126], [128]]], dtype=np.uint8)

        values, scale = \
            mxfp4.CompressedTensorsMxfp4MoEMethod._decode_on_host(
                [jnp.asarray(packed)], [jnp.asarray(scales)],
                (None, None, None), self._mesh())

        assert values.dtype == jnp.float4_e2m1fn
        assert scale.dtype == jnp.float32
        # [E=1, out=2, in=4] checkpoint orientation -> [E, in, out].
        assert values.shape == (1, 4, 2)
        assert scale.shape == (1, 1, 2)
        # Byte 0x2E holds nibbles (low 0xE, high 0x2) -> (-4.0, 1.0): the low
        # nibble is the earlier element. Hand decode, pre-transpose:
        #   row 0: 0x2E, 0x71 -> [-4.0, 1.0, 0.5, 6.0]
        #   row 1: 0x10, 0x9F -> [ 0.0, 0.5, -6.0, -0.5]
        expected = np.array(
            [[[-4.0, 0.0], [1.0, 0.5], [0.5, -6.0], [6.0, -0.5]]],
            dtype=np.float32)
        _assert_bit_equal(values, expected, "values")
        # E8M0 bytes are exponents biased by 127: 126 -> 2^-1, 128 -> 2^1.
        np.testing.assert_array_equal(
            np.asarray(scale), np.array([[[0.5, 2.0]]], dtype=np.float32))

    def test_host_decode_matches_numpy_reference_on_random_data(self):
        rng = np.random.RandomState(7)
        num_experts, out, in_, gs = 3, 4, 16, 8
        packed = rng.randint(0, 256, (num_experts, out, in_ // 2), np.uint8)
        scales = rng.randint(100, 150, (num_experts, out, in_ // gs), np.uint8)

        values, scale = \
            mxfp4.CompressedTensorsMxfp4MoEMethod._decode_on_host(
                [jnp.asarray(packed[e:e + 1]) for e in range(num_experts)],
                [jnp.asarray(scales[e:e + 1]) for e in range(num_experts)],
                (None, None, None), self._mesh())

        _assert_bit_equal(values, np.swapaxes(_ref_unpack_e2m1(packed), 1, 2),
                          "values")
        np.testing.assert_array_equal(
            np.asarray(scale), np.swapaxes(_ref_e8m0_to_fp32(scales), 1, 2))


class TestCompressedTensorsMxfp4ShardVsHostParity:
    """`_decode_sharded` must be bit-equal to the host decode.

    The sharded path exists purely as a memory/traffic optimization, so for
    the same synthetic MXFP4 bytes it must assemble exactly the array
    `_decode_on_host` produces -- same values (down to -0.0), same scales,
    same sharding -- for shards cut along each of the three axes.
    """

    E, OUT, IN, GS = 4, 6, 64, 32

    def _mesh(self):
        devices = jax.devices()[:2]
        if len(devices) < 2:
            pytest.skip(f"needs 2 devices, have {len(devices)}")
        return Mesh(np.array(devices).reshape(1, 2), ("data", "model"))

    @pytest.mark.parametrize(
        "sharding",
        [
            (None, None, "model"),  # output axis
            (None, "model", None),  # decoded (packed) input axis
            ("model", None, None),  # expert axis
        ])
    def test_shard_decode_bit_equals_host_decode(self, sharding):
        mesh = self._mesh()
        rng = np.random.RandomState(11)
        staged_packed = [
            jnp.asarray(
                rng.randint(0, 256, (1, self.OUT, self.IN // 2), np.uint8))
            for _ in range(self.E)
        ]
        staged_scale = [
            jnp.asarray(
                rng.randint(96, 160, (1, self.OUT, self.IN // self.GS),
                            np.uint8)) for _ in range(self.E)
        ]

        host_values, host_scale = \
            mxfp4.CompressedTensorsMxfp4MoEMethod._decode_on_host(
                staged_packed, staged_scale, sharding, mesh)
        shard_values = mxfp4._decode_sharded(staged_packed,
                                             mxfp4.u8_unpack_e2m1, 2, sharding,
                                             mesh)
        shard_scale = mxfp4._decode_sharded(staged_scale, mxfp4.e8m0_to_fp32,
                                            1, sharding, mesh)

        for name, host, shard in (("values", host_values, shard_values),
                                  ("scale", host_scale, shard_scale)):
            assert shard.shape == host.shape, name
            assert shard.dtype == host.dtype, name
            assert shard.sharding == host.sharding, (
                f"{name}: {shard.sharding} vs {host.sharding}")
            _assert_bit_equal(shard, np.asarray(host.astype(jnp.float32)),
                              name)
