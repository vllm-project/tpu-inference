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
"""Unit tests for the JAX-native compressed-tensors config (issue #2261).

These tests exercise the *dispatch* logic (`get_quant_method`): given a
compressed-tensors `quantization_config`, does each layer get routed to the
right JAX quant method (or skipped)? They mirror `test_fp8.py::TestFp8Config`,
which asserts `layer.quant_method` types rather than running a forward pass.
"""

import jax
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.linear import (JaxLinear,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig
from tpu_inference.layers.jax.quantization.fp8 import (
    Fp8BlockwiseLinearMethod, Fp8BlockwiseMergedLinearMethod,
    Fp8TensorwiseLinearMethod)
from tpu_inference.layers.jax.quantization.unquantized import \
    UnquantizedLinearMethod


# A compressed-tensors `quantization_config` modeled on
# RedHatAI/gemma-4-31B-it-FP8-block: fp8 block-quantized weights (128x128) +
# dynamic fp8 activations, applied to Linear layers, with an ignore regex.
# NOTE(verify): exact field values (esp. input-activation strategy) should be
# reconciled with the real checkpoint's config.json when running on a TPU VM.
def _fp8_block_config(ignore=None):
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "block",
                    "block_structure": [128, 128],
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
        "ignore": ignore or [],
    }


# Same as above but per-tensor weights (no block_structure) -> should route to
# the tensorwise method instead of the blockwise one.
def _fp8_tensor_config():
    cfg = _fp8_block_config()
    cfg["config_groups"]["group_0"]["weights"] = {
        "num_bits": 8,
        "type": "float",
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


class _MLP(nnx.Module):
    """Two linear layers, so we can test per-layer routing / skipping."""

    def __init__(self,
                 in_features,
                 out_features,
                 rngs,
                 quant_config,
                 prefix=''):
        # NOTE: blockwise fp8 create_weights runs `kernel_init` on an fp8 dtype
        # at construction. The default variance-scaling init uses
        # truncated_normal -> chlo.erf, which TPU cannot legalize on fp8. Pass
        # `uniform` (erf-free) like test_fp8's blockwise tests do.
        kernel_init = nnx.initializers.uniform()
        self.proj1 = JaxLinear(in_features,
                               out_features,
                               rngs=rngs,
                               quant_config=quant_config,
                               kernel_init=kernel_init,
                               prefix=prefix + ".proj1")
        self.proj2 = JaxLinear(in_features,
                               out_features,
                               rngs=rngs,
                               quant_config=quant_config,
                               kernel_init=kernel_init,
                               prefix=prefix + ".proj2")

    def __call__(self, x):
        return self.proj2(self.proj1(x))


class TestCompressedTensorsConfig:

    def test_parses_config_without_error(self):
        """Smoke test: upstream parsing yields a non-empty target_scheme_map."""
        config = CompressedTensorsConfig(_fp8_block_config())
        assert config._target_scheme_map  # parsed something
        assert "Linear" in config._target_scheme_map

    def test_fp8_block_routes_to_blockwise_method(self, rngs, mesh):
        """A Linear layer under an fp8-block group -> Fp8BlockwiseLinearMethod."""
        config = CompressedTensorsConfig(_fp8_block_config())
        with jax.set_mesh(mesh):
            mlp = _MLP(16, 16, rngs, config, prefix="mlp")
        assert isinstance(mlp.proj1.quant_method, Fp8BlockwiseLinearMethod)
        assert isinstance(mlp.proj2.quant_method, Fp8BlockwiseLinearMethod)

    def test_fp8_block_scale_param_uses_ct_name(self, rngs, mesh):
        """The blockwise scale param must be named `weight_scale`.

        compressed-tensors checkpoints serialize the dequant scale as
        `weight_scale`; the method's default name (`weight_scale_inv`,
        DeepSeek convention) would leave the checkpoint scales with no
        matching param and weight loading would never complete.
        """
        config = CompressedTensorsConfig(_fp8_block_config())
        with jax.set_mesh(mesh):
            mlp = _MLP(16, 16, rngs, config, prefix="mlp")
        assert hasattr(mlp.proj1, "weight_scale")
        assert not hasattr(mlp.proj1, "weight_scale_inv")

    def test_fp8_block_merged_routes_to_merged_method(self, rngs, mesh):
        """A merged (fused gate_up-style) layer under an fp8-block group ->
        Fp8BlockwiseMergedLinearMethod, with per-projection shard slots ready.
        """
        config = CompressedTensorsConfig(_fp8_block_config())
        with jax.set_mesh(mesh):
            layer = JaxMergedColumnParallelLinear(
                32, [16, 16],
                rngs,
                use_bias=False,
                quant_config=config,
                kernel_init=nnx.initializers.uniform(),
                prefix="mlp.gate_up_proj")
        assert isinstance(layer.quant_method, Fp8BlockwiseMergedLinearMethod)
        assert layer.weight.get_metadata("_merged_shards") == [None, None]

    def test_fp8_block_merged_scale_param_uses_ct_name(self, rngs, mesh):
        """The merged blockwise scale param must also be named `weight_scale`.

        Guards against the merged method regressing to the hardcoded DeepSeek
        name (`weight_scale_inv`): the scale loader is attached by looking the
        param up via `weight_scale_name`, so a mismatch would either raise at
        layer construction or leave checkpoint scales with no matching param.
        """
        config = CompressedTensorsConfig(_fp8_block_config())
        with jax.set_mesh(mesh):
            layer = JaxMergedColumnParallelLinear(
                32, [16, 16],
                rngs,
                use_bias=False,
                quant_config=config,
                kernel_init=nnx.initializers.uniform(),
                prefix="mlp.gate_up_proj")
        assert hasattr(layer, "weight_scale")
        assert not hasattr(layer, "weight_scale_inv")
        assert layer.weight_scale.get_metadata("_merged_shards") == [
            None, None
        ]

    def test_fp8_tensor_routes_to_tensorwise_method(self, rngs, mesh):
        """Per-tensor fp8 (no block_structure) -> Fp8TensorwiseLinearMethod."""
        config = CompressedTensorsConfig(_fp8_tensor_config())
        with jax.set_mesh(mesh):
            mlp = _MLP(16, 16, rngs, config, prefix="mlp")
        assert isinstance(mlp.proj1.quant_method, Fp8TensorwiseLinearMethod)

    def test_ignored_layer_is_skipped(self, rngs, mesh):
        """A layer matched by the ignore regex -> UnquantizedLinearMethod."""
        config = CompressedTensorsConfig(
            _fp8_block_config(ignore=["re:.*proj1"]))
        with jax.set_mesh(mesh):
            mlp = _MLP(16, 16, rngs, config, prefix="mlp")
        # proj1 ignored, proj2 still quantized.
        assert isinstance(mlp.proj1.quant_method, UnquantizedLinearMethod)
        assert isinstance(mlp.proj2.quant_method, Fp8BlockwiseLinearMethod)
