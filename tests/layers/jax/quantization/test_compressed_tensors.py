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

import json

import jax
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig
from tpu_inference.layers.jax.quantization.fp8 import (
    Fp8BlockwiseLinearMethod, Fp8TensorwiseLinearMethod)
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


class TestPackedModuleInspection:
    """Which modules the checkpoint itself stores compressed.

    The config's target/ignore lists are not a reliable statement of that (a
    checkpoint may leave a targeted module uncompressed), so the config reads
    the checkpoint's own tensor names. These tests cover the two ways that
    reading has to work: from the safetensors index, and from the shards.
    """

    def _index(self, tmp_path, tensor_names):
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({
                "metadata": {},
                "weight_map": {
                    n: "model-00001-of-00001.safetensors"
                    for n in tensor_names
                },
            }))
        return str(tmp_path)

    def test_reads_the_index_when_the_shards_are_not_local(self, tmp_path):
        """Streamed weights leave only the config files on disk.

        Loading from object storage downloads `*.json` -- the index among them
        -- and streams the shards, so a scan that opens `*.safetensors` finds
        nothing and silently falls back to trusting the config.
        """
        path = self._index(tmp_path, [
            "model.layers.0.mlp.experts.0.w1.weight_packed",
            "model.layers.0.mlp.experts.0.w1.weight_scale",
            "model.layers.0.self_attn.q_proj.weight"
        ])
        assert not list(tmp_path.glob("*.safetensors"))

        config = CompressedTensorsConfig(_fp8_block_config(),
                                         model_name_or_path=path)
        assert config._packed_modules is not None, (
            "the checkpoint was not inspected at all")
        assert config._is_packed_in_checkpoint(
            "model.layers.0.mlp.experts.0.w1")
        # ...including for the parent that owns the per-expert subtree.
        assert config._is_packed_in_checkpoint("model.layers.0.mlp.experts")
        # The attention projection is stored plain, and must stay that way.
        assert not config._is_packed_in_checkpoint(
            "model.layers.0.self_attn.q_proj")

    def test_the_wrapper_prefix_is_stripped_from_index_paths(self, tmp_path):
        """`*ForConditionalGeneration` checkpoints nest the text stack.

        Their tensor names carry a `language_model.` the model's own module
        paths do not, so an unnormalized comparison finds no compressed module
        anywhere and every quantized layer falls back to unquantized.
        """
        path = self._index(
            tmp_path,
            ["language_model.model.layers.0.mlp.experts.0.w1.weight_packed"])
        config = CompressedTensorsConfig(_fp8_block_config(),
                                         model_name_or_path=path)
        # Anti-vacuity: an uninspected checkpoint answers True to everything,
        # so pin that this one was read before believing the True below.
        assert config._packed_modules is not None
        assert not config._is_packed_in_checkpoint("model.layers.0.self_attn")

        assert config._is_packed_in_checkpoint(
            "model.layers.0.mlp.experts.0.w1")
        assert config._is_packed_in_checkpoint("model.layers.0.mlp.experts")

    def test_an_uninspectable_checkpoint_falls_back_to_the_config(
            self, tmp_path):
        """No index and no shards -> no information, trust the config."""
        config = CompressedTensorsConfig(_fp8_block_config(),
                                         model_name_or_path=str(tmp_path))
        assert config._packed_modules is None
        assert config._is_packed_in_checkpoint("anything.at.all")
