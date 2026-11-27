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
"""Unit tests for the opt-in native JAX W4A16 compressed-tensors path.

Covers packed-weight unpacking, per-layer dispatch in
``W4A16CompressedTensorsConfig``, checkpoint loading through the registered
weight loaders (plain, merged column-parallel and multi-axis einsum layers),
numerical agreement with a dense BF16 reference on single- and multi-device
meshes, and validation of the ``jax_w4a16`` option.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32, unpack_from_int32)
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.linear import (JaxEinsum, JaxLinear,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.quantization.compressed_tensors_w4a16 import (
    W4A16CompressedTensorsConfig, W4A16LinearMethod, _unpack_uint4b8)
from tpu_inference.layers.jax.quantization.fp8 import Fp8BlockwiseLinearMethod
from tpu_inference.layers.jax.quantization.unquantized import \
    UnquantizedLinearMethod
from tpu_inference.models.common.w4a16_config import (W4A16Kernel,
                                                      W4A16Options,
                                                      validate_w4a16_request)

GROUP_SIZE = 32


def _w4a16_config(*,
                  group_size=GROUP_SIZE,
                  symmetric=True,
                  actorder=None,
                  ignore=None):
    """compressed-tensors config shaped like a pack-quantized W4A16 export."""
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": symmetric,
                    "strategy": "group",
                    "group_size": group_size,
                    "dynamic": False,
                    "actorder": actorder,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
        "ignore": ignore or [],
    }


def _fp8_block_config():
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
        "ignore": [],
    }


def _quantized_weight(out_features, in_features, seed):
    """Return signed INT4 values, BF16 group scales and the dense BF16 weight.

    All tensors use the checkpoint (PyTorch) layout ``[out, in]``.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(-8,
                           8, (out_features, in_features),
                           generator=generator,
                           dtype=torch.int8)
    scales = (torch.rand(
        out_features, in_features // GROUP_SIZE, generator=generator) * 0.02 +
              1e-3).to(torch.bfloat16)
    dense = (values.reshape(out_features, -1, GROUP_SIZE).to(torch.bfloat16) *
             scales[..., None]).reshape(out_features, in_features)
    return values, scales, dense


def _checkpoint_tensors(values, scales):
    """Serialize like the compressed-tensors ``pack-quantized`` format."""
    return {
        "weight_packed": pack_to_int32(values, 4),
        "weight_scale": scales,
        "weight_shape": torch.tensor(values.shape),
    }


def _load(layer, tensors, shard_id=None):
    for name in ("weight_packed", "weight_scale", "weight_shape"):
        param = getattr(layer, name)
        if shard_id is None:
            param.weight_loader(param, tensors[name])
        else:
            param.weight_loader(param, tensors[name], shard_id)


def _model_mesh(num_devices):
    """All devices on the ``model`` axis; every other mesh axis has size 1."""
    devices = np.array(jax.devices()[:num_devices])
    shape = tuple(num_devices if name == "model" else 1
                  for name in MESH_AXIS_NAMES)
    return Mesh(devices.reshape(shape), axis_names=MESH_AXIS_NAMES)


def _device_counts():
    return sorted({1, jax.device_count()})


def _dense_reference(x, dense):
    """BF16 inputs and weights, FP32 accumulation, BF16 output."""
    weight = jnp.asarray(dense.float().numpy()).astype(jnp.bfloat16)
    return jnp.dot(x, weight.T,
                   preferred_element_type=jnp.float32).astype(jnp.bfloat16)


def _assert_close_bf16(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert actual.shape == expected.shape
    # Accumulation order may differ; allow a couple of BF16 ulps.
    np.testing.assert_allclose(actual,
                               expected,
                               rtol=2**-6,
                               atol=2**-6 * np.abs(expected).max())


@pytest.fixture
def rngs():
    return nnx.Rngs(42)


class TestUnpack:

    @pytest.mark.parametrize("out_features,in_features", [(8, 32), (48, 256)])
    def test_matches_compressed_tensors(self, out_features, in_features):
        values, _, _ = _quantized_weight(out_features, in_features, seed=0)
        packed = pack_to_int32(values, 4)  # [out, in / 8]
        reference = unpack_from_int32(packed, 4, values.shape)
        np.testing.assert_array_equal(reference.numpy(), values.numpy())

        # The JAX parameter stores the packed weight as [in / 8, out].
        unpacked = _unpack_uint4b8(jnp.asarray(packed.numpy().T))
        assert unpacked.dtype == jnp.int8
        np.testing.assert_array_equal(np.asarray(unpacked), values.numpy().T)

    def test_covers_full_int4_range(self):
        values = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 2)  # [4, 32]
        packed = pack_to_int32(values, 4)
        unpacked = _unpack_uint4b8(jnp.asarray(packed.numpy().T))
        np.testing.assert_array_equal(np.asarray(unpacked), values.numpy().T)


class TestDispatch:

    def test_linear_routes_to_w4a16(self, rngs):
        config = W4A16CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(_model_mesh(1)):
            layer = JaxLinear(256,
                              64,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="model.layers.0.mlp.down_proj")
        assert isinstance(layer.quant_method, W4A16LinearMethod)
        assert not hasattr(layer, "weight")
        assert layer.weight_packed[...].shape == (256 // 8, 64)
        assert layer.weight_packed[...].dtype == jnp.int32
        assert layer.weight_scale[...].shape == (256 // GROUP_SIZE, 64)
        assert layer.weight_shape[...].shape == (2, )

    def test_ignored_layer_is_unquantized(self, rngs):
        config = W4A16CompressedTensorsConfig(
            _w4a16_config(ignore=["re:.*vision_tower.*"]))
        with jax.set_mesh(_model_mesh(1)):
            layer = JaxLinear(256,
                              64,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="model.vision_tower.encoder.proj")
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)

    def test_non_w4a16_schemes_use_base_config(self, rngs):
        config = W4A16CompressedTensorsConfig(_fp8_block_config())
        with jax.set_mesh(_model_mesh(1)):
            layer = JaxLinear(256,
                              256,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              kernel_init=nnx.initializers.uniform(),
                              prefix="model.layers.0.mlp.down_proj")
        assert isinstance(layer.quant_method, Fp8BlockwiseLinearMethod)

    @pytest.mark.parametrize("overrides", [
        dict(group_size=128),
        dict(symmetric=False),
        dict(actorder="group"),
    ])
    def test_unsupported_variants_raise(self, rngs, overrides):
        config = W4A16CompressedTensorsConfig(_w4a16_config(**overrides))
        with jax.set_mesh(_model_mesh(1)), pytest.raises(NotImplementedError):
            JaxLinear(256,
                      64,
                      rngs,
                      use_bias=False,
                      quant_config=config,
                      prefix="model.layers.0.mlp.down_proj")


class TestForward:

    @pytest.mark.parametrize("num_devices", _device_counts())
    @pytest.mark.parametrize("sharding", [(None, "model"), ("model", None)])
    @pytest.mark.parametrize("rows", [1, 5])
    def test_linear_matches_dense(self, rngs, num_devices, sharding, rows):
        in_features, out_features = 512, 128
        values, scales, dense = _quantized_weight(out_features,
                                                  in_features,
                                                  seed=1)
        config = W4A16CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(_model_mesh(num_devices)):
            layer = JaxLinear(in_features,
                              out_features,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              kernel_init=nnx.with_partitioning(
                                  nnx.initializers.uniform(), sharding),
                              prefix="model.layers.0.mlp.proj")
            _load(layer, _checkpoint_tensors(values, scales))
            x = jax.random.normal(jax.random.key(0),
                                  (rows, in_features)).astype(jnp.bfloat16)
            output = layer(x)
            expected = _dense_reference(x, dense)
        assert output.dtype == jnp.bfloat16
        _assert_close_bf16(output, expected)

    def test_loaded_weights_dequantize_exactly(self, rngs):
        in_features, out_features = 256, 64
        values, scales, dense = _quantized_weight(out_features,
                                                  in_features,
                                                  seed=2)
        config = W4A16CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(_model_mesh(1)):
            layer = JaxLinear(in_features,
                              out_features,
                              rngs,
                              use_bias=False,
                              quant_config=config,
                              prefix="model.layers.0.mlp.proj")
            _load(layer, _checkpoint_tensors(values, scales))
            q = _unpack_uint4b8(layer.weight_packed[...])
            scale = layer.weight_scale[...]
            weight = (q.reshape(scale.shape[0], GROUP_SIZE, -1).astype(
                jnp.bfloat16) * scale[:, None, :]).reshape(q.shape)
        np.testing.assert_array_equal(np.asarray(weight, dtype=np.float32),
                                      dense.float().numpy().T)
        np.testing.assert_array_equal(np.asarray(layer.weight_shape[...]),
                                      [out_features, in_features])

    @pytest.mark.parametrize("num_devices", _device_counts())
    def test_merged_column_parallel_matches_dense(self, rngs, num_devices):
        in_features, output_sizes = 256, [64, 128]
        shards = [
            _quantized_weight(size, in_features, seed=10 + index)
            for index, size in enumerate(output_sizes)
        ]
        config = W4A16CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(_model_mesh(num_devices)):
            layer = JaxMergedColumnParallelLinear(
                in_features,
                output_sizes,
                rngs,
                use_bias=False,
                quant_config=config,
                kernel_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                  (None, "model")),
                prefix="model.layers.0.mlp.gate_up_proj")
            assert isinstance(layer.quant_method, W4A16LinearMethod)
            # Checkpoints provide gate_proj and up_proj separately.
            for shard_id, (values, scales, _) in enumerate(shards):
                _load(layer, _checkpoint_tensors(values, scales), shard_id)
            x = jax.random.normal(jax.random.key(1),
                                  (3, in_features)).astype(jnp.bfloat16)
            output = layer(x)
            expected = jnp.concatenate(
                [_dense_reference(x, dense) for _, _, dense in shards],
                axis=-1)
        _assert_close_bf16(output, expected)

    @pytest.mark.parametrize("num_devices", _device_counts())
    def test_multi_axis_contraction_matches_dense(self, rngs, num_devices):
        # Attention output projection layout: TNH,NHD->TD with the checkpoint
        # weight stored as [D, N * H].
        heads, head_dim, hidden = 8, 64, 96
        values, scales, dense = _quantized_weight(hidden,
                                                  heads * head_dim,
                                                  seed=3)
        config = W4A16CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(_model_mesh(num_devices)):
            layer = JaxEinsum("TNH,NHD->TD", (heads, head_dim, hidden),
                              rngs,
                              quant_config=config,
                              kernel_init=nnx.with_partitioning(
                                  nnx.initializers.uniform(),
                                  ("model", None, None)),
                              prefix="model.layers.0.self_attn.o_proj")
            _load(layer, _checkpoint_tensors(values, scales))
            x = jax.random.normal(jax.random.key(2),
                                  (4, heads, head_dim)).astype(jnp.bfloat16)
            output = layer(x)
            expected = _dense_reference(x.reshape(4, heads * head_dim), dense)
        assert output.shape == (4, hidden)
        _assert_close_bf16(output, expected)

    @pytest.mark.parametrize("num_devices", _device_counts())
    def test_multi_axis_output_matches_dense(self, rngs, num_devices):
        # Attention query projection layout: TD,DNH->TNH with the checkpoint
        # weight stored as [N * H, D].
        hidden, heads, head_dim = 256, 8, 32
        values, scales, dense = _quantized_weight(heads * head_dim,
                                                  hidden,
                                                  seed=4)
        config = W4A16CompressedTensorsConfig(_w4a16_config())
        with jax.set_mesh(_model_mesh(num_devices)):
            layer = JaxEinsum("TD,DNH->TNH", (hidden, heads, head_dim),
                              rngs,
                              quant_config=config,
                              kernel_init=nnx.with_partitioning(
                                  nnx.initializers.uniform(),
                                  (None, "model", None)),
                              prefix="model.layers.0.self_attn.q_proj")
            _load(layer, _checkpoint_tensors(values, scales))
            x = jax.random.normal(jax.random.key(3),
                                  (2, hidden)).astype(jnp.bfloat16)
            output = layer(x)
            expected = _dense_reference(x, dense).reshape(2, heads, head_dim)
        assert output.shape == (2, heads, head_dim)
        _assert_close_bf16(output, expected)


def _vllm_config(**overrides):
    hf_config = SimpleNamespace(
        architectures=["Gemma4ForCausalLM"],
        text_config=SimpleNamespace(enable_moe_block=False),
        quantization_config=_w4a16_config(),
    )
    values = dict(
        additional_config={"jax_w4a16": {
            "kernel": "jax"
        }},
        speculative_config=None,
        lora_config=None,
        model_config=SimpleNamespace(hf_config=hf_config,
                                     quantization="compressed-tensors",
                                     dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(tensor_parallel_size=8,
                                        data_parallel_size=1,
                                        pipeline_parallel_size=1),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestOptions:

    @pytest.mark.parametrize("value", [{}, {"kernel": "jax"}])
    def test_accepts_jax_kernel(self, value):
        options = W4A16Options.from_config({"jax_w4a16": value})
        assert options.kernel is W4A16Kernel.JAX

    @pytest.mark.parametrize("value", [
        {
            "kernel": "pallas"
        },
        {
            "kernel": "jax",
            "extra": 1
        },
        "jax",
    ])
    def test_rejects_invalid_options(self, value):
        with pytest.raises(ValueError):
            W4A16Options.from_config({"jax_w4a16": value})

    def test_validates_supported_request(self):
        options = validate_w4a16_request(_vllm_config(),
                                         impl="flax_nnx",
                                         is_draft_model=False)
        assert options.kernel is W4A16Kernel.JAX

    @pytest.mark.parametrize("impl,overrides", [
        ("vllm", {}),
        ("flax_nnx", dict(speculative_config=object())),
        ("flax_nnx", dict(lora_config=object())),
        ("flax_nnx",
         dict(parallel_config=SimpleNamespace(tensor_parallel_size=8,
                                              data_parallel_size=2,
                                              pipeline_parallel_size=1))),
    ])
    def test_rejects_unsupported_request(self, impl, overrides):
        with pytest.raises(ValueError):
            validate_w4a16_request(_vllm_config(**overrides),
                                   impl=impl,
                                   is_draft_model=False)

    def test_rejects_non_group32_checkpoint(self):
        config = _vllm_config()
        config.model_config.hf_config.quantization_config = _w4a16_config(
            group_size=128)
        with pytest.raises(ValueError):
            validate_w4a16_request(config,
                                   impl="flax_nnx",
                                   is_draft_model=False)
