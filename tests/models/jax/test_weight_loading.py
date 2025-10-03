# SPDX-License-Identifier: Apache-2.0
# Test for LoRA weight loading API

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax._src import test_util as jtu
from jax.sharding import Mesh
from safetensors.numpy import save_file

from tpu_inference.models.jax.utils.weight_utils import (
    MetadataMap, load_hf_weights, transfer_state_with_mappings)

# ----- nnx.Module Wrappers -----


class SourceLayer(nnx.Module):

    def __init__(self, rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs(), (4, 4)))
        self.bias = nnx.Param(jax.random.normal(rngs(), (4, )))


class SourceModel(nnx.Module):

    def __init__(self, rngs):
        self.src_lm_head = nnx.Param(jax.random.normal(rngs(), (2, 4)))
        self.layers = {0: SourceLayer(rngs)}


class TargetLinear(nnx.Module):

    def __init__(self, rngs):
        self.kernel = nnx.Param(jnp.zeros((4, 4)))
        self.bias = nnx.Param(jnp.zeros((4, )))


class TargetBlock(nnx.Module):

    def __init__(self, rngs):
        self.mlp = {"up_proj": TargetLinear(rngs)}


class TargetModel(nnx.Module):

    def __init__(self, rngs):
        self.tgt_lm_head = nnx.Param(jnp.zeros((2, 4)))
        self.model = {"layers": {0: TargetBlock(rngs)}}


# ----- Test -----
class WeightTransfer(jtu.JaxTestCase):

    def test_transfer_state(self):
        rng = nnx.Rngs(0)
        src_model = SourceModel(rng)
        tgt_model = TargetModel(rng)

        # Get split states
        _, src_state = nnx.split(src_model)
        _, tgt_state = nnx.split(tgt_model)

        # Overwrite known values
        src_state["layers"][0]["kernel"].value = jnp.ones((4, 4)) * 42.0
        src_state["layers"][0]["bias"].value = jnp.ones((4, )) * 7.0
        src_state["src_lm_head"].value = jnp.ones((2, 4)) * 6.0
        # Mapping for both kernel and bias
        mappings = {
            "layers.*.kernel": ("model.layers.*.mlp.up_proj.kernel", (None, )),
            "layers.*.bias": ("model.layers.*.mlp.up_proj.bias", (None, )),
            "src_lm_head": ("tgt_lm_head", (None, None)),
        }

        # Transfer
        new_tgt_state = transfer_state_with_mappings(src_state, tgt_state,
                                                     mappings)

        # Assert correctness
        assert jnp.allclose(
            new_tgt_state["model"]["layers"][0]["mlp"]["up_proj"]
            ["kernel"].value, 42.0)
        assert jnp.allclose(
            new_tgt_state["model"]["layers"][0]["mlp"]["up_proj"]
            ["bias"].value, 7.0)
        assert jnp.allclose(new_tgt_state["tgt_lm_head"].value, 6.0)


# ----- Mocks for dtype test -----


class DtypeTestModel(nnx.Module):

    def __init__(self, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.weight_to_cast = nnx.Param(jnp.zeros((2, 2), dtype=dtype))
        self.weight_to_keep = nnx.Param(jnp.zeros((2, 2), dtype=dtype))


@dataclass
class MockModelConfig:
    model: str
    dtype: jnp.dtype
    hf_config: Any = None

    def get_vocab_size(self):
        return 1

    def get_hidden_size(self):
        return 1

    def get_head_size(self):
        return 1

    is_multimodal_model: bool = False


@dataclass
class MockLoadConfig:
    download_dir: str


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    load_config: MockLoadConfig
    speculative_config: Any = None


class WeightLoadingDtypeTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        # Create dummy safetensors file
        tensors = {
            "weight_to_cast.weight": np.ones((2, 2), dtype=np.float32),
            "weight_to_keep.weight": np.ones((2, 2), dtype=np.float32),
        }
        self.safetensors_path = os.path.join(self.tempdir.name,
                                             "model.safetensors")
        save_file(tensors, self.safetensors_path)

    def test_keep_original_dtype(self):
        rng = nnx.Rngs(0)
        model_dtype = jnp.bfloat16
        model = DtypeTestModel(dtype=model_dtype, rngs=rng)

        mock_model_config = MockModelConfig(model=self.tempdir.name,
                                            dtype=model_dtype)
        mock_load_config = MockLoadConfig(download_dir=self.tempdir.name)
        vllm_config = MockVllmConfig(model_config=mock_model_config,
                                     load_config=mock_load_config)

        mesh = Mesh(jax.devices(), ("model", ))

        name_map = {
            "weight_to_cast": "weight_to_cast",
            "weight_to_keep": "weight_to_keep",
        }
        metadata_map = MetadataMap(name_map=name_map)

        keep_original_dtype_keys_regex = [r"weight_to_keep.*"]

        load_hf_weights(
            vllm_config=vllm_config,
            model=model,
            metadata_map=metadata_map,
            mesh=mesh,
            keep_original_dtype_keys_regex=keep_original_dtype_keys_regex,
        )

        self.assertEqual(model.weight_to_cast.value.dtype, model_dtype)
        self.assertEqual(model.weight_to_keep.value.dtype, jnp.float32)
