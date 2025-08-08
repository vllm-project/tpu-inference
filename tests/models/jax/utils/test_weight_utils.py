# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax._src import test_util as jtu
from safetensors.torch import save_file

from tpu_commons.models.jax.utils.weight_utils import (
    load_hf_weights, load_hf_weights_on_thread, transfer_state_with_mappings)

# Test for LoRA weight loading API

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


# ----- Mocks for HF Weight Loading Test -----

HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_HEADS = 8
NUM_KV_HEADS = 4  # For GQA
HEAD_DIM = 16
VOCAB_SIZE = 1000
TP_SIZE = 2


class MockAttention(nnx.Module):

    def __init__(self, rngs):
        # Shapes are post-transformation
        self.q_proj = nnx.Param(jnp.zeros((HIDDEN_SIZE, NUM_HEADS, HEAD_DIM)))
        # K/V heads are repeated for GQA
        kv_heads_padded = NUM_KV_HEADS * (TP_SIZE // NUM_KV_HEADS)
        self.k_proj = nnx.Param(
            jnp.zeros((HIDDEN_SIZE, kv_heads_padded, HEAD_DIM)))
        self.v_proj = nnx.Param(
            jnp.zeros((HIDDEN_SIZE, kv_heads_padded, HEAD_DIM)))
        self.o_proj = nnx.Param(jnp.zeros((NUM_HEADS, HEAD_DIM, HIDDEN_SIZE)))


class MockMLP(nnx.Module):

    def __init__(self, rngs):
        self.gate_proj = nnx.Param(jnp.zeros((HIDDEN_SIZE, INTERMEDIATE_SIZE)))
        self.down_proj = nnx.Param(jnp.zeros((INTERMEDIATE_SIZE, HIDDEN_SIZE)))


class MockBlock(nnx.Module):

    def __init__(self, rngs):
        self.attention = MockAttention(rngs)
        self.mlp = MockMLP(rngs)


class MockModel(nnx.Module):

    def __init__(self, rngs):
        self.embed_tokens = nnx.Param(jnp.zeros((VOCAB_SIZE, HIDDEN_SIZE)))
        self.layers = [MockBlock(rngs)]
        self.lm_head = nnx.Param(jnp.zeros((HIDDEN_SIZE, VOCAB_SIZE)))


class HFWeightLoadingTest(jtu.JaxTestCase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # Mock devices to control TP size
        self.mock_devices = [MagicMock(platform='cpu', device_kind='cpu')
                             ] * TP_SIZE
        self.original_jax_devices = jax.devices
        jax.devices = lambda: self.mock_devices

        self.mesh = jax.sharding.Mesh(np.array(jax.devices()), ("model", ))

        # Create mock vllm_config
        self.vllm_config = MagicMock()
        hf_config = MagicMock()
        hf_config.num_attention_heads = NUM_HEADS
        hf_config.num_key_value_heads = NUM_KV_HEADS
        model_config = MagicMock()
        model_config.hf_config = hf_config
        model_config.get_hidden_size.return_value = HIDDEN_SIZE
        model_config.get_head_size.return_value = HEAD_DIM
        model_config.is_multimodal_model = False
        self.vllm_config.model_config = model_config

        # Create mock model and get its state
        rng = nnx.Rngs(0)
        self.model = MockModel(rng)
        self.params = nnx.state(self.model)

        # Create dummy safetensors file with HF-style weights
        self.weights_file = os.path.join(self.temp_dir.name,
                                         "model.safetensors")
        dummy_weights = {
            "model.layers.0.self_attn.q_proj.weight":
            torch.ones((NUM_HEADS * HEAD_DIM, HIDDEN_SIZE)),
            "model.layers.0.self_attn.k_proj.weight":
            torch.ones((NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)) * 2,
            "model.layers.0.mlp.gate_proj.weight":
            torch.ones((INTERMEDIATE_SIZE, HIDDEN_SIZE)) * 3,
            "lm_head.weight":
            torch.ones((VOCAB_SIZE, HIDDEN_SIZE)) * 4,
        }
        save_file(dummy_weights, self.weights_file)

        # Define mappings
        self.mappings = {
            "model.layers.*.self_attn.q_proj": "layers.*.attention.q_proj",
            "model.layers.*.self_attn.k_proj": "layers.*.attention.k_proj",
            "model.layers.*.mlp.gate_proj": "layers.*.mlp.gate_proj",
            "lm_head": "lm_head",
        }

    def tearDown(self):
        jax.devices = self.original_jax_devices
        super().tearDown()

    @patch("tpu_commons.utils.get_padded_head_dim", lambda x: HEAD_DIM)
    def test_load_hf_weights_on_thread(self):
        """Tests loading weights and applying transformations in a single thread."""
        # Call the function under test
        load_hf_weights_on_thread(self.vllm_config, self.params, self.mappings,
                                  self.mesh, self.weights_file)

        # --- Assertions ---

        # 1. Test lm_head (transpose only)
        lm_head_val = self.params["lm_head"].value
        hf_lm_head = jnp.ones(
            (VOCAB_SIZE, HIDDEN_SIZE), dtype=lm_head_val.dtype) * 4
        expected_lm_head = jnp.transpose(hf_lm_head, (1, 0))
        self.assertArraysEqual(lm_head_val, expected_lm_head)

        # 2. Test gate_proj (transpose only)
        gate_proj_val = self.params["layers"][0]["mlp"]["gate_proj"].value
        hf_gate_proj = jnp.ones(
            (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=gate_proj_val.dtype) * 3
        expected_gate_proj = jnp.transpose(hf_gate_proj, (1, 0))
        self.assertArraysEqual(gate_proj_val, expected_gate_proj)

        # 3. Test q_proj (reshape and transpose)
        q_proj_val = self.params["layers"][0]["attention"]["q_proj"].value
        hf_q_proj = jnp.ones((NUM_HEADS * HEAD_DIM, HIDDEN_SIZE),
                             dtype=q_proj_val.dtype)
        expected_q_proj = jnp.reshape(hf_q_proj,
                                      (NUM_HEADS, HEAD_DIM, HIDDEN_SIZE))
        expected_q_proj = jnp.transpose(expected_q_proj, (2, 0, 1))
        self.assertArraysEqual(q_proj_val, expected_q_proj)

        # 4. Test k_proj (reshape, transpose, and repeat for GQA)
        k_proj_val = self.params["layers"][0]["attention"]["k_proj"].value
        hf_k_proj = jnp.ones(
            (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE), dtype=k_proj_val.dtype) * 2
        expected_k_proj = jnp.reshape(hf_k_proj,
                                      (NUM_KV_HEADS, HEAD_DIM, HIDDEN_SIZE))
        expected_k_proj = jnp.transpose(expected_k_proj, (2, 0, 1))

        # Repeat KV heads to match TP size
        repeats = TP_SIZE // NUM_KV_HEADS
        if repeats > 1:
            expected_k_proj = jnp.repeat(expected_k_proj, repeats, axis=1)

        self.assertArraysEqual(k_proj_val, expected_k_proj)

    @patch(
        "tpu_commons.models.jax.utils.weight_utils.load_hf_weights_on_thread")
    @patch("tpu_commons.models.jax.utils.weight_utils.get_model_weights_files")
    def test_load_hf_weights_multithreaded(self, mock_get_files,
                                           mock_load_on_thread):
        """Tests the multi-threaded wrapper to ensure it calls the single-thread loader."""
        # Arrange
        model_path = "/fake/model/path"
        self.vllm_config.model_config.model = model_path
        mock_files = [
            "/fake/model.safetensors", "/fake/model-00001.safetensors"
        ]
        mock_get_files.return_value = mock_files

        # Act
        load_hf_weights(self.vllm_config, self.model, self.mappings, self.mesh)

        # Assert
        mock_get_files.assert_called_once_with(model_path)
        self.assertEqual(mock_load_on_thread.call_count, len(mock_files))

        # Check that it was called with the correct arguments for each file
        for i, mock_file in enumerate(mock_files):
            call_args = mock_load_on_thread.call_args_list[i].args
            self.assertIs(call_args[0], self.vllm_config)
            # nnx.State doesn't have a good __eq__, so we check identity
            self.assertIs(call_args[1], self.params)
            self.assertIs(call_args[2], self.mappings)
            self.assertIs(call_args[3], self.mesh)
            self.assertEqual(call_args[4], mock_file)
