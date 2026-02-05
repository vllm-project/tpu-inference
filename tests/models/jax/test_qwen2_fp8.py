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

import json
import os
import tempfile
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from safetensors.torch import save_file
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.fp8 import Fp8LinearMethod
from tpu_inference.models.jax.qwen2 import Qwen2ForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches


class MockVllmConfig:
    """A mock VllmConfig sufficient for testing the Qwen2 model with FP8."""

    def __init__(self,
                 model_path: str,
                 quantization: str | None = None,
                 dtype="float32"):
        self.model_config = ModelConfig(model_path)
        self.model_config.dtype = jnp.float32 if dtype == "float32" else jnp.bfloat16
        self.model_config.quantization = quantization

        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.load_config.load_format = "safetensors"

        self.cache_config = MagicMock(cache_dtype="auto")
        self.cache_config.block_size = 16

        self.speculative_config = None
        self.parallel_config = MagicMock()
        self.parallel_config.tensor_parallel_size = 1
        self.parallel_config.enable_expert_parallel = False

        self.scheduler_config = MagicMock()
        self.scheduler_config.max_num_batched_tokens = 2048

        self.compilation_config = MagicMock()
        self.compilation_config.pass_config.enable_sp = False

        self.quant_config = get_tpu_quantization_config(self)


@pytest.fixture(scope="module", autouse=True)
def init_pp():
    """Initialize pipeline parallel distributed environment for tests."""
    init_pp_distributed_environment(ip="",
                                    rank=0,
                                    world_size=1,
                                    device=None,
                                    need_pp=False)


@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    m = Mesh(device_mesh, axis_names=('data', 'attn_dp', 'expert', 'model'))
    with jax.set_mesh(m):
        yield m


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_model_inputs():
    num_tokens = 16
    num_reqs = 1
    input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
    positions = jnp.ones((num_tokens, ), dtype=jnp.int32)

    # Needs valid block tables for attention kernel
    max_num_blocks = 4
    block_tables = jnp.zeros((num_reqs * max_num_blocks), dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
        query_start_loc=jnp.array([0, num_tokens], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 0], dtype=jnp.int32),
    )

    return (input_ids, attention_metadata)


class TestQwen2Fp8E2E:

    def test_qwen2_f32_to_fp8_requantization_and_inference(
            self, mesh, rng, mock_model_inputs, monkeypatch):
        """Tests FP8 requantization from Float32 weights and inference."""

        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "256")

        # Create dummy model weights and config
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config.json
            hidden_size = 5120
            head_dim = 128
            num_heads = 40
            num_kv_heads = 40
            vocab_size = 5120
            intermediate_size = 5120

            config_data = {
                "model_type": "qwen2",
                "hidden_size": hidden_size,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "num_hidden_layers": 2,
                "vocab_size": vocab_size,
                "intermediate_size": intermediate_size,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "head_dim": head_dim,
                "tie_word_embeddings": False,
                "architectures": ["Qwen2ForCausalLM"],
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_data, f)

            # Creating random Float32 weights (to test requantization to FP8)
            state_dict = {}
            state_dict["model.embed_tokens.weight"] = torch.randn(
                vocab_size, hidden_size)

            # Layers
            for i in range(2):
                # Attn
                # q_proj
                state_dict[
                    f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(
                        num_heads * head_dim, hidden_size)
                state_dict[
                    f"model.layers.{i}.self_attn.q_proj.bias"] = torch.randn(
                        num_heads * head_dim)

                # k_proj
                state_dict[
                    f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(
                        num_kv_heads * head_dim, hidden_size)
                state_dict[
                    f"model.layers.{i}.self_attn.k_proj.bias"] = torch.randn(
                        num_kv_heads * head_dim)

                # v_proj
                state_dict[
                    f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(
                        num_kv_heads * head_dim, hidden_size)
                state_dict[
                    f"model.layers.{i}.self_attn.v_proj.bias"] = torch.randn(
                        num_kv_heads * head_dim)

                # o_proj
                state_dict[
                    f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(
                        hidden_size, num_heads * head_dim)

                # MLP
                # gate_proj: (Intermediate, Hidden)
                state_dict[
                    f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(
                        intermediate_size, hidden_size)
                state_dict[
                    f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(
                        intermediate_size, hidden_size)
                state_dict[
                    f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(
                        hidden_size, intermediate_size)

                # Norms
                state_dict[
                    f"model.layers.{i}.input_layernorm.weight"] = torch.ones(
                        hidden_size)
                state_dict[
                    f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(
                        hidden_size)

            state_dict["model.norm.weight"] = torch.ones(hidden_size)
            state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

            save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))

            # Initialize FP8 Model
            mock_config = MockVllmConfig(tmpdir,
                                         quantization="fp8",
                                         dtype="bfloat16")

            model = Qwen2ForCausalLM(mock_config, rng, mesh)
            # Manually set mesh on FP8 layers (Qwen2ForCausalLM doesn't propagate it yet)
            layers_to_fix = []
            for layer in model.model.layers:
                layers_to_fix.extend([
                    layer.self_attn.q_proj, layer.self_attn.k_proj,
                    layer.self_attn.v_proj, layer.self_attn.o_proj,
                    layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj
                ])

            if hasattr(model, 'lm_head'):
                layers_to_fix.append(model.lm_head)

            for proj in layers_to_fix:
                if isinstance(proj.quant_method, Fp8LinearMethod):
                    proj.quant_method.linear_config.mesh = mesh
                    proj.weight.mesh = mesh
                    proj.weight_scale_inv.mesh = mesh

            q_proj = model.model.layers[0].self_attn.q_proj
            assert isinstance(q_proj.quant_method, Fp8LinearMethod)

            # Load weights and trigger requantization
            loader = get_model_loader(LoadConfig(load_format="safetensors"))
            loader.load_weights(model, mock_config.model_config)

            # Verify weights are quantized
            assert q_proj.weight.value.dtype == jnp.float8_e4m3fn

            input_ids, attention_metadata = mock_model_inputs
            kv_caches = create_kv_caches(
                num_blocks=2,
                block_size=mock_config.cache_config.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                mesh=mesh,
                layer_names=["model.layers.0", "model.layers.1"],
                cache_dtype=jnp.bfloat16)

            kv_caches, hidden_states, _ = model(kv_caches, input_ids,
                                                attention_metadata)

            # Check output shape
            assert hidden_states.shape == (16, hidden_size)
            assert not jnp.isnan(hidden_states).any()

            # Compute logits
            logits = model.compute_logits(hidden_states)
            assert logits.shape == (16, vocab_size)

    def test_qwen2_fp8_checkpoint_loading(self, mesh, rng, mock_model_inputs,
                                          monkeypatch):
        """Tests loading a pre-quantized FP8 checkpoint.

        Simulates a real FP8 checkpoint (e.g. Qwen3-4B-FP8) where weights
        are stored as float8_e4m3fn (viewed as uint8 in safetensors) and
        scales are stored as weight_scale_inv.
        """
        block_size = 256
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        with tempfile.TemporaryDirectory() as tmpdir:
            hidden_size = 5120
            head_dim = 128
            num_heads = 40
            num_kv_heads = 40
            vocab_size = 5120
            intermediate_size = 5120
            n_blocks = hidden_size // block_size  # 20

            config_data = {
                "model_type": "qwen2",
                "hidden_size": hidden_size,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "num_hidden_layers": 2,
                "vocab_size": vocab_size,
                "intermediate_size": intermediate_size,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "head_dim": head_dim,
                "tie_word_embeddings": False,
                "architectures": ["Qwen2ForCausalLM"],
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [block_size, block_size],
                },
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_data, f)

            # Create FP8 weights (stored as uint8 view in safetensors)
            state_dict = {}
            state_dict["model.embed_tokens.weight"] = torch.randn(
                vocab_size, hidden_size)

            for i in range(2):
                for proj, out_size in [
                    ("q_proj", num_heads * head_dim),
                    ("k_proj", num_kv_heads * head_dim),
                    ("v_proj", num_kv_heads * head_dim),
                ]:
                    # FP8 weight as uint8 view
                    w = torch.randn(out_size,
                                    hidden_size).to(torch.float8_e4m3fn)
                    state_dict[
                        f"model.layers.{i}.self_attn.{proj}.weight"] = w.view(
                            torch.uint8)
                    # Blockwise scale_inv: (n_blocks, out_size)
                    state_dict[
                        f"model.layers.{i}.self_attn.{proj}.weight_scale_inv"] = torch.ones(
                            n_blocks, out_size, dtype=torch.float32)

                    if proj in ("q_proj", "k_proj", "v_proj"):
                        state_dict[
                            f"model.layers.{i}.self_attn.{proj}.bias"] = torch.randn(
                                out_size)

                # o_proj: (hidden, heads*head_dim)
                n_blocks_o = (num_heads * head_dim) // block_size
                w_o = torch.randn(hidden_size,
                                  num_heads * head_dim).to(torch.float8_e4m3fn)
                state_dict[
                    f"model.layers.{i}.self_attn.o_proj.weight"] = w_o.view(
                        torch.uint8)
                state_dict[
                    f"model.layers.{i}.self_attn.o_proj.weight_scale_inv"] = torch.ones(
                        n_blocks_o, hidden_size, dtype=torch.float32)

                # MLP
                for proj, in_size, out_size in [
                    ("gate_proj", hidden_size, intermediate_size),
                    ("up_proj", hidden_size, intermediate_size),
                    ("down_proj", intermediate_size, hidden_size),
                ]:
                    n_blk = in_size // block_size
                    w = torch.randn(out_size, in_size).to(torch.float8_e4m3fn)
                    state_dict[f"model.layers.{i}.mlp.{proj}.weight"] = w.view(
                        torch.uint8)
                    state_dict[
                        f"model.layers.{i}.mlp.{proj}.weight_scale_inv"] = torch.ones(
                            n_blk, out_size, dtype=torch.float32)

                # Norms
                state_dict[
                    f"model.layers.{i}.input_layernorm.weight"] = torch.ones(
                        hidden_size)
                state_dict[
                    f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(
                        hidden_size)

            state_dict["model.norm.weight"] = torch.ones(hidden_size)
            state_dict["lm_head.weight"] = torch.randn(vocab_size, hidden_size)

            save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))

            mock_config = MockVllmConfig(tmpdir,
                                         quantization="fp8",
                                         dtype="bfloat16")

            model = Qwen2ForCausalLM(mock_config, rng, mesh)
            for layer in model.model.layers:
                for proj in [
                        layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj,
                        layer.mlp.gate_proj, layer.mlp.up_proj,
                        layer.mlp.down_proj
                ]:
                    if isinstance(proj.quant_method, Fp8LinearMethod):
                        proj.quant_method.linear_config.mesh = mesh
                        proj.weight.mesh = mesh
                        proj.weight_scale_inv.mesh = mesh

            loader = get_model_loader(LoadConfig(load_format="safetensors"))
            loader.load_weights(model, mock_config.model_config)

            # Verify weights loaded as FP8
            q_proj = model.model.layers[0].self_attn.q_proj
            assert q_proj.weight.value.dtype == jnp.float8_e4m3fn

            # Run inference
            input_ids, attention_metadata = mock_model_inputs
            kv_caches = create_kv_caches(
                num_blocks=2,
                block_size=mock_config.cache_config.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                mesh=mesh,
                layer_names=["model.layers.0", "model.layers.1"],
                cache_dtype=jnp.bfloat16)

            kv_caches, hidden_states, _ = model(kv_caches, input_ids,
                                                attention_metadata)

            assert hidden_states.shape == (16, hidden_size)
            assert not jnp.isnan(hidden_states).any()
