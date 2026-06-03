# ruff: noqa: E402
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

from functools import partial
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
# Mock vLLM TP state to bypass assertions
import vllm.distributed.parallel_state as parallel_state
from flax import nnx
from jax.sharding import Mesh

_ORIGINAL_GET_TENSOR_MODEL_PARALLEL_RANK = (
    parallel_state.get_tensor_model_parallel_rank)
_ORIGINAL_GET_TENSOR_MODEL_PARALLEL_WORLD_SIZE = (
    parallel_state.get_tensor_model_parallel_world_size)
_ORIGINAL_GET_TP_GROUP = parallel_state.get_tp_group

parallel_state.get_tensor_model_parallel_rank = lambda: 0
parallel_state.get_tensor_model_parallel_world_size = lambda: 1
parallel_state.get_tp_group = lambda: MagicMock(rank_in_group=0, world_size=1)


def teardown_module(module):
    parallel_state.get_tensor_model_parallel_rank = (
        _ORIGINAL_GET_TENSOR_MODEL_PARALLEL_RANK)
    parallel_state.get_tensor_model_parallel_world_size = (
        _ORIGINAL_GET_TENSOR_MODEL_PARALLEL_WORLD_SIZE)
    parallel_state.get_tp_group = _ORIGINAL_GET_TP_GROUP


from torchax.interop import torch_view
from torchax.ops.mappings import t2j
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from vllm.config.compilation import CompilationMode
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.qwen3_vl import \
    Qwen3_VisionBlock as TorchVisionBlock
from vllm.model_executor.models.qwen3_vl import \
    Qwen3_VisionMLP as TorchVisionMLP
from vllm.model_executor.models.qwen3_vl import \
    Qwen3_VisionPatchMerger as TorchVisionPatchMerger
from vllm.model_executor.models.qwen3_vl import \
    Qwen3_VisionTransformer as TorchVisionTransformer

from tpu_inference.models.jax.qwen3_vl import \
    Qwen3_VisionBlock as JaxVisionBlock
from tpu_inference.models.jax.qwen3_vl import Qwen3_VisionMLP as JaxVisionMLP
from tpu_inference.models.jax.qwen3_vl import \
    Qwen3_VisionPatchMerger as JaxVisionPatchMerger
from tpu_inference.models.jax.qwen3_vl import \
    Qwen3_VisionTransformer as JaxVisionTransformer
from tpu_inference.models.jax.qwen3_vl import copy_weights_to_jax_vision_tower


# --- Mocks & Configs ---
class MockModelConfig:

    def __init__(self, hf_config, dtype):
        self.hf_config = hf_config
        self.dtype = dtype
        self.model = "mock_qwen3_vl"

    def get_head_size(self):
        return 16 // 4

    def is_multimodal_model(self):
        return True


class MockCompilationConfig:

    def __init__(self):
        self.custom_ops = ["all"]
        self.enabled_custom_ops = set()
        self.disabled_custom_ops = set()
        self.mode = CompilationMode.NONE
        self.backend = "eager"


class MockVllmConfig:

    def __init__(self):
        vision_config = {
            "hidden_size": 16,
            "intermediate_size": 64,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "window_size": 16,
            "spatial_merge_size": 2,
            "fullatt_block_indexes": list(range(8)),  # 8 layers
            "out_hidden_size": 16,
            "depth": 8,
            "hidden_act": "gelu",
            "num_heads": 4,
            "deepstack_visual_indexes":
            [2, 4, 6],  # realistic intermediate extraction indexes
            "num_position_embeddings": 1024,
        }
        hf_config = Qwen3VLConfig(
            vision_config=vision_config,
            hidden_size=16,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            rms_norm_eps=1e-6,
            image_token_id=200000,
            video_token_id=200001,
            vocab_size=32000,
            rope_theta=1000000.0,
        )
        self.model_config = MockModelConfig(hf_config, jnp.float32)
        self.additional_config = {"enable_dynamic_image_sizes": False}
        self.compilation_config = MockCompilationConfig()
        self.quant_config = None


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices())
    return Mesh(devices.reshape((len(devices), 1, 1)),
                axis_names=('data', 'attn_dp', 'model'))


@pytest.fixture
def rngs() -> nnx.Rngs:
    return nnx.Rngs(params=jax.random.PRNGKey(42))


@pytest.fixture
def mock_vllm_config() -> MockVllmConfig:
    return MockVllmConfig()


# --- Equivalence Tests ---


class TestQwen3_VisionMLP_Equivalence:

    def test_equivalence(self, mock_vllm_config: MockVllmConfig,
                         rngs: nnx.Rngs, mesh: Mesh):
        vc = mock_vllm_config.model_config.hf_config.vision_config
        dtype = jnp.float32  # Use float32 for parity verification

        # 1. Instantiate JAX MLP
        with jax.set_mesh(mesh):
            jax_mlp = JaxVisionMLP(vc, dtype, rngs)

        # 2. Instantiate PyTorch MLP safely inside current vLLM config context
        with set_current_vllm_config(mock_vllm_config):
            torch_mlp = TorchVisionMLP(
                in_features=vc.hidden_size,
                hidden_features=vc.intermediate_size,
                bias=True,
                act_fn=torch.nn.functional.gelu,
            )
        torch.nn.init.normal_(torch_mlp.linear_fc1.weight, std=0.1)
        torch.nn.init.zeros_(torch_mlp.linear_fc1.bias)
        torch.nn.init.normal_(torch_mlp.linear_fc2.weight, std=0.1)
        torch.nn.init.zeros_(torch_mlp.linear_fc2.bias)

        # 3. Copy weights from PyTorch to JAX (using [...] setter to bypass Flax NNX deprecation warnings)
        jax_mlp.linear_fc1.kernel[...] = jnp.array(
            torch_mlp.linear_fc1.weight.detach().cpu().numpy().T)
        jax_mlp.linear_fc1.bias[...] = jnp.array(
            torch_mlp.linear_fc1.bias.detach().cpu().numpy())
        jax_mlp.linear_fc2.kernel[...] = jnp.array(
            torch_mlp.linear_fc2.weight.detach().cpu().numpy().T)
        jax_mlp.linear_fc2.bias[...] = jnp.array(
            torch_mlp.linear_fc2.bias.detach().cpu().numpy())

        # 4. Generate Input (well-bounded)
        x_np = (np.random.randn(5, vc.hidden_size) * 0.1).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.from_numpy(x_np)

        # 5. Run forward pass
        y_jax = jax_mlp(x_jax)
        y_torch = torch_mlp(x_torch)

        # 6. Compare
        np.testing.assert_allclose(np.array(y_jax),
                                   y_torch.detach().cpu().numpy(),
                                   rtol=1e-3,
                                   atol=1e-3)


class TestQwen3_VisionPatchMerger_Equivalence:

    def test_equivalence(self, mock_vllm_config: MockVllmConfig,
                         rngs: nnx.Rngs, mesh: Mesh):
        vc = mock_vllm_config.model_config.hf_config.vision_config
        dtype = jnp.float32

        # 1. JAX Merger (use_postshuffle_norm=True)
        with jax.set_mesh(mesh):
            jax_merger = JaxVisionPatchMerger(
                d_model=vc.out_hidden_size,
                context_dim=vc.hidden_size,
                norm_layer=partial(nnx.LayerNorm, epsilon=1e-6),
                spatial_merge_size=vc.spatial_merge_size,
                use_postshuffle_norm=True,
                dtype=dtype,
                rngs=rngs)

        # 2. Torch Merger safely inside current vLLM config context
        with set_current_vllm_config(mock_vllm_config):
            torch_merger = TorchVisionPatchMerger(
                d_model=vc.out_hidden_size,
                context_dim=vc.hidden_size,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                spatial_merge_size=vc.spatial_merge_size,
                use_postshuffle_norm=True,
            )
        torch.nn.init.ones_(torch_merger.norm.weight)
        torch.nn.init.zeros_(torch_merger.norm.bias)
        torch.nn.init.normal_(torch_merger.linear_fc1.weight, std=0.1)
        torch.nn.init.zeros_(torch_merger.linear_fc1.bias)
        torch.nn.init.normal_(torch_merger.linear_fc2.weight, std=0.1)
        torch.nn.init.zeros_(torch_merger.linear_fc2.bias)

        # 3. Align weights (using [...] setter to bypass Flax NNX deprecation warnings)
        jax_merger.norm.scale[...] = jnp.array(
            torch_merger.norm.weight.detach().cpu().numpy())
        jax_merger.norm.bias[...] = jnp.array(
            torch_merger.norm.bias.detach().cpu().numpy())

        jax_merger.mlp_fc1.kernel[...] = jnp.array(
            torch_merger.linear_fc1.weight.detach().cpu().numpy().T)
        jax_merger.mlp_fc1.bias[...] = jnp.array(
            torch_merger.linear_fc1.bias.detach().cpu().numpy())

        jax_merger.mlp_fc2.kernel[...] = jnp.array(
            torch_merger.linear_fc2.weight.detach().cpu().numpy().T)
        jax_merger.mlp_fc2.bias[...] = jnp.array(
            torch_merger.linear_fc2.bias.detach().cpu().numpy())

        # 4. Inputs
        x_np = (np.random.randn(5, vc.spatial_merge_size**2, vc.hidden_size) *
                0.1).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.from_numpy(x_np)

        # 5. Forward
        y_jax = jax_merger(x_jax)
        y_torch = torch_merger(x_torch)

        # 6. Compare
        np.testing.assert_allclose(np.array(y_jax),
                                   y_torch.detach().cpu().numpy(),
                                   rtol=1e-3,
                                   atol=1e-3)


class TestQwen3_VisionBlock_Equivalence:

    def test_equivalence(self, mock_vllm_config: MockVllmConfig,
                         rngs: nnx.Rngs, mesh: Mesh):
        config = mock_vllm_config.model_config.hf_config
        vc = config.vision_config
        dtype = jnp.float32

        # 1. JAX VisionBlock
        with jax.set_mesh(mesh):
            jax_block = JaxVisionBlock(config=config,
                                       norm_eps=1e-6,
                                       dtype=dtype,
                                       rngs=rngs,
                                       mesh=mesh)

        # 2. Torch VisionBlock and initialize safely inside config context
        with set_current_vllm_config(mock_vllm_config):
            torch_block = TorchVisionBlock(
                dim=vc.hidden_size,
                num_heads=vc.num_heads,
                mlp_hidden_dim=vc.intermediate_size,
                act_fn=torch.nn.functional.gelu,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            )
        torch.nn.init.ones_(torch_block.norm1.weight)
        torch.nn.init.zeros_(torch_block.norm1.bias)
        torch.nn.init.ones_(torch_block.norm2.weight)
        torch.nn.init.zeros_(torch_block.norm2.bias)

        torch.nn.init.normal_(torch_block.attn.qkv.weight, std=0.1)
        torch.nn.init.zeros_(torch_block.attn.qkv.bias)
        torch.nn.init.normal_(torch_block.attn.proj.weight, std=0.1)
        torch.nn.init.zeros_(torch_block.attn.proj.bias)

        torch.nn.init.normal_(torch_block.mlp.linear_fc1.weight, std=0.1)
        torch.nn.init.zeros_(torch_block.mlp.linear_fc1.bias)
        torch.nn.init.normal_(torch_block.mlp.linear_fc2.weight, std=0.1)
        torch.nn.init.zeros_(torch_block.mlp.linear_fc2.bias)

        # 3. Align Weights (using [...] setter to bypass Flax NNX deprecation warnings)
        jax_block.norm1.scale[...] = jnp.array(
            torch_block.norm1.weight.detach().cpu().numpy())
        jax_block.norm1.bias[...] = jnp.array(
            torch_block.norm1.bias.detach().cpu().numpy())
        jax_block.norm2.scale[...] = jnp.array(
            torch_block.norm2.weight.detach().cpu().numpy())
        jax_block.norm2.bias[...] = jnp.array(
            torch_block.norm2.bias.detach().cpu().numpy())

        jax_block.attn.qkv_proj.kernel[...] = jnp.array(
            torch_block.attn.qkv.weight.detach().cpu().numpy().T)
        jax_block.attn.qkv_proj.bias[...] = jnp.array(
            torch_block.attn.qkv.bias.detach().cpu().numpy())
        jax_block.attn.proj.kernel[...] = jnp.array(
            torch_block.attn.proj.weight.detach().cpu().numpy().T)
        jax_block.attn.proj.bias[...] = jnp.array(
            torch_block.attn.proj.bias.detach().cpu().numpy())

        jax_block.mlp.linear_fc1.kernel[...] = jnp.array(
            torch_block.mlp.linear_fc1.weight.detach().cpu().numpy().T)
        jax_block.mlp.linear_fc1.bias[...] = jnp.array(
            torch_block.mlp.linear_fc1.bias.detach().cpu().numpy())
        jax_block.mlp.linear_fc2.kernel[...] = jnp.array(
            torch_block.mlp.linear_fc2.weight.detach().cpu().numpy().T)
        jax_block.mlp.linear_fc2.bias[...] = jnp.array(
            torch_block.mlp.linear_fc2.bias.detach().cpu().numpy())

        # 4. Inputs (B=1, T=10, D=16)
        x_np = (np.random.randn(10, 1, vc.hidden_size) * 0.1).astype(
            np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.from_numpy(x_np)

        rotary_pos_emb = jnp.ones((10, vc.hidden_size // vc.num_heads // 2))

        # Mock JAX attention
        jax_block.attn.flash_attention = MagicMock(
            side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        # Evaluation mode and direct attention forward method patch for single PyTorch block
        torch_block = torch_block.to(torch.float32).eval()
        orig_forward = torch_block.attn.attn.forward
        torch_block.attn.attn.forward = lambda query, key, value, cu_seqlens, max_seqlen, sequence_lengths: torch.ones_like(
            query)

        # Construct PyTorch block inputs
        cu_seqlens = torch.tensor([0, 10], dtype=torch.int32)
        rotary_pos_emb_cos = torch.ones(10,
                                        vc.hidden_size // vc.num_heads // 2)
        rotary_pos_emb_sin = torch.ones(10,
                                        vc.hidden_size // vc.num_heads // 2)
        max_seqlen = torch.tensor(10, dtype=torch.int32)

        try:
            with torch.no_grad():
                y_torch = torch_block(x_torch,
                                      cu_seqlens=cu_seqlens,
                                      rotary_pos_emb_cos=rotary_pos_emb_cos,
                                      rotary_pos_emb_sin=rotary_pos_emb_sin,
                                      max_seqlen=max_seqlen,
                                      sequence_lengths=None)
        finally:
            torch_block.attn.attn.forward = orig_forward

        # Pass actual window segment lengths to JAX block to bypass searchsorted NoneType errors
        cu_window_seqlens = jnp.array([0, 10], dtype=jnp.int32)
        y_jax = jax_block(x_jax,
                          rotary_pos_emb,
                          cu_window_seqlens=cu_window_seqlens,
                          use_fullattn=True)

        np.testing.assert_allclose(np.array(y_jax),
                                   y_torch.detach().cpu().numpy(),
                                   rtol=1e-3,
                                   atol=1e-3)


class TestQwen3_VisionTransformer_Equivalence:

    def test_equivalence(self, mock_vllm_config: MockVllmConfig,
                         rngs: nnx.Rngs, mesh: Mesh):
        config = mock_vllm_config.model_config.hf_config
        vc = config.vision_config
        # 1. Instantiate PyTorch visual tower inside current vLLM config context
        with set_current_vllm_config(mock_vllm_config):
            torch_visual = TorchVisionTransformer(
                vision_config=vc,
                norm_eps=1e-6,
            ).to(dtype=torch.float32).eval()

        # 2. Instantiate JAX visual tower
        with jax.set_mesh(mesh):
            jax_visual = JaxVisionTransformer(
                vllm_config=mock_vllm_config,
                rngs=rngs,
                mesh=mesh,
            )

        # Mock flash_attention on all blocks to bypass TPU-only Pallas kernel on CPU
        for blk in jax_visual.blocks:
            blk.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        # Mock PyTorch flash attention to also return ones (for equivalent dummy behavior)
        for blk in torch_visual.blocks:
            blk.attn.attn.forward = lambda query, key, value, cu_seqlens, max_seqlen, sequence_lengths: torch.ones_like(
                query)

        # 3. Align all weights using copy_weights_to_jax_vision_tower
        copy_weights_to_jax_vision_tower(torch_visual, jax_visual)

        # 4. Generate inputs (1 image: t=1, h=4, w=4 -> 16 patches -> 4 output tokens)
        # Inputs must match the patch projection dimensions (flat patch pixels = 3 * 2 * 16 * 16 = 1536)
        flat_patch_dim = 3 * 2 * 16 * 16
        x_np = (np.random.randn(16, flat_patch_dim) * 0.1).astype(np.float32)
        x_jax = jnp.array(x_np)
        x_torch = torch.from_numpy(x_np)

        # 5. Run forward pass
        grid_thw = ((1, 4, 4), )

        # JAX visual tower call
        y_jax = jax_visual(x_jax, grid_thw)

        # PyTorch visual tower call
        with torch.no_grad():
            grid_thw_tensor = torch.tensor([[1, 4, 4]], dtype=torch.int32)
            y_torch = torch_visual(x_torch, grid_thw=grid_thw_tensor)

        # 6. Compare outputs
        np.testing.assert_allclose(np.array(y_jax),
                                   y_torch.detach().cpu().numpy(),
                                   rtol=1e-3,
                                   atol=1e-3)


class TestQwen3_VisionTransformer_Compilation:

    def test_compilation_behavior_with_dynamic_shapes(
            self, mock_vllm_config: MockVllmConfig, rngs: nnx.Rngs,
            mesh: Mesh):
        # 1. Enable dynamic image sizes
        mock_vllm_config.additional_config = {
            "enable_dynamic_image_sizes": True
        }
        vc = mock_vllm_config.model_config.hf_config.vision_config
        vc.window_size = 128  # Fix ZeroDivisionError: ensure vit_merger_window_size is non-zero
        # 2. Instantiate Qwen3_VisionTransformer
        with jax.set_mesh(mesh):
            jax_visual = JaxVisionTransformer(
                vllm_config=mock_vllm_config,
                rngs=rngs,
                mesh=mesh,
            )

        # Mock flash_attention on all blocks to bypass TPU-only Pallas kernel on CPU
        for blk in jax_visual.blocks:
            blk.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        # 3. Set up compilation tracking hook on compute_hidden_states
        compilation_count = 0
        orig_compute_hidden_states = jax_visual.compute_hidden_states

        def track_compute_hidden_states(*args, **kwargs):
            nonlocal compilation_count
            compilation_count += 1
            return orig_compute_hidden_states(*args, **kwargs)

        jax_visual.compute_hidden_states = track_compute_hidden_states

        # Helper function to run model
        def run_model(grid_thw, values_multiplier=1.0):
            # Calculate total sequence length of patches (before spatial merge)
            total_patches = 0
            for t, h, w in grid_thw:
                total_patches += t * h * w

            bucket_patches = 1 << (total_patches - 1).bit_length()

            # Calculate flat patch dimension required by vision patch embed
            flat_patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

            # Create inputs
            x = jnp.ones((total_patches, flat_patch_dim)) * values_multiplier
            x_padded = jnp.pad(x,
                               ((0, bucket_patches - total_patches), (0, 0)))
            return jax_visual(x_padded, grid_thw)

        # --- Run 1: 1 image of 2x2 patches (total 4 patches) ---
        # Should trigger compilation (Compile count: 1)
        run_model(((1, 2, 2), ))
        assert compilation_count == 1

        # --- Run 2: Same shape, different pixel values ---
        # Should reuse cache (Compile count remains 1)
        run_model(((1, 2, 2), ), values_multiplier=2.0)
        assert compilation_count == 1

        # --- Run 3: Different size (4x4 = 16 patches, bucket size 16) ---
        # Should trigger a new compilation (Compile count: 2)
        run_model(((1, 4, 4), ))
        assert compilation_count == 2

        # --- Run 4: Different grid shape (2x6 = 12 patches, bucket size 16) ---
        # Since 12 patches falls into the same bucket size 16 as Run 3,
        # it should reuse the compiled JIT function (Compile count remains 2)
        run_model(((1, 2, 6), ))
        assert compilation_count == 2

        # --- Run 5: 2 images, each 2x2 patches (total 8 patches, bucket size 8) ---
        # Different bucket size (8) and image count -> Triggers compilation (Compile count: 3)
        run_model(((1, 2, 2), (1, 2, 2)))
        assert compilation_count == 3

        # --- Run 6: Same 2 images, different pixel values ---
        # Should reuse cache (Compile count remains 3)
        run_model(((1, 2, 2), (1, 2, 2)), values_multiplier=3.0)
        assert compilation_count == 3

        # --- Run 7: 2 images: 2x2 and 2x4 patches (total 12 patches, bucket size 16) ---
        # Different bucket size (16) -> Triggers compilation (Compile count: 4)
        run_model(((1, 2, 2), (1, 2, 4)))
        assert compilation_count == 4

        # --- Run 8: Different 2 images: 2x4 and 2x4 patches (total 16 patches, bucket size 16) ---
        # Since total patches still falls into bucket size 16, and number of images is still 2,
        # it shares the exact same static dimensions and JAX shapes as Run 7.
        # Should reuse the compiled JIT function from Run 7 (Compile count remains 4)!
        run_model(((1, 2, 4), (1, 2, 4)))
        assert compilation_count == 4

        # --- Run 9: Same 2 images as Run 8, different pixel values ---
        # Should reuse cache (Compile count remains 4)
        run_model(((1, 2, 4), (1, 2, 4)), values_multiplier=4.0)
        assert compilation_count == 4

    def test_instantiation_with_pytorch_dtype(self,
                                              mock_vllm_config: MockVllmConfig,
                                              rngs: nnx.Rngs, mesh: Mesh):
        # Mock model_config.dtype to be a PyTorch dtype
        mock_vllm_config.model_config.dtype = torch.bfloat16

        # Should instantiate successfully without raising TypeError
        with jax.set_mesh(mesh):
            jax_visual = JaxVisionTransformer(
                vllm_config=mock_vllm_config,
                rngs=rngs,
                mesh=mesh,
            )
        assert jax_visual.dtype == jnp.bfloat16

    def test_copy_weights_bfloat16(self, mock_vllm_config: MockVllmConfig,
                                   rngs: nnx.Rngs, mesh: Mesh):
        from tpu_inference.models.jax.qwen3_vl import \
            copy_weights_to_jax_vision_tower

        # 1. Create a mock PyTorch visual tower with bfloat16 parameters
        with set_current_vllm_config(mock_vllm_config):
            torch_visual = TorchVisionTransformer(
                vision_config=mock_vllm_config.model_config.hf_config.
                vision_config,
                norm_eps=1e-6,
            ).to(dtype=torch.bfloat16)

        # 2. Instantiate JAX visual tower
        with jax.set_mesh(mesh):
            jax_visual = JaxVisionTransformer(
                vllm_config=mock_vllm_config,
                rngs=rngs,
                mesh=mesh,
            )

        # Mock flash_attention on all blocks to bypass TPU-only Pallas kernel on CPU
        for blk in jax_visual.blocks:
            blk.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        # 3. Run weight copy. This should execute successfully for bfloat16 parameters!
        copy_weights_to_jax_vision_tower(torch_visual, jax_visual)

    def test_copy_weights_torchax_tensors(self,
                                          mock_vllm_config: MockVllmConfig,
                                          rngs: nnx.Rngs, mesh: Mesh):
        # 1. Create a mock PyTorch visual tower with bfloat16 parameters
        with set_current_vllm_config(mock_vllm_config):
            torch_visual = TorchVisionTransformer(
                vision_config=mock_vllm_config.model_config.hf_config.
                vision_config,
                norm_eps=1e-6,
            ).to(dtype=torch.bfloat16)

        # 2. Convert all parameters to TorchAX Tensors to simulate TorchAX compilation!
        mocked_params = []
        for name, param in torch_visual.named_parameters():
            # We convert torch -> jax -> torchax tensor
            jax_val = t2j(param.detach())
            torchax_tensor = torch_view(jax_val)
            mocked_params.append((name, torchax_tensor))
        torch_visual.named_parameters = lambda: mocked_params

        # 3. Instantiate JAX visual tower
        with jax.set_mesh(mesh):
            jax_visual = JaxVisionTransformer(
                vllm_config=mock_vllm_config,
                rngs=rngs,
                mesh=mesh,
            )

        # Mock flash_attention on all blocks to bypass TPU-only Pallas kernel on CPU
        for blk in jax_visual.blocks:
            blk.attn.flash_attention = MagicMock(
                side_effect=lambda q, k, v, seg: jnp.ones_like(q))

        # 4. Run weight copy. This should execute successfully for TorchAX parameters!
        copy_weights_to_jax_vision_tower(torch_visual, jax_visual)
