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

from unittest.mock import MagicMock, patch

import jax
import torch
import torchax
import vllm.model_executor.models.qwen3_vl as qwen3_vl_mod
from jax.sharding import Mesh, NamedSharding
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration, Qwen3VLMultiModalProcessor)

from tests.models.jax.test_qwen3_vl import MockVllmConfig
from tpu_inference import envs
from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import (
    TorchAXJaxVisionTowerBridge, _patched_flatten_embeddings,
    _patched_get_deepstack, apply_qwen3_vl_patches,
    maybe_apply_qwen3_vl_patches)
from tpu_inference.models.vllm.experimental.vision_tower_jit import GridTHW


def test_patched_flatten_embeddings_2d():
    # Standard 2D embedding tensor (num_tokens, hidden_dim)
    t = torch.ones((5, 10))
    res = _patched_flatten_embeddings(t)

    # Should flatten all but the last dimension.
    # For 2D (5, 10), all but last dim is just dim 0 (size 5).
    # Result shape should be (5, 10).
    assert res.shape == (5, 10)
    assert torch.allclose(res, t)


def test_patched_flatten_embeddings_3d():
    # 3D embedding tensor (batch, num_tokens, hidden_dim)
    t = torch.ones((2, 5, 10))
    res = _patched_flatten_embeddings(t)

    # Should flatten dimensions 0 and 1 -> 2 * 5 = 10.
    # Result shape should be (10, 10).
    assert res.shape == (10, 10)


def test_patched_flatten_embeddings_nested_list():
    # Nested list/tuple of tensors
    t1 = torch.ones((2, 5))
    t2 = torch.ones((3, 5))
    embeddings = [t1, t2]

    res = _patched_flatten_embeddings(embeddings)
    # Should flatten each and concatenate them -> (2+3, 5) = (5, 5)
    assert res.shape == (5, 5)


def test_patched_flatten_embeddings_empty_and_1d():
    # 1D tensor of shape (L,) representing edge cases / 1D sequences
    t = torch.ones((5, ))
    res = _patched_flatten_embeddings(t)

    # ndim = 1. start_dim = 0, end_dim = -2.
    # start_dim = 0. end_dim = -2 + 1 = -1.
    # Returns t.flatten(0, -1) which is a 1D tensor.
    assert res.shape == (5, )

    # Empty 1D tensor
    t_empty = torch.ones((0, ))
    res_empty = _patched_flatten_embeddings(t_empty)
    assert res_empty.shape == (0, )


def test_patched_get_deepstack_none():

    class MockModel:
        pass

    model = MockModel()

    # Case 1: orig_get_deepstack returns None, and there are no cached deepstack tensors
    def orig_get_deepstack(tokens):
        return None

    res = _patched_get_deepstack(model, orig_get_deepstack, 10)
    assert res is None


def test_patched_get_deepstack_with_cache():

    class MockModel:
        _deepstack_tensors = {"deepstack_input_embeds_0": torch.ones((5, 10))}

    model = MockModel()

    # Case 2: cached deepstack tensors exist, orig_get_deepstack should not be called
    def orig_get_deepstack(tokens):
        raise AssertionError("Should not be called")

    res = _patched_get_deepstack(model, orig_get_deepstack, 10)
    assert res is not None
    assert "deepstack_input_embeds_0" in res.tensors


def test_patched_get_deepstack_fallback():

    class MockModel:
        pass

    model = MockModel()

    # Case 3: no cached deepstack tensors, orig_get_deepstack returns a dictionary
    t1 = torch.ones((5, 10))

    def orig_get_deepstack(tokens):
        return {"layer_0": t1}

    res = _patched_get_deepstack(model, orig_get_deepstack, 10)
    assert res is not None
    assert "layer_0" in res.tensors


def _trigger_qwen3_vl_patches():

    class MockModel:
        use_deepstack = True

        def _set_deepstack_input_embeds(self, embeds):
            pass

        def _get_deepstack_input_embeds(self, tokens):
            pass

        def embed_input_ids(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            pass

    mock_model = MockModel()
    original_val = envs.VLLM_TPU_ENABLE_FLAX_ENCODER
    envs.VLLM_TPU_ENABLE_FLAX_ENCODER = False
    try:
        apply_qwen3_vl_patches(mock_model)
    finally:
        envs.VLLM_TPU_ENABLE_FLAX_ENCODER = original_val


def test_patched_merge_multimodal_embeddings_empty():
    _trigger_qwen3_vl_patches()
    inputs_embeds = torch.ones((5, 10))
    is_multimodal = torch.zeros((5, ), dtype=torch.bool)

    # When multimodal_embeddings is empty, it should return inputs_embeds directly
    res = qwen3_vl_mod._merge_multimodal_embeddings(inputs_embeds, [],
                                                    is_multimodal)
    assert res is inputs_embeds


def test_patched_merge_multimodal_embeddings_standard():
    _trigger_qwen3_vl_patches()
    inputs_embeds = torch.ones((10, 4)) * 1.0
    multimodal_embeddings = torch.ones((10, 4)) * 2.0
    is_multimodal = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False])

    res = qwen3_vl_mod._merge_multimodal_embeddings(inputs_embeds,
                                                    multimodal_embeddings,
                                                    is_multimodal)

    assert res.shape == (10, 4)
    # Should select multimodal_embeddings (2.0) where is_multimodal is True
    # and inputs_embeds (1.0) where is_multimodal is False
    assert torch.allclose(res[0:2], torch.tensor(1.0))
    assert torch.allclose(res[2:4], torch.tensor(2.0))
    assert torch.allclose(res[4:5], torch.tensor(1.0))
    assert torch.allclose(res[5:8], torch.tensor(2.0))
    assert torch.allclose(res[8:], torch.tensor(1.0))


def test_patched_merge_multimodal_embeddings_list():
    _trigger_qwen3_vl_patches()
    inputs_embeds = torch.ones((10, 4)) * 1.0

    # Pre-aligned list of features. The combined features total 10 rows.
    # The actual features (2.0) are placed at the mask-matching positions
    # (indices 2,3 for first image, and 5,6,7 for second image).
    img1_padded = torch.zeros((5, 4))
    img1_padded[2:4] = 2.0

    img2_padded = torch.zeros((5, 4))
    img2_padded[0:3] = 3.0  # Corresponds to indices 5,6,7 of combined sequence

    multimodal_embeddings = [img1_padded, img2_padded]
    is_multimodal = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False])

    res = qwen3_vl_mod._merge_multimodal_embeddings(inputs_embeds,
                                                    multimodal_embeddings,
                                                    is_multimodal)

    assert res.shape == (10, 4)
    # Text positions: should keep inputs_embeds (1.0)
    # First image positions (2,3): should get img1_padded features (2.0)
    # Second image positions (5,6,7): should get img2_padded features (3.0)
    assert torch.allclose(res[0:2], torch.tensor(1.0))
    assert torch.allclose(res[2:4], torch.tensor(2.0))
    assert torch.allclose(res[4:5], torch.tensor(1.0))
    assert torch.allclose(res[5:8], torch.tensor(3.0))
    assert torch.allclose(res[8:], torch.tensor(1.0))


def test_qwen3_vl_compute_deepstack_embeds():
    _trigger_qwen3_vl_patches()

    # 1. Import the official method from Qwen3-VL class
    # 2. Create a mock self with required attributes for Qwen3-VL Deepstack
    class MockQwen3VL:
        visual_dim = 4
        multiscale_dim = 8  # deepstack_num_level * visual_dim = 2 * 4 = 8
        deepstack_num_level = 2

    mock_self = MockQwen3VL()

    # 3. Create dummy inputs:
    # inputs_embeds (text embeddings): shape (10, 4)
    inputs_embeds = torch.ones((10, 4)) * 1.0

    # multimodal_embeddings (visual embeddings combining visual_dim & multiscale_dim):
    # list of 1 element of shape (10, visual_dim + multiscale_dim) = (10, 12)
    # We pre-align the multiscale features (3.0) to indices 2,3 and 5,6,7
    vision_features = torch.zeros((10, 12))
    # Set main visual features (visual_dim)
    vision_features[:, :4] = 2.0
    # Set multiscale visual features (multiscale_dim) at mask positions
    vision_features[2:4, 4:] = 3.0
    vision_features[5:8, 4:] = 3.0

    multimodal_embeddings = [vision_features]

    # mask for placeholders: True at indices [2, 3, 5, 6, 7]
    is_multimodal = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False])

    # 4. Call the official Qwen3-VL _compute_deepstack_embeds method!
    deepstack_input_embeds, main_embeds = Qwen3VLForConditionalGeneration._compute_deepstack_embeds(
        mock_self, inputs_embeds, multimodal_embeddings, is_multimodal)

    # 5. Verify the outputs:
    # main_embeds should contain the split main visual features: shape (10, 4)
    assert len(main_embeds) == 1
    assert main_embeds[0].shape == (10, 4)
    assert torch.allclose(main_embeds[0], torch.tensor(2.0))

    # deepstack_input_embeds should contain multiscale features mapped to levels:
    # shape (deepstack_num_level, seq_len, visual_dim) = (2, 10, 4)
    # Level 0 and 1 should have 3.0 at indices 2,3 and 5,6,7, and 0.0 at other positions
    assert deepstack_input_embeds.shape == (2, 10, 4)
    for level in range(2):
        level_embeds = deepstack_input_embeds[level]
        assert torch.allclose(level_embeds[0:2], torch.tensor(0.0))
        assert torch.allclose(level_embeds[2:4], torch.tensor(3.0))
        assert torch.allclose(level_embeds[4:5], torch.tensor(0.0))
        assert torch.allclose(level_embeds[5:8], torch.tensor(3.0))
        assert torch.allclose(level_embeds[8:], torch.tensor(0.0))


def test_qwen3_vl_create_final_video_embeddings():
    _trigger_qwen3_vl_patches()

    # 1. Import necessary Qwen3-VL modules
    # 2. Create mock structures for Qwen3VL class
    class MockConfig:
        vision_start_token_id = 151857
        vision_end_token_id = 151858
        video_token_id = 151859

    class MockLanguageModel:

        def embed_input_ids(self, token_ids):
            # Returns text embeddings of shape (len(token_ids), 4) with values 1.0
            return torch.ones((len(token_ids), 4)) * 1.0

    class MockQwen3VL:
        config = MockConfig()
        _tokenizer = None
        is_multimodal_pruning_enabled = False
        use_deepstack = False

        def get_language_model(self):
            return MockLanguageModel()

    mock_self = MockQwen3VL()

    # 3. Mock the processor static method get_video_repl
    mock_video_repl = MagicMock()
    # Returns a tokenized list: [start_token, video_token, end_token]
    mock_video_repl.full = [151857, 151859, 151858]

    with patch.object(Qwen3VLMultiModalProcessor,
                      "get_video_repl",
                      return_value=mock_video_repl):
        # 4. Prepare input video embeddings: 1 video token of shape (1, 4) with value 2.0
        video_embeddings = torch.ones((1, 4)) * 2.0

        # Call the official Qwen3-VL _create_final_video_embeddings method!
        res = Qwen3VLForConditionalGeneration._create_final_video_embeddings(
            mock_self,
            video_embeddings=video_embeddings,
            num_tokens_per_frame=[1],
            timestamps=[0.0],
            video_grid_thw=[1, 1, 1],
            retention_mask=None)

        # 5. Assertions:
        # Result shape should be (3, 4)
        assert res.shape == (3, 4)
        # Row 0: text embedding (1.0)
        # Row 1: video embedding (2.0)
        # Row 2: text embedding (1.0)
        assert torch.allclose(res[0], torch.tensor(1.0))
        assert torch.allclose(res[1], torch.tensor(2.0))
        assert torch.allclose(res[2], torch.tensor(1.0))


def test_torchax_jax_vision_tower_bridge():
    # A mock JAX model that takes a JAX array and the grid_thw tuple,
    # and returns a JAX array.
    def mock_jax_model(pixel_values_jax, grid_thw_tuple):
        assert isinstance(grid_thw_tuple, tuple)
        assert all(isinstance(x, tuple) for x in grid_thw_tuple)
        return pixel_values_jax * 2.0

    mock_jax_model.dtype = jax.numpy.bfloat16

    class MockConfig:
        spatial_merge_size = 2
        out_hidden_size = 2048

    mock_jax_model.config = MockConfig()

    with torchax.default_env():
        bridge = TorchAXJaxVisionTowerBridge(mock_jax_model)
        assert bridge.dtype == torch.bfloat16
        assert bridge.device == torch.device("jax")
        assert bridge.spatial_merge_size == 2
        assert bridge.out_hidden_size == 2048

        pixel_values = torch_view(t2j(torch.ones((2, 3))))
        grid_thw = [[1, 2, 3], [4, 5, 6]]
        res = bridge(pixel_values, grid_thw)
        assert isinstance(res, torch.Tensor)
        assert res.shape == (2, 3)
        assert torch.allclose(res.to("cpu"), torch.ones((2, 3)) * 2.0)


def test_torchax_jax_vision_tower_bridge_functional_call():

    class MockJaxModel:
        pass

    # Define a parent module to wrap the bridge, mimicking self.model/vllm_model
    class MockParentModule(torch.nn.Module):

        def __init__(self, bridge):
            super().__init__()
            self.visual = bridge

        def forward(self, x):
            return self.visual(x, [[1, 1, 1]])

    def mock_jax_model(pixel_values_jax, grid_thw_tuple):
        return pixel_values_jax

    mock_jax_model.dtype = jax.numpy.bfloat16

    class MockConfig:
        spatial_merge_size = 2
        out_hidden_size = 2048

    mock_jax_model.config = MockConfig()

    with torchax.default_env():
        bridge = TorchAXJaxVisionTowerBridge(mock_jax_model)
        parent = MockParentModule(bridge)

        # Construct a parameters dictionary that includes paths mimicking original parameters
        # from all submodules of Qwen3-VL visual tower
        params = {
            "visual.patch_embed.proj.weight":
            torch_view(t2j(torch.ones((2, 2, 2, 2, 2)))),
            "visual.patch_embed.proj.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.pos_embed.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.merger.norm.weight":
            torch_view(t2j(torch.ones((2, )))),
            "visual.merger.norm.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.merger.linear_fc1.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.merger.linear_fc1.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.merger.linear_fc2.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.merger.linear_fc2.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.norm1.weight":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.norm1.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.norm2.weight":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.norm2.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.attn.qkv.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.blocks.0.attn.qkv.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.attn.proj.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.blocks.0.attn.proj.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.mlp.linear_fc1.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.blocks.0.mlp.linear_fc1.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.mlp.linear_fc2.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.blocks.0.mlp.linear_fc2.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.deepstack_merger_list.0.norm.weight":
            torch_view(t2j(torch.ones((2, )))),
            "visual.deepstack_merger_list.0.norm.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.deepstack_merger_list.0.linear_fc1.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.deepstack_merger_list.0.linear_fc1.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.deepstack_merger_list.0.linear_fc2.weight":
            torch_view(t2j(torch.ones((2, 2)))),
            "visual.deepstack_merger_list.0.linear_fc2.bias":
            torch_view(t2j(torch.ones((2, )))),
            "visual.blocks.0.attn.rotary_pos_emb.cos_sin_cache":
            torch_view(t2j(torch.ones((2, 2)))),
        }

        # Test that torch.func.functional_call successfully runs without AttributeError
        res = torch.func.functional_call(
            parent,
            params,
            args=(torch_view(t2j(torch.ones((2, 2)))), ),
            tie_weights=False,
        )
        assert isinstance(res, torch.Tensor)


def test_torchax_jax_vision_tower_bridge_process_image_input_math():

    class MockJaxModel:
        pass

    class MockConfig:
        spatial_merge_size = 2
        out_hidden_size = 2048

    mock_jax_model = MockJaxModel()
    mock_jax_model.config = MockConfig()

    with torchax.default_env():
        bridge = TorchAXJaxVisionTowerBridge(mock_jax_model)
        merge_size = bridge.spatial_merge_size

        # Case A: grid_thw is a GridTHW instance (precompilation JIT pathway)
        grid_thw_jit = GridTHW([(1, 16, 16)])
        sizes_jit = (grid_thw_jit.prod(-1) // merge_size //
                     merge_size).tolist()
        assert sizes_jit == [64]

        # Case B: grid_thw is a torchax.Tensor (eager/warmup execution pathway)
        grid_thw_tensor = torch_view(t2j(torch.tensor([[1, 16, 16]])))
        res_tensor = (grid_thw_tensor.prod(-1) // merge_size // merge_size)
        assert res_tensor.tolist() == [64]


def test_qwen3_vl_process_image_input_compatibility():

    class MockConfig:
        spatial_merge_size = 2
        out_hidden_size = 2048

    # Define mock model class mimicking Qwen3VLForConditionalGeneration
    class MockQwen3VL:
        use_data_parallel = False

    with torchax.default_env():
        # Create mock JAX forward function returning expected output size (64 tokens, 2048 channels)
        def mock_jax_model_func(pixel_values_jax, grid_thw_tuple):
            return jax.numpy.ones((64, 2048), dtype=jax.numpy.bfloat16)

        mock_jax_model_func.config = MockConfig()
        mock_jax_model_func.dtype = jax.numpy.bfloat16

        bridge = TorchAXJaxVisionTowerBridge(mock_jax_model_func)

        mock_self = MockQwen3VL()
        mock_self.visual = bridge

        pixel_values = torch_view(
            t2j(torch.ones((256, 1536), dtype=torch.bfloat16)))

        # Case A: GridTHW (JIT precompilation pathway)
        grid_thw_jit = GridTHW([(1, 16, 16)])
        image_input_jit = {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw_jit,
        }

        # Call the official Qwen3-VL _process_image_input method!
        res_split_jit = Qwen3VLForConditionalGeneration._process_image_input(
            mock_self, image_input_jit)
        assert len(res_split_jit) == 1
        assert res_split_jit[0].shape == (64, 2048)

        # Case B: torchax.Tensor (eager/warmup pathway)
        grid_thw_tensor = torch_view(t2j(torch.tensor([[1, 16, 16]])))
        image_input_tensor = {
            "type": "pixel_values",
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw_tensor,
        }

        # Call the official Qwen3-VL _process_image_input method!
        res_split_tensor = Qwen3VLForConditionalGeneration._process_image_input(
            mock_self, image_input_tensor)
        assert len(res_split_tensor) == 1
        assert res_split_tensor[0].shape == (64, 2048)


def test_torchax_jax_vision_tower_bridge_padding_and_slicing():

    class MockConfig:
        spatial_merge_size = 2
        out_hidden_size = 2048

    # Mock JAX model that verifies it received padded inputs
    def mock_jax_model_func(pixel_values_jax, grid_thw_tuple):
        # Original num_patches was 12, it should have been padded to 16!
        assert pixel_values_jax.shape == (16, 1536)
        # Return fully padded JAX outputs (e.g. 4 padded tokens)
        return jax.numpy.ones((4, 2048), dtype=jax.numpy.bfloat16)

    mock_jax_model_func.config = MockConfig()
    mock_jax_model_func.dtype = jax.numpy.bfloat16

    with torchax.default_env():
        bridge = TorchAXJaxVisionTowerBridge(mock_jax_model_func)

        # Generate 16 pre-padded patches (padded from 12 to bucket size 16)
        pixel_values = torch_view(
            t2j(torch.ones((16, 1536), dtype=torch.bfloat16)))
        grid_thw = [[1, 2,
                     6]]  # total_tokens = 1 * (2 // 2) * (6 // 2) = 3 tokens

        # Execute bridge!
        res = bridge(pixel_values, grid_thw)

        # Assertions:
        # 1. Bridge outputs fully padded JAX outputs directly!
        assert res.shape == (4, 2048)
        assert torch.allclose(res.to("cpu").float(), torch.ones((4, 2048)))


def test_maybe_apply_qwen3_vl_patches_deepstack_input_embeds():
    mock_vllm_config = MockVllmConfig()

    # Define a Mock Qwen3-VL model inheriting from torch.nn.Module to avoid initialization crashes
    class MockQwen3VL(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.use_deepstack = True

            class MockTextConfig:
                rms_norm_eps = 1e-6

            class MockConfig:
                text_config = MockTextConfig()
                vision_config = mock_vllm_config.model_config.hf_config.vision_config

            self.config = MockConfig()
            self.deepstack_input_embeds = [
                torch.ones((2, 4), dtype=torch.bfloat16)
            ]
            self.visual = torch.nn.Linear(2, 2)

    mock_model = MockQwen3VL()

    # Define a JAX mesh context for the test with 'model' axis to match model sharding specs
    devices = jax.local_devices()[:1]
    mesh = Mesh(devices, ('model', ))

    original_val = envs.VLLM_TPU_ENABLE_FLAX_ENCODER
    envs.VLLM_TPU_ENABLE_FLAX_ENCODER = True

    try:
        with jax.set_mesh(mesh), torchax.default_env(
        ), set_current_vllm_config(mock_vllm_config):
            # Call the method under test with patched architecture check
            with patch(
                    "tpu_inference.models.vllm.experimental.qwen3_vl_patcher.is_qwen3_vl",
                    return_value=True):
                maybe_apply_qwen3_vl_patches(mock_model)

            # Check that deepstack_input_embeds was converted to a JAX-backed Torchax view tensor
            assert hasattr(mock_model.deepstack_input_embeds[0], "jax_device")

            # Fetch the underlying JAX Array using jax_view
            jax_array = jax_view(mock_model.deepstack_input_embeds[0])

            # Assert that the JAX Array's sharding matches our test mesh NamedSharding!
            assert isinstance(jax_array.sharding, NamedSharding)
            assert jax_array.sharding.mesh == mesh
    finally:
        envs.VLLM_TPU_ENABLE_FLAX_ENCODER = original_val


def test_maybe_apply_qwen3_vl_patches_precompile_vision_encoder():
    mock_vllm_config = MockVllmConfig()

    class MockSchedulerConfig:
        max_num_batched_tokens = 2048

    mock_vllm_config.scheduler_config = MockSchedulerConfig()

    # Define a Mock Qwen3-VL model inheriting from torch.nn.Module to avoid initialization crashes
    class MockQwen3VL(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.use_deepstack = True

            class MockTextConfig:
                rms_norm_eps = 1e-6

            class MockConfig:
                text_config = MockTextConfig()
                vision_config = mock_vllm_config.model_config.hf_config.vision_config

            self.config = MockConfig()
            self.deepstack_input_embeds = [
                torch.ones((2, 4), dtype=torch.bfloat16)
            ]
            self.visual = torch.nn.Linear(2, 2)

    mock_model = MockQwen3VL()

    with patch(
            "tpu_inference.models.vllm.experimental.qwen3_vl_patcher.is_qwen3_vl",
            return_value=True):
        maybe_apply_qwen3_vl_patches(mock_model)

        # Verify that the precompile_vision_encoder method is patched on the model
        assert hasattr(mock_model, "precompile_vision_encoder")
        assert callable(mock_model.precompile_vision_encoder)

        # Explicitly verify it is a bound method and its __self__ is the model instance
        import types
        assert isinstance(mock_model.precompile_vision_encoder,
                          types.MethodType)
        assert mock_model.precompile_vision_encoder.__self__ is mock_model

        # Test precompile execution when wrapper and params are bound
        class MockWrapper:
            vllm_config = mock_vllm_config

            def wrap_embed_multimodal_func(self):
                return lambda *args, **kwargs: None

        mock_model._wrapper = MockWrapper()
        mock_model._params = {"dummy": None}

        # Call precompile_vision_encoder and make sure it runs without throwing exceptions
        run_compilation_fn = MagicMock()
        mock_model.precompile_vision_encoder(run_compilation_fn)
        assert run_compilation_fn.called
