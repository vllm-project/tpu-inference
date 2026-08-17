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

from unittest import mock

import torch
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration

from tpu_inference.models.vllm.experimental.gemma4_mm_patcher import (
    apply_gemma4_mm_patches, maybe_apply_gemma4_mm_patches)

IMAGE_TOKEN_ID = 7
AUDIO_TOKEN_ID = 8
NUM_LAYERS = 3
PLE_DIM = 4


def _make_mock_gemma4(ple_dim=PLE_DIM):
    # spec= makes isinstance(model, Gemma4ForConditionalGeneration) true but
    # blocks reads of instance-only attributes, so attach them explicitly.
    model = mock.MagicMock(spec=Gemma4ForConditionalGeneration)
    model.config = mock.MagicMock()
    model.config.image_token_id = IMAGE_TOKEN_ID
    model.config.audio_token_id = AUDIO_TOKEN_ID
    model.config.text_config.num_hidden_layers = NUM_LAYERS
    model.config.text_config.hidden_size_per_layer_input = ple_dim
    model.language_model = mock.MagicMock()
    model._clear_mm_prefix_for_full_attn_layers = mock.MagicMock()
    model.per_layer_embeddings = None if ple_dim is None else torch.zeros(
        16, NUM_LAYERS, ple_dim)
    return model


def test_maybe_apply_skips_non_gemma4():
    model = mock.MagicMock()
    orig_forward = model.forward
    maybe_apply_gemma4_mm_patches(model)
    assert model.forward is orig_forward


def test_maybe_apply_skips_variant_without_ple():
    model = _make_mock_gemma4(ple_dim=None)
    orig_forward = model.forward
    maybe_apply_gemma4_mm_patches(model)
    assert model.forward is orig_forward


def test_apply_drops_buffer_and_replaces_forward():
    model = _make_mock_gemma4()
    orig_forward = model.forward
    maybe_apply_gemma4_mm_patches(model)

    # The plain-tensor CUDA-graph buffer must be gone: it is invisible to
    # shard_model_to_tpu and crashes _aten_copy under torchax.
    assert model.per_layer_embeddings is None
    assert model.forward is not orig_forward
    assert model._tpu_ple_mask_token_ids == (IMAGE_TOKEN_ID, AUDIO_TOKEN_ID)


def test_patched_forward_computes_masked_ple_inline():
    model = _make_mock_gemma4()
    apply_gemma4_mm_patches(model)

    num_tokens = 6
    input_ids = torch.tensor(
        [1, IMAGE_TOKEN_ID, 2, AUDIO_TOKEN_ID, 3, IMAGE_TOKEN_ID])
    positions = torch.arange(num_tokens)
    inputs_embeds = torch.ones(num_tokens, 5)

    seen_ple_ids = []

    def fake_get_per_layer_inputs(ids):
        seen_ple_ids.append(ids)
        return torch.ones(num_tokens, NUM_LAYERS * PLE_DIM)

    model.language_model.model.get_per_layer_inputs.side_effect = \
        fake_get_per_layer_inputs

    model.forward(input_ids, positions, inputs_embeds=inputs_embeds)

    # Reference semantics: embed_input_ids masks multimodal placeholder
    # positions to token 0 before the PLE lookup.
    assert len(seen_ple_ids) == 1
    assert seen_ple_ids[0].tolist() == [1, 0, 2, 0, 3, 0]

    # The language model receives the reshaped PLE tensor, not a buffer read.
    lm_call = model.language_model.model.call_args
    ple = lm_call.kwargs['per_layer_inputs']
    assert ple.shape == (num_tokens, NUM_LAYERS, PLE_DIM)
    assert lm_call.kwargs['inputs_embeds'] is inputs_embeds

    # mm_prefix_range clearing must still run outside the compiled region.
    model._clear_mm_prefix_for_full_attn_layers.assert_called_once()


def test_patched_forward_skips_ple_without_inputs_embeds():
    model = _make_mock_gemma4()
    apply_gemma4_mm_patches(model)

    input_ids = torch.tensor([1, 2, 3])
    positions = torch.arange(3)

    model.forward(input_ids, positions, inputs_embeds=None)

    model.language_model.model.get_per_layer_inputs.assert_not_called()
    assert model.language_model.model.call_args.kwargs[
        'per_layer_inputs'] is None
