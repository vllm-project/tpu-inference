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

from unittest.mock import MagicMock

import pytest

from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper


@pytest.mark.parametrize(
    "is_multimodal_model, multimodal_config, expected",
    [
        (False, None, False),
        (True, None, True),
        (True, MagicMock(language_model_only=False), True),
        (True, MagicMock(language_model_only=True), False),
    ],
)
def test_is_multimodal_enabled(is_multimodal_model, multimodal_config,
                               expected):
    wrapper = object.__new__(VllmModelWrapper)
    wrapper.vllm_config = MagicMock()
    wrapper.vllm_config.model_config.is_multimodal_model = is_multimodal_model
    wrapper.vllm_config.model_config.multimodal_config = multimodal_config

    assert wrapper._is_multimodal_enabled() is expected


def test_language_model_only_disables_multimodal_wrappers():
    wrapper = object.__new__(VllmModelWrapper)
    wrapper.vllm_config = MagicMock()
    wrapper.vllm_config.model_config.is_multimodal_model = True
    wrapper.vllm_config.model_config.multimodal_config.language_model_only = True
    wrapper._mm_encoder_jit_manager = None

    assert wrapper.wrap_precompile_vision_encoder_fn(MagicMock()) is None
    assert wrapper.wrap_embed_multimodal_func() is None
    assert wrapper.wrap_embed_input_ids_func() is None
