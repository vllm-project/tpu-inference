# Copyright 2025 Google LLC
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

import torch
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.models.vllm.vllm_model_loader import \
    _patch_tied_embedding_mapper


class _LanguageModel(torch.nn.Module):

    def __init__(self, with_lm_head: bool):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        if with_lm_head:
            self.lm_head = torch.nn.Linear(4, 8, bias=False)


class _Wrapper(torch.nn.Module):
    """Multimodal-style wrapper: the lm_head, if any, lives in a submodule."""

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    })

    def __init__(self, with_lm_head: bool):
        super().__init__()
        self.language_model = _LanguageModel(with_lm_head)


class _TopLevel(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": "model."})

    def __init__(self):
        super().__init__()
        self.model = _LanguageModel(with_lm_head=False)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


def test_untied_lm_head_in_submodule_is_kept():
    # e.g. Qwen3_5ForConditionalGeneration with tie_word_embeddings=False.
    model = _Wrapper(with_lm_head=True)
    _patch_tied_embedding_mapper(model)
    assert (model.hf_to_vllm_mapper.map_name("lm_head.weight") ==
            "language_model.lm_head.weight")


def test_lm_head_dropped_when_model_has_none():
    model = _Wrapper(with_lm_head=False)
    _patch_tied_embedding_mapper(model)
    assert model.hf_to_vllm_mapper.map_name("lm_head.weight") is None
    assert (model.hf_to_vllm_mapper.map_name(
        "model.language_model.embed_tokens.weight") ==
            "language_model.model.embed_tokens.weight")


def test_top_level_lm_head_untouched():
    model = _TopLevel()
    mapper = model.hf_to_vllm_mapper
    _patch_tied_embedding_mapper(model)
    assert model.hf_to_vllm_mapper is mapper
