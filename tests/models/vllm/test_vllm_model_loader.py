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
"""Unit tests for `_patch_tied_embedding_mapper` in the vLLM model loader."""

import torch
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.models.vllm import vllm_model_loader

_LM_HEAD = "lm_head.weight"


class _LanguageModel(torch.nn.Module):
    """Text model with its own embedding and an untied `lm_head`."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class _MultimodalWrapper(torch.nn.Module):
    """Keeps `lm_head` on the inner language model, like Qwen3.5's wrapper."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        })

    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()


class _CausalLM(torch.nn.Module):
    """Text model with `lm_head` on the top-level module."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."})

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class _TiedWithoutLmHead(torch.nn.Module):
    """Ties word embeddings and defines no `lm_head` module."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."})

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)


class _MultimodalTiedWithoutLmHead(torch.nn.Module):
    """Multimodal wrapper around a tied model that has no `lm_head`."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "language_model.model."})

    def __init__(self):
        super().__init__()
        self.language_model = _TiedWithoutLmHead()


def test_keeps_nested_lm_head_weights():
    model = _MultimodalWrapper()

    vllm_model_loader._patch_tied_embedding_mapper(model)

    assert (model.hf_to_vllm_mapper.map_name(_LM_HEAD) ==
            "language_model.lm_head.weight")


def test_keeps_top_level_lm_head_weights():
    model = _CausalLM()

    vllm_model_loader._patch_tied_embedding_mapper(model)

    assert model.hf_to_vllm_mapper.map_name(_LM_HEAD) == _LM_HEAD


def test_skips_lm_head_weights_without_lm_head():
    model = _TiedWithoutLmHead()

    vllm_model_loader._patch_tied_embedding_mapper(model)

    assert model.hf_to_vllm_mapper.map_name(_LM_HEAD) is None
    assert (model.hf_to_vllm_mapper.map_name(
        "model.language_model.embed_tokens.weight") ==
            "model.embed_tokens.weight")


def test_skips_lm_head_weights_without_nested_lm_head():
    model = _MultimodalTiedWithoutLmHead()

    vllm_model_loader._patch_tied_embedding_mapper(model)

    assert model.hf_to_vllm_mapper.map_name(_LM_HEAD) is None


def test_ignores_models_without_mapper():
    model = torch.nn.Module()

    vllm_model_loader._patch_tied_embedding_mapper(model)

    assert not hasattr(model, "hf_to_vllm_mapper")
