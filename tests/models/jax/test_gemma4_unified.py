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
"""Gemma4UnifiedForConditionalGeneration (Gemma 4 12B) on the JAX path."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tpu_inference.models.common.model_loader import _get_model_architecture
from tpu_inference.models.jax.gemma4 import Gemma4ForCausalLM
from tpu_inference.models.jax.gemma4_unified import \
    Gemma4UnifiedTextForCausalLM


def _vllm_config(limits, token_ids=("image", "audio", "video")):
    cfg = MagicMock()
    cfg.model_config.multimodal_config.limit_per_prompt = {
        k: SimpleNamespace(count=v)
        for k, v in limits.items()
    }
    # The 12B config declares image, audio and video placeholder tokens.
    cfg.model_config.hf_config = SimpleNamespace(**{
        f"{m}_token_id": 100 + i
        for i, m in enumerate(token_ids)
    })
    return cfg


def test_unified_architecture_is_registered():
    config = SimpleNamespace(
        architectures=["Gemma4UnifiedForConditionalGeneration"])
    assert _get_model_architecture(config) is Gemma4UnifiedTextForCausalLM
    assert issubclass(Gemma4UnifiedTextForCausalLM, Gemma4ForCausalLM)


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {
            "image": 1,
            "audio": 0,
            "video": 0
        },
        {
            "image": 0,
            "audio": 0,
            "video": 2
        },
        # vLLM treats an unlisted modality as unlimited: video stays open.
        {
            "image": 0,
            "audio": 0
        },
    ])
def test_multimodal_limits_are_rejected(limits):
    with pytest.raises(NotImplementedError, match="text-only"):
        Gemma4UnifiedTextForCausalLM(_vllm_config(limits), MagicMock(),
                                     MagicMock())


def test_only_declared_modalities_must_be_listed():
    with patch.object(Gemma4ForCausalLM, "__init__", return_value=None):
        Gemma4UnifiedTextForCausalLM(
            _vllm_config({"image": 0}, token_ids=("image", )), "rng", "mesh")


def test_all_zero_limits_build_the_decoder():
    with patch.object(Gemma4ForCausalLM, "__init__",
                      return_value=None) as base_init:
        cfg = _vllm_config({"image": 0, "audio": 0, "video": 0})
        Gemma4UnifiedTextForCausalLM(cfg, "rng", "mesh")
    base_init.assert_called_once_with(cfg, "rng", "mesh")
