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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper


def _build_wrapper(vllm_config):
    wrapper = object.__new__(VllmModelWrapper)
    wrapper.vllm_config = vllm_config
    wrapper.model_config = vllm_config.model_config
    wrapper.mesh = MagicMock()
    wrapper.is_draft_model = False
    return wrapper


@patch('tpu_inference.models.vllm.vllm_model_wrapper.envs.NEW_MODEL_DESIGN',
       True)
def test_load_weights_keeps_expert_parallel_enabled_for_new_model_design():
    vllm_config = MagicMock()
    vllm_config.device_config = SimpleNamespace(device='jax', slice=None)
    vllm_config.model_config = SimpleNamespace(dtype=torch.float32,
                                               is_multimodal_model=False)
    vllm_config.compilation_config = SimpleNamespace(static_forward_context={},
                                                     static_all_moe_layers=[])
    vllm_config.parallel_config = SimpleNamespace(enable_expert_parallel=True)
    # Opt in to the PR's streaming loader (default is 'auto' / upstream
    # DefaultLoader, which OOMs on 744 GB FP8 MoE checkpoints; the PR's
    # tpu_streaming_loader path is the one the test actually covers).
    vllm_config.load_config = SimpleNamespace(
        load_format='tpu_streaming_loader')
    vllm_config.lora_config = None
    vllm_config.speculative_config = None

    wrapper = _build_wrapper(vllm_config)

    with patch('tpu_inference.models.vllm.vllm_model_wrapper.copy.deepcopy',
               return_value=vllm_config), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.nullcontext',
               side_effect=lambda: nullcontext()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.jax.default_device',
               side_effect=lambda *_args, **_kwargs: nullcontext()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.jax.devices',
               return_value=[MagicMock()]), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.set_current_vllm_config',
               side_effect=lambda *_args, **_kwargs: nullcontext()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.vllm_get_model',
               return_value=MagicMock()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.shard_model_to_tpu',
               return_value=MagicMock()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper._VllmRunner',
               side_effect=lambda model: SimpleNamespace(pooler=None,
                                                         vllm_model=model)), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.maybe_apply_qwen3_vl_patches',
               return_value=None):
        wrapper.load_weights()

    assert vllm_config.parallel_config.enable_expert_parallel is True


@patch('tpu_inference.models.vllm.vllm_model_wrapper.envs.NEW_MODEL_DESIGN',
       False)
def test_load_weights_disables_expert_parallel_for_legacy_path():
    vllm_config = MagicMock()
    vllm_config.device_config = SimpleNamespace(device='jax', slice=None)
    vllm_config.model_config = SimpleNamespace(dtype=torch.float32,
                                               is_multimodal_model=False)
    vllm_config.compilation_config = SimpleNamespace(static_forward_context={},
                                                     static_all_moe_layers=[])
    vllm_config.parallel_config = SimpleNamespace(enable_expert_parallel=True)
    # Opt in to the PR's streaming loader (default is 'auto' / upstream
    # DefaultLoader, which OOMs on 744 GB FP8 MoE checkpoints; the PR's
    # tpu_streaming_loader path is the one the test actually covers).
    vllm_config.load_config = SimpleNamespace(
        load_format='tpu_streaming_loader')
    vllm_config.lora_config = None
    vllm_config.speculative_config = None

    wrapper = _build_wrapper(vllm_config)

    with patch('tpu_inference.models.vllm.vllm_model_wrapper.copy.deepcopy',
               return_value=vllm_config), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.nullcontext',
               side_effect=lambda: nullcontext()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.jax.default_device',
               side_effect=lambda *_args, **_kwargs: nullcontext()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.jax.devices',
               return_value=[MagicMock()]), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.set_current_vllm_config',
               side_effect=lambda *_args, **_kwargs: nullcontext()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.vllm_get_model',
               return_value=MagicMock()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.shard_model_to_tpu',
               return_value=MagicMock()), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper._VllmRunner',
               side_effect=lambda model: SimpleNamespace(pooler=None,
                                                         vllm_model=model)), \
         patch('tpu_inference.models.vllm.vllm_model_wrapper.maybe_apply_qwen3_vl_patches',
               return_value=None):
        wrapper.load_weights()

    assert vllm_config.parallel_config.enable_expert_parallel is False
