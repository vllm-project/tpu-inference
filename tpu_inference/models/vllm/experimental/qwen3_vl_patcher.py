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
"""Qwen3-VL Patches for running vLLM Qwen3-VL model via TorchAX.

This file provides patches to make the Deepstack feature in vLLM Qwen3-VL compatible
with JIT compilation on TPUs.
"""

import torch
import torch.nn as nn
import vllm.model_executor.models.qwen3_vl as qwen3_vl_mod
import vllm.model_executor.models.utils as vllm_utils
from torchax.interop import jax_view, torch_view
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.multimodal import NestedTensors
from vllm.sequence import IntermediateTensors

from tpu_inference.distributed.jax_parallel_state import \
    get_pp_group as jax_get_pp_group
from tpu_inference.logger import init_logger
from tpu_inference.utils import t2j

logger = init_logger(__name__)


def _patched_set_deepstack(vllm_model, deepstack_input_embeds):
    """Intercepts Deepstack embeddings to store them in a JAX-friendly cached tensors."""
    if deepstack_input_embeds is None:
        vllm_model._deepstack_tensors = {}
        return

    if not hasattr(vllm_model, "_deepstack_tensors"):
        vllm_model._deepstack_tensors = {}

    if isinstance(deepstack_input_embeds, dict):
        vllm_model._deepstack_tensors.update(deepstack_input_embeds)
    elif isinstance(deepstack_input_embeds, (list, tuple)):
        for idx, v in enumerate(deepstack_input_embeds):
            vllm_model._deepstack_tensors[f"deepstack_input_embeds_{idx}"] = v


def _patched_get_deepstack(vllm_model, orig_get_deepstack, num_tokens: int):
    """Retrieves Deepstack embeddings, preferring JAX-compatible cached tensors."""
    if getattr(vllm_model, "_deepstack_tensors", None):
        return IntermediateTensors(vllm_model._deepstack_tensors)

    orig_output = orig_get_deepstack(num_tokens)
    if orig_output is None:
        return None
    converted = {
        k:
        torch_view(t2j(v, use_dlpack=False))
        if not v.__class__.__module__.startswith("torchax") else v
        for k, v in orig_output.items()
    }
    return IntermediateTensors(converted)


def _patched_embed_input_ids(vllm_model, orig_embed_input_ids, *args,
                             **kwargs):
    """Appends Deepstack features to the main text embeddings."""
    inputs_embeds = orig_embed_input_ids(*args, **kwargs)

    deepstack_input_embeds = getattr(vllm_model, "deepstack_input_embeds",
                                     None)
    if deepstack_input_embeds is not None:
        try:
            import jax.numpy as jnp
            stacked = torch.stack(deepstack_input_embeds, dim=0)
            cur_tokens = inputs_embeds.size(0)
            packed = stacked[:, :cur_tokens, :].transpose(0, 1).reshape(
                cur_tokens, -1)

            # Concatenate using raw JAX arrays to avoid mixed-math dispatcher errors.
            jax_inputs = jax_view(inputs_embeds)
            jax_packed = t2j(packed, use_dlpack=False)
            jax_combined = jnp.concatenate([jax_inputs, jax_packed], axis=-1)
            inputs_embeds = torch_view(jax_combined)

        except (TypeError, RuntimeError):
            pass

    return inputs_embeds


def _patched_forward(vllm_model,
                     orig_forward,
                     input_ids,
                     positions,
                     intermediate_tensors,
                     inputs_embeds=None,
                     **kwargs):
    """Unpacks vision features and ensures metadata is passed as Tensors."""
    if inputs_embeds is not None and jax_get_pp_group().is_first_rank:
        if getattr(vllm_model, "use_deepstack",
                   False) and inputs_embeds.shape[-1] > vllm_model.visual_dim:
            packed_dim = inputs_embeds.shape[-1] - vllm_model.visual_dim

            jax_inputs_embeds = jax_view(inputs_embeds)
            jax_deepstack_packed = jax_inputs_embeds[...,
                                                     vllm_model.visual_dim:]
            jax_inputs_embeds = jax_inputs_embeds[..., :vllm_model.visual_dim]

            deepstack_input_embeds = {}
            num_levels = getattr(vllm_model, "deepstack_num_level", 1)
            per_level_dim = packed_dim // num_levels
            indexes = getattr(vllm_model.config.vision_config,
                              "deepstack_visual_indexes", [])

            for idx, _ in enumerate(indexes):
                start = idx * per_level_dim
                end = (idx + 1) * per_level_dim
                deepstack_input_embeds[
                    f"deepstack_input_embeds_{idx}"] = torch_view(
                        jax_deepstack_packed[..., start:end])

            inputs_embeds = torch_view(jax_inputs_embeds)
            vllm_model._set_deepstack_input_embeds(deepstack_input_embeds)

    # Bare minimum fix for XLA segfault on v7x: convert JIT-friendly tuples back to Tensors
    # before calling the original model. Standard Tensors have memory buffers XLA can verify.
    for k in ("image_grid_thw", "video_grid_thw", "grid_thw"):
        if k in kwargs and isinstance(kwargs[k], tuple):
            kwargs[k] = torch.tensor(kwargs[k], device=input_ids.device)

    return orig_forward(input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **kwargs)


def _patched_flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """Patched version of vLLM's `_flatten_embeddings` for Torchax/JAX compatibility."""
    if isinstance(embeddings, torch.Tensor):
        if embeddings.ndim < 2:
            return embeddings
        if embeddings.numel() == 0 or 0 in embeddings.shape:
            return embeddings.view(0, embeddings.shape[-1])
        ndim = embeddings.ndim
        return embeddings.flatten(0, ndim - 2)
    return torch.cat(tuple(_patched_flatten_embeddings(t) for t in embeddings))


def apply_qwen3_vl_patches(vllm_model):
    """Apply Qwen3-VL specific patches."""
    if not getattr(vllm_model, "use_deepstack", False):
        return

    vllm_model._set_deepstack_input_embeds = lambda embeds: _patched_set_deepstack(
        vllm_model, embeds)

    orig_get_deepstack = getattr(vllm_model, "_get_deepstack_input_embeds",
                                 None)
    if orig_get_deepstack is not None:
        vllm_model._get_deepstack_input_embeds = lambda num_tokens: _patched_get_deepstack(
            vllm_model, orig_get_deepstack, num_tokens)

    orig_embed_input_ids = getattr(vllm_model, "embed_input_ids", None)
    if orig_embed_input_ids is not None:
        vllm_model.embed_input_ids = lambda *args, **kwargs: _patched_embed_input_ids(
            vllm_model, orig_embed_input_ids, *args, **kwargs)

    orig_forward = vllm_model.forward
    vllm_model.forward = lambda *args, **kwargs: _patched_forward(
        vllm_model, orig_forward, *args, **kwargs)

    # 4. Patch embed_multimodal to convert JIT-friendly tuples back to Tensors
    # for model-internal processing (avoids AttributeError: tuple has no attr ndim).
    orig_embed_mm = vllm_model.embed_multimodal

    def _patched_embed_mm(**kwargs):
        for k in ("image_grid_thw", "video_grid_thw", "grid_thw"):
            if k in kwargs and isinstance(kwargs[k], tuple):
                kwargs[k] = torch.tensor(kwargs[k],
                                         device=next(
                                             vllm_model.parameters()).device)
        return orig_embed_mm(**kwargs)

    vllm_model.embed_multimodal = _patched_embed_mm

    vllm_utils._flatten_embeddings = _patched_flatten_embeddings
    qwen3_vl_mod.HAS_TRITON = False


def is_qwen3_vl(vllm_model) -> bool:
    """Check if the given vLLM model is of architecture Qwen3VLForConditionalGeneration."""
    return isinstance(vllm_model, Qwen3VLForConditionalGeneration)


def maybe_apply_qwen3_vl_patches(vllm_model: nn.Module) -> None:
    if is_qwen3_vl(vllm_model):
        apply_qwen3_vl_patches(vllm_model)

        if hasattr(vllm_model, "deepstack_input_embeds"):
            target_device = next(vllm_model.parameters()).device
            vllm_model.deepstack_input_embeds = [
                t.to(device=target_device)
                for t in vllm_model.deepstack_input_embeds
            ]
