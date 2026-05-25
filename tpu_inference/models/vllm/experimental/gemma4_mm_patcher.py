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
"""Gemma 4 Multimodal patches for running PLE statelessly via TorchAX/JAX."""

import torch
from torchax.interop import jax_view, torch_view


def _patched_gemma4_embed_input_ids(
    vllm_model,
    orig_embed_input_ids,
    input_ids: torch.Tensor,
    multimodal_embeddings=None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes and packs PLE statelessly at the end of embeds."""
    text_config = vllm_model.config.text_config
    ple_dim = text_config.hidden_size_per_layer_input
    num_layers = text_config.num_hidden_layers

    # 1. Temporarily disable per_layer_embeddings to bypass the stateful in-place .copy_()
    cached_ple_buffer = vllm_model.per_layer_embeddings
    vllm_model.per_layer_embeddings = None

    try:
        # 2. Call original embed_input_ids to get base text/visual embeddings safely
        inputs_embeds = orig_embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
    finally:
        # 3. Restore the original property state
        vllm_model.per_layer_embeddings = cached_ple_buffer

    # 4. Statelessly compute only Track A (embedding lookup table)
    if is_multimodal is not None:
        ple_input_ids = torch.where(
            is_multimodal.to(input_ids.device, non_blocking=True),
            torch.zeros_like(input_ids),
            input_ids,
        )
    else:
        ple_input_ids = input_ids

    # get_per_layer_inputs represents only Track A (Discrete Token ID lookup)
    per_layer_inputs = vllm_model.language_model.model.get_per_layer_inputs(
        ple_input_ids
    )

    # 5. Pack PLE values at the end of inputs_embeds along the hidden dimension (dim=-1)
    if per_layer_inputs is not None:
        per_layer_inputs = per_layer_inputs.reshape(
            -1,
            num_layers * ple_dim
        )
        per_layer_inputs = per_layer_inputs.to(inputs_embeds.device)
        inputs_embeds = torch.cat([inputs_embeds, per_layer_inputs], dim=-1)

    return inputs_embeds


def _patched_gemma4_forward(
    vllm_model,
    orig_forward,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs,
):
    """Unpacks PLE Track A values and restores them statelessly just-in-time."""
    text_config = vllm_model.config.text_config
    base_hidden_size = text_config.hidden_size
    ple_dim = text_config.hidden_size_per_layer_input
    num_layers = text_config.num_hidden_layers

    if inputs_embeds is not None and ple_dim > 0:
        expected_combined_dim = base_hidden_size + (num_layers * ple_dim)
        
        # Check if embeddings contain packed PLE features
        if inputs_embeds.shape[-1] == expected_combined_dim:
            # 1. Unpack embeddings: split base text embeds and packed PLE Track A embeds
            packed_ple = inputs_embeds[..., base_hidden_size:]
            inputs_embeds = inputs_embeds[..., :base_hidden_size]

            # 2. Reshape PLE back to [num_tokens, num_hidden_layers, hidden_size_per_layer_input]
            ple_unpacked = packed_ple.reshape(
                inputs_embeds.shape[0],
                num_layers,
                ple_dim
            )

            # 3. Cache the unpacked PLE Track A Torchax tensor back into model attribute
            vllm_model.per_layer_embeddings = ple_unpacked

    # 4. Call the original forward pass with standard text embeddings and restored buffer
    return orig_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def apply_gemma4_mm_patches(vllm_model):
    """Stateless PLE patch application for Gemma-4 MultiModal model."""
    text_config = getattr(vllm_model.config, "text_config", None)
    ple_dim = getattr(text_config, "hidden_size_per_layer_input", 0)
    if ple_dim == 0:
        return

    # Patch embed_input_ids
    orig_embed_input_ids = getattr(vllm_model, "embed_input_ids", None)
    if orig_embed_input_ids is not None:
        vllm_model.embed_input_ids = lambda *args, **kwargs: _patched_gemma4_embed_input_ids(
            vllm_model, orig_embed_input_ids, *args, **kwargs)

    # Patch forward
    orig_forward = vllm_model.forward
    vllm_model.forward = lambda *args, **kwargs: _patched_gemma4_forward(
        vllm_model, orig_forward, *args, **kwargs)


def maybe_apply_gemma4_mm_patches(vllm_model):
    """Applies patches if model is a Gemma-4 Multimodal instance."""
    if hasattr(vllm_model, "config") and hasattr(vllm_model.config, "architectures"):
        if "Gemma4ForConditionalGeneration" in vllm_model.config.architectures:
            apply_gemma4_mm_patches(vllm_model)
