# Copyright 2026 Google LLC
"""Gemma-4 Patches for running vLLM Gemma-4 model via TorchAX on TPU with full PLE support."""

import torch
import torch.accelerator
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _safe_get_memory_info(*args, **kwargs):
    """Fallback for TPU where PyTorch accelerator memory queries are not supported."""
    return (32 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024)


def _patched_gemma4_embed_input_ids(vllm_model, orig_embed_input_ids, input_ids, *args, **kwargs):
    """Computes PLE Table Lookup and packs it into inputs_embeds across the JIT boundary."""
    inputs_embeds = orig_embed_input_ids(input_ids, *args, **kwargs)

    text_cfg = getattr(vllm_model.config, "text_config", vllm_model.config)
    ple_dim = getattr(text_cfg, "hidden_size_per_layer_input", None)
    if ple_dim is None or ple_dim <= 0:
        return inputs_embeds

    is_multimodal = kwargs.get("is_multimodal", None)
    if is_multimodal is not None:
        ple_input_ids = torch.where(
            is_multimodal.to(input_ids.device, non_blocking=True),
            torch.zeros_like(input_ids),
            input_ids,
        )
    else:
        ple_input_ids = input_ids

    per_layer_inputs = vllm_model.language_model.model.get_per_layer_inputs(ple_input_ids)
    if per_layer_inputs is None:
        return inputs_embeds

    total_ple_dim = text_cfg.num_hidden_layers * ple_dim
    per_layer_flat = per_layer_inputs.reshape(inputs_embeds.shape[0], total_ple_dim)
    packed = torch.cat([inputs_embeds, per_layer_flat], dim=-1)

    print(f"\n[TPU VERIFY: PACKING] Base Embeds: {inputs_embeds.shape} + PLE Table: {per_layer_flat.shape} -> Packed: {packed.shape}\n", flush=True)
    return packed


def _patched_gemma4_forward(
    vllm_model,
    orig_forward,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    **kwargs,
):
    """Unpacks PLE Table Lookup and forwards directly to language model without duplicate kwargs."""
    text_cfg = getattr(vllm_model.config, "text_config", vllm_model.config)
    ple_dim = getattr(text_cfg, "hidden_size_per_layer_input", None)
    hidden_size = getattr(text_cfg, "hidden_size", 1536)

    per_layer_inputs = None
    if inputs_embeds is not None and ple_dim is not None and ple_dim > 0:
        total_ple_dim = text_cfg.num_hidden_layers * ple_dim
        expected_packed_dim = hidden_size + total_ple_dim

        if inputs_embeds.shape[-1] == expected_packed_dim:
            base_embeds = inputs_embeds[..., :hidden_size]
            ple_flat = inputs_embeds[..., hidden_size:]
            per_layer_inputs = ple_flat.reshape(
                *inputs_embeds.shape[:-1],
                text_cfg.num_hidden_layers,
                ple_dim,
            )
            inputs_embeds = base_embeds
            print(f"\n[TPU VERIFY: UNPACKING] Successfully unpacked -> Base Embeds: {inputs_embeds.shape} | PLE Inputs: {per_layer_inputs.shape}\n", flush=True)

    if intermediate_tensors is not None:
        inputs_embeds = None

    if hasattr(vllm_model, "_clear_mm_prefix_for_full_attn_layers"):
        vllm_model._clear_mm_prefix_for_full_attn_layers()

    # Forward directly with unpacked per_layer_inputs
    return vllm_model.language_model.model(
        input_ids,
        positions,
        per_layer_inputs=per_layer_inputs,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def maybe_apply_gemma4_patches(vllm_model) -> None:
    """Apply Gemma-4 specific patches for TorchAX TPU execution."""
    # 1. Patch torch.accelerator.get_memory_info for TPU compatibility
    try:
        torch.accelerator.get_memory_info()
    except Exception:
        torch.accelerator.get_memory_info = _safe_get_memory_info
        logger.info("Patched torch.accelerator.get_memory_info for TPU execution.")

    # 2. Disable the static CUDA buffer that breaks XLA
    if hasattr(vllm_model, "per_layer_embeddings"):
        vllm_model.per_layer_embeddings = None
        logger.info("Disabled static CUDA PLE buffer for TPU execution.")

    # 3. Patch embed_input_ids and forward for functional pack/unpack
    if hasattr(vllm_model, "embed_input_ids") and not getattr(vllm_model, "_is_gemma4_patched", False):
        orig_embed = vllm_model.embed_input_ids
        orig_forward = vllm_model.forward

        vllm_model.embed_input_ids = lambda *a, **kw: _patched_gemma4_embed_input_ids(
            vllm_model, orig_embed, *a, **kw
        )
        vllm_model.forward = lambda *a, **kw: _patched_gemma4_forward(
            vllm_model, orig_forward, *a, **kw
        )
        vllm_model._is_gemma4_patched = True
        logger.info("Applied Gemma-4 functional PLE pack/unpack patch for TPU.")