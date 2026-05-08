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
"""Gemma-4 patches for running vLLM Gemma-4 model via TorchAX.

Why this is needed:
The vLLM `Gemma4ForConditionalGeneration` (vllm/model_executor/models/gemma4_mm.py)
implements Per-Layer Embeddings (PLE — used by E2B / E4B variants) by:

  1. Pre-allocating `self.per_layer_embeddings` as a torch zeros buffer at
     init time (gemma4_mm.py:993).
  2. Inside `embed_input_ids`, computing per-layer features and copying them
     into the buffer **in-place** with `.copy_()` (gemma4_mm.py:1322).
  3. Inside `forward`, reading `self.per_layer_embeddings[:N]` and passing
     to the language model.

The in-place `.copy_()` does not survive torchax's `__torch_dispatch__` —
the destination buffer was created by torch.zeros (no `_elem` attribute
required by torchax) but the source is a torchax tensor. The dispatch
fails with `AttributeError: 'Tensor' object has no attribute '_elem'`.

The fix:
Replace `embed_input_ids` with a version that:
  - Computes per-layer features the same way as the original.
  - **Rebinds** `self.per_layer_embeddings` (Python attribute assignment,
    not an in-place tensor op) to the freshly-computed tensor.
  - Sets `self.per_layer_embeddings = None` while calling the original
    `embed_input_ids` so the original's broken in-place branch is skipped.
  - Restores the rebound attribute after the call so `forward` reads it.

This is structurally similar to `qwen3_vl_patcher.py` (which solves the
analogous deepstack stateful-write problem), but simpler — Gemma-4's PLE
flows through `self.per_layer_embeddings` and is read as a plain tensor
in `forward`, so we don't need to pack/unpack into `inputs_embeds`.

Wire this in via `maybe_apply_gemma4_patches(vllm_model)` from
`vllm_model_wrapper.py` next to `maybe_apply_qwen3_vl_patches`.
"""

import torch

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _patched_embed_input_ids(
    vllm_model,
    orig_method,
    input_ids: torch.Tensor,
    multimodal_embeddings=None,
    *,
    is_multimodal: torch.Tensor | None = None,
):
    """Patched embed_input_ids that avoids the in-place .copy_() on
    self.per_layer_embeddings.

    Mirrors the original logic from vllm gemma4_mm.py but rebinds the
    attribute instead of doing an in-place buffer write.
    """
    # We replicate the PLE branch first, then call the original method
    # with self.per_layer_embeddings temporarily set to None so the
    # broken in-place .copy_() block at gemma4_mm.py:1320-1323 is
    # skipped.
    text_config = vllm_model.config.text_config
    has_ple = vllm_model.per_layer_embeddings is not None

    new_ple = None
    if has_ple:
        if is_multimodal is not None:
            ple_input_ids = torch.where(
                is_multimodal.to(input_ids.device, non_blocking=True),
                torch.zeros_like(input_ids),
                input_ids,
            )
        else:
            ple_input_ids = input_ids

        per_layer_inputs = (
            vllm_model.language_model.model.get_per_layer_inputs(ple_input_ids))
        if per_layer_inputs is not None:
            new_ple = per_layer_inputs.reshape(
                -1,
                text_config.num_hidden_layers,
                text_config.hidden_size_per_layer_input,
            )

    # Temporarily mask out the broken branch by setting the attribute to
    # None for the duration of the call. Restore (with the new tensor)
    # afterwards so `forward` reads the right values.
    saved_buf = vllm_model.per_layer_embeddings
    vllm_model.per_layer_embeddings = None
    try:
        if multimodal_embeddings is None or is_multimodal is None:
            result = orig_method(input_ids)
        else:
            result = orig_method(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
    finally:
        # Rebind to the freshly-computed PLE tensor (or original buffer
        # if we didn't compute one, e.g. variant without PLE).
        vllm_model.per_layer_embeddings = (new_ple
                                            if new_ple is not None else saved_buf)
    return result


def apply_gemma4_patches(vllm_model):
    """Apply Gemma-4 specific patches for stateless PLE handling."""
    cfg = getattr(vllm_model, "config", None)
    if cfg is None:
        return
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is None:
        return

    ple_dim = getattr(text_cfg, "hidden_size_per_layer_input", 0)
    if not ple_dim:
        # Server-family Gemma-4 variants (26B / 31B) — PLE is off,
        # nothing to patch.
        logger.info(
            "Gemma-4 patcher: hidden_size_per_layer_input=%s, no PLE patch needed.",
            ple_dim)
        return

    orig_embed = getattr(vllm_model, "embed_input_ids", None)
    if orig_embed is None:
        logger.warning(
            "Gemma-4 patcher: vllm_model has no embed_input_ids; skipping patch."
        )
        return

    vllm_model.embed_input_ids = (
        lambda input_ids, multimodal_embeddings=None, *, is_multimodal=None:
        _patched_embed_input_ids(
            vllm_model,
            orig_embed,
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        ))
    logger.info(
        "Gemma-4 patcher: applied PLE in-place-copy avoidance "
        "(hidden_size_per_layer_input=%d).", ple_dim)


def maybe_apply_gemma4_patches(vllm_model):
    """Conditionally apply Gemma-4 patches based on the model architecture."""
    cfg = getattr(vllm_model, "config", None)
    if cfg is None:
        return
    arches = getattr(cfg, "architectures", []) or []
    if "Gemma4ForConditionalGeneration" in arches:
        apply_gemma4_patches(vllm_model)
