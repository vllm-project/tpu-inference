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
"""Gemma4 multimodal patches for running the vLLM model via torchax.

Why this exists:
vLLM's Gemma4ForConditionalGeneration caches per-layer embeddings (PLE) in a
pre-allocated instance attribute for CUDA-graph compatibility:

1. ``__init__`` creates ``self.per_layer_embeddings`` with a bare
   ``torch.zeros(...)`` — a plain attribute, NOT a registered buffer. Our
   ``shard_model_to_tpu`` only converts ``named_parameters()`` and
   ``named_buffers()`` to torchax tensors, so this attribute stays a plain
   CPU ``torch.Tensor``, and ``embed_input_ids``'s
   ``self.per_layer_embeddings[:n].copy_(...)`` crashes inside torchax's
   ``_aten_copy`` with ``AttributeError: 'Tensor' object has no attribute
   '_elem'`` (destination has no jax array behind it).
2. Even with the buffer converted, the write cannot work here: writes into
   a slice-view mutate a temporary torchax tensor, and the jitted forward
   receives ``params_and_buffers`` materialized once at load time — the
   stateful write never crosses the jit boundary.

Fix: drop the buffer and compute PLE *inside* the jitted forward from
``input_ids`` (which the forward already receives as a jit argument). This
is a pure embedding lookup and reproduces the reference computation exactly:
``embed_input_ids`` masks multimodal placeholder positions to token 0 before
calling ``get_per_layer_inputs``; those positions are exactly the ones whose
``input_ids`` equal the image/audio placeholder token ids, so masking by
token id is equivalent.
"""

from types import MethodType

import torch
import torch.nn as nn
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.sequence import IntermediateTensors

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _ple_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: object,
) -> IntermediateTensors:
    """Replaces Gemma4ForConditionalGeneration.forward on the TPU path.

    Identical to the vLLM forward except PLE is computed inline from
    input_ids instead of being read from the pre-allocated
    ``per_layer_embeddings`` CUDA-graph buffer (see module docstring).

    Provenance, for review against the upstream source
    (vllm/model_executor/models/gemma4_mm.py): everything here is copied
    verbatim from ``Gemma4ForConditionalGeneration.forward`` except the
    ``per_layer_inputs`` block, which replaces upstream's buffer read
    (``self.per_layer_embeddings[:n]``) with the producer computation
    copied from ``embed_input_ids``. Each section is marked below.
    """
    # [upstream forward, verbatim]
    if intermediate_tensors is not None:
        inputs_embeds = None

    # [replaces upstream forward's buffer read] Upstream forward does
    #   per_layer_inputs = self.per_layer_embeddings[:inputs_embeds.shape[0]]
    #   (guarded on per_layer_embeddings/inputs_embeds not None).
    # Here PLE is computed inline instead, under the equivalent guard
    # (input_ids replaces the buffer as the data source).
    per_layer_inputs = None
    if inputs_embeds is not None and input_ids is not None:
        # [new — replaces upstream embed_input_ids' is_multimodal mask]
        # Upstream masks multimodal placeholder positions to token 0 via the
        # is_multimodal tensor, which forward does not receive. Those
        # positions hold the image/audio placeholder token ids, so masking
        # by token id selects the same positions.
        ple_input_ids = input_ids
        for token_id in self._tpu_ple_mask_token_ids:
            ple_input_ids = torch.where(
                ple_input_ids == token_id,
                torch.zeros_like(ple_input_ids),
                ple_input_ids,
            )
        # [upstream embed_input_ids, verbatim] The PLE producer computation
        # (lookup + reshape), minus the final buffer copy_.
        per_layer_inputs = self.language_model.model.get_per_layer_inputs(
            ple_input_ids)
        if per_layer_inputs is not None:
            per_layer_inputs = per_layer_inputs.reshape(
                -1,
                self.config.text_config.num_hidden_layers,
                self.config.text_config.hidden_size_per_layer_input,
            )

    # [upstream forward, verbatim]
    self._clear_mm_prefix_for_full_attn_layers()

    hidden_states = self.language_model.model(
        input_ids,
        positions,
        per_layer_inputs=per_layer_inputs,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return hidden_states


def apply_gemma4_mm_patches(vllm_model: nn.Module) -> None:
    # Dropping the buffer makes embed_input_ids skip its
    # per_layer_embeddings[:n].copy_(...) block (guarded by
    # ``if self.per_layer_embeddings is not None``), which would crash on
    # the plain CPU tensor under torchax.
    vllm_model.per_layer_embeddings = None

    mask_token_ids = []
    for attr in ("image_token_id", "audio_token_id"):
        token_id = getattr(vllm_model.config, attr, None)
        if token_id is not None:
            mask_token_ids.append(token_id)
    vllm_model._tpu_ple_mask_token_ids = tuple(mask_token_ids)

    vllm_model.forward = MethodType(_ple_forward, vllm_model)
    logger.info(
        "[gemma4-patch] Replaced per_layer_embeddings CUDA-graph buffer with "
        "inline PLE computation in forward (mask_token_ids=%s).",
        mask_token_ids,
    )


def maybe_apply_gemma4_mm_patches(vllm_model: nn.Module) -> None:
    if not isinstance(vllm_model, Gemma4ForConditionalGeneration):
        return
    ple_dim = getattr(vllm_model.config.text_config,
                      "hidden_size_per_layer_input", None)
    if ple_dim is None or ple_dim <= 0:
        # Variant without PLE (e.g. 26B/31B): per_layer_embeddings is None
        # and the forward's buffer read is already skipped. Nothing to patch.
        return
    apply_gemma4_mm_patches(vllm_model)
