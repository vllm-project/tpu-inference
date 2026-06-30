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

Deepstack Implementation in vLLM Qwen3-VL:
The vLLM model implements Deepstack by storing the deepstack features directly on the
model object (as an instance variable) after the vision encoder is executed. Later, during the
language model's forward pass, it reads these stored features directly from the model object
instead of receiving them statelessly as function arguments.

Why it Breaks JAX Compatibility:
JAX requires functions to be pure and stateless for JIT compilation (`jax.jit`).
vLLM's approach of reading hidden features from instance variables breaks this rule.

How these Patches Work:
To maintain JAX compatibility, these patches change how these features are passed,
making sure they go through standard function arguments:
1. **Getter/Setter Override**: We intercept Deepstack features and store them in a
   JAX-friendly cache (`_deepstack_tensors`) to avoid breaking JIT with PyTorch-specific
   stateful updates.
2. **Packing**: In `embed_input_ids`, we concatenate the Deepstack vision features
   to the end of the standard text embeddings (`inputs_embeds`) along the hidden dimension.
   This forces the vision features to pass through the LLM's JIT boundary as part of the
   explicit inputs.
3. **Unpacking**: In the model's `forward` pass, we split the combined embeddings
   back into text embeddings and vision features. We restore the vision features to the
   model's cache just-in-time for the execution, and proceed with the original forward pass.
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

logger = init_logger(__name__)


def _patched_set_deepstack(vllm_model, deepstack_input_embeds):
    """Intercepts Deepstack embeddings to store them in a JAX-friendly cached tensors (`_deepstack_tensors`).

    This avoids stateful updates of vLLM deepstack variables that would break JAX JIT tracing.
    """
    if deepstack_input_embeds is None:
        vllm_model._deepstack_tensors = {}
        return

    if not hasattr(vllm_model, "_deepstack_tensors"):
        vllm_model._deepstack_tensors = {}

    # Case A: We are restoring unpacked features in `_patched_forward` (passed as a dict)
    if isinstance(deepstack_input_embeds, dict):
        vllm_model._deepstack_tensors.update(deepstack_input_embeds)
    # Case B: vLLM is providing them during vision encoding (passed as a list of tensors)
    elif isinstance(deepstack_input_embeds, (list, tuple)):
        for idx, v in enumerate(deepstack_input_embeds):
            key = f"deepstack_input_embeds_{idx}"
            vllm_model._deepstack_tensors[key] = v
    elif torch.is_tensor(deepstack_input_embeds):
        if deepstack_input_embeds.ndim == 3:
            num_levels = deepstack_input_embeds.size(0)
            for idx in range(num_levels):
                key = f"deepstack_input_embeds_{idx}"
                vllm_model._deepstack_tensors[key] = deepstack_input_embeds[
                    idx]
        else:
            vllm_model._deepstack_tensors[
                "deepstack_input_embeds_0"] = deepstack_input_embeds


def _convert_to_torchax_tensor(v):
    """Converts a PyTorch tensor to a Torchax tensor, ensuring JAX compatibility."""
    if not v.__class__.__module__.startswith("torchax"):
        try:
            # Try zero-copy view first
            return torch_view(jax_view(v))
        except Exception:
            # Fallback to CPU copy if zero-copy fails
            import jax
            val_f32 = v.detach().cpu().float().numpy()
            jax_arr = jax.device_put(val_f32).astype(jax.numpy.bfloat16)
            return torch_view(jax_arr)
    return v


def _patched_get_deepstack(vllm_model, orig_get_deepstack, num_tokens: int):
    """Retrieves Deepstack embeddings, preferring JAX-compatible cached tensors.
    
    This method is called by the language model layers to retrieve the
    intermediate vision features.
    """
    # Default: Use tensors cached locally in `_deepstack_tensors`.
    # These are already converted to Torchax tensors and JIT-compatible.
    if getattr(vllm_model, "_deepstack_tensors", None):
        return IntermediateTensors(vllm_model._deepstack_tensors)

    # Fallback: Retrieve deepstack features from vLLM and convert.
    # If the cache is empty (e.g., during eager initialization), we fetch the raw vLLM
    # PyTorch tensors and convert them to Torchax tensors to ensure JIT compatibility.
    orig_output = orig_get_deepstack(num_tokens)
    if orig_output is None:
        return None
    converted = {
        k: _convert_to_torchax_tensor(v)
        for k, v in orig_output.items()
    }
    return IntermediateTensors(converted)


def _patched_embed_input_ids(vllm_model, orig_embed_input_ids, *args,
                             **kwargs):
    """Appends Deepstack features to the main text embeddings.
    
    We concatenate deepstack features to the end of the text embeddings so they can
    pass through the JIT boundary of the LLM model.
    """
    # 1. Get the base text embeddings from the native model.
    inputs_embeds = orig_embed_input_ids(*args, **kwargs)

    # 2. Check if there are any deepstack features to pack.
    deepstack_input_embeds = getattr(vllm_model, "deepstack_input_embeds",
                                     None)
    if deepstack_input_embeds is not None:
        packed = None
        try:
            # Deepstack features are typically a list of tensors (one per layer).
            # Combine the list into a single tensor.
            stacked = torch.stack(
                deepstack_input_embeds,
                dim=0)  # Shape: [num_layers, total_tokens, hidden_dim]

            # Slice to number of tokens processed in current execution step.
            # This handles cases where chunked prefill might be active.
            cur_tokens = inputs_embeds.size(0)
            packed = stacked[:, :cur_tokens, :].transpose(0, 1).reshape(
                cur_tokens, -1)
        except (TypeError, RuntimeError):
            # Fallback if it's already a single tensor (not a list).
            if torch.is_tensor(deepstack_input_embeds):
                cur_tokens = inputs_embeds.size(0)
                packed = deepstack_input_embeds[:, :cur_tokens, :].transpose(
                    0, 1).reshape(cur_tokens, -1)

        # 3. Concatenate the vision features to the end of the text embeddings.
        if packed is not None:
            packed = packed.to(inputs_embeds.device)
            inputs_embeds = torch.cat([inputs_embeds, packed], dim=-1)

    return inputs_embeds


def _patched_forward(vllm_model,
                     orig_forward,
                     input_ids,
                     positions,
                     intermediate_tensors,
                     inputs_embeds=None,
                     **kwargs):
    """Unpacks vision features from combined embeddings and restores them to model state.
    
    This reverses the packing done in `_patched_embed_input_ids` before passing
    the execution to the original model forward pass.
    """
    if inputs_embeds is not None and jax_get_pp_group().is_first_rank:
        # Check if the embeddings contain packed vision features (indicated by dimension larger than visual_dim)
        if getattr(vllm_model, "use_deepstack",
                   False) and inputs_embeds.shape[-1] > vllm_model.visual_dim:
            packed_dim = inputs_embeds.shape[-1] - vllm_model.visual_dim

            # Split combined embeddings back into text embeddings and packed vision features
            deepstack_packed = inputs_embeds[..., vllm_model.visual_dim:]
            inputs_embeds = inputs_embeds[..., :vllm_model.visual_dim]

            # Unpack the stacked vision features back into per-layer tensors
            deepstack_input_embeds = {}
            num_levels = getattr(vllm_model, "deepstack_num_level", 1)
            per_level_dim = packed_dim // num_levels
            indexes = getattr(vllm_model.config.vision_config,
                              "deepstack_visual_indexes", [])

            for idx, layer_idx in enumerate(indexes):
                start = idx * per_level_dim
                end = (idx + 1) * per_level_dim
                sliced = deepstack_packed[..., start:end]
                # Convert to torchax view for JIT compatibility
                sliced = torch_view(jax_view(sliced))
                deepstack_input_embeds[
                    f"deepstack_input_embeds_{idx}"] = sliced

            # Restore the unpacked features to the model's cache
            vllm_model._set_deepstack_input_embeds(deepstack_input_embeds)

    # Call the original forward pass with separated text embeddings
    return orig_forward(input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **kwargs)


def _patched_flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """Patched version of vLLM's `_flatten_embeddings` to prevent JAX/Torchax ZeroDivisionError.

    Why this patch is necessary:
    At model warmup/precompilation or during empty multimodal inputs, the sequence dimension
    can evaluate to 0 (e.g., shape `(num_tokens,)` where `num_tokens=0`, yielding a 1D or empty
    dimension tensor).
    
    When `_flatten_embeddings` calls standard PyTorch `embeddings.flatten(0, -2)` to flatten
    all but the last dimension, `torchax.Tensor.flatten` fails to map the negative `end_dim=-2`
    to a positive index (it only maps `end_dim=-1`). This leaves `end_dim=-2` unresolved.

    Under JAX/Torchax tracing:
    - `end_dim + 1` becomes `-1`.
    - `self._elem.shape[end_dim + 1 :]` resolves to `shape[-1:]` -> `(0,)` for a 1D or 0-size tensor.
    - The resulting target shape becomes `(-1, 0)`.
    - During shape resolution, JAX calculates the product of non-negative dimensions (`0`),
      and checks `arr.size % math.prod(other_sizes) != 0`, which evaluates to `0 % 0` and
      raises `ZeroDivisionError: integer modulo by zero`.

    By pre-converting negative `start_dim` and `end_dim` to positive index equivalents based
    on the tensor's `ndim` before calling `.flatten`, we pass positive arguments to Torchax.
    This allows it to compute the target shape correctly without producing 0-size dimensions,
    preventing the JAX compiler from crashing at startup.
    """
    if isinstance(embeddings, torch.Tensor):
        if embeddings.ndim < 2:
            return embeddings
        if embeddings.numel() == 0 or 0 in embeddings.shape:
            return embeddings.view(0, embeddings.shape[-1])
        ndim = embeddings.ndim
        start_dim = 0
        end_dim = -2
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim
        return embeddings.flatten(start_dim, end_dim)
    return torch.cat(tuple(_patched_flatten_embeddings(t) for t in embeddings))


def apply_qwen3_vl_patches(vllm_model):
    """Apply Qwen3-VL specific patches for stateless Deepstack support."""
    if not getattr(vllm_model, "use_deepstack", False):
        return

    # 1. Override deepstack setter to avoid stateful updates of native vLLM variables
    orig_set_deepstack = getattr(vllm_model, "_set_deepstack_input_embeds",
                                 None)
    if orig_set_deepstack is not None:
        vllm_model._set_deepstack_input_embeds = lambda embeds: _patched_set_deepstack(
            vllm_model, embeds)

    # 2. Override deepstack getter to prefer JAX-compatible cached tensors
    orig_get_deepstack = getattr(vllm_model, "_get_deepstack_input_embeds",
                                 None)
    if orig_get_deepstack is not None:
        vllm_model._get_deepstack_input_embeds = lambda num_tokens: _patched_get_deepstack(
            vllm_model, orig_get_deepstack, num_tokens)

    # 3. Patch embed_input_ids to pack Deepstack vision features into main text embeddings
    orig_embed_input_ids = getattr(vllm_model, "embed_input_ids", None)
    if orig_embed_input_ids is not None:
        vllm_model.embed_input_ids = lambda *args, **kwargs: _patched_embed_input_ids(
            vllm_model, orig_embed_input_ids, *args, **kwargs)

    # 4. Patch forward to unpack vision features and restore them before execution
    orig_forward = vllm_model.forward
    vllm_model.forward = lambda *args, **kwargs: _patched_forward(
        vllm_model, orig_forward, *args, **kwargs)

    # 5. Patch _flatten_embeddings in vllm utils to handle negative indexes correctly in torchax
    vllm_utils._flatten_embeddings = _patched_flatten_embeddings

    # 6. Force HAS_TRITON to False to prevent JAX/Triton active driver crash on TPU
    qwen3_vl_mod.HAS_TRITON = False


def is_qwen3_vl(vllm_model) -> bool:
    """Check if the given vLLM model is of architecture Qwen3VLForConditionalGeneration."""
    return isinstance(vllm_model, Qwen3VLForConditionalGeneration)


def maybe_apply_qwen3_vl_patches(vllm_model: nn.Module) -> None:
    if is_qwen3_vl(vllm_model):
        apply_qwen3_vl_patches(vllm_model)

        if hasattr(vllm_model, "deepstack_input_embeds"):
            # Force the deepstack placeholder buffers to the correct TPU device
            # (via Torchax/JAX) to resolve device mismatch during the forward pass.
            # This is required because the ModelForEmbedding adapter can cause
            # the standard weight loader to skip these non-parameter buffers.
            target_device = next(vllm_model.parameters()).device
            logger.info(
                f"Patching Qwen3-VL deepstack buffers to device: {target_device}"
            )
            vllm_model.deepstack_input_embeds = [
                t.to(device=target_device)
                for t in vllm_model.deepstack_input_embeds
            ]
