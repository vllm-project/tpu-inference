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

import os

import torch
import torch.nn as nn
import vllm.model_executor.models.qwen3_vl as qwen3_vl_mod
import vllm.model_executor.models.utils as vllm_utils
from torchax.interop import jax_view, torch_view
from vllm.config import get_current_vllm_config
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
    # The tensors may be torchax.view.View objects (produced by slicing inside the
    # JIT-compiled mm-encoder). Materialize them into proper torchax.tensor.Tensor
    # so that _deepstack_tensors never contains Views, which break AOT lower.
    elif isinstance(deepstack_input_embeds, (list, tuple)):
        for idx, v in enumerate(deepstack_input_embeds):
            key = f"deepstack_input_embeds_{idx}"
            vllm_model._deepstack_tensors[key] = _convert_to_torchax_tensor(v)
    elif torch.is_tensor(deepstack_input_embeds):
        # Slicing a 3D torchax Tensor produces View objects. Materialize each
        # slice into a proper torchax.tensor.Tensor so _deepstack_tensors never
        # holds Views, which cause TypeError during JAX AOT lower.
        if deepstack_input_embeds.ndim == 3:
            num_levels = deepstack_input_embeds.size(0)
            for idx in range(num_levels):
                key = f"deepstack_input_embeds_{idx}"
                vllm_model._deepstack_tensors[
                    key] = _convert_to_torchax_tensor(
                        deepstack_input_embeds[idx])
        else:
            vllm_model._deepstack_tensors[
                "deepstack_input_embeds_0"] = _convert_to_torchax_tensor(
                    deepstack_input_embeds)


def _convert_to_torchax_tensor(v):
    """Converts a tensor to a proper torchax.tensor.Tensor, materializing Views.

    torchax.view.View objects (produced by slicing inside JAX JIT) hold stale
    JAX abstract tracers from the mm-encoder's completed trace. They must be
    materialized into torchax.tensor.Tensor (concrete JAX arrays) before being
    stored in _deepstack_tensors, otherwise using them in the LLM's AOT lower
    trace raises TypeError (stale tracer from a dead JAX scope).
    """
    try:
        # Always run through torch_view(jax_view(v)) — this materializes both
        # regular torch.Tensor and torchax.view.View into torchax.tensor.Tensor.
        return torch_view(jax_view(v))
    except Exception:
        if not v.__class__.__module__.startswith("torchax"):
            # Fallback: CPU copy for non-JAX tensors that can't be zero-copy viewed
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
    # Escape hatch: DISABLE_QWEN3_VL_DEEPSTACK=1 disables deepstack entirely
    # by returning None here, which causes the LM to skip the
    # `hidden_states + deepstack_input_embeds[key]` add at
    # `qwen3_vl_moe.py:118`. Deepstack currently trips a torchax/dynamo
    # `TypeError: unsupported operand type(s) for +: 'View' and 'Tensor'`
    # during the backbone-with-embeds precompile, so this is the only way
    # to reach warmup completion on that code path. Vision quality
    # degrades because the deepstack residual is dropped, but the server
    # comes up.
    if os.environ.get("DISABLE_QWEN3_VL_DEEPSTACK", "0") == "1":
        return None
    # Default: Use tensors cached locally in `_deepstack_tensors`.
    # These are already converted to Torchax tensors and JIT-compatible.
    if getattr(vllm_model, "_deepstack_tensors", None):
        return IntermediateTensors(vllm_model._deepstack_tensors)

    # Fallback: Retrieve deepstack features from vLLM and convert.
    # If the cache is empty (e.g., during h_size=visual_dim compilation tasks that
    # don't have packed deepstack), fetch the vLLM placeholder buffers and convert
    # them to torchax.Tensors for JIT compatibility.
    orig_output = orig_get_deepstack(num_tokens)
    if orig_output is None:
        return None
    return IntermediateTensors({
        k: _convert_to_torchax_tensor(v)
        for k, v in orig_output.items()
    })


def _patched_embed_input_ids(vllm_model, orig_embed_input_ids, *args,
                             **kwargs):
    """Appends Deepstack features to the main text embeddings.

    We concatenate deepstack features to the end of the text embeddings so they can
    pass through the JIT boundary of the LLM model.
    """
    # 1. Get the base text embeddings from the native model.
    inputs_embeds = orig_embed_input_ids(*args, **kwargs)

    # Escape hatch: DISABLE_QWEN3_VL_DEEPSTACK=1 short-circuits packing so
    # the JIT signature stays visual_dim wide and the LM never sees a
    # non-None deepstack argument.
    if os.environ.get("DISABLE_QWEN3_VL_DEEPSTACK", "0") == "1":
        return inputs_embeds

    # 2. Check if there are any deepstack features to pack.
    # Read from _deepstack_tensors (our JAX-compatible cache) rather than
    # vllm_model.deepstack_input_embeds (the pre-allocated placeholder buffer that
    # is never updated, because our patched _set_deepstack_input_embeds stores to
    # _deepstack_tensors instead of doing the in-place copy_ that breaks JAX JIT).
    deepstack_cache = getattr(vllm_model, "_deepstack_tensors", None)
    if not deepstack_cache:
        return inputs_embeds

    num_levels = getattr(vllm_model, "deepstack_num_level", 1)
    cur_tokens = inputs_embeds.size(0)

    level_tensors = []
    for idx in range(num_levels):
        key = f"deepstack_input_embeds_{idx}"
        if key not in deepstack_cache:
            return inputs_embeds  # incomplete cache, skip packing
        v = deepstack_cache[key]
        sliced = v[:cur_tokens]
        # Slicing a torchax Tensor produces a View; materialize it so torch.stack works.
        if sliced.__class__.__module__.startswith("torchax"):
            sliced = _convert_to_torchax_tensor(sliced)
        level_tensors.append(sliced)

    try:
        # 3. Stack levels and reshape: [num_levels, cur_tokens, hdim] → [cur_tokens, num_levels*hdim]
        stacked = torch.stack(level_tensors, dim=0)
        packed = stacked.transpose(0, 1).reshape(cur_tokens, -1)
        try:
            packed = packed.to(inputs_embeds.device)
        except Exception:
            pass  # tensors already on the correct device
        inputs_embeds = torch.cat([inputs_embeds, packed], dim=-1)
    except (TypeError, RuntimeError) as e:
        logger.warning(
            "_patched_embed_input_ids: failed to pack deepstack: %s", e)

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
    if os.environ.get("DISABLE_QWEN3_VL_DEEPSTACK", "0") == "1":
        # Escape hatch: clear any deepstack tensor stash so
        # `orig_forward -> self._get_deepstack_input_embeds(...)` returns
        # None, and the LM's `hidden_states + deepstack_input_embeds[key]`
        # add at qwen3_vl_moe.py:118 is skipped entirely. torch.compile
        # may inline the original bound method rather than our monkey-
        # patched lambda, so patching `_get_deepstack_input_embeds` alone
        # isn't sufficient — clearing the state at the outer forward
        # ensures the LM sees None regardless of how the getter resolves.
        vllm_model._deepstack_tensors = {}
        if hasattr(vllm_model, "deepstack_input_embeds"):
            vllm_model.deepstack_input_embeds = None
        if hasattr(vllm_model, "deepstack_input_embeds_num_tokens"):
            vllm_model.deepstack_input_embeds_num_tokens = 0
    if inputs_embeds is not None and jax_get_pp_group().is_first_rank:
        if getattr(vllm_model, "use_deepstack", False):
            if inputs_embeds.shape[-1] > vllm_model.visual_dim:
                # Detect whether we are inside a JAX JIT trace by inspecting the tensor
                # type. During JAX JIT, inputs_embeds is a torchax type (Tensor or View).
                # In eager (non-JIT) calls, it is a plain torch.Tensor. This distinction
                # is critical: torch_view(jax_view(slice)) on an eager torch.Tensor
                # produces a torchax.Tensor that is incompatible with the eager
                # hidden_states (plain torch.Tensor) inside the language model, causing:
                #   TypeError: unsupported operand type(s) for +: 'Tensor' and 'Tensor'
                is_torchax_context = inputs_embeds.__class__.__module__.startswith(
                    "torchax")

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
                    if is_torchax_context:
                        # Inside JAX JIT: convert View → Tensor so the JAX trace
                        # sees concrete operations (not View arithmetic).
                        sliced = torch_view(jax_view(sliced))
                    # In eager mode: keep as plain torch.Tensor so hidden_states + sliced
                    # remains a torch.Tensor operation (no type mismatch).
                    deepstack_input_embeds[
                        f"deepstack_input_embeds_{idx}"] = sliced

                # Restore the unpacked features to the model's cache
                vllm_model._set_deepstack_input_embeds(deepstack_input_embeds)
            else:
                # No deepstack packing in this call (inputs_embeds.shape[-1] == visual_dim).
                # Clear any stale torchax.Tensors left by a previous h_size=16384
                # compilation task. Without this, _patched_get_deepstack returns those
                # concrete torchax.Tensors into the new JAX trace, causing:
                #   TypeError: unsupported operand type(s) for +: 'Tensor' and 'Tensor'
                # at the deepstack addition (qwen3_vl.py:1546).
                vllm_model._deepstack_tensors = {}

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


def _patched_get_encoder_cudagraph_config(vllm_model, orig_fn):
    """Filter modalities down to those actually enabled by the user.

    Upstream `get_encoder_cudagraph_config` hard-codes
    `modalities = ["image", "video"]` and then unconditionally calls
    `self.get_max_frames_per_video()` when 'video' is in that list. On
    workloads that disable video (e.g. `--limit-mm-per-prompt
    '{"video":0}'`) the video path is a dead branch that still crashes
    `get_max_frames_per_video` because `self.model_config` is not
    initialized on `Qwen3VLMoeForConditionalGeneration` (its `__init__`
    calls `super(Qwen3VLForConditionalGeneration, self).__init__()`
    which SKIPS `Qwen3VLForConditionalGeneration.__init__` where
    `self.model_config = vllm_config.model_config` would happen). Drop
    disabled modalities BEFORE calling the original so both problems
    disappear together.
    """
    vllm_config = get_current_vllm_config()
    mm_config = getattr(vllm_config.model_config, "multimodal_config", None)
    if mm_config is not None:
        # `mm_config.get_limit_per_prompt` returns a plain int regardless
        # of whether `limit_per_prompt` is stored as a bare int or as a
        # VideoDummyOptions-typed object. Keep video only if the user
        # permitted at least one.
        if mm_config.get_limit_per_prompt("video") == 0:
            # Temporarily shadow the offending pathway by making the
            # video-branch predicate false. We patch at the module scope
            # by wrapping `orig_fn` to run in a context where the
            # multimodal_config reports video=0. Simplest reliable path:
            # replace get_max_frames_per_video with a return-1 constant
            # for the duration of the call, since upstream only uses it
            # when "video" is in modalities (the list it builds is
            # ["image", "video"] regardless of user config).
            #
            # We mutate the bound method rather than the class so other
            # models aren't affected. Restore after.
            orig_max_frames = vllm_model.get_max_frames_per_video
            vllm_model.get_max_frames_per_video = (
                lambda: 1)  # video disabled -> value never actually used
            try:
                return orig_fn()
            finally:
                vllm_model.get_max_frames_per_video = orig_max_frames
    return orig_fn()


def _patched_get_max_frames_per_video(vllm_model, orig_fn):
    """Fall back to the ambient VllmConfig when `self.model_config` is
    absent.

    `Qwen3VLMoeForConditionalGeneration.__init__` skips
    `Qwen3VLForConditionalGeneration.__init__` (via `super(Qwen3VLForCondGen,
    self).__init__()`), so `self.model_config` is never set on MoE
    variants. The upstream implementation dereferences
    `self.model_config.max_model_len` and
    `mm_registry.get_processing_info(self.model_config)` and raises
    AttributeError. Route both through `get_current_vllm_config()` when
    the attribute is missing.
    """
    if getattr(vllm_model, "model_config", None) is not None:
        return orig_fn()
    from vllm.multimodal import MULTIMODAL_REGISTRY
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    info = MULTIMODAL_REGISTRY.get_processing_info(model_config)
    return info.get_num_frames_with_most_features(
        seq_len=model_config.max_model_len,
        mm_counts={
            "video":
            model_config.multimodal_config.get_limit_per_prompt("video")
        },
    )


def _patched_get_encoder_cudagraph_budget_range(vllm_model, orig_fn,
                                                vllm_config):
    """Prefer the passed `vllm_config.model_config.max_model_len` over
    `self.model_config.max_model_len`. Upstream uses `self.model_config`
    which is not present on `Qwen3VLMoeForConditionalGeneration`.
    """
    if getattr(vllm_model, "model_config", None) is not None:
        return orig_fn(vllm_config)
    # Bind `model_config` on the instance for the duration of the call.
    vllm_model.model_config = vllm_config.model_config
    try:
        return orig_fn(vllm_config)
    finally:
        del vllm_model.model_config


def apply_qwen3_vl_patches(vllm_model):
    """Apply Qwen3-VL specific patches for stateless Deepstack support."""
    # ---- Fixes for Qwen3VLMoe missing `self.model_config` -----------------
    # These patches are safe to apply for ALL Qwen3VL variants (both the
    # dense Qwen3VLForConditionalGeneration and the MoE subclass) — the
    # helpers early-return when `self.model_config` is already set, so the
    # dense path is a no-op. The MoE subclass hits the fix path.
    orig_get_encoder_cfg = getattr(vllm_model, "get_encoder_cudagraph_config",
                                   None)
    if orig_get_encoder_cfg is not None:
        vllm_model.get_encoder_cudagraph_config = (
            lambda: _patched_get_encoder_cudagraph_config(
                vllm_model, orig_get_encoder_cfg))

    orig_get_max_frames = getattr(vllm_model, "get_max_frames_per_video", None)
    if orig_get_max_frames is not None:
        vllm_model.get_max_frames_per_video = (
            lambda: _patched_get_max_frames_per_video(vllm_model,
                                                      orig_get_max_frames))

    orig_get_budget_range = getattr(vllm_model,
                                    "get_encoder_cudagraph_budget_range", None)
    if orig_get_budget_range is not None:
        vllm_model.get_encoder_cudagraph_budget_range = (
            lambda vc: _patched_get_encoder_cudagraph_budget_range(
                vllm_model, orig_get_budget_range, vc))

    # ---- Deepstack support (only when the model actually uses it) --------
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

    # 7. Disable dynamo compilation for Qwen3LLMModel so JAX JIT can trace through it.
    # During _flush_compilations, model_fn is called inside JAX JIT for shapes that
    # failed AOT lower. _patched_forward stores JAX abstract tracers in _deepstack_tensors,
    # then language_model.model (wrapped by TorchCompileWithNoGuardsWrapper) tries to run
    # dynamo on forward(), which can't handle torchax abstract tracers → TypeError at
    # deepstack addition (qwen3_vl.py:1546). Setting do_not_compile=True makes __call__
    # go directly to forward(), letting JAX trace through as pure JAX ops instead.
    llm_model = getattr(getattr(vllm_model, "language_model", None), "model",
                        None)
    if llm_model is not None and hasattr(llm_model, "do_not_compile"):
        llm_model.do_not_compile = True
        logger.info(
            "Disabled dynamo for Qwen3LLMModel; JAX JIT handles outer compilation"
        )


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
