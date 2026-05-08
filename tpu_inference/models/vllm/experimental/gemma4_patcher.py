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
uses Per-Layer Embeddings (PLE — used by E2B / E4B variants) by:
  1. Pre-allocating `self.per_layer_embeddings` as a torch.zeros buffer at
     init time (gemma4_mm.py:993).
  2. Inside `embed_input_ids`, computing per-layer features and copying
     them into the buffer in-place with `.copy_()` (gemma4_mm.py:1322).
  3. Inside `forward`, reading `self.per_layer_embeddings[:N]` and
     passing to `language_model.model(...)` via the `per_layer_inputs`
     kwarg.

Two problems for torchax:
  (a) The in-place `.copy_()` does not survive `__torch_dispatch__`:
      destination buffer was created by torch.zeros (no `_elem` attr) but
      the source is a torchax tensor. Result:
      `AttributeError: 'Tensor' object has no attribute '_elem'`.
  (b) Even if (a) is bypassed, JIT captures `self.per_layer_embeddings`
      at trace time. Subsequent Python attribute assignments don't
      propagate across the JIT boundary, so `forward` still reads the
      original zeros and the LM gets no PLE → garbage outputs.

Fix (mirrors qwen3_vl_patcher.py's deepstack pack/unpack strategy):
  PLE flows through **function arguments**, not state.

  - `_patched_embed_input_ids`: compute per_layer_inputs the same way as
    the original; PACK them into the returned `inputs_embeds` tensor by
    concatenating along the hidden dim. Skip the broken `.copy_()` by
    setting `self.per_layer_embeddings = None` for the duration of the
    delegated call (so the upstream PLE branch is short-circuited).
  - `_patched_forward`: detect packed `inputs_embeds` (hidden dim larger
    than `text_config.hidden_size`); split text embeds vs packed PLE;
    reshape PLE to `[N, num_layers, ple_dim]`; call
    `language_model.model(...)` directly with `per_layer_inputs=<the
    reshaped PLE>` and `inputs_embeds=<text embeds>`. Bypass the
    upstream `forward`'s read of `self.per_layer_embeddings`.

Wire via `maybe_apply_gemma4_patches(vllm_model)` from
`vllm_model_wrapper.py` next to `maybe_apply_qwen3_vl_patches`.
"""

import torch
from torchax.interop import jax_view, torch_view

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
    """Pack PLE into the returned inputs_embeds so it crosses the JIT
    boundary as an explicit value (not as model state)."""
    text_config = vllm_model.config.text_config
    has_ple = (vllm_model.per_layer_embeddings is not None
               and getattr(text_config, "hidden_size_per_layer_input", 0) > 0)

    per_layer_inputs = None
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
            per_layer_inputs = per_layer_inputs.reshape(
                -1,
                text_config.num_hidden_layers,
                text_config.hidden_size_per_layer_input,
            )

    # Set per_layer_embeddings = None so the orig method's broken
    # in-place .copy_() block (gemma4_mm.py:1320-1323) is short-circuited.
    saved_buf = vllm_model.per_layer_embeddings
    vllm_model.per_layer_embeddings = None
    try:
        if multimodal_embeddings is None or is_multimodal is None:
            inputs_embeds = orig_method(input_ids)
        else:
            inputs_embeds = orig_method(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
    finally:
        vllm_model.per_layer_embeddings = saved_buf

    # Pack PLE into the returned embeds along the hidden dim. The
    # patched forward will detect the larger-than-expected hidden dim
    # and unpack.
    if per_layer_inputs is not None:
        cur_tokens = inputs_embeds.size(0)
        ple_slice = per_layer_inputs[:cur_tokens]
        # Reshape: [N, num_layers, ple_dim] -> [N, num_layers*ple_dim]
        packed = ple_slice.reshape(cur_tokens, -1)
        # Avoid torch/torchax type mixing in torch.cat: bring `packed` into
        # the same tensor world as `inputs_embeds`. inputs_embeds is a
        # torchax tensor (since the runner handed jax arrays in); packed
        # is whatever get_per_layer_inputs returned. Force both through
        # torchax view.
        if not packed.__class__.__module__.startswith("torchax"):
            packed = torch_view(jax_view(packed))
        inputs_embeds = torch.cat([inputs_embeds, packed], dim=-1)
        logger.warning_once(
            "Gemma-4 patcher: packed PLE into inputs_embeds; "
            "shape now %s (text=%d + ple_packed=%d).",
            str(inputs_embeds.shape), text_config.hidden_size,
            packed.shape[-1])

    return inputs_embeds


def _patched_forward(
    vllm_model,
    orig_forward,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors=None,
    inputs_embeds=None,
    **kwargs,
):
    """Unpack PLE from inputs_embeds, then call the language model's
    underlying forward directly with per_layer_inputs as an argument."""
    text_config = vllm_model.config.text_config
    text_hidden = text_config.hidden_size
    num_layers = text_config.num_hidden_layers
    ple_dim = getattr(text_config, "hidden_size_per_layer_input", 0)

    # If embeddings carry packed PLE (and we're in the prefill path,
    # not intermediate-tensors propagation), split them.
    per_layer_inputs = None
    if (intermediate_tensors is None and inputs_embeds is not None
            and ple_dim > 0
            and inputs_embeds.shape[-1] > text_hidden):
        packed_dim = inputs_embeds.shape[-1] - text_hidden
        expected_packed = num_layers * ple_dim
        if packed_dim == expected_packed:
            packed_ple = inputs_embeds[..., text_hidden:]
            inputs_embeds = inputs_embeds[..., :text_hidden]
            per_layer_inputs = packed_ple.reshape(-1, num_layers, ple_dim)
        else:
            logger.warning(
                "Gemma-4 patcher: unexpected packed dim %d (expected %d); "
                "falling back to original forward.", packed_dim,
                expected_packed)

    if intermediate_tensors is not None:
        inputs_embeds = None

    # Preserve the upstream side effect that runs outside the compiled
    # subgraph (it mutates Python state on the language model so that
    # full-attention layers can clear the multi-modal prefix range).
    if hasattr(vllm_model, "_clear_mm_prefix_for_full_attn_layers"):
        vllm_model._clear_mm_prefix_for_full_attn_layers()

    if per_layer_inputs is not None:
        # We've split out PLE; call the language model directly with
        # the unpacked per_layer_inputs to avoid the upstream forward's
        # `self.per_layer_embeddings[:N]` read.
        logger.warning_once(
            "Gemma-4 patcher: forward unpacked PLE; "
            "inputs_embeds=%s, per_layer_inputs=%s.",
            str(inputs_embeds.shape), str(per_layer_inputs.shape))
        hidden_states = vllm_model.language_model.model(
            input_ids,
            positions,
            per_layer_inputs=per_layer_inputs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return hidden_states

    # No PLE in this call (e.g., 26B/31B variants without PLE, or
    # decode steps that don't pass inputs_embeds). Defer to the
    # original forward.
    logger.warning_once(
        "Gemma-4 patcher: forward fallback to orig_forward "
        "(intermediate=%s, inputs_embeds=%s).",
        intermediate_tensors is not None,
        None if inputs_embeds is None else str(inputs_embeds.shape))
    return orig_forward(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )


def _coerce_scale_buffers_to_torchax(text_model):
    """vllm gemma4 registers persistent=False scalar buffers
    (per_layer_input_scale, per_layer_projection_scale,
    embed_scale_per_layer). These buffers are torch.Tensor scalars that
    do NOT get shipped through torchax sharding (because persistent=False
    means they're NOT in state_dict, so shard_model_to_tpu skips them).

    At inference, when they multiply a torchax tensor (e.g.,
    per_layer_projection + per_layer_inputs * scale), torchax raises:

      AssertionError: Expect a Tensor or a View but got
        <class 'torch.Tensor'>; usually this means there is a mixed
        math between XLATensor and torch.Tensor

    Fix: convert each buffer's value into a torchax tensor (via
    jax.device_put + torch_view). The result IS a torch.Tensor (so
    torch.func.functional_call's buffer-swap machinery still works) AND
    it's torchax-compatible (so forward math doesn't crash).

    Replacing buffers with plain Python floats does NOT work — it breaks
    torch.func.functional_call which requires buffers to be torch.Tensor.
    """
    import jax

    for attr in (
            "per_layer_input_scale",
            "per_layer_projection_scale",
            "embed_scale_per_layer",
    ):
        buf = getattr(text_model, attr, None)
        if buf is None:
            continue
        if not isinstance(buf, torch.Tensor):
            continue
        # Already torchax? Nothing to do.
        if buf.__class__.__module__.startswith("torchax"):
            continue
        try:
            arr = buf.detach().cpu().numpy()
            jax_arr = jax.device_put(arr)
            new_buf = torch_view(jax_arr)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Gemma-4 patcher: could not convert %s to torchax: %s", attr, e)
            continue

        # Replace via _buffers dict. nn.Module.register_buffer would
        # also work but might trigger validation checks.
        if attr in getattr(text_model, "_buffers", {}):
            text_model._buffers[attr] = new_buf
        else:
            # Not in _buffers? Try plain attribute set as fallback.
            setattr(text_model, attr, new_buf)
        logger.info(
            "Gemma-4 patcher: converted %s buffer to torchax (was %s, now %s)",
            attr, buf.__class__.__module__, new_buf.__class__.__module__)


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

    # Convert non-persistent scale buffers to torchax tensors so they
    # interoperate with the torchax tensor world at math time.
    text_model = getattr(getattr(vllm_model, "language_model", None),
                         "model", None)
    if text_model is not None:
        _coerce_scale_buffers_to_torchax(text_model)

    orig_embed = getattr(vllm_model, "embed_input_ids", None)
    orig_forward = getattr(vllm_model, "forward", None)
    if orig_embed is None or orig_forward is None:
        logger.warning(
            "Gemma-4 patcher: vllm_model missing embed_input_ids/forward; "
            "skipping patch.")
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

    vllm_model.forward = (lambda *args, **kwargs: _patched_forward(
        vllm_model, orig_forward, *args, **kwargs))

    logger.info(
        "Gemma-4 patcher: applied PLE pack/unpack via inputs_embeds "
        "(hidden_size_per_layer_input=%d, num_hidden_layers=%d).", ple_dim,
        text_cfg.num_hidden_layers)


def maybe_apply_gemma4_patches(vllm_model):
    """Conditionally apply Gemma-4 patches based on the model architecture."""
    cfg = getattr(vllm_model, "config", None)
    if cfg is None:
        return
    arches = getattr(cfg, "architectures", []) or []
    if "Gemma4ForConditionalGeneration" in arches:
        apply_gemma4_patches(vllm_model)
