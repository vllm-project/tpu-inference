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

import functools

import jax
import jax.numpy as jnp
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
from tpu_inference.layers.common.sharding import ShardingAxisName
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


@jax.jit
def _jit_masked_merge(inputs_embeds, is_multimodal, mm_embeds_parts):
    """Pure-JAX equivalent of `inputs_embeds[is_multimodal] = cat(mm_embeds_parts)`.

    `embed_input_ids_func` (vllm_model_wrapper.py) is deliberately not
    jax.jit-wrapped, so torchax's per-call boolean-mask __setitem__ dispatch
    (torchax.tensor.Tensor.__setitem__ -> _shape_static_boolean_index_put) was
    recompiling on every call even for shapes it had already seen, since eager
    op-by-op dispatch outside a persistent jax.jit doesn't reuse JAX's compile
    cache reliably. Defining this as a module-level @jax.jit function (called
    the same way every time) restores proper cache reuse across calls with
    identical shapes. Concatenating `mm_embeds_parts` here too (rather than
    eagerly beforehand) keeps that op inside the same cached program instead
    of it recompiling separately every call.

    Under `enable_dp_attention` (ambient mesh has a real 'data' axis, e.g.
    size 4), this function had no sharding annotations, so JAX's default
    GSPMD inference left `inputs_embeds`/`is_multimodal` fully replicated
    (spec=P()) across the whole mesh -- even though `inputs_embeds`'s
    leading dim is already the DP-combined size (e.g. 4x max_model_len).
    Every device held a full copy of that combined array instead of its own
    1/data_size slice, multiplying this function's memory footprint by the
    DP degree on top of the shape-diversity cost, and was the dominant
    reason DP=4 OOMs in ~24-38 requests while plain TP=2 (no data axis to
    replicate across) runs hundreds cleanly. `with_sharding_constraint`
    (not shard_map) is used deliberately: it only declares the desired
    layout, and GSPMD inserts whatever collectives are needed (e.g. a
    distributed prefix-sum for `cumsum` sharded along 'data') to keep the
    result correct -- no manual per-shard logic to get wrong.
    """
    inputs_embeds = jax.lax.with_sharding_constraint(
        inputs_embeds,
        jax.sharding.PartitionSpec(ShardingAxisName.ATTN_DATA, None))
    is_multimodal = jax.lax.with_sharding_constraint(
        is_multimodal, jax.sharding.PartitionSpec(ShardingAxisName.ATTN_DATA))

    mm_embeds_flat = jnp.concatenate(mm_embeds_parts, axis=0)
    idx = jnp.cumsum(is_multimodal.astype(jnp.int32)) - 1
    idx = jnp.clip(idx, 0, mm_embeds_flat.shape[0] - 1)
    gathered = mm_embeds_flat[idx]
    result = jnp.where(is_multimodal[:, None], gathered, inputs_embeds)
    return jax.lax.with_sharding_constraint(
        result, jax.sharding.PartitionSpec(ShardingAxisName.ATTN_DATA, None))


def _patched_merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings,
                                         is_multimodal):
    """Drop-in replacement for vLLM's `_merge_multimodal_embeddings`.

    Routes both the concatenation of per-item embeddings and the boolean-mask
    merge through the persistent `_jit_masked_merge`, to avoid the
    recompile-on-every-call behavior described there.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    input_dtype = inputs_embeds.dtype
    if all(torch.is_tensor(t) for t in multimodal_embeddings):
        # Fast path: what Qwen3-VL always produces in practice -- a flat
        # list/tuple of one 2D tensor per image/video. Hand the parts to the
        # jit uncatted so concatenation happens inside the cached program.
        mm_parts = tuple(
            jax_view(t.to(dtype=input_dtype)) for t in multimodal_embeddings)
    else:
        # Fallback for arbitrarily nested NestedTensors -- not exercised by
        # Qwen3-VL today, but kept for correctness/generality. Uses the
        # existing (already correct) recursive flattener eagerly.
        mm_parts = (jax_view(
            _patched_flatten_embeddings(multimodal_embeddings).to(
                dtype=input_dtype)), )
    merged = _jit_masked_merge(jax_view(inputs_embeds),
                               jax_view(is_multimodal), mm_parts)
    return torch_view(merged)


@jax.jit
def _jit_pack_deepstack(inputs_embeds, level_tensors):
    """Pure-JAX equivalent of the deepstack stack/transpose/reshape/cat below.

    Same rationale as `_jit_masked_merge`: a persistent @jax.jit function so
    repeated shapes hit JAX's compile cache instead of recompiling per call.
    `level_tensors` has a fixed length (deepstack_num_level, a model constant),
    so its pytree structure is stable across calls. Also shards along 'data'
    like `_jit_masked_merge` -- see that function's docstring for why
    unsharded (fully-replicated) arrays under `enable_dp_attention` are a
    major OOM contributor.
    """
    data_spec = jax.sharding.PartitionSpec(ShardingAxisName.ATTN_DATA, None)
    inputs_embeds = jax.lax.with_sharding_constraint(inputs_embeds, data_spec)
    level_tensors = tuple(
        jax.lax.with_sharding_constraint(t, data_spec) for t in level_tensors)
    stacked = jnp.stack(level_tensors,
                        axis=0)  # [num_levels, cur_tokens, hdim]
    packed = jnp.transpose(stacked, (1, 0, 2)).reshape(stacked.shape[1], -1)
    result = jnp.concatenate([inputs_embeds, packed], axis=-1)
    return jax.lax.with_sharding_constraint(result, data_spec)


@functools.partial(jax.jit, static_argnames=("num_levels", "visual_dim"))
def _jit_deepstack_reshape_permute(deepstack_input_embeds, num_levels,
                                   visual_dim):
    """Pure-JAX equivalent of vLLM's `.view(...).permute(1, 0, 2)` in
    `_compute_deepstack_embeds` (qwen3_vl.py).

    This op runs eagerly (outside any persistent jax.jit) in vLLM's own
    `_compute_deepstack_embeds`, called before our merge-step fix above even
    gets a chance to run. Same recompile-every-call/no-cache-reuse problem as
    `_jit_masked_merge`, but on the much larger deepstack tensor (shape
    (cur_tokens, deepstack_num_level * hidden) -- video requests have far
    more tokens than image ones), so it accumulates device memory fast
    enough to OOM within ~24 requests under sustained video load rather than
    the ~1500+ requests images needed to trigger the same class of crash.
    `num_levels`/`visual_dim` are static: fixed by model config, never vary
    across calls, so this specializes to one cached compilation. Sharded
    along 'data' for the same reason as `_jit_masked_merge` -- the token
    axis moves from dim 0 (input) to dim 1 after the permute (output).
    """
    deepstack_input_embeds = jax.lax.with_sharding_constraint(
        deepstack_input_embeds,
        jax.sharding.PartitionSpec(ShardingAxisName.ATTN_DATA, None))
    cur_tokens = deepstack_input_embeds.shape[0]
    reshaped = deepstack_input_embeds.reshape(cur_tokens, num_levels,
                                              visual_dim)
    result = jnp.transpose(reshaped, (1, 0, 2))
    return jax.lax.with_sharding_constraint(
        result,
        jax.sharding.PartitionSpec(None, ShardingAxisName.ATTN_DATA, None))


def _patched_compute_deepstack_embeds(vllm_model,
                                      orig_compute_deepstack_embeds,
                                      inputs_embeds, multimodal_embeddings,
                                      is_multimodal):
    """Drop-in replacement for vLLM's `_compute_deepstack_embeds`.

    Identical computation, except the boolean-mask merge and the final
    reshape+permute are routed through persistent jax.jit functions instead
    of eager torchax dispatch -- see `_jit_deepstack_reshape_permute` for why.
    """
    visual_lens = [len(x) for x in multimodal_embeddings]
    multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

    multimodal_embeddings_main, multimodal_embeddings_multiscale = torch.split(
        multimodal_embeddings_cat,
        [vllm_model.visual_dim, vllm_model.multiscale_dim],
        dim=-1,
    )

    multimodal_embeddings = torch.split(multimodal_embeddings_main,
                                        visual_lens,
                                        dim=0)
    multimodal_embeddings_multiscale = torch.split(
        multimodal_embeddings_multiscale, visual_lens, dim=0)

    deepstack_input_embeds = inputs_embeds.new_zeros(
        inputs_embeds.size(0),
        vllm_model.deepstack_num_level * inputs_embeds.size(1))

    deepstack_input_embeds = _patched_merge_multimodal_embeddings(
        inputs_embeds=deepstack_input_embeds,
        multimodal_embeddings=multimodal_embeddings_multiscale,
        is_multimodal=is_multimodal,
    )
    permuted_jax = _jit_deepstack_reshape_permute(
        jax_view(deepstack_input_embeds),
        num_levels=vllm_model.deepstack_num_level,
        visual_dim=vllm_model.visual_dim,
    )
    deepstack_input_embeds = torch_view(permuted_jax)

    return deepstack_input_embeds, multimodal_embeddings


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
        # 3. Stack levels, reshape, and append via the persistent jit helper:
        # [num_levels, cur_tokens, hdim] -> [cur_tokens, num_levels*hdim] -> cat.
        packed_jax = _jit_pack_deepstack(
            jax_view(inputs_embeds), tuple(jax_view(t) for t in level_tensors))
        inputs_embeds = torch_view(packed_jax)
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


def apply_qwen3_vl_patches(vllm_model: Qwen3VLForConditionalGeneration):
    """Apply Qwen3-VL specific patches for stateless Deepstack support."""
    if not vllm_model.use_deepstack:
        return

    # 1. Override deepstack setter to avoid stateful updates of native vLLM variables
    vllm_model._set_deepstack_input_embeds = lambda embeds: _patched_set_deepstack(
        vllm_model, embeds)

    # 2. Override deepstack getter to prefer JAX-compatible cached tensors
    orig_get_deepstack = vllm_model._get_deepstack_input_embeds
    vllm_model._get_deepstack_input_embeds = lambda num_tokens: _patched_get_deepstack(
        vllm_model, orig_get_deepstack, num_tokens)

    # 3. Patch embed_input_ids to pack Deepstack vision features into main text embeddings
    orig_embed_input_ids = vllm_model.embed_input_ids
    vllm_model.embed_input_ids = lambda *args, **kwargs: _patched_embed_input_ids(
        vllm_model, orig_embed_input_ids, *args, **kwargs)

    # 3b. Patch _compute_deepstack_embeds (called from within embed_input_ids,
    # before our patch above even runs) to route its eager reshape+permute
    # through a persistent jax.jit -- see _jit_deepstack_reshape_permute for
    # why this is needed in addition to the embed_input_ids/merge patches.
    orig_compute_deepstack_embeds = vllm_model._compute_deepstack_embeds
    vllm_model._compute_deepstack_embeds = lambda *args, **kwargs: _patched_compute_deepstack_embeds(
        vllm_model, orig_compute_deepstack_embeds, *args, **kwargs)

    # 4. Patch forward to unpack vision features and restore them before execution
    orig_forward = vllm_model.forward
    vllm_model.forward = lambda *args, **kwargs: _patched_forward(
        vllm_model, orig_forward, *args, **kwargs)

    # 5. Patch _flatten_embeddings in vllm utils to handle negative indexes correctly in torchax
    vllm_utils._flatten_embeddings = _patched_flatten_embeddings

    # 5b. Patch _merge_multimodal_embeddings (imported into qwen3_vl_mod's own
    # namespace, so it must be patched there, not on vllm_utils) to route the
    # boolean-mask merge through a persistent jax.jit instead of torchax's
    # per-call eager dispatch -- see _jit_masked_merge for why.
    qwen3_vl_mod._merge_multimodal_embeddings = _patched_merge_multimodal_embeddings

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
