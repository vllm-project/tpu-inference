# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MM-encoder JIT manager — TPU subclass of vLLM's EncoderCudaGraphManager.

The GPU manager (``vllm/v1/worker/encoder_cudagraph.py``) records one
``torch.cuda.CUDAGraph`` per token budget and replays it by mutating
persistent device buffers in place. TPU/XLA has no CUDA graph; its
equivalent is the XLA compile cache keyed on input *shape*. So instead of
re-forking the manager we **subclass** it and override only the two
device-bound methods, inheriting everything algorithmic:

  Inherited unchanged
    * ``__init__`` budget derivation + validation (reads the upstream
      ``compilation_config.encoder_cudagraph_*`` knobs).
    * ``_generate_budgets`` / ``_find_smallest_fitting_budget_given_tokens``.
    * ``_execute_local`` — the greedy smallest-first bin-packing loop.
    * ``supports_modality`` / ``get_cumulative_stats`` / ``capture``.

  Overridden for XLA
    * ``_capture_budget_graph`` — prime the ``jax.jit`` cache for a budget
      (the XLA analog of CUDA-graph capture) instead of recording a graph.
    * ``_run_budget_graph`` — host-pad the replay buffers to the budget's
      template shape, then call the once-built ``jax.jit`` closure. The XLA
      cache hits on the per-budget shape signature regardless of the actual
      ``image_grid_thw`` values.
    * ``execute`` — run the inherited ``_execute_local`` in plain-torch
      context (NO outer torchax env): item selection and replay-buffer prep
      stay as normal torch (they index model Parameters), while the per-budget
      JIT and the eager fallback each enter the torchax env locally and emit
      ``jax.Array``. The encoder outputs flow as jax arrays end-to-end, so the
      result is already the ``list[jax.Array]`` the caller expects.

The inherited ``_execute_local`` calls ``model.encoder_eager_forward`` and
``model.postprocess_encoder_output``. On TPU the model's own weights are
meta/empty — the real params live in ``params_and_buffers`` — so we pass a
thin ``_TorchaxEncoderModelAdapter`` as the ``model`` to the parent: it
bridges ``encoder_eager_forward`` through ``functional_call`` (returning a
``jax.Array``), provides a jax-array ``postprocess_encoder_output`` that
scatters per-item slices, and delegates every other protocol method to the
real vLLM model.

The JIT closure is built once at init (the v7 cache-share lesson — see
``round2_bench/perf_report.html`` §6.6 for the per-call PjitFunction trap
this avoids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.ir import enable_torch_wrap
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from tpu_inference.logger import init_logger
from tpu_inference.utils import to_torch_dtype

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class _TorchaxEncoderModelAdapter:
    """Wrap the vLLM model so the inherited ``_execute_local`` eager
    fallback runs through the torchax runner with externally-held weights.

    The inherited bin-packing loop calls ``model.encoder_eager_forward`` for
    single items that exceed the largest budget. On TPU the model module's
    parameters are meta/empty (the real arrays live in
    ``params_and_buffers``), so a direct call would use wrong weights. This
    adapter overrides only ``encoder_eager_forward`` — routing it through
    ``torch.func.functional_call`` on the torchax-wrapped runner — and
    delegates every other ``SupportsEncoderCudaGraph`` method (config, item
    specs, item selection, replay-buffer prep, postprocess) to the real
    model via ``__getattr__``.
    """

    def __init__(self, vllm_model: Any, vllm_runner: torch.nn.Module,
                 params_and_buffers: Any):
        self._model = vllm_model
        self._runner = vllm_runner
        self._params = params_and_buffers
        self._jit_forward = jax.jit(self._build_forward_fn())

    # ----- JIT forward closure -----

    def _build_forward_fn(self) -> Callable:
        """Build the closure that gets ``jax.jit``-wrapped exactly once.

        Inputs: ``params_jax`` (the model weights) + ``mm_kwargs_jax`` (the
        padded multimodal input dict, e.g. pixel_values + image_grid_thw)
        + ``buffers_jax`` (the padded encoder metadata buffers, e.g.
        pos_embeds / rotary_pos_emb_{cos,sin} / cu_seqlens / max_seqlen /
        sequence_lengths).

        Inside: bridge jax -> torchax with ``torch_view``, dispatch via
        ``functional_call(call_method="encoder_cudagraph_forward")``
        with the (mm_kwargs, buffers) two-arg signature vLLM adopted in
        PR #41234 ("Simplify ViT CUDA graph interfaces"), bridge torchax
        output back to jax with ``jax_view``.
        """
        vllm_runner = self._runner

        def _forward(params_jax: Any, mm_kwargs_jax: dict[str, jax.Array],
                     buffers_jax: dict[str, jax.Array]):
            params_torchax = torch_view(params_jax)
            mm_kwargs_torchax = {
                k: jax.tree.map(torch_view, v)
                for k, v in mm_kwargs_jax.items()
            }
            buffers_torchax = {
                k: jax.tree.map(torch_view, v)
                for k, v in buffers_jax.items()
            }
            out_torch = torch.func.functional_call(
                vllm_runner,
                params_torchax,
                kwargs={
                    "call_method": "encoder_cudagraph_forward",
                    "call_args": (mm_kwargs_torchax, buffers_torchax),
                    "call_kwargs": {},
                },
                tie_weights=False,
            )
            return jax_view(out_torch)

        return _forward

    def run_budget_forward(
        self,
        padded_mm_kwargs: dict[str, Any],
        padded_buffers: dict[str, Any],
    ) -> jax.Array:
        # Convert + JIT — INSIDE the env (the closure bridges torchax<->jax).
        with torchax.default_env(), enable_torch_wrap(False):
            mm_kwargs_jax = {
                k: jax.tree.map(self._t2j_if_tensor, v)
                for k, v in padded_mm_kwargs.items()
            }
            buffers_jax = {
                k: jax.tree.map(self._t2j_if_tensor, v)
                for k, v in padded_buffers.items()
            }
            return self._jit_forward(self._params, mm_kwargs_jax, buffers_jax)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any]) -> jax.Array:
        # Bridge plain-torch mm_kwargs -> torchax, dispatch the model's eager
        # vision forward via functional_call (binds the real TPU weights),
        # and return a jax.Array. The torchax env is entered locally here so
        # the inherited _execute_local can stay in plain-torch context (its
        # replay-buffer prep indexes model Parameters, which must NOT run
        # under the torchax dispatch).
        with torchax.default_env():
            torchax_kwargs = {
                k: jax.tree.map(self._torchax_view_if_torch, v)
                for k, v in mm_kwargs.items()
            }
            out_torch = torch.func.functional_call(
                self._runner,
                torch_view(self._params),
                kwargs={
                    "call_method": "encoder_eager_forward",
                    "call_args": (torchax_kwargs, ),
                    "call_kwargs": {},
                },
                tie_weights=False,
            )
            return jax_view(out_torch)

    def postprocess_encoder_output(self,
                                   output: jax.Array,
                                   indices: list[int],
                                   per_item_out_tokens: list[int],
                                   dest,
                                   clone: bool = False,
                                   batch_mm_kwargs=None) -> None:
        # jax-array analog of the model's default postprocess (which calls
        # scatter_output_slices + torch .clone()). The encoder output is a
        # jax.Array here, so slice per item and scatter; jax arrays are
        # immutable, so ``clone`` is a no-op.
        offset = 0
        for idx in indices:
            n = per_item_out_tokens[idx]
            dest[idx] = output[offset:offset + n]
            offset += n

    @staticmethod
    def _t2j_if_tensor(v: torch.Tensor | jax.Array) -> jax.Array:
        """Tree-map helper — convert leaf torch.Tensors to jax.Array."""
        if isinstance(v, torch.Tensor):
            return t2j(v, use_dlpack=False)
        return v

    @staticmethod
    def _torchax_view_if_torch(v):
        """Tree-map helper for the eager path."""
        if isinstance(v, torch.Tensor):
            return torch_view(t2j(v, use_dlpack=False))
        return v

    def __getattr__(self, name: str) -> Any:
        # Delegate all non-overridden protocol methods to the real model.
        return getattr(self._model, name)


class JaxEncoderModelAdapter:
    """Wrap a JAX/flax SupportsEncoderCudaGraph model for MMEncoderJITManager.
    Mirrors ``_TorchaxEncoderModelAdapter`` but routes budget execution and
    eager fallback directly through the JAX model.
    """

    def __init__(self, jax_model: Any):
        self._model = jax_model

    def run_budget_forward(self, padded_torch: dict[str, Any]) -> jax.Array:
        jax_inputs = {}
        for k, v in padded_torch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.bfloat16:
                    jax_inputs[k] = jnp.asarray(v.contiguous().view(
                        torch.int16).numpy().view(jnp.bfloat16))
                else:
                    jax_inputs[k] = jnp.asarray(v.numpy())
            else:
                jax_inputs[k] = v
        return self._model.encoder_cudagraph_forward(jax_inputs)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any]) -> jax.Array:
        return self._model.encoder_eager_forward(mm_kwargs)

    def postprocess_encoder_output(self,
                                   output: jax.Array,
                                   indices: list[int],
                                   per_item_out_tokens: list[int],
                                   dest,
                                   clone: bool = False,
                                   batch_mm_kwargs=None) -> None:
        self._model.postprocess_encoder_output(output, indices,
                                               per_item_out_tokens, dest,
                                               clone, batch_mm_kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class MMEncoderJITManager(EncoderCudaGraphManager):
    """Per-budget XLA-cache manager for the vision encoder forward."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        vllm_runner: torch.nn.Module | None,
        vllm_model: Any,
        params_and_buffers: Any,
    ):
        """
        Args:
          vllm_config: The vllm config object. Budget knobs are read by the
              inherited ``EncoderCudaGraphManager.__init__`` from
              ``compilation_config.encoder_cudagraph_token_budgets`` /
              ``encoder_cudagraph_max_vision_items_per_batch`` /
              ``encoder_cudagraph_max_frames_per_batch`` (same as GPU).
          vllm_runner: The torchax-wrapped ``_VllmRunner`` (provides
              ``forward(call_method=..., call_args=...)``-style dispatch
              required by ``torch.func.functional_call``). Pass ``None``
              for the JAX/flax path — selects ``JaxEncoderModelAdapter``.
          vllm_model: The underlying vllm model (e.g.
              ``Qwen3VLForConditionalGeneration``) or a JAX flax model.
              Must implement ``SupportsEncoderCudaGraph``.
          params_and_buffers: The model's loaded weights as a pytree of
              JAX arrays. Bound into ``functional_call`` per request.
              Pass ``None`` for the JAX/flax path.
        """
        # The parent calls model.{get_encoder_cudagraph_config,
        # get_encoder_cudagraph_budget_range}; the inherited _execute_local
        # later calls model.{select_encoder_cudagraph_items,
        # encoder_eager_forward, postprocess_encoder_output}. Route eager
        # forward and budget execution through the appropriate adapter.
        if vllm_runner is None:
            # JAX/flax path
            adapter = JaxEncoderModelAdapter(vllm_model)
        else:
            # torchax path: functional_call through the torchax runner.
            adapter = _TorchaxEncoderModelAdapter(vllm_model, vllm_runner,
                                                  params_and_buffers)

        # Reuse upstream budget derivation + validation. Capture inputs are
        # built on CPU; the JIT path moves them to TPU via t2j, so we never
        # run the encoder on a CUDA device.
        torch_dtype = to_torch_dtype(vllm_config.model_config.dtype)
        super().__init__(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            dtype=torch_dtype,
            model=adapter,
        )

        # Capture templates per budget — shape signature reference for
        # host-side padding. The values inside templates are dummy; only
        # tensor.shape / tensor.dtype matter (to us and to XLA's cache key).
        # As of vLLM PR #41234 the capture inputs split into two dicts
        # (`mm_kwargs` = pixel_values + grid metadata, `buffers` = the
        # precomputed encoder metadata that the CUDA-graph path would
        # replay). We keep them separate here because
        # `encoder_cudagraph_forward` now takes them as two distinct args.
        capture_device = torch.device("cpu")
        capture_dtype = torch_dtype
        self.mm_kwargs_templates: dict[int, dict[str, torch.Tensor]] = {}
        self.buffers_templates: dict[int, dict[str, torch.Tensor]] = {}
        for budget in self.token_budgets:
            capture = self.model.prepare_encoder_cudagraph_capture_inputs(
                budget, self.max_batch_size, self.max_frames_per_batch,
                capture_device, capture_dtype)
            self.mm_kwargs_templates[budget] = capture.mm_kwargs
            self.buffers_templates[budget] = capture.buffers

        logger.info(
            "[mm_encoder_jit] budgets=%s max_batch_size=%d "
            "max_frames_per_batch=%d mm_keys=%s buffer_keys=%s",
            self.token_budgets, self.max_batch_size, self.max_frames_per_batch,
            list(next(iter(self.mm_kwargs_templates.values())).keys()),
            list(next(iter(self.buffers_templates.values())).keys()))

    # ----- Padding (per-key) -----

    def _pad_dict_to_template(
        self,
        replay_values: dict[str, torch.Tensor],
        template: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Zero-and-copy each replay tensor into a template-shaped buffer.

        Mirrors the GPU manager's ``_copy_padded_buffer`` default (the
        parent's ``_run_budget_graph`` does this per buffer_key). Here we pad
        *every* template key because the jit call needs a fresh, fully
        materialised input dict each step (no persistent buffers).
        cu_seqlens / scalars pass through because ``prepare_encoder_metadata``
        already padded them.
        """
        padded: dict[str, torch.Tensor] = {}
        for key, tmpl in template.items():
            src = replay_values.get(key)
            if src is None:
                # Some keys (e.g. ``sequence_lengths``) are backend-conditional
                # — replay returned None, template also has None / unused.
                padded[key] = tmpl
                continue
            if not hasattr(src, "shape") or src.ndim == 0:
                # Scalars (e.g. ``max_seqlen``) — use template value directly
                # so the JIT sees a static budget-sized scalar.
                padded[key] = tmpl
                continue
            if src.shape == tmpl.shape:
                # Already template-shaped (e.g. cu_seqlens padded by
                # max_batch_size at metadata-prep time).
                padded[key] = src
                continue
            # General case: zero buffer, then slice-copy src onto its head.
            buf = torch.zeros_like(tmpl)
            n = src.shape[0]
            buf[:n] = src.to(dtype=tmpl.dtype, device=tmpl.device)
            padded[key] = buf
        return padded

    def _prepare_padded_torch(
        self,
        mm_kwargs: dict[str, Any],
        replay_buffers: dict[str, torch.Tensor | None] | None,
        token_budget: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build padded (plain-torch) mm_kwargs and buffers dicts for one
        batch.

        Runs entirely OUTSIDE the torchax env: `replay_buffers` (when None)
        comes from the model's `prepare_encoder_cudagraph_replay_buffers`
        which indexes model Parameters (`fast_pos_embed_interpolate`) and
        must dispatch as normal torch, not through torchax. The t2j
        conversion to jax happens later, inside `_run_budget_graph`'s env
        block.

        `mm_kwargs` (the actual batch inputs) are padded against
        `mm_kwargs_templates[token_budget]`. `replay_buffers` (or the
        model-computed replay values) are padded against
        `buffers_templates[token_budget]`. Both dicts must match the
        template shape signature for the JIT cache to hit.

        TODO(lk-chen): cache the non-pixel metadata tensors keyed on
        (token_budget, grid_thw) to skip the expensive
        fast_pos_embed_interpolate + rot_pos_emb host CPU work on repeated
        requests with the same resolution. pixel_values must always come
        from a fresh call. Invalidate when token_budget or grid_thw
        changes.
        """
        if replay_buffers is None:
            # Compatibility path for callers that don't have replay_buffers
            # handy (e.g. our own `_capture_budget_graph` warmup). Compute
            # them from the model just like the upstream `_execute_local`
            # loop does before calling `_run_budget_graph`.
            replay = self.model.prepare_encoder_cudagraph_replay_buffers(
                mm_kwargs, self.max_batch_size, self.max_frames_per_batch)
            replay_buffers = replay.buffers
        padded_mm = self._pad_dict_to_template(
            mm_kwargs, self.mm_kwargs_templates[token_budget])
        padded_buffers = self._pad_dict_to_template(
            replay_buffers, self.buffers_templates[token_budget])
        return padded_mm, padded_buffers

    # ----- Overrides of the CUDA-graph device hooks -----

    def _capture_budget_graph(self, token_budget: int) -> None:
        """XLA-cache analog of ``torch.cuda.graph`` capture.

        Primes the ``jax.jit`` cache for ``token_budget`` by calling the
        forward closure once on the budget's dummy templates, so the first
        runtime call is a cache hit instead of paying compile time.
        Invoked by the inherited ``capture()`` loop (optional — the cache
        also fills lazily on first ``execute``).
        """
        mm_template = self.mm_kwargs_templates[token_budget]
        buffers_template = self.buffers_templates[token_budget]
        out = self.model.run_budget_forward(mm_template, buffers_template)
        jax.block_until_ready(out)
        # Mark captured so inherited capture()/get_cumulative_stats count it.
        self.budget_graphs[token_budget] = (mm_template, buffers_template)

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
        replay_buffers: dict[str, torch.Tensor | None] | None = None,
    ) -> jax.Array | None:
        """XLA-cache analog of CUDA-graph replay.

        Host-pads the replay buffers to the budget template shape (plain
        torch, outside the env) and delegates to the adapter's
        ``run_budget_forward``. Returns a **jax.Array** that the inherited
        ``_execute_local`` slices via the adapter's jax-friendly
        ``postprocess_encoder_output`` — no outer torchax env required.

        Signature matches vLLM's parent post-PR #41234:
        ``(mm_kwargs, token_budget, replay_buffers)`` where
        ``replay_buffers`` is the dict returned by
        ``model.prepare_encoder_cudagraph_replay_buffers(...).buffers``.
        Older callers that pass only ``(mm_kwargs, token_budget)`` still
        work via the ``replay_buffers=None`` fallback which recomputes
        them from the model.
        """
        num_items = len(self._get_item_specs(mm_kwargs))
        if token_budget not in self.mm_kwargs_templates:
            # No template for this budget (rarely happens — budget comes
            # from _find_smallest_fitting_budget over self.token_budgets).
            self.graph_misses += num_items
            return None

        # Prep in plain torch (touches model Parameters) — OUTSIDE the env.
        padded_mm, padded_buffers = self._prepare_padded_torch(
            mm_kwargs, replay_buffers, token_budget)
        out_jax = self.model.run_budget_forward(padded_mm, padded_buffers)
        self.graph_hits += num_items
        return out_jax

    def execute(self, mm_kwargs: dict[str, Any]) -> list[jax.Array]:
        """Run the encoder on one MM batch and return per-item outputs.

        Reuses the inherited greedy bin-packing ``_execute_local`` wholesale.
        It runs in plain-torch context (no outer torchax env): item
        selection and replay-buffer prep stay as normal torch (they index
        model Parameters), while the per-budget JIT forward and the eager
        fallback each enter the torchax env locally and return ``jax.Array``.
        The encoder outputs flow as jax arrays through the adapter's
        ``postprocess_encoder_output`` / ``scatter_output_slices``, so the
        result is already a ``list[jax.Array]`` — matching what the caller
        expects from ``runner.embed_multimodal_fn(...)``.
        """
        return self._execute_local(mm_kwargs)

    def precompile_vision_encoder(self, run_compilation: Callable) -> None:
        for budget in self.token_budgets:
            run_compilation(
                "mm_encoder_jit",
                self._capture_budget_graph,
                budget,
                budget=budget,
            )


def maybe_create_mm_encoder_jit_manager(
    vllm_config: "VllmConfig",
    vllm_model: Any,
    vllm_runner: "torch.nn.Module | None",
    params_and_buffers: Any,
) -> "MMEncoderJITManager | None":
    if not vllm_config.compilation_config.cudagraph_mm_encoder:
        return None
    if not supports_encoder_cudagraph(vllm_model):
        return None
    return MMEncoderJITManager(
        vllm_config=vllm_config,
        vllm_runner=vllm_runner,
        vllm_model=vllm_model,
        params_and_buffers=params_and_buffers,
    )
