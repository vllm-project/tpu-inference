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
    * ``execute`` — run the inherited ``_execute_local`` inside one
      ``torchax.default_env()`` (plain-torch prep falls through untouched;
      the encoder output flows as torchax tensors through
      ``postprocess_encoder_output`` / ``scatter_output_slices``) and bridge
      the per-item outputs back to ``jax.Array`` at the boundary.

The inherited ``_execute_local`` calls ``model.encoder_eager_forward`` for
the oversize-item fallback. On TPU the model's own weights are meta/empty —
the real params live in ``params_and_buffers`` — so we pass a thin
``_TorchaxEncoderModelAdapter`` as the ``model`` to the parent: it bridges
``encoder_eager_forward`` through ``functional_call`` and delegates every
other protocol method to the real vLLM model.

The JIT closure is built once at init (the v7 cache-share lesson — see
``round2_bench/perf_report.html`` §6.6 for the per-call PjitFunction trap
this avoids).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

import jax
import torch
import torchax
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from tpu_inference.logger import init_logger

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

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        # Bridge plain-torch mm_kwargs -> torchax, dispatch the model's
        # eager vision forward via functional_call (binds the real TPU
        # weights), return a torchax tensor. The caller (inherited
        # _execute_local) slices/clones it under the active torchax env.
        torchax_kwargs = {
            k: jax.tree.map(_torchax_view_if_torch, v)
            for k, v in mm_kwargs.items()
        }
        return torch.func.functional_call(
            self._runner,
            torch_view(self._params),
            kwargs={
                "call_method": "encoder_eager_forward",
                "call_args": (torchax_kwargs, ),
                "call_kwargs": {},
            },
            tie_weights=False,
        )

    def __getattr__(self, name: str) -> Any:
        # Delegate all non-overridden protocol methods to the real model.
        return getattr(self._model, name)


class MMEncoderJITManager(EncoderCudaGraphManager):
    """Per-budget XLA-cache manager for the vision encoder forward."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        vllm_runner: torch.nn.Module,
        vllm_model: Any,
        params_and_buffers: Any,
        token_budget_override: int = 0,
        max_batch_size_override: int = 0,
    ):
        """
        Args:
          vllm_config: The vllm config object (budget knobs come from
              ``compilation_config.encoder_cudagraph_*``).
          vllm_runner: The torchax-wrapped ``_VllmRunner`` (provides
              ``forward(call_method=..., call_args=...)``-style dispatch
              required by ``torch.func.functional_call``).
          vllm_model: The underlying vllm model (e.g.
              ``Qwen3VLForConditionalGeneration``). Must implement
              ``SupportsEncoderCudaGraph``.
          params_and_buffers: The model's loaded weights as a pytree of
              JAX arrays. Bound into ``functional_call`` per request.
          token_budget_override / max_batch_size_override: 0 to keep the
              upstream-derived values. Non-zero forces a single budget /
              fixed batch size (env-driven A/B knobs).
        """
        from tpu_inference.utils import to_jax_dtype

        self.vllm_runner = vllm_runner
        self.params_and_buffers = params_and_buffers
        self._jax_dtype = to_jax_dtype(vllm_config.model_config.dtype)

        # The parent calls model.{get_encoder_cudagraph_config,
        # get_encoder_cudagraph_budget_range}; the inherited _execute_local
        # later calls model.{select_encoder_cudagraph_items,
        # encoder_eager_forward, postprocess_encoder_output}. Route eager
        # forward through the torchax runner via the adapter.
        adapter = _TorchaxEncoderModelAdapter(vllm_model, vllm_runner,
                                              params_and_buffers)

        # Reuse upstream budget derivation + validation. Capture inputs are
        # built on CPU; the JIT path moves them to TPU via t2j, so we never
        # run the encoder on a CUDA device.
        super().__init__(
            vllm_config=vllm_config,
            device=torch.device("cpu"),
            dtype=vllm_config.model_config.dtype,
            model=adapter,
        )
        # Keep the raw model for the grid/pixel helper calls in the
        # metadata-cache path (the adapter would delegate, but referencing
        # the model directly is clearer).
        self.vllm_model = vllm_model

        # Local single-budget / fixed-batch overrides (env-driven A/B). 0
        # keeps the upstream-derived values from super().__init__.
        if token_budget_override > 0:
            self.token_budgets = [token_budget_override]
        if max_batch_size_override > 0:
            self.max_batch_size = max_batch_size_override
        # Re-assert the parent's invariant (max_batch_size <= min budget) so
        # per_item_output >= 1 in the capture grid.
        min_tb = min(self.token_budgets)
        if self.max_batch_size > min_tb:
            logger.warning(
                "[mm_encoder_jit] capping max_batch_size %d -> %d to "
                "keep per_item_output >= 1 for smallest budget %d",
                self.max_batch_size, min_tb, min_tb)
            self.max_batch_size = min_tb

        # Capture templates per budget — shape signature reference for
        # host-side padding. The values inside templates are dummy; only
        # tensor.shape / tensor.dtype matter (to us and to XLA's cache key).
        capture_device = torch.device("cpu")
        capture_dtype = vllm_config.model_config.dtype
        self.budget_templates: dict[int, dict[str, torch.Tensor]] = {}
        for budget in self.token_budgets:
            capture = vllm_model.prepare_encoder_cudagraph_capture_inputs(
                budget, self.max_batch_size, self.max_frames_per_batch,
                capture_device, capture_dtype)
            self.budget_templates[budget] = capture.values

        logger.info(
            "[mm_encoder_jit] budgets=%s max_batch_size=%d "
            "max_frames_per_batch=%d template_keys=%s", self.token_budgets,
            self.max_batch_size, self.max_frames_per_batch,
            list(next(iter(self.budget_templates.values())).keys()))

        # Hoist the JIT-wrap once (v7 cache-share fix — PjitFunction lives
        # on this instance and accumulates per-shape entries from here).
        self._jit_forward = jax.jit(self._build_forward_closure())

        # Per-shape JAX-side metadata cache used when
        # ``USE_ENCODER_METADATA_CACHE=1``. Maps cache_key -> {key:
        # jax.Array} for all keys EXCEPT pixel_values. See
        # _prepare_padded_jax for the cache_key construction.
        self._cache_metadata_jax: dict[Any, dict] = {}

    # ----- JIT forward closure -----

    def _build_forward_closure(self) -> Callable:
        """Build the closure that gets ``jax.jit``-wrapped exactly once.

        Inputs: ``params_jax`` (the model weights) + ``values_jax`` (the
        full padded buffer dict including ``pixel_values``).

        Inside: bridge jax -> torchax with ``torch_view``, dispatch via
        ``functional_call(call_method="encoder_cudagraph_forward")``,
        bridge torchax output back to jax with ``jax_view``.
        """
        vllm_runner = self.vllm_runner

        def _forward(params_jax: Any, values_jax: dict[str, jax.Array]):
            params_torchax = torch_view(params_jax)
            values_torchax = {
                k: jax.tree.map(torch_view, v)
                for k, v in values_jax.items()
            }
            out_torch = torch.func.functional_call(
                vllm_runner,
                params_torchax,
                kwargs={
                    "call_method": "encoder_cudagraph_forward",
                    "call_args": (values_torchax, ),
                    "call_kwargs": {},
                },
                tie_weights=False,
            )
            return jax_view(out_torch)

        return _forward

    # ----- Padding (per-key) -----

    def _pad_to_template(
        self,
        replay_values: dict[str, torch.Tensor],
        budget: int,
    ) -> dict[str, torch.Tensor]:
        """Zero-and-copy each replay tensor into a template-shaped buffer.

        Mirrors the GPU manager's ``_copy_padded_buffer`` default (the
        parent's ``_run_budget_graph`` does this per buffer_key). Here we pad
        *every* template key because the jit call needs a fresh, fully
        materialised input dict each step (no persistent buffers).
        cu_seqlens / scalars pass through because ``prepare_encoder_metadata``
        already padded them.
        """
        template = self.budget_templates[budget]
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

    def _prepare_padded_jax(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
    ) -> dict[str, jax.Array]:
        """Build the full padded, jax-converted buffer dict for one batch.

        Optional metadata cache (``USE_ENCODER_METADATA_CACHE=1``): skip the
        expensive ``prepare_encoder_cudagraph_replay_buffers`` call
        (fast_pos_embed_interpolate + rot_pos_emb host CPU work) when we have
        seen this (budget, grid_thw) before. Only the (h,w)-derived METADATA
        tensors are cached; ``pixel_values`` always comes fresh from the real
        call to avoid the stale-input bug from earlier probe versions.

        Must be called inside an active ``torchax.default_env()``.
        """
        use_cache = os.environ.get("USE_ENCODER_METADATA_CACHE", "0") == "1"
        _grid = self.vllm_model._get_grid_thw_by_modality(mm_kwargs)
        # grid_thw arrives as list[list[int]] or torch.Tensor; lift to nested
        # tuple so it can hash as a dict key.
        cache_key = (token_budget,
                     tuple(
                         tuple(g) if isinstance(g, list) else g
                         for g in _grid))

        if use_cache and cache_key in self._cache_metadata_jax:
            cached_jax = self._cache_metadata_jax[cache_key]
            # Build (fresh pixel_values) + (cached metadata).
            fresh_pv = self.vllm_model._get_pixel_values_by_modality(mm_kwargs)
            tmpl_pv = self.budget_templates[token_budget].get("pixel_values")
            if tmpl_pv is not None and fresh_pv.shape != tmpl_pv.shape:
                buf = torch.zeros_like(tmpl_pv)
                buf[:fresh_pv.shape[0]] = fresh_pv.to(dtype=tmpl_pv.dtype,
                                                      device=tmpl_pv.device)
                pv_padded = buf
            else:
                pv_padded = fresh_pv
            padded_jax = dict(cached_jax)
            padded_jax["pixel_values"] = jax.tree.map(_t2j_if_tensor,
                                                      pv_padded)
            return padded_jax

        replay = self.model.prepare_encoder_cudagraph_replay_buffers(
            mm_kwargs, self.max_batch_size, self.max_frames_per_batch)
        padded_torch = self._pad_to_template(replay.values, token_budget)
        padded_jax = {
            k: jax.tree.map(_t2j_if_tensor, v)
            for k, v in padded_torch.items()
        }
        if use_cache:
            # Cache ONLY metadata; pixel_values is replaced by a fresh
            # extraction next time.
            self._cache_metadata_jax[cache_key] = {
                k: v
                for k, v in padded_jax.items() if k != "pixel_values"
            }
        return padded_jax

    # ----- Overrides of the CUDA-graph device hooks -----

    def _capture_budget_graph(self, token_budget: int) -> None:
        """XLA-cache analog of ``torch.cuda.graph`` capture.

        Primes the ``jax.jit`` cache for ``token_budget`` by calling the
        forward closure once on the budget's dummy template, so the first
        runtime call is a cache hit instead of paying compile time. Invoked
        by the inherited ``capture()`` loop (optional — the cache also fills
        lazily on first ``execute``).
        """
        from vllm.ir import enable_torch_wrap

        template = self.budget_templates[token_budget]
        with torchax.default_env(), enable_torch_wrap(False):
            values_jax = {
                k: jax.tree.map(_t2j_if_tensor, v)
                for k, v in template.items()
            }
            out = self._jit_forward(self.params_and_buffers, values_jax)
            jax.block_until_ready(out)
        # Mark captured so inherited capture()/get_cumulative_stats count it.
        self.budget_graphs[token_budget] = template

    def _run_budget_graph(
        self,
        mm_kwargs: dict[str, Any],
        token_budget: int,
    ) -> torch.Tensor | None:
        """XLA-cache analog of CUDA-graph replay.

        Host-pads the replay buffers to the budget template shape and calls
        the once-built ``jax.jit`` closure. Returns the encoder output as a
        **torchax tensor** so the inherited ``_execute_local`` can slice and
        clone it through ``postprocess_encoder_output`` under the active
        torchax env. The jax bridge happens at the ``execute`` boundary.

        Must be called inside an active ``torchax.default_env()``.
        """
        from vllm.ir import enable_torch_wrap

        num_items = len(self._get_item_specs(mm_kwargs))
        if token_budget not in self.budget_templates:
            # No template for this budget (shouldn't happen — budget comes
            # from _find_smallest_fitting_budget over self.token_budgets).
            self.graph_misses += num_items
            return None

        padded_jax = self._prepare_padded_jax(mm_kwargs, token_budget)
        with enable_torch_wrap(False):
            out_jax = self._jit_forward(self.params_and_buffers, padded_jax)
        self.graph_hits += num_items
        return torch_view(out_jax)

    def execute(self, mm_kwargs: dict[str, Any]) -> list[jax.Array]:
        """Run the encoder on one MM batch and return per-item outputs.

        Reuses the inherited greedy bin-packing ``_execute_local`` wholesale,
        running it inside one ``torchax.default_env()``: plain-torch prep
        (item selection, replay-buffer prep, host padding) falls through
        untouched, while the encoder output flows as torchax tensors through
        ``postprocess_encoder_output`` / ``scatter_output_slices``. Outputs
        are bridged to ``jax.Array`` at the boundary so the caller behaves
        like ``runner.embed_multimodal_fn(...)``.
        """
        assert not self.use_dp, (
            "[mm_encoder_jit] data-parallel encoder path uses torch "
            "collectives and is unsupported on the TPU manager")
        with torchax.default_env():
            torch_outputs = self._execute_local(mm_kwargs)
            return [jax_view(t) for t in torch_outputs]


def _t2j_if_tensor(v):
    """Tree-map helper — convert leaf torch.Tensors to jax.Array."""
    if isinstance(v, torch.Tensor):
        return t2j(v, use_dlpack=False)
    return v


def _torchax_view_if_torch(v):
    """Tree-map helper for the eager path."""
    if isinstance(v, torch.Tensor):
        return torch_view(t2j(v, use_dlpack=False))
    return v
