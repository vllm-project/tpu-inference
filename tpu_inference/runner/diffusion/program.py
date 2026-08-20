# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.runner.diffusion.algorithm import (
    CommitDiagnostics, CommitDiagnosticsFn, CommitFn, _exclude_mask_token,
    confidence_threshold_with_log_bias)
from tpu_inference.runner.diffusion.config import (LogitAlignment,
                                                   NextBlockPolicy)

BlockForwardFn = Callable[
    [Any, jax.Array, jax.Array, Any, jax.Array, Any],
    tuple[jax.Array, Any],
]

AlignedSubBlockForwardFn = Callable[
    [Any, jax.Array, jax.Array, Any, jax.Array, Any, jax.Array],
    tuple[jax.Array, Any],
]

FinalBlockForwardFn = Callable[
    [Any, jax.Array, jax.Array, Any, jax.Array, Any],
    tuple[jax.Array, Any],
]

DUAL_CACHE_Q32 = 0
DUAL_CACHE_Q8 = 1


class DenoiseBlockOutput(NamedTuple):
    canvas: jax.Array
    next_anchor: jax.Array
    denoise_steps: jax.Array
    kv_caches: Any
    stopped_rows: jax.Array


class DualCacheAcceptanceTrace(NamedTuple):
    count: jax.Array
    block_start: jax.Array
    sub_block_starts: jax.Array
    iterations: jax.Array
    forward_kinds: jax.Array
    row0_token_ids: jax.Array
    row0_eligible: jax.Array
    row0_commit: jax.Array
    row0_remaining: jax.Array
    row0_forced_anchor: jax.Array
    row0_selected_log_confidence: jax.Array
    row0_threshold_margin: jax.Array


class DualCacheDenoiseBlockOutput(NamedTuple):
    canvas: jax.Array
    next_anchor: jax.Array
    denoise_steps: jax.Array
    kv_caches: Any
    q32_forward_calls: jax.Array
    q8_forward_calls: jax.Array
    final_q32_forward_calls: jax.Array
    forced_q32_anchor_commits: jax.Array
    stopped_rows: jax.Array
    acceptance_trace: DualCacheAcceptanceTrace | None


def _empty_dual_cache_acceptance_trace(
    positions: jax.Array,
    max_steps: int,
    sub_block_size: int,
) -> DualCacheAcceptanceTrace:
    step_shape = (max_steps, )
    row_shape = (max_steps, sub_block_size)
    return DualCacheAcceptanceTrace(
        count=jnp.array(0, dtype=jnp.int32),
        block_start=positions[0, 0].astype(jnp.int32),
        sub_block_starts=jnp.zeros(step_shape, dtype=jnp.int32),
        iterations=jnp.zeros(step_shape, dtype=jnp.int32),
        forward_kinds=jnp.zeros(step_shape, dtype=jnp.int8),
        row0_token_ids=jnp.zeros(row_shape, dtype=jnp.int32),
        row0_eligible=jnp.zeros(row_shape, dtype=bool),
        row0_commit=jnp.zeros(row_shape, dtype=bool),
        row0_remaining=jnp.zeros(row_shape, dtype=bool),
        row0_forced_anchor=jnp.zeros(step_shape, dtype=bool),
        row0_selected_log_confidence=jnp.zeros(row_shape, dtype=jnp.float32),
        row0_threshold_margin=jnp.zeros(row_shape, dtype=jnp.float32),
    )


def _record_dual_cache_acceptance_step(
    trace: DualCacheAcceptanceTrace,
    sub_block_start: jax.Array,
    iteration: jax.Array,
    forward_kind: int,
    token_ids: jax.Array,
    eligible: jax.Array,
    remaining: jax.Array,
    forced_anchor_rows: jax.Array,
    diagnostics: CommitDiagnostics,
) -> DualCacheAcceptanceTrace:
    index = trace.count
    committed = eligible & ~remaining
    return DualCacheAcceptanceTrace(
        count=index + 1,
        block_start=trace.block_start,
        sub_block_starts=trace.sub_block_starts.at[index].set(sub_block_start),
        iterations=trace.iterations.at[index].set(iteration),
        forward_kinds=trace.forward_kinds.at[index].set(forward_kind),
        row0_token_ids=trace.row0_token_ids.at[index].set(token_ids[0]),
        row0_eligible=trace.row0_eligible.at[index].set(eligible[0]),
        row0_commit=trace.row0_commit.at[index].set(committed[0]),
        row0_remaining=trace.row0_remaining.at[index].set(remaining[0]),
        row0_forced_anchor=trace.row0_forced_anchor.at[index].set(
            forced_anchor_rows[0]),
        row0_selected_log_confidence=(
            trace.row0_selected_log_confidence.at[index].set(
                diagnostics.selected_log_confidence[0])),
        row0_threshold_margin=trace.row0_threshold_margin.at[index].set(
            diagnostics.threshold_margin[0]),
    )


def _resolved_eos_rows(
    canvas: jax.Array,
    mask: jax.Array,
    generation_positions: jax.Array,
    stop_on_eos_rows: jax.Array,
    eos_token_ids: tuple[int, ...],
) -> jax.Array:
    if not eos_token_ids:
        return jnp.zeros((canvas.shape[0], ), dtype=bool)

    is_eos = jnp.zeros_like(mask)
    for token_id in eos_token_ids:
        is_eos |= canvas == token_id
    is_eos &= generation_positions

    has_eos = jnp.any(is_eos, axis=-1)
    first_eos = jnp.argmax(is_eos, axis=-1)
    positions = jnp.arange(canvas.shape[1], dtype=jnp.int32)
    unresolved_before_eos = jnp.any(
        mask & generation_positions &
        (positions[None, :] < first_eos[:, None]),
        axis=-1,
    )
    return stop_on_eos_rows & has_eos & ~unresolved_before_eos


def select_aligned_hidden_states(
    hidden_states: jax.Array,
    batch_size: int,
    query_length: int,
    start: jax.Array,
    sub_block_size: int,
    alignment: LogitAlignment,
    *,
    local_alignment: bool,
) -> jax.Array:
    hidden_states = hidden_states.reshape(batch_size, query_length, -1)
    indices = jnp.arange(sub_block_size, dtype=jnp.int32)
    if not local_alignment:
        indices += start
    if alignment is LogitAlignment.SHIFTED:
        indices = jnp.maximum(indices - 1, 0)
    selected = jnp.take(hidden_states, indices, axis=1)
    return selected.reshape(-1, hidden_states.shape[-1])


def _align_logits(
    logits: jax.Array,
    alignment: LogitAlignment,
) -> jax.Array:
    if alignment is LogitAlignment.SAME_POSITION:
        return logits
    if alignment is LogitAlignment.SHIFTED:
        indices = jnp.maximum(
            jnp.arange(logits.shape[1], dtype=jnp.int32) - 1, 0)
        return logits[:, indices, :]
    raise ValueError(f"Unsupported logit alignment: {alignment}")


@functools.partial(
    jax.jit,
    static_argnames=(
        "forward_fn",
        "commit_fn",
        "logit_alignment",
        "next_block_policy",
        "mask_token_id",
        "sub_block_size",
        "max_denoise_steps",
        "eos_token_ids",
    ),
)
def _denoise_block_jit(
        forward_fn: BlockForwardFn,
        commit_fn: CommitFn,
        model_state: Any,
        initial_canvas: jax.Array,
        initial_mask: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        needs_final_forward: jax.Array,
        stop_on_eos_rows: jax.Array,
        confidence_threshold: jax.Array,
        temperature: jax.Array,
        forward_context: Any,
        *,
        logit_alignment: LogitAlignment,
        next_block_policy: NextBlockPolicy,
        mask_token_id: int,
        sub_block_size: int,
        max_denoise_steps: int = 0,
        eos_token_ids: tuple[int, ...] = (),
) -> DenoiseBlockOutput:
    """Jitted block denoising implementation.

    ``forward_fn`` must leave cache entries for false ``active_rows`` unchanged.
    The cache is opaque to this program, so inactive-row isolation belongs to
    the forward implementation that maps logical rows to physical cache pages.

    ``max_denoise_steps`` limits while-loop iterations for each sub-block, not
    for the full model block. Zero uses ``sub_block_size`` iterations. The full
    block can therefore run ``num_sub_blocks * steps_per_sub_block`` denoising
    forwards, followed by one final forward that commits the completed KV state.
    """
    if initial_canvas.ndim != 2:
        raise ValueError("initial_canvas must have shape [batch, block_size]")
    if initial_mask.shape != initial_canvas.shape:
        raise ValueError("initial_mask must match initial_canvas")
    if positions.shape != initial_canvas.shape:
        raise ValueError("positions must match initial_canvas")
    if sub_block_size < 1 or initial_canvas.shape[1] % sub_block_size:
        raise ValueError("sub_block_size must divide the model block size")

    batch_size, block_size = initial_canvas.shape
    del batch_size
    active_rows = jnp.asarray(active_rows, dtype=bool)
    canvas = initial_canvas.astype(jnp.int32)
    mask = jnp.asarray(initial_mask, dtype=bool) & active_rows[:, None]
    generation_positions = mask
    stop_on_eos_rows = jnp.asarray(stop_on_eos_rows, dtype=bool) & active_rows
    stopped_rows = jnp.zeros_like(active_rows)
    denoise_steps = jnp.zeros((canvas.shape[0], ), dtype=jnp.int32)
    steps_per_sub_block = (sub_block_size
                           if max_denoise_steps <= 0 else max_denoise_steps)
    num_sub_blocks = block_size // sub_block_size

    def denoise_sub_block(sub_block_index, carry):
        canvas, mask, denoise_steps, kv, stopped_rows = carry
        start = sub_block_index * sub_block_size
        sub_block_positions = (
            (jnp.arange(block_size) >= start) &
            (jnp.arange(block_size) < start + sub_block_size))
        eligible = mask & sub_block_positions[None, :]
        last_tokens = canvas

        state = (
            canvas,
            mask,
            denoise_steps,
            kv,
            eligible,
            last_tokens,
            jnp.array(0, dtype=jnp.int32),
            stopped_rows,
        )

        def has_work(state):
            _, _, _, _, eligible, _, iteration, _ = state
            return ((iteration < steps_per_sub_block) & jnp.any(eligible))

        def denoise_step(state):
            (canvas, mask, steps, kv, eligible, _, iteration,
             stopped_rows) = state
            row_has_work = jnp.any(eligible, axis=-1)
            logits, kv = forward_fn(model_state, canvas, positions, kv,
                                    row_has_work, forward_context)
            aligned_logits = _align_logits(logits, logit_alignment)
            token_ids, remaining = commit_fn(
                aligned_logits,
                eligible,
                active_rows,
                confidence_threshold,
                temperature,
                mask_token_id,
            )
            committed = eligible & ~remaining
            canvas = jnp.where(committed, token_ids, canvas)
            mask &= ~committed
            newly_stopped = _resolved_eos_rows(
                canvas,
                mask,
                generation_positions,
                stop_on_eos_rows,
                eos_token_ids,
            )
            stopped_rows |= newly_stopped
            mask &= ~stopped_rows[:, None]
            remaining &= ~stopped_rows[:, None]
            steps += row_has_work.astype(jnp.int32)
            return (canvas, mask, steps, kv, remaining, token_ids,
                    iteration + 1, stopped_rows)

        state = jax.lax.while_loop(has_work, denoise_step, state)
        (canvas, mask, denoise_steps, kv, remaining, last_tokens, _,
         stopped_rows) = state

        canvas = jnp.where(remaining, last_tokens, canvas)
        mask &= ~remaining
        newly_stopped = _resolved_eos_rows(
            canvas,
            mask,
            generation_positions,
            stop_on_eos_rows,
            eos_token_ids,
        )
        stopped_rows |= newly_stopped
        mask &= ~stopped_rows[:, None]
        return canvas, mask, denoise_steps, kv, stopped_rows

    canvas, mask, denoise_steps, kv_caches, stopped_rows = jax.lax.fori_loop(
        0,
        num_sub_blocks,
        denoise_sub_block,
        (canvas, mask, denoise_steps, kv_caches, stopped_rows),
    )

    final_rows = active_rows & needs_final_forward & ~stopped_rows

    def run_final_forward(kv):
        final_logits, next_kv = forward_fn(model_state, canvas, positions, kv,
                                           final_rows, forward_context)
        if next_block_policy is NextBlockPolicy.LAST_LOGIT_ANCHOR:
            anchor_logits = _exclude_mask_token(final_logits[:, -1:, :],
                                                mask_token_id)
            next_anchor = jnp.argmax(anchor_logits[:, 0, :],
                                     axis=-1).astype(jnp.int32)
        elif next_block_policy is NextBlockPolicy.ALL_MASKED:
            next_anchor = jnp.zeros((canvas.shape[0], ), dtype=jnp.int32)
        else:
            raise ValueError(
                f"Unsupported next block policy: {next_block_policy}")
        return jnp.where(final_rows, next_anchor, 0), next_kv

    def skip_final_forward(kv):
        return jnp.zeros((canvas.shape[0], ), dtype=jnp.int32), kv

    next_anchor, kv_caches = jax.lax.cond(jnp.any(final_rows),
                                          run_final_forward,
                                          skip_final_forward, kv_caches)
    return DenoiseBlockOutput(
        canvas=canvas,
        next_anchor=next_anchor,
        denoise_steps=denoise_steps,
        kv_caches=kv_caches,
        stopped_rows=stopped_rows,
    )


def denoise_block(
        forward_fn: BlockForwardFn,
        commit_fn: CommitFn,
        model_state: Any,
        initial_canvas: jax.Array,
        initial_mask: jax.Array,
        positions: jax.Array,
        kv_caches: Any,
        active_rows: jax.Array,
        confidence_threshold: jax.Array,
        temperature: jax.Array,
        forward_context: Any,
        *,
        logit_alignment: LogitAlignment,
        next_block_policy: NextBlockPolicy,
        mask_token_id: int,
        sub_block_size: int,
        max_denoise_steps: int = 0,
        needs_final_forward: jax.Array | None = None,
        stop_on_eos_rows: jax.Array | None = None,
        eos_token_ids: tuple[int, ...] = (),
) -> DenoiseBlockOutput:
    """Validate and denoise one model block.

    Shifted logits predict canvas position ``i + 1`` from logits position ``i``.
    Position zero therefore has no aligned prediction and must be an immutable
    seed for every active row. This host-side check fails before compilation or
    token commitment instead of silently consuming the wrong logits.
    """
    if initial_canvas.ndim != 2:
        raise ValueError("initial_canvas must have shape [batch, block_size]")
    if initial_mask.shape != initial_canvas.shape:
        raise ValueError("initial_mask must match initial_canvas")
    if active_rows.shape != (initial_canvas.shape[0], ):
        raise ValueError("active_rows must match the canvas batch size")
    if needs_final_forward is None:
        needs_final_forward = active_rows
    elif needs_final_forward.shape != active_rows.shape:
        raise ValueError("needs_final_forward must match active_rows")
    if stop_on_eos_rows is None:
        stop_on_eos_rows = jnp.zeros_like(active_rows, dtype=bool)
    elif stop_on_eos_rows.shape != active_rows.shape:
        raise ValueError("stop_on_eos_rows must match active_rows")
    if logit_alignment is LogitAlignment.SHIFTED:
        seed_mask, active = jax.device_get((initial_mask[:, 0], active_rows))
        if np.any(
                np.asarray(seed_mask, dtype=bool)
                & np.asarray(active, dtype=bool)):
            raise ValueError(
                "Shifted logit alignment requires position 0 to be an "
                "unmasked seed for every active row")

    return _denoise_block_jit(
        forward_fn,
        commit_fn,
        model_state,
        initial_canvas,
        initial_mask,
        positions,
        kv_caches,
        active_rows,
        needs_final_forward,
        stop_on_eos_rows,
        confidence_threshold,
        temperature,
        forward_context,
        logit_alignment=logit_alignment,
        next_block_policy=next_block_policy,
        mask_token_id=mask_token_id,
        sub_block_size=sub_block_size,
        max_denoise_steps=max_denoise_steps,
        eos_token_ids=eos_token_ids,
    )


@functools.partial(
    jax.jit,
    donate_argnums=(9, ),
    # Model forwards are nested no-options JITs, so TPU compiler options must
    # live on this top-level denoise-loop JIT.
    compiler_options={
        "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
        "xla_tpu_reduce_scatter_collective_matmul_mode":
        "post_spmd_conservative",
        "xla_tpu_use_minor_sharding_for_major_trivial_input": "true",
    },
    static_argnames=(
        "full_forward_fn",
        "partial_forward_fn",
        "final_forward_fn",
        "commit_fn",
        "commit_diagnostics_fn",
        "next_block_policy",
        "mask_token_id",
        "sub_block_size",
        "max_denoise_steps",
        "eos_token_ids",
        "trace_acceptance_steps",
        "q8_log_confidence_bias",
        "force_q32_anchor_commit",
    ),
)
def _denoise_block_dual_cache_jit(
    full_forward_fn: AlignedSubBlockForwardFn,
    partial_forward_fn: AlignedSubBlockForwardFn,
    final_forward_fn: FinalBlockForwardFn,
    commit_fn: CommitFn,
    commit_diagnostics_fn: CommitDiagnosticsFn | None,
    model_state: Any,
    initial_canvas: jax.Array,
    initial_mask: jax.Array,
    positions: jax.Array,
    kv_caches: Any,
    active_rows: jax.Array,
    needs_final_forward: jax.Array,
    stop_on_eos_rows: jax.Array,
    confidence_threshold: jax.Array,
    temperature: jax.Array,
    forward_context: Any,
    *,
    next_block_policy: NextBlockPolicy,
    mask_token_id: int,
    sub_block_size: int,
    max_denoise_steps: int = 0,
    eos_token_ids: tuple[int, ...] = (),
    trace_acceptance_steps: bool = False,
    q8_log_confidence_bias: float = 0.0,
    force_q32_anchor_commit: bool = False,
) -> DualCacheDenoiseBlockOutput:
    batch_size, block_size = initial_canvas.shape
    active_rows = jnp.asarray(active_rows, dtype=bool)
    canvas = initial_canvas.astype(jnp.int32)
    mask = jnp.asarray(initial_mask, dtype=bool) & active_rows[:, None]
    generation_positions = mask
    stop_on_eos_rows = jnp.asarray(stop_on_eos_rows, dtype=bool) & active_rows
    stopped_rows = jnp.zeros_like(active_rows)
    denoise_steps = jnp.zeros((batch_size, ), dtype=jnp.int32)
    steps_per_sub_block = (sub_block_size
                           if max_denoise_steps <= 0 else max_denoise_steps)
    num_sub_blocks = block_size // sub_block_size
    q8_confidence_threshold = confidence_threshold_with_log_bias(
        confidence_threshold, q8_log_confidence_bias)
    acceptance_trace = (_empty_dual_cache_acceptance_trace(
        positions,
        num_sub_blocks * steps_per_sub_block,
        sub_block_size,
    ) if trace_acceptance_steps else None)

    def record_acceptance_step(
        trace,
        start,
        iteration,
        forward_kind,
        logits,
        token_ids,
        eligible,
        remaining,
        forced_anchor_rows,
        step_confidence_threshold,
    ):
        if not trace_acceptance_steps:
            return trace
        assert commit_diagnostics_fn is not None
        diagnostics = commit_diagnostics_fn(
            logits,
            step_confidence_threshold,
            temperature,
            mask_token_id,
        )
        return _record_dual_cache_acceptance_step(
            trace,
            positions[0, 0] + start,
            iteration,
            forward_kind,
            token_ids,
            eligible,
            remaining,
            forced_anchor_rows,
            diagnostics,
        )

    def denoise_sub_block(sub_block_index, carry):
        (canvas, mask, denoise_steps, kv, q32_calls, q8_calls,
         forced_anchor_commits, stopped_rows, acceptance_trace) = carry
        start = sub_block_index * sub_block_size
        sub_canvas = jax.lax.dynamic_slice(canvas, (0, start),
                                           (batch_size, sub_block_size))
        eligible = jax.lax.dynamic_slice(mask, (0, start),
                                         (batch_size, sub_block_size))

        def no_work(_):
            return (canvas, mask, denoise_steps, kv, q32_calls, q8_calls,
                    forced_anchor_commits, stopped_rows, acceptance_trace)

        def work(_):

            def apply_eos_stop(current_sub_canvas, current_sub_mask,
                               current_remaining, current_stopped_rows):
                current_canvas = jax.lax.dynamic_update_slice(
                    canvas, current_sub_canvas, (0, start))
                current_mask = jax.lax.dynamic_update_slice(
                    mask, current_sub_mask, (0, start))
                newly_stopped = _resolved_eos_rows(
                    current_canvas,
                    current_mask,
                    generation_positions,
                    stop_on_eos_rows,
                    eos_token_ids,
                )
                current_stopped_rows |= newly_stopped
                current_sub_mask &= ~current_stopped_rows[:, None]
                current_remaining &= ~current_stopped_rows[:, None]
                return (current_sub_canvas, current_sub_mask,
                        current_remaining, current_stopped_rows)

            row_has_work = jnp.any(eligible, axis=-1)
            with jax.named_scope("diffusion_dual_cache_q32"):
                logits, next_kv = full_forward_fn(
                    model_state,
                    canvas,
                    positions,
                    kv,
                    row_has_work,
                    forward_context,
                    start,
                )
            token_ids, remaining = commit_fn(
                logits,
                eligible,
                row_has_work,
                confidence_threshold,
                temperature,
                mask_token_id,
            )
            if force_q32_anchor_commit:
                forced_anchor_rows = remaining[:, 0]
                remaining = remaining.at[:, 0].set(False)
            else:
                forced_anchor_rows = jnp.zeros_like(remaining[:, 0])
            next_forced_anchor_commits = (
                forced_anchor_commits +
                jnp.sum(forced_anchor_rows.astype(jnp.int32)))
            next_acceptance_trace = record_acceptance_step(
                acceptance_trace,
                start,
                jnp.array(0, dtype=jnp.int32),
                DUAL_CACHE_Q32,
                logits,
                token_ids,
                eligible,
                remaining,
                forced_anchor_rows,
                confidence_threshold,
            )
            committed = eligible & ~remaining
            next_sub_canvas = jnp.where(committed, token_ids, sub_canvas)
            next_sub_mask = eligible & ~committed
            (next_sub_canvas, next_sub_mask, remaining,
             next_stopped_rows) = apply_eos_stop(next_sub_canvas,
                                                 next_sub_mask, remaining,
                                                 stopped_rows)
            next_steps = denoise_steps + row_has_work.astype(jnp.int32)

            state = (
                next_sub_canvas,
                next_sub_mask,
                next_steps,
                next_kv,
                remaining,
                token_ids,
                jnp.array(1, dtype=jnp.int32),
                q32_calls + 1,
                q8_calls,
                next_stopped_rows,
                next_acceptance_trace,
            )

            def needs_full_refresh(state):
                _, _, _, _, remaining, _, iteration, _, _, _, _ = state
                return ((iteration < steps_per_sub_block)
                        & jnp.any(remaining[:, 0]))

            def full_refresh_step(state):
                (current_sub_canvas, current_sub_mask, current_steps,
                 current_kv, remaining, _, iteration, current_q32_calls,
                 current_q8_calls, current_stopped_rows,
                 current_acceptance_trace) = state
                current_canvas = jax.lax.dynamic_update_slice(
                    canvas, current_sub_canvas, (0, start))
                row_has_work = jnp.any(remaining, axis=-1)
                with jax.named_scope("diffusion_dual_cache_q32"):
                    logits, current_kv = full_forward_fn(
                        model_state,
                        current_canvas,
                        positions,
                        current_kv,
                        row_has_work,
                        forward_context,
                        start,
                    )
                token_ids, next_remaining = commit_fn(
                    logits,
                    remaining,
                    row_has_work,
                    confidence_threshold,
                    temperature,
                    mask_token_id,
                )
                current_acceptance_trace = record_acceptance_step(
                    current_acceptance_trace,
                    start,
                    iteration,
                    DUAL_CACHE_Q32,
                    logits,
                    token_ids,
                    remaining,
                    next_remaining,
                    jnp.zeros((batch_size, ), dtype=bool),
                    confidence_threshold,
                )
                committed = remaining & ~next_remaining
                current_sub_canvas = jnp.where(committed, token_ids,
                                               current_sub_canvas)
                current_sub_mask &= ~committed
                (current_sub_canvas, current_sub_mask, next_remaining,
                 current_stopped_rows) = apply_eos_stop(
                     current_sub_canvas, current_sub_mask, next_remaining,
                     current_stopped_rows)
                current_steps += row_has_work.astype(jnp.int32)
                return (
                    current_sub_canvas,
                    current_sub_mask,
                    current_steps,
                    current_kv,
                    next_remaining,
                    token_ids,
                    iteration + 1,
                    current_q32_calls + 1,
                    current_q8_calls,
                    current_stopped_rows,
                    current_acceptance_trace,
                )

            state = jax.lax.while_loop(needs_full_refresh, full_refresh_step,
                                       state)

            def has_work(state):
                _, _, _, _, remaining, _, iteration, _, _, _, _ = state
                return ((iteration < steps_per_sub_block) & jnp.any(remaining))

            def denoise_step(state):
                (current_sub_canvas, current_sub_mask, current_steps,
                 current_kv, remaining, _, iteration, current_q32_calls,
                 current_q8_calls, current_stopped_rows,
                 current_acceptance_trace) = state
                current_canvas = jax.lax.dynamic_update_slice(
                    canvas, current_sub_canvas, (0, start))
                row_has_work = jnp.any(remaining, axis=-1)
                with jax.named_scope("diffusion_dual_cache_q8"):
                    logits, current_kv = partial_forward_fn(
                        model_state,
                        current_canvas,
                        positions,
                        current_kv,
                        row_has_work,
                        forward_context,
                        start,
                    )
                token_ids, next_remaining = commit_fn(
                    logits,
                    remaining,
                    row_has_work,
                    q8_confidence_threshold,
                    temperature,
                    mask_token_id,
                )
                current_acceptance_trace = record_acceptance_step(
                    current_acceptance_trace,
                    start,
                    iteration,
                    DUAL_CACHE_Q8,
                    logits,
                    token_ids,
                    remaining,
                    next_remaining,
                    jnp.zeros((batch_size, ), dtype=bool),
                    q8_confidence_threshold,
                )
                committed = remaining & ~next_remaining
                current_sub_canvas = jnp.where(committed, token_ids,
                                               current_sub_canvas)
                current_sub_mask &= ~committed
                (current_sub_canvas, current_sub_mask, next_remaining,
                 current_stopped_rows) = apply_eos_stop(
                     current_sub_canvas, current_sub_mask, next_remaining,
                     current_stopped_rows)
                current_steps += row_has_work.astype(jnp.int32)
                return (
                    current_sub_canvas,
                    current_sub_mask,
                    current_steps,
                    current_kv,
                    next_remaining,
                    token_ids,
                    iteration + 1,
                    current_q32_calls,
                    current_q8_calls + 1,
                    current_stopped_rows,
                    current_acceptance_trace,
                )

            state = jax.lax.while_loop(has_work, denoise_step, state)
            (next_sub_canvas, next_sub_mask, next_steps, next_kv, remaining,
             last_tokens, _, next_q32_calls, next_q8_calls, next_stopped_rows,
             next_acceptance_trace) = state
            next_sub_canvas = jnp.where(remaining, last_tokens,
                                        next_sub_canvas)
            next_sub_mask &= ~remaining
            (next_sub_canvas, next_sub_mask, _,
             next_stopped_rows) = apply_eos_stop(
                 next_sub_canvas,
                 next_sub_mask,
                 jnp.zeros_like(remaining),
                 next_stopped_rows,
             )
            next_canvas = jax.lax.dynamic_update_slice(canvas, next_sub_canvas,
                                                       (0, start))
            next_mask = jax.lax.dynamic_update_slice(mask, next_sub_mask,
                                                     (0, start))
            return (next_canvas, next_mask, next_steps, next_kv,
                    next_q32_calls, next_q8_calls, next_forced_anchor_commits,
                    next_stopped_rows, next_acceptance_trace)

        return jax.lax.cond(jnp.any(eligible), work, no_work, operand=None)

    (canvas, mask, denoise_steps, kv_caches, q32_forward_calls,
     q8_forward_calls, forced_q32_anchor_commits, stopped_rows,
     acceptance_trace) = jax.lax.fori_loop(
         0,
         num_sub_blocks,
         denoise_sub_block,
         (
             canvas,
             mask,
             denoise_steps,
             kv_caches,
             jnp.array(0, dtype=jnp.int32),
             jnp.array(0, dtype=jnp.int32),
             jnp.array(0, dtype=jnp.int32),
             stopped_rows,
             acceptance_trace,
         ),
     )

    final_rows = active_rows & needs_final_forward & ~stopped_rows

    def run_final_forward(kv):
        with jax.named_scope("diffusion_dual_cache_final_q32"):
            anchor_logits, next_kv = final_forward_fn(
                model_state,
                canvas,
                positions,
                kv,
                final_rows,
                forward_context,
            )
        if next_block_policy is NextBlockPolicy.LAST_LOGIT_ANCHOR:
            anchor_logits = _exclude_mask_token(anchor_logits[:, None, :],
                                                mask_token_id)[:, 0, :]
            next_anchor = jnp.argmax(anchor_logits, axis=-1).astype(jnp.int32)
        elif next_block_policy is NextBlockPolicy.ALL_MASKED:
            next_anchor = jnp.zeros((canvas.shape[0], ), dtype=jnp.int32)
        else:
            raise ValueError(
                f"Unsupported next block policy: {next_block_policy}")
        return (jnp.where(final_rows, next_anchor,
                          0), next_kv, jnp.array(1, dtype=jnp.int32))

    def skip_final_forward(kv):
        return (jnp.zeros((canvas.shape[0], ),
                          dtype=jnp.int32), kv, jnp.array(0, dtype=jnp.int32))

    next_anchor, kv_caches, final_q32_forward_calls = jax.lax.cond(
        jnp.any(final_rows), run_final_forward, skip_final_forward, kv_caches)
    q32_forward_calls += final_q32_forward_calls
    return DualCacheDenoiseBlockOutput(
        canvas=canvas,
        next_anchor=next_anchor,
        denoise_steps=denoise_steps,
        kv_caches=kv_caches,
        q32_forward_calls=q32_forward_calls,
        q8_forward_calls=q8_forward_calls,
        final_q32_forward_calls=final_q32_forward_calls,
        forced_q32_anchor_commits=forced_q32_anchor_commits,
        stopped_rows=stopped_rows,
        acceptance_trace=acceptance_trace,
    )


def denoise_block_dual_cache(
    full_forward_fn: AlignedSubBlockForwardFn,
    partial_forward_fn: AlignedSubBlockForwardFn,
    final_forward_fn: FinalBlockForwardFn,
    commit_fn: CommitFn,
    model_state: Any,
    initial_canvas: jax.Array,
    initial_mask: jax.Array,
    positions: jax.Array,
    kv_caches: Any,
    active_rows: jax.Array,
    confidence_threshold: jax.Array,
    temperature: jax.Array,
    forward_context: Any,
    *,
    logit_alignment: LogitAlignment,
    next_block_policy: NextBlockPolicy,
    mask_token_id: int,
    sub_block_size: int,
    max_denoise_steps: int = 0,
    needs_final_forward: jax.Array | None = None,
    stop_on_eos_rows: jax.Array | None = None,
    eos_token_ids: tuple[int, ...] = (),
    commit_diagnostics_fn: CommitDiagnosticsFn | None = None,
    trace_acceptance_steps: bool = False,
    q8_log_confidence_bias: float = 0.0,
    force_q32_anchor_commit: bool = False,
) -> DualCacheDenoiseBlockOutput:
    """Denoise with aligned q32 refreshes followed by q8 replacements."""
    if initial_canvas.ndim != 2:
        raise ValueError("initial_canvas must have shape [batch, block_size]")
    if initial_mask.shape != initial_canvas.shape:
        raise ValueError("initial_mask must match initial_canvas")
    if positions.shape != initial_canvas.shape:
        raise ValueError("positions must match initial_canvas")
    if active_rows.shape != (initial_canvas.shape[0], ):
        raise ValueError("active_rows must match the canvas batch size")
    if needs_final_forward is None:
        needs_final_forward = active_rows
    elif needs_final_forward.shape != active_rows.shape:
        raise ValueError("needs_final_forward must match active_rows")
    if stop_on_eos_rows is None:
        stop_on_eos_rows = jnp.zeros_like(active_rows, dtype=bool)
    elif stop_on_eos_rows.shape != active_rows.shape:
        raise ValueError("stop_on_eos_rows must match active_rows")
    if sub_block_size < 1 or initial_canvas.shape[1] % sub_block_size:
        raise ValueError("sub_block_size must divide the model block size")
    if trace_acceptance_steps and commit_diagnostics_fn is None:
        raise ValueError(
            "trace_acceptance_steps requires commit_diagnostics_fn")
    if (not np.isfinite(q8_log_confidence_bias)
            or q8_log_confidence_bias < 0.0):
        raise ValueError(
            "q8_log_confidence_bias must be finite and non-negative")
    if q8_log_confidence_bias > 0.5:
        raise ValueError("q8_log_confidence_bias must not exceed 0.5")
    if (force_q32_anchor_commit
            and logit_alignment is not LogitAlignment.SHIFTED):
        raise ValueError(
            "force_q32_anchor_commit requires shifted logit alignment")
    if logit_alignment is LogitAlignment.SHIFTED:
        seed_mask, active = jax.device_get((initial_mask[:, 0], active_rows))
        if np.any(
                np.asarray(seed_mask, dtype=bool)
                & np.asarray(active, dtype=bool)):
            raise ValueError(
                "Shifted logit alignment requires position 0 to be an "
                "unmasked seed for every active row")

    return _denoise_block_dual_cache_jit(
        full_forward_fn,
        partial_forward_fn,
        final_forward_fn,
        commit_fn,
        commit_diagnostics_fn,
        model_state,
        initial_canvas,
        initial_mask,
        positions,
        kv_caches,
        active_rows,
        needs_final_forward,
        stop_on_eos_rows,
        confidence_threshold,
        temperature,
        forward_context,
        next_block_policy=next_block_policy,
        mask_token_id=mask_token_id,
        sub_block_size=sub_block_size,
        max_denoise_steps=max_denoise_steps,
        eos_token_ids=eos_token_ids,
        trace_acceptance_steps=trace_acceptance_steps,
        q8_log_confidence_bias=q8_log_confidence_bias,
        force_q32_anchor_commit=force_q32_anchor_commit,
    )
