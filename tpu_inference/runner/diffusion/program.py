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

from tpu_inference.runner.diffusion.algorithm import (CommitFn,
                                                      _exclude_mask_token)
from tpu_inference.runner.diffusion.config import (LogitAlignment,
                                                   NextBlockPolicy)

BlockForwardFn = Callable[
    [Any, jax.Array, jax.Array, Any, jax.Array, Any],
    tuple[jax.Array, Any],
]


class DenoiseBlockOutput(NamedTuple):
    canvas: jax.Array
    next_anchor: jax.Array
    denoise_steps: jax.Array
    kv_caches: Any


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
    confidence_threshold: jax.Array,
    temperature: jax.Array,
    forward_context: Any,
    *,
    logit_alignment: LogitAlignment,
    next_block_policy: NextBlockPolicy,
    mask_token_id: int,
    sub_block_size: int,
    max_denoise_steps: int = 0,
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
    denoise_steps = jnp.zeros((canvas.shape[0], ), dtype=jnp.int32)
    steps_per_sub_block = (sub_block_size
                           if max_denoise_steps <= 0 else max_denoise_steps)
    num_sub_blocks = block_size // sub_block_size

    def denoise_sub_block(sub_block_index, carry):
        canvas, mask, denoise_steps, kv = carry
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
        )

        def has_work(state):
            _, _, _, _, eligible, _, iteration = state
            return ((iteration < steps_per_sub_block) & jnp.any(eligible))

        def denoise_step(state):
            canvas, mask, steps, kv, eligible, _, iteration = state
            row_has_work = jnp.any(eligible, axis=-1)
            logits, kv = forward_fn(model_state, canvas, positions, kv,
                                    active_rows, forward_context)
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
            steps += row_has_work.astype(jnp.int32)
            return (canvas, mask, steps, kv, remaining, token_ids,
                    iteration + 1)

        state = jax.lax.while_loop(has_work, denoise_step, state)
        canvas, mask, denoise_steps, kv, remaining, last_tokens, _ = state

        canvas = jnp.where(remaining, last_tokens, canvas)
        mask &= ~remaining
        return canvas, mask, denoise_steps, kv

    canvas, mask, denoise_steps, kv_caches = jax.lax.fori_loop(
        0,
        num_sub_blocks,
        denoise_sub_block,
        (canvas, mask, denoise_steps, kv_caches),
    )

    final_logits, kv_caches = forward_fn(model_state, canvas, positions,
                                         kv_caches, active_rows,
                                         forward_context)
    if next_block_policy is NextBlockPolicy.LAST_LOGIT_ANCHOR:
        anchor_logits = _exclude_mask_token(final_logits[:, -1:, :],
                                            mask_token_id)
        next_anchor = jnp.argmax(anchor_logits[:, 0, :],
                                 axis=-1).astype(jnp.int32)
    elif next_block_policy is NextBlockPolicy.ALL_MASKED:
        next_anchor = jnp.zeros((canvas.shape[0], ), dtype=jnp.int32)
    else:
        raise ValueError(f"Unsupported next block policy: {next_block_policy}")
    next_anchor = jnp.where(active_rows, next_anchor, 0)
    return DenoiseBlockOutput(
        canvas=canvas,
        next_anchor=next_anchor,
        denoise_steps=denoise_steps,
        kv_caches=kv_caches,
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
        confidence_threshold,
        temperature,
        forward_context,
        logit_alignment=logit_alignment,
        next_block_policy=next_block_policy,
        mask_token_id=mask_token_id,
        sub_block_size=sub_block_size,
        max_denoise_steps=max_denoise_steps,
    )
