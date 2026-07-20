# Copyright 2025 Google LLC
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
"""Block-diffusion denoising decode loop (Fast_dLLM / Diffusion-GR2).

This implements a *standalone* block-diffusion decode path that is fully gated
behind ``enable_diffusion_decode`` (see ``tpu_runner.py``); the existing
autoregressive (AR) decode path is unchanged when the flag is off.

Reference semantics (SGLang / HF ``Fast_dLLM``):

  * A *canvas* of ``block_size`` positions is seeded with the first AR token at
    index 0 and ``mask_id`` everywhere else.
  * Each denoise iteration forwards the model over the canvas, attending to the
    cached prompt prefix + canvas *bidirectionally* (``use_causal_mask=False``),
    then commits the high-confidence positions (plus one forced-argmax position
    to guarantee progress) via
    :func:`tpu_inference.layers.jax.sample.sampling.diffusion_commit`.
  * Logits use the *shifted* convention: the token at canvas position ``i`` is
    predicted from hidden ``i-1`` (clamped ``>= 0``), matching the AR next-token
    training objective.
  * Iterate until the block is fully committed or ``max_denoise_steps`` is
    reached, then advance to the next block whose first token is the argmax of
    the *last* committed position (i.e. the model's next-token prediction from
    the final canvas hidden state).

Loop design
-----------
The **inner** denoise iterations for a single block run as an on-device
``jax.lax.while_loop`` (see :func:`denoise_block`) — mirroring
``runner.decode_loop.continue_decode`` — so the per-iteration model forward +
threshold commit do not each incur a separate Pathways dispatch. The **outer**
block-advancement loop (:func:`diffusion_decode`) runs host-side because block
count is inherently dynamic (bounded by ``max_tokens`` / EOS) and each block is
a separate fused on-device program invocation, exactly like how the runner calls
``continue_decode`` once per scheduler step.

Model coupling is via an injected ``forward_fn`` (dependency injection) so the
whole seed -> forward -> shift -> commit -> advance loop is exercisable on CPU
with a stub, without the real model, RPA kernel, or paged KV cache:

    forward_fn(canvas_tokens, canvas_positions, kv_caches)
        -> (full_logits, kv_caches)

where ``canvas_tokens`` is ``(block_size,) int32``, ``canvas_positions`` is the
matching ``(block_size,) int32`` absolute positions, ``full_logits`` is the
*unshifted* per-position logits ``(block_size, vocab)``, and ``kv_caches`` is an
opaque pytree threaded through the loop (may be ``None`` for the stub/tests).
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.jax.sample.sampling import diffusion_commit

# forward_fn(canvas_tokens, canvas_positions, kv_caches) -> (full_logits, kv_caches)
ForwardFn = Callable[[jax.Array, jax.Array, Any], Tuple[jax.Array, Any]]


@dataclass
class DiffusionDecodeResult:
    """Result of a host-side block-diffusion decode run for one request."""
    # Flat generated token stream. ``tokens[0]`` == the seeding ``first_token``;
    # the remainder are the denoised / block-chained tokens in emission order.
    tokens: List[int]
    # Number of blocks actually denoised.
    num_blocks: int
    # Denoise iterations used per block (len == num_blocks).
    steps_per_block: List[int]
    # The next block's seed token (argmax of the last committed position of the
    # final block); useful if the caller wants to continue decoding.
    next_seed: int
    # Threaded KV caches (unchanged pytree structure; ``None`` for the stub).
    kv_caches: Any
    # True if an EOS token was committed (decode stopped early).
    hit_eos: bool


def _shifted_logit_indices(block_size: int) -> jax.Array:
    """Indices implementing the shifted-logit convention.

    ``shifted[i] = full_logits[max(i - 1, 0)]`` so the token at canvas position
    ``i`` is predicted from hidden ``i-1`` (clamped ``>= 0`` for position 0).
    """
    return jnp.maximum(jnp.arange(block_size, dtype=jnp.int32) - 1, 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "forward_fn",
        "block_size",
        "mask_id",
        "threshold",
        "temperature",
        "max_denoise_steps",
    ),
)
def denoise_block(
    forward_fn: ForwardFn,
    seed_token: jax.Array,
    positions: jax.Array,
    kv_caches: Any = None,
    *,
    block_size: int,
    mask_id: int,
    threshold: float,
    temperature: float,
    max_denoise_steps: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, Any]:
    """Denoise a single block on-device via ``jax.lax.while_loop``.

    Args:
        forward_fn: Injected model forward. See module docstring for signature.
        seed_token: Scalar int token placed at canvas index 0 (already committed).
        positions: ``(block_size,)`` int32 absolute positions for the canvas.
        kv_caches: Opaque KV-cache pytree threaded through the loop (may be None).
        block_size: Canvas length (static).
        mask_id: Token id used to mark not-yet-committed positions (static).
        threshold: Confidence threshold forwarded to ``diffusion_commit``.
        temperature: Softmax temperature forwarded to ``diffusion_commit``.
        max_denoise_steps: Static upper bound on denoise iterations.

    Returns:
        ``(canvas, next_seed, steps_used, kv_caches)`` where ``canvas`` is the
        fully-committed ``(block_size,)`` int32 block (index 0 == ``seed_token``),
        ``next_seed`` is the scalar argmax of the last committed position, and
        ``steps_used`` is the scalar iteration count.
    """
    shift_idx = _shifted_logit_indices(block_size)

    canvas0 = jnp.full((block_size, ), mask_id, dtype=jnp.int32)
    canvas0 = canvas0.at[0].set(seed_token.astype(jnp.int32))
    # Index 0 is the given AR seed (already committed); the rest start masked.
    mask0 = jnp.arange(block_size, dtype=jnp.int32) > 0

    # carry = (canvas, mask, step, next_seed, last_committed, kv_caches)
    init = (
        canvas0,
        mask0,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        canvas0,
        kv_caches,
    )

    def cond_fn(carry):
        _, mask, step, _, _, _ = carry
        under_cap = step < max_denoise_steps
        # Force at least one forward (step 0) so ``next_seed`` is always set,
        # even in the degenerate block_size == 1 case; otherwise stop once the
        # block is fully committed.
        more_work = jnp.logical_or(step == 0, jnp.any(mask))
        return jnp.logical_and(under_cap, more_work)

    def body_fn(carry):
        canvas, mask, step, _, _, kv = carry
        full_logits, kv = forward_fn(canvas, positions, kv)
        full_logits = full_logits.astype(jnp.float32)
        shifted = full_logits[shift_idx]
        committed_tok, new_mask = diffusion_commit(shifted, mask, threshold,
                                                   temperature)
        newly = jnp.logical_and(mask, jnp.logical_not(new_mask))
        canvas = jnp.where(newly, committed_tok, canvas)
        # Next block's first token = model's next-token prediction from the
        # final canvas position's hidden state (unshifted logits at the tail).
        next_seed = jnp.argmax(full_logits[block_size - 1]).astype(jnp.int32)
        return (canvas, new_mask, step + 1, next_seed, committed_tok, kv)

    canvas, mask, steps, next_seed, last_committed, kv_caches = jax.lax.while_loop(
        cond_fn, body_fn, init)

    # If the step cap was hit before full commitment, force-fill any still-masked
    # positions with their argmax so ``mask_id`` never leaks into the output.
    canvas = jnp.where(mask, last_committed, canvas)
    return canvas, next_seed, steps, kv_caches


def diffusion_decode(
        forward_fn: ForwardFn,
        first_token: int,
        prefix_len: int,
        kv_caches: Any = None,
        *,
        block_size: int,
        mask_id: int,
        max_tokens: int,
        threshold: float = 0.9,
        temperature: float = 0.0,
        max_denoise_steps: int = 0,
        eos_token_id: Tuple[int, ...] = (),
) -> DiffusionDecodeResult:
    """Host-side block-diffusion decode: iterate blocks until done.

    Seeds block 0 with ``first_token`` at canvas index 0, denoises it on-device
    via :func:`denoise_block`, then advances to the next block whose first token
    is the previous block's ``next_seed``. The returned ``tokens`` stream is the
    concatenation of the committed block canvases (so ``tokens[0] ==
    first_token``), truncated at ``max_tokens`` or the first committed EOS.

    Args:
        forward_fn: Injected model forward (see module docstring).
        first_token: The AR token (e.g. from prefill) that seeds block 0.
        prefix_len: Number of cached prompt tokens preceding block 0.
        kv_caches: Opaque KV-cache pytree threaded through decode (may be None).
        block_size: Canvas length per block.
        mask_id: Token id marking not-yet-committed canvas positions.
        max_tokens: Cap on the number of generated tokens.
        threshold: Confidence threshold for committing a position.
        temperature: Softmax temperature for the commit step.
        max_denoise_steps: Per-block denoise iteration cap; ``<= 0`` means use
            ``block_size`` (which fully commits since the forced-argmax progress
            guarantee commits >= 1 masked position per iteration).
        eos_token_id: EOS token id(s); decode stops after the first is committed.

    Returns:
        A :class:`DiffusionDecodeResult`.
    """
    if block_size < 1:
        raise ValueError(f"{block_size=} must be >= 1")

    eff_max_steps = block_size if max_denoise_steps <= 0 else int(
        max_denoise_steps)
    eos_set = set(
        int(t) for t in np.atleast_1d(np.asarray(
            eos_token_id)).ravel()) if len(eos_token_id) else set()

    tokens: List[int] = []
    steps_per_block: List[int] = []
    seed = int(first_token)
    prefix = int(prefix_len)
    hit_eos = False

    while len(tokens) < max_tokens and not hit_eos:
        positions = jnp.arange(block_size, dtype=jnp.int32) + jnp.array(
            prefix, dtype=jnp.int32)
        canvas, next_seed, steps, kv_caches = denoise_block(
            forward_fn,
            jnp.array(seed, dtype=jnp.int32),
            positions,
            kv_caches,
            block_size=block_size,
            mask_id=mask_id,
            threshold=threshold,
            temperature=temperature,
            max_denoise_steps=eff_max_steps,
        )
        canvas_host = np.asarray(canvas).tolist()
        steps_per_block.append(int(steps))

        for tok in canvas_host:
            tokens.append(int(tok))
            if int(tok) in eos_set:
                hit_eos = True
                break
            if len(tokens) >= max_tokens:
                break

        seed = int(next_seed)
        prefix += block_size

    tokens = tokens[:max_tokens]
    return DiffusionDecodeResult(
        tokens=tokens,
        num_blocks=len(steps_per_block),
        steps_per_block=steps_per_block,
        next_seed=seed,
        kv_caches=kv_caches,
        hit_eos=hit_eos,
    )
