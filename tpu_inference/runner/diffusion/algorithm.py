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

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from tpu_inference.runner.diffusion.config import DiffusionAlgorithm

CommitFn = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int],
    tuple[jax.Array, jax.Array],
]


class CommitDiagnostics(NamedTuple):
    selected_log_confidence: jax.Array
    threshold_margin: jax.Array


CommitDiagnosticsFn = Callable[
    [jax.Array, jax.Array, jax.Array, int],
    CommitDiagnostics,
]


def confidence_threshold_with_log_bias(
    confidence_threshold: jax.Array,
    log_confidence_bias: float,
) -> jax.Array:
    """Lower a probability threshold by an additive log-confidence bias."""
    if log_confidence_bias == 0.0:
        return confidence_threshold
    threshold = jnp.asarray(confidence_threshold, dtype=jnp.float32)
    bias = jnp.asarray(log_confidence_bias, dtype=jnp.float32)
    biased_threshold = threshold * jnp.exp(-bias)
    return jnp.where((threshold > 0.0) & (threshold < 1.0), biased_threshold,
                     threshold)


def _exclude_mask_token(
    logits: jax.Array,
    mask_token_id: int,
) -> jax.Array:
    """Remove the diffusion MASK token from the candidate distribution."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, length, vocab]")
    vocab_size = logits.shape[-1]
    if vocab_size < 2:
        raise ValueError("logits must contain at least two vocabulary entries")
    if not 0 <= mask_token_id < vocab_size:
        raise ValueError("mask_token_id must be within the logits vocabulary")

    candidate_logits = logits.astype(jnp.float32)
    candidate_logits = candidate_logits.at[..., mask_token_id].set(-jnp.inf)

    # Keep one non-MASK candidate finite so argmax cannot select MASK even when
    # every model logit is -inf.
    fallback_token_id = 1 if mask_token_id == 0 else 0
    fallback_logits = candidate_logits[..., fallback_token_id]
    fallback_logits = jnp.where(
        jnp.isneginf(fallback_logits) | jnp.isnan(fallback_logits),
        jnp.finfo(jnp.float32).min,
        fallback_logits,
    )
    return candidate_logits.at[..., fallback_token_id].set(fallback_logits)


def _low_confidence_scores(
    logits: jax.Array,
    confidence_threshold: jax.Array,
    temperature: jax.Array,
    mask_token_id: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    confidence_threshold = jnp.asarray(confidence_threshold, dtype=jnp.float32)
    temperature = jnp.asarray(temperature, dtype=jnp.float32)
    safe_temperature = jnp.where(temperature > 0.0, temperature, 1.0)
    scaled_logits = logits.astype(jnp.float32) / safe_temperature[:, None,
                                                                  None]
    scaled_logits = _exclude_mask_token(scaled_logits, mask_token_id)

    token_ids = jnp.argmax(scaled_logits, axis=-1).astype(jnp.int32)
    max_logits = jnp.max(scaled_logits, axis=-1)
    log_confidence = max_logits - jax.nn.logsumexp(scaled_logits, axis=-1)
    log_threshold = jnp.log(jnp.clip(confidence_threshold, min=0.0,
                                     max=1.0))[:, None]
    return token_ids, log_confidence, log_threshold


def low_confidence_diagnostics(
    logits: jax.Array,
    confidence_threshold: jax.Array,
    temperature: jax.Array,
    mask_token_id: int,
) -> CommitDiagnostics:
    _, log_confidence, log_threshold = _low_confidence_scores(
        logits,
        confidence_threshold,
        temperature,
        mask_token_id,
    )
    return CommitDiagnostics(
        selected_log_confidence=log_confidence,
        threshold_margin=log_confidence - log_threshold,
    )


def low_confidence_commit(
    logits: jax.Array,
    eligible_mask: jax.Array,
    active_rows: jax.Array,
    confidence_threshold: jax.Array,
    temperature: jax.Array,
    mask_token_id: int,
) -> tuple[jax.Array, jax.Array]:
    """Select greedy tokens and retain positions that need more denoising.

    Temperature rescales confidence and can therefore change which positions
    cross the commit threshold. Token selection remains deterministic argmax:
    this algorithm has no RNG and temperature does not sample alternate tokens.
    """
    if eligible_mask.shape != logits.shape[:2]:
        raise ValueError("eligible_mask must match logits [batch, length]")

    active_rows = jnp.asarray(active_rows, dtype=bool)
    token_ids, log_confidence, log_threshold = _low_confidence_scores(
        logits,
        confidence_threshold,
        temperature,
        mask_token_id,
    )

    eligible = jnp.asarray(eligible_mask, dtype=bool) & active_rows[:, None]
    commit = eligible & (log_confidence > log_threshold)

    masked_confidence = jnp.where(eligible, log_confidence, -jnp.inf)
    forced_indices = jnp.argmax(masked_confidence, axis=-1)
    forced = jax.nn.one_hot(forced_indices, logits.shape[1], dtype=bool)
    forced &= jnp.any(eligible, axis=-1)[:, None]
    commit |= forced

    remaining = eligible & ~commit
    return token_ids, remaining


_COMMIT_ALGORITHMS: dict[DiffusionAlgorithm, CommitFn] = {
    DiffusionAlgorithm.LOW_CONFIDENCE: low_confidence_commit,
}

_COMMIT_DIAGNOSTICS: dict[DiffusionAlgorithm, CommitDiagnosticsFn] = {
    DiffusionAlgorithm.LOW_CONFIDENCE: low_confidence_diagnostics,
}


def get_commit_algorithm(algorithm: DiffusionAlgorithm) -> CommitFn:
    try:
        return _COMMIT_ALGORITHMS[algorithm]
    except KeyError as exc:
        raise ValueError(
            f"No commit implementation is registered for {algorithm.value!r}"
        ) from exc


def get_commit_diagnostics_algorithm(
        algorithm: DiffusionAlgorithm) -> CommitDiagnosticsFn:
    try:
        return _COMMIT_DIAGNOSTICS[algorithm]
    except KeyError as exc:
        raise ValueError(
            "No commit diagnostics implementation is registered for "
            f"{algorithm.value!r}") from exc
