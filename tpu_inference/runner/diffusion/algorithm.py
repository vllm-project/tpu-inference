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

import jax
import jax.numpy as jnp

from tpu_inference.runner.diffusion.config import DiffusionAlgorithm

CommitFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
                    tuple[jax.Array, jax.Array]]


def low_confidence_commit(
    logits: jax.Array,
    eligible_mask: jax.Array,
    active_rows: jax.Array,
    confidence_threshold: jax.Array,
    temperature: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Select greedy tokens and retain positions that need more denoising."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, length, vocab]")
    if eligible_mask.shape != logits.shape[:2]:
        raise ValueError("eligible_mask must match logits [batch, length]")

    active_rows = jnp.asarray(active_rows, dtype=bool)
    confidence_threshold = jnp.asarray(confidence_threshold, dtype=jnp.float32)
    temperature = jnp.asarray(temperature, dtype=jnp.float32)
    safe_temperature = jnp.where(temperature > 0.0, temperature, 1.0)
    scaled_logits = logits.astype(jnp.float32) / safe_temperature[:, None,
                                                                  None]

    token_ids = jnp.argmax(scaled_logits, axis=-1).astype(jnp.int32)
    max_logits = jnp.max(scaled_logits, axis=-1)
    log_confidence = max_logits - jax.nn.logsumexp(scaled_logits, axis=-1)
    log_threshold = jnp.log(jnp.clip(confidence_threshold, min=0.0,
                                     max=1.0))[:, None]

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


def get_commit_algorithm(algorithm: DiffusionAlgorithm) -> CommitFn:
    try:
        return _COMMIT_ALGORITHMS[algorithm]
    except KeyError as exc:
        raise ValueError(
            f"No commit implementation is registered for {algorithm.value!r}"
        ) from exc
