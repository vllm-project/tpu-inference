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
"""Host-side pooling paths for flax_nnx pooling models (`--runner pooling`).

The pooling runner copies the step's hidden states to the host and runs
vLLM's pooler on CPU. Two optional speed-ups live here:

* `pool_on_host`: the same D2H copy and the same vLLM pooler call as before,
  but the pooler runs as plain torch on the host instead of under torchax's
  dispatch modes, which intercept every torch op in Python (for a pooler call
  with only CPU tensors that costs ~1.4 ms per step and changes nothing).
  Selected by `TPU_POOLING_FAST_PATH` (default on).
* `DeviceMeanPooler`: mean pooling as one small jitted reduction on the TPU.
  Only [num_reqs, hidden] is copied to the host (instead of [num_tokens,
  hidden]); vLLM's pooler then runs on those means as length-1 prompts, so
  its embedding head (normalization, `dimensions`, ST projector) is applied
  exactly as before. Opt-in, for models that declare
  `supports_device_mean_pooling` (JinaBert, `JINA_BERT_DEVICE_POOLING=1`).
"""

import dataclasses
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import Mesh, NamedSharding

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Smallest request bucket of the device pooling function (requests per step
# are padded up to a power of two >= this).
MIN_REQ_BUCKET = 8
# Steps with more requests than this fall back to the CPU pooler.
MAX_DEVICE_POOLING_REQS = 256
# Maximum requests per step computed by the in-model fused pooling output
# (`JINA_BERT_FUSED_POOLING=1`); steps with more requests fall back to the
# standalone jitted reduction in `DeviceMeanPooler`.
FUSED_POOLING_MAX_REQS = 32


def pool_on_host(pooler: Any, hidden_states: jax.Array, pooling_metadata: Any,
                 seq_lens: np.ndarray,
                 num_scheduled_tokens: np.ndarray) -> Any:
    """Copies `hidden_states` to the host and runs vLLM's pooler on it.

    Same conversion and pooler call as the original path, minus torchax's
    dispatch modes around the (CPU-only) pooler: `Tensor.to('cpu')` on a
    torchax view returns a plain torch CPU tensor, so the pooler's ops are
    ordinary torch ops either way and the result is bitwise identical.
    """
    import torchax
    from torchax.interop import torch_view

    torch_states = torch_view(hidden_states)
    with torchax.default_env():
        torch_states = torch_states.to('cpu', non_blocking=True)
    pooling_metadata.build_pooling_cursor(
        num_scheduled_tokens,
        torch.tensor(seq_lens),
        device=torch_states.device,
    )
    return pooler(torch_states, pooling_metadata)


def _segment_means_and_norm(
    hidden: jax.Array, bounds: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Per-request mean and L2-normalized mean of `hidden` rows.

    Args:
      hidden: [T, D] hidden states of the step.
      bounds: int32 [2, R]: request r owns rows [bounds[0, r], bounds[1, r]);
        unused slots have an empty range.

    Returns:
      (means, normed_means), each float32 [R, D]; zeros for unused slots.
    """
    num_tokens = hidden.shape[0]
    starts, ends = bounds[0][:, None], bounds[1][:, None]  # [R, 1]
    tok = lax.broadcasted_iota(jnp.int32, (1, num_tokens), 1)
    member = ((tok >= starts) & (tok < ends)).astype(jnp.float32)  # [R, T]
    # Rows past the last request are padding with unspecified contents:
    # zero them so that a non-finite value there cannot reach any mean
    # through the (0 * x) terms of the matmul.
    row = lax.broadcasted_iota(jnp.int32, (num_tokens, 1), 0)
    h = jnp.where(row < jnp.max(bounds[1]), hidden.astype(jnp.float32), 0.0)
    # 0/1 weights are exact in every MXU pass, so HIGHEST gives float32
    # sums (only the summation order differs from the CPU pooler's).
    sums = lax.dot(member,
                   h,
                   precision=lax.Precision.HIGHEST,
                   preferred_element_type=jnp.float32)
    counts = (bounds[1] - bounds[0]).astype(jnp.float32)[:, None]
    means = sums / jnp.maximum(counts, 1.0)
    norms = jnp.linalg.norm(means, axis=-1, keepdims=True)
    normed = means / jnp.maximum(norms, 1e-12)
    return means, normed


def _segment_means(hidden: jax.Array, bounds: jax.Array) -> jax.Array:
    """Per-request mean of `hidden` rows."""
    return _segment_means_and_norm(hidden, bounds)[0]


def mean_pool_from_seq_lens(
    hidden: jax.Array,
    seq_lens: jax.Array,
    max_reqs: int = FUSED_POOLING_MAX_REQS,
) -> tuple[jax.Array, jax.Array]:
    """Per-request mean and L2-normalized mean of `hidden` from `seq_lens`.

    Used when mean pooling is fused into the primary model JIT (`run_model`):
    `seq_lens` is already on the device inside `AttentionMetadata`, so the
    reduction and L2 normalization run in the same XLA launch as the encoder
    without a second dispatch or H2D bounds transfer.
    """
    r = min(max_reqs, seq_lens.shape[0])
    lens = jnp.maximum(seq_lens[:r].astype(jnp.int32), 0)
    ends = jnp.cumsum(lens, dtype=jnp.int32)
    starts = ends - lens
    bounds = jnp.stack([starts, ends], axis=0)
    return _segment_means_and_norm(hidden, bounds)


def _req_bucket(num_reqs: int) -> int:
    return max(MIN_REQ_BUCKET, 1 << (max(num_reqs, 1) - 1).bit_length())


class DeviceMeanPooler:
    """Mean pooling on the device for a vLLM `DispatchPooler`'s embed task.

    Use `maybe_create`, which returns None where this does not apply.
    """

    def __init__(self,
                 pooler: Any,
                 max_num_reqs: int,
                 fused_max_reqs: int = 0):
        from vllm.model_executor.layers.pooler.activations import \
            PoolerNormalize
        from vllm.model_executor.layers.pooler.seqwise.heads import \
            EmbeddingPoolerHead

        self.pooler = pooler
        self.max_reqs = min(MAX_DEVICE_POOLING_REQS,
                            _req_bucket(max(max_num_reqs, 1)))
        self.fused_max_reqs = min(max(fused_max_reqs, 0), self.max_reqs)
        self._means_and_norm = jax.jit(_segment_means_and_norm)
        self._means = jax.jit(_segment_means)

        embed = getattr(pooler, "poolers_by_task", {}).get("embed")
        head = getattr(embed, "head", None)
        self._can_bypass_cpu_head = (
            isinstance(head, EmbeddingPoolerHead) and head.projector is None
            and head.head_dtype in (None, torch.float32, "float32")
            and (head.activation is None
                 or isinstance(head.activation, PoolerNormalize)))
        self._has_normalize_activation = (head is not None and isinstance(
            head.activation, PoolerNormalize))

    @classmethod
    def maybe_create(cls,
                     pooler: Any,
                     mesh: Mesh,
                     max_num_reqs: int,
                     fused_max_reqs: int = 0) -> Optional["DeviceMeanPooler"]:
        from vllm.model_executor.layers.pooler.seqwise.methods import MeanPool
        from vllm.model_executor.layers.pooler.seqwise.poolers import \
            SequencePooler

        embed = getattr(pooler, "poolers_by_task", {}).get("embed")
        if not (isinstance(embed, SequencePooler)
                and type(embed.pooling) is MeanPool):
            logger.warning(
                "Device mean pooling requested but the embed pooler is %r, "
                "not mean pooling; using vLLM's CPU pooler.", embed)
            return None
        if mesh.size != 1:
            logger.warning(
                "Device mean pooling supports a single-device mesh only (got "
                "%d devices); using vLLM's CPU pooler.", mesh.size)
            return None
        logger.info(
            "Mean pooling runs on the device (up to %d requests per "
            "step, fused_max_reqs=%d).",
            min(MAX_DEVICE_POOLING_REQS, _req_bucket(max_num_reqs)),
            min(max(fused_max_reqs, 0),
                min(MAX_DEVICE_POOLING_REQS, _req_bucket(max_num_reqs))))
        return cls(pooler, max_num_reqs, fused_max_reqs=fused_max_reqs)

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: Any,
        num_scheduled_tokens: np.ndarray,
        aux_hidden_states: Optional[Sequence[jax.Array]] = None,
    ) -> Optional[list]:
        """Pooler output for this step, or None to use the CPU pooler."""
        num_reqs = len(num_scheduled_tokens)
        if not 0 < num_reqs <= self.max_reqs:
            return None
        if any(task != "embed" for task in pooling_metadata.tasks):
            return None
        lens = np.asarray(num_scheduled_tokens, dtype=np.int64)
        # Partial prefills: leave them to the CPU pooler (which rejects mean
        # pooling of partial prompts exactly as before).
        if not np.array_equal(lens, pooling_metadata.prompt_lens.numpy()):
            return None
        ends = np.cumsum(lens)
        if ends[-1] > hidden_states.shape[0]:
            return None

        if (aux_hidden_states is not None and len(aux_hidden_states) == 2
                and num_reqs <= aux_hidden_states[0].shape[0]):
            means_dev, normed_dev = aux_hidden_states
        else:
            bounds = np.zeros((2, _req_bucket(num_reqs)), np.int32)
            bounds[0, :num_reqs] = ends - lens
            bounds[1, :num_reqs] = ends
            means_dev, normed_dev = self._means_and_norm(hidden_states, bounds)

        pooling_params = pooling_metadata.pooling_params
        if (self._can_bypass_cpu_head
                and all(p.dimensions is None for p in pooling_params)):
            flags = [
                self._has_normalize_activation and bool(p.use_activation)
                for p in pooling_params
            ]
            if all(flags):
                arr = np.array(np.asarray(normed_dev)[:num_reqs], copy=True)
                return list(torch.from_numpy(arr))
            if not any(flags):
                arr = np.array(np.asarray(means_dev)[:num_reqs], copy=True)
                return list(torch.from_numpy(arr))

        means = np.asarray(means_dev)[:num_reqs]
        pooled = torch.from_numpy(np.array(means, copy=True))

        # vLLM's pooler on the means as length-1 prompts: the mean of one
        # row is that row exactly, then the embedding head runs as usual.
        ones = np.ones(num_reqs, np.int32)
        metadata = dataclasses.replace(
            pooling_metadata,
            prompt_lens=torch.ones(num_reqs,
                                   dtype=pooling_metadata.prompt_lens.dtype),
            pooling_cursor=None)
        metadata.build_pooling_cursor(ones,
                                      torch.from_numpy(ones),
                                      device=pooled.device)
        return self.pooler(pooled, metadata)

    def precompile(self, num_tokens_paddings: Sequence[int], hidden_size: int,
                   dtype: Any, sharding: NamedSharding) -> None:
        """Compiles the reduction for every (token bucket, request bucket)."""
        for num_tokens in num_tokens_paddings:
            hidden = jax.device_put(
                jnp.zeros((num_tokens, hidden_size), dtype), sharding)
            for reqs in self._req_buckets(num_tokens):
                if reqs <= self.fused_max_reqs:
                    continue
                bounds = np.zeros((2, reqs), np.int32)
                jax.block_until_ready(self._means_and_norm(hidden, bounds))

    def _req_buckets(self, num_tokens: int):
        # A step of `num_tokens` (padded) tokens holds at most that many
        # requests.
        limit = min(self.max_reqs, _req_bucket(num_tokens))
        reqs = MIN_REQ_BUCKET
        while reqs <= limit:
            yield reqs
            reqs *= 2
