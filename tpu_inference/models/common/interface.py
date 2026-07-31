# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

import jax
import numpy as np
from flax.nnx import GraphState
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata


class PoolerFunc(Protocol):
    """The wrapped pooler interface.

    Accept hidden-state, pooling-metadata and sequence lengths.
    Returns pooler output as a list of tensors, one per request.

    The contract is dependent on vLLM lib.
    """

    def __call__(
        self,
        hidden_states: jax.Array,
        pooling_metadata: PoolingMetadata,
        seq_lens: np.ndarray,
        num_scheduled_tokens: np.ndarray | None = None,
    ) -> PoolerOutput:
        ...


@dataclass(frozen=True)
class MultiModalInterface:
    precompile_vision_encoder_fn: Callable | None
    embed_multimodal_fn: Callable | None
    embed_input_ids_fn: Callable | None
    get_mrope_input_positions_fn: Callable | None


@dataclass(frozen=True)
class ModelInterface:
    model_fn: Callable
    compute_logits_fn: Callable
    pooler_fn: Callable
    combine_hidden_states_fn: Callable
    multimodal_fns: MultiModalInterface
    # For flax_nnx path, this is a `GraphState` containing the model's parameters and state.
    # For the vllm-impl path, this is a dict containing the model's parameters
    state: Dict | GraphState
    # Pre-flattened leaves of `state` (a tuple of `jax.Array`s for the
    # flax_nnx path, the same dict as `state` for the vllm-impl path). The
    # dispatch-side fns (`model_fn`, `compute_logits_fn`, etc.) accept this
    # form as their first positional arg so per-call jit dispatch skips the
    # `nnx.Variable` pytree traversal.
    state_leaves: Any
    lora_manager: Callable
    model: Any
    # Model forward callable safe to invoke from an enclosing JAX program. It
    # omits compiler options that JAX only permits on a top-level jit.
    model_fn_no_options: Callable | None = None
