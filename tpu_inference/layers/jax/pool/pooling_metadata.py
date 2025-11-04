from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from tpu_inference.runner.input_batch_jax import InputBatch
from tpu_inference.utils import device_array

logger = logging.getLogger(__name__)


SUPPORTED_POOLING_TASKS = {"embed"}


def build_pooling_cursor(
    num_scheduled_tokens: list[int],
    padded_num_seqs: int,
    prompt_lens: jax.Array,
):
    assert len(prompt_lens) == len(num_scheduled_tokens)

    n_seq = len(num_scheduled_tokens)
    num_sched_tokens_padded = jnp.zeros(padded_num_seqs)
    num_sched_tokens_padded = num_sched_tokens_padded.at[:n_seq].set(
        jnp.asarary(num_scheduled_tokens, dtype=jnp.int32)
    )
    cumsum = jnp.cumsum(num_scheduled_tokens)
    first_token_indices = jnp.concatenate((jnp.asarray(0), cumsum[:-1]))
    last_token_indices = first_token_indices + num_sched_tokens_padded - 1
    last_token_indices = jnp.where(
        num_sched_tokens_padded > 0, last_token_indices, first_token_indices
    )
    return first_token_indices, last_token_indices


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "prompt_lens",
        "normalize",
        "num_reqs",
        "padded_num_reqs",
    ),
    meta_fields=("task_id",),
)
@dataclass
class TPUSupportedPoolingMetadata:
    """Device metadata required for pooling computations."""

    prompt_lens: jax.Array
    first_token_indices: jax.Array
    last_token_indices: jax.Array
    normalize: jax.Array
    num_reqs: int
    padded_num_reqs: int
    task: str

    @classmethod
    def from_input_batch(
        cls,
        mesh: Mesh,
        input_batch: InputBatch,
        num_scheduled_tokens: list[int],
        padded_num_reqs: int,
    ) -> TPUSupportedPoolingMetadata:
        pooling_params_list = input_batch.get_pooling_params()

        num_reqs = input_batch.num_reqs
        assert len(pooling_params_list) == num_reqs

        padded_prompt_lens_np = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_prompt_lens_np[:num_reqs] = input_batch.num_prompt_tokens[:num_reqs]

        normalize = np.full(padded_num_reqs, -1, dtype=np.int8)

        # Instead of shutting down the whole program, we should just ignore it and make it return 'embed' by default,
        # but provide a warning.
        for idx, params in enumerate(pooling_params_list):
            if params.normalize is True:
                normalize[idx] = 1
            elif params.normalize is False:
                normalize[idx] = 0

            if (task := params.task) not in SUPPORTED_POOLING_TASKS:
                logger.warning(
                    f"Unsupported pooling task '{task}'. Supported tasks: {sorted(SUPPORTED_POOLING_TASKS)}. Defaulting to 'embed'."
                )

        # maybe in the future if we need to support multiple tasks in one batch, we need to make sure each batch has only one task
        # if not task_values:
        #     raise ValueError("Pooling metadata requires at least one request")
        # if any(task != task_values[0] for task in task_values):
        #     raise ValueError("Mixed pooling tasks within the same batch are not supported yet")

        task = "embed"
        first_token_indices, last_token_indices = build_pooling_cursor(
            num_scheduled_tokens, padded_num_reqs, padded_prompt_lens_np[:num_reqs]
        )

        prompt_lens, normalize, first_token_indices, last_token_indices = device_array(
            mesh,
            (padded_prompt_lens_np, normalize, first_token_indices, last_token_indices),
        )

        return cls(
            prompt_lens=prompt_lens,
            first_token_indices=first_token_indices,
            last_token_indices=last_token_indices,
            normalize=normalize,
            task=task,
            num_reqs=num_reqs,
            padded_num_reqs=padded_num_reqs,
        )
