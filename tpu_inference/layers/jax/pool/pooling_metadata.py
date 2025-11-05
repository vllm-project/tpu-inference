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
    padded_num_reqs: int,
):

    n_seq = len(num_scheduled_tokens)
    padded_num_scheduled_tokens = jnp.zeros(padded_num_reqs)
    padded_num_scheduled_tokens = padded_num_scheduled_tokens.at[:n_seq].set(
        jnp.asarray(num_scheduled_tokens, dtype=jnp.int32)
    )
    cumsum = jnp.cumsum(padded_num_scheduled_tokens, dtype = jnp.int64)
    first_token_indices = jnp.concatenate((jnp.asarray((0,)), cumsum[:-1]))
    last_token_indices = (first_token_indices + padded_num_scheduled_tokens - 1).astype(jnp.int64)
    last_token_indices = jnp.where(
        padded_num_scheduled_tokens > 0, last_token_indices, first_token_indices
    )
    return first_token_indices, last_token_indices, padded_num_scheduled_tokens


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "prompt_lens",
        "first_token_indices",
        "last_token_indices",
        "num_scheduled_tokens", 
    ),
    meta_fields = (),
)
@dataclass
class TPUSupportedPoolingMetadata:
    """Device metadata required for pooling computations."""

    prompt_lens: jax.Array
    first_token_indices: jax.Array
    last_token_indices: jax.Array
    num_scheduled_tokens: jax.Array

    @classmethod
    def from_input_batch(
        cls,
        mesh: Mesh,
        input_batch: InputBatch,
        padded_num_scheduled_tokens: list[int],
        padded_num_reqs: int,
    ) -> TPUSupportedPoolingMetadata:
        pooling_params_list = input_batch.get_pooling_params()

        num_reqs = input_batch.num_reqs
        assert len(pooling_params_list) == num_reqs
        assert len(input_batch.num_prompt_tokens[:num_reqs]) == len(padded_num_scheduled_tokens)

        padded_prompt_lens= jnp.zeros(padded_num_reqs, dtype=np.int32)
        padded_prompt_lens= padded_prompt_lens.at[:num_reqs].set(input_batch.num_prompt_tokens[:num_reqs])

        first_token_indices, last_token_indices, padded_num_scheduled_tokens = build_pooling_cursor(
            padded_num_scheduled_tokens, padded_num_reqs
        )

        prompt_lens, first_token_indices, last_token_indices, num_scheduled_tokens = device_array(
            mesh,
            (padded_prompt_lens, first_token_indices, last_token_indices, padded_num_scheduled_tokens),
        )

        #everything in pooling_metadata is padded.
        return cls(
            prompt_lens=prompt_lens,
            first_token_indices=first_token_indices,
            last_token_indices=last_token_indices,
            num_scheduled_tokens = num_scheduled_tokens,
        )


def is_partial_prefill(pooling_metadata: TPUSupportedPoolingMetadata):
    return not jnp.all(pooling_metadata.prompt_lens == pooling_metadata.num_scheduled_tokens)
