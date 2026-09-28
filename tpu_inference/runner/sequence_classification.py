# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from vllm.v1.outputs import ModelRunnerOutput

from tpu_inference.layers.common.sharding import ShardingAxisName

if TYPE_CHECKING:
    from tpu_inference.runner.tpu_runner import TPUModelRunner


def prewarm_sequence_classification(runner: "TPUModelRunner") -> None:
    """Warm the request buckets and their small device-to-host transfers."""
    import torchax
    from torchax.interop import torch_view

    mesh = runner.mesh
    hidden_size = runner.model_config.get_hidden_size()
    hidden_sharding = NamedSharding(
        mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
    indices_sharding = NamedSharding(mesh,
                                     PartitionSpec(ShardingAxisName.ATTN_DATA))
    hidden = jax.device_put(
        jnp.zeros((runner.num_tokens_paddings[-1], hidden_size),
                  dtype=jnp.bfloat16), hidden_sharding)
    pcp_size = runner.vllm_config.sharding_config.prefill_cp_size
    # The compilation manager covers gather's other token/request bucket pairs.
    with torchax.default_env():
        for padded_reqs in runner.num_reqs_paddings:
            indices = jax.device_put(
                jnp.full((padded_reqs, ), -1, dtype=jnp.int32),
                indices_sharding)
            rows = runner._select_from_array_fn(hidden, indices, mesh,
                                                pcp_size)
            logits = runner.compute_logits_fn(runner.state_leaves, rows)
            torch_view(logits).to("cpu", non_blocking=False)


def pool_sequence_classification(
    runner: "TPUModelRunner",
    hidden_states: jax.Array,
    logits_indices: jax.Array,
    logits_indices_selector,
    scheduler_output,
) -> ModelRunnerOutput:
    """Select LAST rows and complete only requests with a finished prompt."""
    if logits_indices_selector is not None:
        raise RuntimeError("Native sequence classification requires request "
                           "order without in-process DP reordering")
    batch = runner.input_batch
    num_reqs = batch.num_reqs
    metadata = batch.get_pooling_metadata()
    rows = runner._select_from_array_fn(
        hidden_states, logits_indices, runner.mesh,
        runner.vllm_config.sharding_config.prefill_cp_size)
    logits = runner.compute_logits_fn(runner.state_leaves, rows)
    outputs = runner.pooler_fn(logits, metadata)
    scheduled = np.array([
        scheduler_output.num_scheduled_tokens[req_id]
        for req_id in batch.req_ids[:num_reqs]
    ],
                         dtype=np.int32)
    # The buffer's seq_lens view is not the region written by _prepare_inputs.
    completed = (batch.num_computed_tokens_cpu[:num_reqs] + scheduled
                 >= batch.num_prompt_tokens[:num_reqs])
    return ModelRunnerOutput(
        req_ids=list(batch.req_ids),
        req_id_to_index=dict(batch.req_id_to_index),
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[
            out if done else None for out, done in zip(outputs, completed)
        ],
    )
