# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, SpeculativeConfig,
                         VllmConfig)
from vllm.config.load import LoadConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer

# Use a real model dir for config, but we will mock model loading/execution
model_dir = "meta-llama/Llama-3.1-8B-Instruct"
eagle3_dir = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"


def _create_proposer(
    method: str,
    num_speculative_tokens: int,
) -> Eagle3Proposer:
    model_config = ModelConfig(model=model_dir,
                               runner="generate",
                               max_model_len=8192,
                               seed=42)

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=eagle3_dir,
        method=method,
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="tpu"),
        parallel_config=ParallelConfig(pipeline_parallel_size=1,
                                       tensor_parallel_size=1),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=8192,
                                         max_num_seqs=128))

    # Mock the runner, as the proposer needs it for initialization
    mock_runner = mock.MagicMock()
    # Create a real mesh for testing sharding-related logic
    devices = np.array(jax.devices())
    mock_runner.mesh = jax.sharding.Mesh(devices, axis_names=('model', ))
    mock_runner.max_num_tokens = 8192
    mock_runner.max_model_len = 8192
    mock_runner.kv_cache_config.kv_cache_groups = [mock.MagicMock()]
    mock_runner.input_batch = mock.MagicMock()

    return Eagle3Proposer(vllm_config=vllm_config, runner=mock_runner)


def test_prepare_inputs():
    """
    Mirrors the GPU test for prepare_inputs, adapted for JAX.
    - cu_target_query_lens: [0, a, a + b, a + b + c]
    - num_rejected_tokens: [n1, n2, n3]
    - num_tokens_per_req: [a - n1, b - n2, c - n3]
    - cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    - token_indices: [0, ..., a - n1 - 1, a, ..., a + b - n2 - 1, ...]
    """
    proposer = _create_proposer("eagle3", 1)
    num_reqs = 3
    max_num_seqs = 128

    # Mock runner attributes
    proposer.runner.input_batch.num_reqs = num_reqs
    qsl_cpu = np.zeros(max_num_seqs + 1, dtype=np.int32)
    query_lens = np.zeros(max_num_seqs, dtype=np.int32)
    query_lens[:num_reqs] = [4, 7, 5]
    qsl_cpu[1:] = np.cumsum(query_lens)

    sl_cpu = np.zeros(max_num_seqs, dtype=np.int32)
    sl_cpu[:num_reqs] = [4, 7, 5]

    # Inputs
    total_tokens = 16
    hidden_size = 128
    # The input_ids should be large enough to be indexed by token_indices,
    # which can access up to total_tokens for padded requests.
    input_ids = jnp.arange(total_tokens + 1)
    aux_hidden_states = (jnp.ones((total_tokens + 1, hidden_size)),
                         jnp.ones((total_tokens + 1, hidden_size)),
                         jnp.ones((total_tokens + 1, hidden_size)))

    num_rejected_tokens_cpu = np.zeros(max_num_seqs, dtype=np.int32)
    num_rejected_tokens_cpu[:num_reqs] = [1, 3, 2]
    num_rejected_tokens = jnp.array(num_rejected_tokens_cpu)

    attn_metadata = AttentionMetadata(
        seq_lens=jnp.array(sl_cpu),
        input_positions=jnp.arange(total_tokens),
        query_start_loc=jnp.array(qsl_cpu),
        block_tables=jnp.array([]),
        request_distribution=None,
    )
    attn_metadata.query_start_loc_cpu = qsl_cpu
    attn_metadata.seq_lens_cpu = sl_cpu

    # Expected results
    expected_new_qsl = np.zeros(max_num_seqs + 1, dtype=np.int32)
    num_tokens_per_req = np.zeros(max_num_seqs, dtype=np.int32)
    num_tokens_per_req[:num_reqs] = [3, 4, 3]
    # The implementation sets padded query lengths to 1, and rejected tokens
    # are 0 for padded requests.
    num_tokens_per_req[num_reqs:] = 1
    expected_new_qsl[1:] = np.cumsum(num_tokens_per_req)

    expected_new_seq_lens = np.zeros(max_num_seqs, dtype=np.int32)
    expected_new_seq_lens[:num_reqs] = [3, 4, 3]

    expected_total_tokens = int(expected_new_qsl[-1])

    # Execute
    updated_metadata, target_token_ids, target_hidden_states = (
        proposer.prepare_inputs(attn_metadata, input_ids, aux_hidden_states,
                                num_rejected_tokens))

    # Assertions
    assert jnp.array_equal(updated_metadata.query_start_loc,
                           jnp.array(expected_new_qsl))
    assert jnp.array_equal(updated_metadata.seq_lens,
                           jnp.array(expected_new_seq_lens))
    assert target_token_ids.shape == (expected_total_tokens, )
    # NOTE: We don't check the content of target_token_ids for padded requests
    # as it's complicated to construct the expected tensor. The shape check
    # and the qsl/seq_len checks are sufficient to validate the logic.
    assert target_hidden_states.shape == (expected_total_tokens,
                                          hidden_size * 3)


@pytest.mark.parametrize("method", ["eagle3"])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3, 8])
def test_propose(method, num_speculative_tokens):
    proposer = _create_proposer(method, num_speculative_tokens)

    # Mock the JAX model functions
    hidden_size = 128
    vocab_size = 100
    batch_size = 2
    seq_len_1 = 5
    seq_len_2 = 3
    total_tokens = seq_len_1 + seq_len_2
    base_token_ids = [42, 60]

    def mock_model_fn(state, kv_caches, input_ids, hidden_states,
                      attn_metadata):
        num_tokens = input_ids.shape[0]
        new_hidden_states = jnp.zeros((num_tokens, hidden_size))

        if num_tokens == total_tokens:
            # First call in propose. Set hidden states for last tokens
            # to produce the first draft tokens.
            last_token_indices = attn_metadata.query_start_loc[1:] - 1
            # The proposer uses next_token_ids to set the last token of each
            # sequence in input_ids. We mock this behavior by directly using
            # next_token_ids to generate the first draft tokens.
            # The mock `compute_logits` will use hidden_states[:, 0]
            # to generate tokens.
            new_hidden_states = new_hidden_states.at[
                last_token_indices, 0].set(jnp.array(base_token_ids))
        else:  # Subsequent calls in the loop
            new_hidden_states = new_hidden_states.at[:, 0].set(input_ids + 1)

        return kv_caches, new_hidden_states, new_hidden_states

    def mock_compute_logits_fn(state, hidden_states, lora_metadata):
        # Create deterministic logits from hidden_states.
        token_ids = hidden_states[:, 0].astype(jnp.int32)
        return jax.nn.one_hot(token_ids, vocab_size)

    def mock_combine_hidden_states_fn(state, hidden_states):
        # Passthrough, as the mock doesn't need combination.
        return hidden_states

    proposer.model_fn = mock_model_fn
    proposer.compute_logits_fn = mock_compute_logits_fn
    proposer.combine_hidden_states_fn = mock_combine_hidden_states_fn
    proposer.state = None  # Mock state

    # Inputs
    kv_caches = [None] * 1  # Mock kv_caches
    next_token_ids = jnp.array(base_token_ids, dtype=jnp.int32)
    attn_metadata = AttentionMetadata(
        seq_lens=jnp.array([seq_len_1, seq_len_2]),
        input_positions=jnp.concatenate(
            [jnp.arange(seq_len_1),
             jnp.arange(seq_len_2)]),
        query_start_loc=jnp.array([0, seq_len_1, total_tokens]),
        block_tables=jnp.zeros((2, 10), dtype=jnp.int32),
        request_distribution=None,
    )
    input_ids = jnp.zeros(total_tokens, dtype=jnp.int32)
    target_token_ids = jnp.zeros(total_tokens, dtype=jnp.int32)
    target_hidden_states = jnp.zeros((total_tokens, hidden_size))

    # Mock runner for block tables
    proposer.runner.input_batch.block_table = [mock.MagicMock()]
    (proposer.runner.input_batch.block_table[0].get_device_tensor.return_value
     ) = attn_metadata.block_tables

    # Execute
    _, draft_token_ids = proposer.propose(
        kv_caches,
        next_token_ids,
        attn_metadata,
        input_ids,
        target_token_ids,
        target_hidden_states,
    )

    # Assertions
    assert draft_token_ids.shape == (batch_size, num_speculative_tokens)

    expected_tokens = np.zeros((batch_size, num_speculative_tokens),
                               dtype=np.int64)
    for i in range(batch_size):
        for j in range(num_speculative_tokens):
            expected_tokens[i, j] = base_token_ids[i] + j

    assert jnp.array_equal(draft_token_ids, jnp.array(expected_tokens))
