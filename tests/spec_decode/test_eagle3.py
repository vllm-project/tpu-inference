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
from tpu_inference.runner import utils as runner_utils
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

    vllm_config = VllmConfig(model_config=model_config,
                             cache_config=CacheConfig(block_size=16),
                             speculative_config=speculative_config,
                             device_config=DeviceConfig(device="tpu"),
                             parallel_config=ParallelConfig(
                                 pipeline_parallel_size=1,
                                 tensor_parallel_size=1),
                             load_config=LoadConfig(),
                             scheduler_config=SchedulerConfig(
                                 max_num_batched_tokens=8192,
                                 max_num_seqs=128,
                                 max_model_len=model_config.max_model_len,
                                 is_encoder_decoder=False))

    # Mock the runner, as the proposer needs it for initialization
    mock_runner = mock.MagicMock()
    # Create a real mesh for testing sharding-related logic
    devices = np.array(jax.devices()).reshape((1, len(jax.devices())))
    mock_runner.mesh = jax.sharding.Mesh(devices, axis_names=('data', 'model'))
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
    max_num_blocks_per_req = 10  # Mock value

    # Mock runner attributes
    proposer.runner.input_batch.num_reqs = num_reqs
    proposer.runner.num_tokens_paddings = runner_utils.get_token_paddings(
        min_token_size=16, max_token_size=1024, padding_gap=0)

    # Mocks required by _prepare_draft_inputs helper
    proposer.combine_hidden_states_fn = lambda state, h: h  # Mock passthrough
    proposer.state = None  # Mock state
    proposer.state_leaves = None  # Mock state leaves
    proposer.runner.input_batch.block_table = [mock.MagicMock()]
    # Mock the block table return value (2D array)
    (proposer.runner.input_batch.block_table[0].get_cpu_tensor.return_value
     ) = jnp.zeros((num_reqs, max_num_blocks_per_req), dtype=jnp.int32)

    # --- Setup sequence data ---
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
    # This is only used in the _prepare_input_ids helper
    # It must be padded to max_num_seqs (128) to match the mask in jnp.where
    last_sampled_token_id_cpu = np.zeros(max_num_seqs, dtype=np.int32)
    last_sampled_token_id_cpu[:num_reqs] = [1, 2,
                                            3]  # Valid tokens for active reqs
    last_sampled_token_id = jnp.array(last_sampled_token_id_cpu)
    next_prompt_token_id = jnp.zeros(max_num_seqs, dtype=jnp.int32)
    # All requests are in decode (not prefill), so last_sampled_token_id is used.
    is_in_prefill = jnp.zeros(max_num_seqs, dtype=jnp.bool_)

    attn_metadata = AttentionMetadata(
        seq_lens=jnp.array(sl_cpu),
        input_positions=jnp.arange(total_tokens),
        query_start_loc=jnp.array(qsl_cpu),
        block_tables=jnp.array([]),  # This will be replaced by the mock
        request_distribution=None,
    )

    # Expected results
    expected_new_qsl = np.zeros(max_num_seqs + 1, dtype=np.int32)
    num_tokens_per_req = np.zeros(max_num_seqs, dtype=np.int32)
    num_tokens_per_req[:num_reqs] = [3, 4, 3]
    # The implementation sets padded query lengths to 0, and rejected tokens
    # are 0 for padded requests.
    num_tokens_per_req[num_reqs:] = 0
    expected_new_qsl[1:] = np.cumsum(num_tokens_per_req)

    expected_new_seq_lens = np.zeros(max_num_seqs, dtype=np.int32)
    expected_new_seq_lens[:num_reqs] = [3, 4, 3]

    expected_total_tokens = input_ids.shape[0]

    expected_last_token_indices = jnp.array(expected_new_qsl[1:] - 1)

    # Execute
    num_reqs_dp = jnp.array([num_reqs], dtype=jnp.int32)
    with jax.set_mesh(proposer.mesh):
        target_hidden_states, input_ids, last_token_indices, updated_metadata = (
            proposer.prepare_inputs(attn_metadata, input_ids,
                                    aux_hidden_states, last_sampled_token_id,
                                    next_prompt_token_id, is_in_prefill,
                                    num_rejected_tokens, num_reqs_dp))

    # Assertions
    assert jnp.array_equal(updated_metadata.query_start_loc,
                           jnp.array(expected_new_qsl))
    assert jnp.array_equal(updated_metadata.seq_lens,
                           jnp.array(expected_new_seq_lens))

    assert jnp.array_equal(last_token_indices, expected_last_token_indices)

    assert input_ids.shape == (expected_total_tokens, )
    # NOTE: We don't check the content of target_token_ids for padded requests
    # as it's complicated to construct the expected tensor. The shape check
    # and the qsl/seq_len checks are sufficient to validate the logic.
    # The concatenated hidden state shape should be (..., hidden_size * 3)
    assert target_hidden_states.shape == (expected_total_tokens,
                                          hidden_size * 3)


@pytest.mark.parametrize("method", ["eagle3", "mtp"])
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

    def mock_model_fn(state, kv_caches, input_ids, target_hidden_states,
                      attn_metadata, layer_name_to_kvcache_index,
                      spec_step_idx):
        """
        Mock model_fn.
        Returns: (kv_caches, hidden_states_for_logits, residual_tuple)

        - On first call (num_tokens == total_tokens):
          Populate hidden_states_for_logits[last_token_indices] with base_token_ids.
          Populate residual_tuple[0][last_token_indices] with base_token_ids.
        - On loop calls (num_tokens == batch_size):
          Use input_ids (previous draft token) to generate new token (input_ids + 1).
          Populate hidden_states_for_logits with (input_ids + 1).
          Populate residual_tuple[0] with (input_ids + 1).
        """
        num_tokens = input_ids.shape[0]

        # This will be used for logits (output 2)
        hidden_states_for_logits = jnp.zeros((num_tokens, hidden_size))
        # This will be fed into the next step (output 3, item 0)
        residual_hidden_states = jnp.zeros((num_tokens, hidden_size))

        if num_tokens == total_tokens:
            # First call in propose.
            # `propose` will select from last_token_indices.
            last_token_indices = attn_metadata.query_start_loc[1:] - 1

            # Set logits output
            hidden_states_for_logits = hidden_states_for_logits.at[
                last_token_indices, 0].set(jnp.array(base_token_ids))

            # Set residual for next step
            residual_hidden_states = residual_hidden_states.at[
                last_token_indices, 0].set(jnp.array(base_token_ids))
        else:
            # Subsequent calls in the loop
            # input_ids is the previous draft token (shape `batch_size`)
            # Mock logic: next token = previous token + 1
            next_token_ids_encoded = input_ids + 1

            # Set logits output
            hidden_states_for_logits = hidden_states_for_logits.at[:, 0].set(
                next_token_ids_encoded)

            # Set residual for next step
            residual_hidden_states = residual_hidden_states.at[:, 0].set(
                next_token_ids_encoded)

        # Return (kv_caches, hidden_states, residual_tuple, None)
        return kv_caches, hidden_states_for_logits, (
            residual_hidden_states, ), None

    def mock_compute_logits_fn(state, hidden_states, lora_metadata):
        # Create deterministic logits from hidden_states.
        # Takes the value from hidden_states[:, 0]
        token_ids = hidden_states[:, 0].astype(jnp.int32)
        return jax.nn.one_hot(token_ids, vocab_size)

    def mock_combine_hidden_states_fn(state, hidden_states):
        # Passthrough, as the mock doesn't need combination.
        return hidden_states

    proposer.model_fn = mock_model_fn
    proposer.compute_logits_fn = mock_compute_logits_fn
    proposer.combine_hidden_states_fn = mock_combine_hidden_states_fn
    proposer.state = None  # Mock state
    proposer.state_leaves = None  # Mock state leaves

    # Inputs
    kv_caches = [None] * 1  # Mock kv_caches

    # Create the 2D table first, as this is what the (unused) mock expects
    block_tables_2d = jnp.zeros((batch_size, 10), dtype=jnp.int32)

    attn_metadata = AttentionMetadata(
        seq_lens=jnp.array([seq_len_1, seq_len_2]),
        input_positions=jnp.concatenate(
            [jnp.arange(seq_len_1),
             jnp.arange(seq_len_2)]),
        query_start_loc=jnp.array([0, seq_len_1, total_tokens]),
        # Pass the FLATTENED table to simulate output of prepare_inputs
        block_tables=block_tables_2d.reshape(-1),
        request_distribution=None,
    )

    # These are the inputs to `propose`
    # input_ids (from prepare_inputs)
    target_token_ids = jnp.zeros(total_tokens, dtype=jnp.int32)
    # target_hidden_states (from prepare_inputs)
    target_hidden_states = jnp.zeros((total_tokens, hidden_size))
    # last_token_indices (from prepare_inputs)
    last_token_indices = attn_metadata.query_start_loc[1:] - 1

    # Mock runner for block tables
    # This mock isn't actually used by propose(), but we'll set it
    # to the 2D table for correctness, as that's what
    # _prepare_draft_inputs (called by prepare_inputs) would expect.
    proposer.runner.input_batch.num_reqs = batch_size
    proposer.runner.input_batch.block_table = [mock.MagicMock()]
    (proposer.runner.input_batch.block_table[0].get_device_tensor.return_value
     ) = block_tables_2d

    # Execute
    _, draft_token_ids = proposer.propose(
        kv_caches,
        target_token_ids,
        attn_metadata,
        last_token_indices,
        target_hidden_states,
    )

    if draft_token_ids.ndim == 1:
        draft_token_ids = jnp.expand_dims(draft_token_ids, axis=-1)
    # Assertions
    assert draft_token_ids.shape == (batch_size, num_speculative_tokens)

    # Check the generated tokens
    # Step 0: base_token_ids [42, 60]
    # Step 1: [43, 61]
    # Step 2: [44, 62]
    # ...
    expected_tokens = np.zeros((batch_size, num_speculative_tokens),
                               dtype=np.int64)
    for i in range(batch_size):
        for j in range(num_speculative_tokens):
            expected_tokens[i, j] = base_token_ids[i] + j

    assert jnp.array_equal(draft_token_ids, jnp.array(expected_tokens))


def test_update_inputs_for_loop_speculation_mrope():
    proposer = _create_proposer("mtp", 2)

    # Setup 2D positions (M-RoPE)
    max_model_len = 10
    proposer.runner.max_model_len = max_model_len

    # positions shape: (3, num_reqs) -> (3, 2)
    positions = jnp.array([[1, 5], [2, 6], [3, 7]])

    seq_lens = jnp.array([5, 8])

    # block_tables shape: (num_reqs * max_num_blocks_per_req,)
    # Let's say max_num_blocks_per_req = 2. So shape (4,)
    block_tables = jnp.array([1, 2, 3, 4])

    # Execute
    positions_out, clamped_positions, new_seq_lens, query_start_loc, new_block_tables = proposer._update_inputs_for_loop_speculation(
        positions, seq_lens, block_tables)

    # Assertions
    # 1. positions_out should be positions + 1
    expected_positions_out = positions + 1
    assert jnp.array_equal(positions_out, expected_positions_out)

    # Let's test a case where it DOES exceed!
    positions_exceed = jnp.array([[1, 9], [2, 9], [3, 9]])

    positions_out, clamped_positions, new_seq_lens, query_start_loc, new_block_tables = proposer._update_inputs_for_loop_speculation(
        positions_exceed, seq_lens, block_tables)

    # exceeds_max_model_len_reduced should be [False, True]
    # So for request 1 (index 1):
    # new_seq_lens[1] should be set to 1!
    # clamped_positions[:, 1] should be set to 0!
    # new_block_tables for request 1 should be set to -1!

    assert new_seq_lens[1] == 1
    assert jnp.array_equal(clamped_positions[:, 1], jnp.array([0, 0, 0]))

    # block_tables for request 1 are at indices 2, 3. They should be -1.
    assert new_block_tables[2] == -1
    assert new_block_tables[3] == -1

    # For request 0 (index 0), it does not exceed.
    assert jnp.array_equal(clamped_positions[:, 0], jnp.array([2, 3, 4]))
    assert new_seq_lens[0] == 6
    assert new_block_tables[0] == 1
    assert new_block_tables[1] == 2


@pytest.mark.parametrize("method", ["eagle3", "mtp"])
def test_select_inputs_for_loop_speculation_mrope(method):
    """Regression for the (16,)-vs-(128,) broadcasting bug that appeared on
    Qwen3-VL + Eagle3 under DP-attention.

    Before the fix, the Eagle3 branch of `_select` did
    ``positions[last_token_indices]`` unconditionally. For 2D M-RoPE positions
    of shape ``(3, N)`` that indexed axis 0 with a length-B index array and
    produced ``(B, N)`` instead of ``(3, B)``. The wrongly shaped positions
    were then fed into ``_update_inputs_for_loop_speculation``, whose
    ``jnp.where(exceeds_max_model_len_reduced, 1, new_seq_lens)`` failed with
    "Incompatible shapes for broadcasting" on the first serving request.

    Only the ``method="eagle3"`` case hit the bug in production. But since
    ``SpeculativeConfig(model=<eagle3_dir>, method="mtp")`` auto-normalises
    to ``"eagle3"`` internally, we force ``proposer.method`` after construction
    so each parametrisation actually exercises the branch it names.
    """
    proposer = _create_proposer(method, 2)
    proposer.method = method
    proposer.runner.max_model_len = 32

    # Mock draft-token selection so we do not need a real model.
    proposer._select_draft_token_ids = mock.MagicMock(
        return_value=jnp.zeros(2, dtype=jnp.int32))

    num_reqs = 2
    total_tokens = 8
    hidden_size = 4

    # 2D M-RoPE positions: (3, total_tokens).
    positions = jnp.stack([jnp.arange(total_tokens)] * 3, axis=0)
    residual = jnp.ones((total_tokens, hidden_size))
    hidden_states = jnp.ones((total_tokens, hidden_size))
    # Pick the last token of each of the two requests.
    last_token_indices = jnp.array([3, 7], dtype=jnp.int32)

    with jax.set_mesh(proposer.mesh):
        selected_positions, selected_residual, _ = (
            proposer._select_inputs_for_loop_speculation(
                state_leaves=None,
                positions=positions,
                residual=residual,
                hidden_states=hidden_states,
                last_token_indices=last_token_indices,
            ))

    # Positions must stay 2D shape (3, num_reqs); the pre-fix bug produced
    # (num_reqs, total_tokens) here.
    assert selected_positions.shape == (3, num_reqs), (
        f"expected (3, {num_reqs}), got {selected_positions.shape} — the "
        "select branch is silently mangling 2D M-RoPE positions")
    # Values should be [3, 7] for each of the 3 M-RoPE axes.
    assert jnp.array_equal(selected_positions,
                           jnp.array([[3, 7], [3, 7], [3, 7]]))
    assert selected_residual.shape == (num_reqs, hidden_size)
