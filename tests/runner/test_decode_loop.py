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

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.runner.decode_loop import (TpuSamplingState, _split_rngs,
                                              _update_loop_state,
                                              continue_decode)


def test_update_loop_state_basic():
    # 2 requests, DP = 1, pad_len = 0
    next_tokens = jnp.array([42, 99], dtype=jnp.int32)  # 99 is EOS
    active_mask = jnp.array([True, True], dtype=jnp.bool_)
    input_positions = jnp.array([10, 20], dtype=jnp.int32)
    seq_lens = jnp.array([11, 21], dtype=jnp.int32)
    eos_token_id = (99, )
    padding_token_id = -1
    dp_size = 1
    pad_len = 0

    (
        new_active_mask,
        next_input_ids,
        new_positions,
        new_seq_lens,
        step_record_tokens,
        any_hit_eos,
    ) = _update_loop_state(
        next_tokens,
        active_mask,
        input_positions,
        seq_lens,
        eos_token_id,
        padding_token_id,
        dp_size,
        pad_len,
    )

    assert np.array_equal(new_active_mask, [True, False])
    assert np.array_equal(next_input_ids, [42, -1])
    assert np.array_equal(new_positions, [11, 20])
    assert np.array_equal(new_seq_lens, [12, 21])
    assert np.array_equal(step_record_tokens, [42, 99])
    assert any_hit_eos


def test_update_loop_state_dp_padding():
    # 4 requests total (2 per DP rank), DP = 2, pad_len = 1
    next_tokens = jnp.array([42, 99, 43, 44], dtype=jnp.int32)  # 99 is EOS
    active_mask = jnp.array([True, True, True, True], dtype=jnp.bool_)
    input_positions = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    # seq_lens: [11, 21, 0,  31, 41, 0]  (last one in each DP rank is padding)
    seq_lens = jnp.array([11, 21, 0, 31, 41, 0], dtype=jnp.int32)
    eos_token_id = (99, )
    padding_token_id = -1
    dp_size = 2
    pad_len = 1

    (
        new_active_mask,
        next_input_ids,
        new_positions,
        new_seq_lens,
        step_record_tokens,
        any_hit_eos,
    ) = _update_loop_state(
        next_tokens,
        active_mask,
        input_positions,
        seq_lens,
        eos_token_id,
        padding_token_id,
        dp_size,
        pad_len,
    )

    assert np.array_equal(new_active_mask, [True, False, True, True])
    assert np.array_equal(next_input_ids, [42, -1, 43, 44])
    assert np.array_equal(new_positions, [11, 20, 31, 41])
    assert np.array_equal(new_seq_lens, [12, 21, 0, 32, 42, 0])
    assert np.array_equal(step_record_tokens, [42, 99, 43, 44])
    assert any_hit_eos


def test_split_rngs():
    rng = jax.random.PRNGKey(42)
    static_size = 5
    dynamic_size = 3

    step_keys, final_key = _split_rngs(rng, static_size, dynamic_size)

    assert step_keys.shape[0] == static_size
    assert final_key.shape == rng.shape


def test_continue_decode_early_exit():
    batch_size = 2
    max_decode_steps = 5
    static_max_decode_steps = 5

    init_tokens = jnp.array([10, 20], dtype=jnp.int32)
    active_mask = jnp.array([True, True], dtype=jnp.bool_)

    # attn_metadata
    input_positions = jnp.array([0, 0], dtype=jnp.int32)
    block_tables = jnp.zeros((2, 16), dtype=jnp.int32)
    seq_lens = jnp.array([1, 1], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 1, 2], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0], dtype=jnp.int32)
    mamba_state_indices = None

    attn_metadata = AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
        mamba_state_indices=mamba_state_indices,
    )

    init_state = TpuSamplingState(
        current_tokens=init_tokens,
        active_mask=active_mask,
        attn_metadata=attn_metadata,
        step_counter=jnp.array(0, dtype=jnp.int32),
    )

    kv_caches = [jnp.zeros((2, 10))]  # dummy kv caches

    def mock_model_fn(state, kv_caches, current_tokens, attn_metadata, *args,
                      **kwargs):
        # Pass position in hidden_states so compute_logits can use it
        hidden_states = attn_metadata.input_positions.astype(jnp.float32)[:,
                                                                          None,
                                                                          None]
        return kv_caches, hidden_states, None, None

    def mock_compute_logits_fn(state, hidden_states, _):
        # Reshape to (batch_size, 1) and pad to (batch_size, 100)
        pos = hidden_states[:, 0, 0]
        logits = jnp.zeros((batch_size, 100))
        logits = logits.at[:, 0].set(pos)
        return logits

    def mock_sample_fn(rng, mesh, logits, sampling_metadata):
        pos = logits[:, 0].astype(jnp.int32)
        token_table = jnp.array(
            [
                [42, 43],  # pos 0
                [44, 99],  # pos 1 (req 1 hits EOS)
                [99, 99],  # pos 2 (req 0 hits EOS)
            ],
            dtype=jnp.int32,
        )
        batch_idx = jnp.arange(batch_size)
        next_tokens = token_table[pos, batch_idx]
        return next_tokens, None

    rng = jax.random.PRNGKey(0)

    (
        token_buffer,
        final_kv_caches,
        final_state,
        current_rng,
        all_expert_indices,
        logprobs_tensors,
    ) = continue_decode(
        state={},
        model_fn=mock_model_fn,
        compute_logits_fn=mock_compute_logits_fn,
        sample_fn=mock_sample_fn,
        init_state=init_state,
        kv_caches=kv_caches,
        max_decode_steps=max_decode_steps,
        static_max_decode_steps=static_max_decode_steps,
        eos_token_id=(99, ),
        padding_token_id=-1,
        rng=rng,
        mesh=None,
        sampling_metadata=None,
    )

    # Verify early exit at step 2 (because step 1 hit EOS)
    assert int(final_state.step_counter) == 2

    # Expected tokens:
    # Step 0: [42, 43]
    # Step 1: [44, 99]
    # Step 2: [-1, -1] (padding)
    # Step 3: [-1, -1] (padding)
    # Step 4: [-1, -1] (padding)
    expected_tokens = np.array(
        [
            [42, 43],
            [44, 99],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(token_buffer, expected_tokens)

    # Verify final state (state at the start of step 2, which did not run)
    assert np.array_equal(final_state.active_mask, [True, False])
    assert np.array_equal(final_state.current_tokens, [44, -1])
    assert np.array_equal(final_state.attn_metadata.input_positions, [2, 1])
    assert np.array_equal(final_state.attn_metadata.seq_lens, [3, 2])
    assert all_expert_indices is None
    assert logprobs_tensors is None


def test_continue_decode_with_experts():
    batch_size = 2
    max_decode_steps = 3
    static_max_decode_steps = 3

    init_tokens = jnp.array([10, 20], dtype=jnp.int32)
    active_mask = jnp.array([True, True], dtype=jnp.bool_)

    attn_metadata = AttentionMetadata(
        input_positions=jnp.array([0, 0], dtype=jnp.int32),
        block_tables=jnp.zeros((2, 16), dtype=jnp.int32),
        seq_lens=jnp.array([1, 1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0], dtype=jnp.int32),
        mamba_state_indices=None,
    )

    init_state = TpuSamplingState(
        current_tokens=init_tokens,
        active_mask=active_mask,
        attn_metadata=attn_metadata,
        step_counter=jnp.array(0, dtype=jnp.int32),
    )

    kv_caches = [jnp.zeros((2, 10))]

    def mock_model_fn(state, kv_caches, current_tokens, attn_metadata, *args,
                      **kwargs):
        hidden_states = current_tokens.astype(jnp.float32)[:, None, None]
        pos = attn_metadata.input_positions
        experts = (jnp.ones(
            (2, batch_size, 4), dtype=jnp.int32) * pos[None, :, None])
        return kv_caches, hidden_states, None, experts

    def mock_compute_logits_fn(state, hidden_states, _):
        return jnp.zeros((batch_size, 100))

    def mock_sample_fn(rng, mesh, logits, sampling_metadata):
        return jnp.array([42, 43], dtype=jnp.int32), None

    rng = jax.random.PRNGKey(0)

    (
        token_buffer,
        final_kv_caches,
        final_state,
        current_rng,
        all_expert_indices,
        logprobs_tensors,
    ) = continue_decode(
        state={},
        model_fn=mock_model_fn,
        compute_logits_fn=mock_compute_logits_fn,
        sample_fn=mock_sample_fn,
        init_state=init_state,
        kv_caches=kv_caches,
        max_decode_steps=max_decode_steps,
        static_max_decode_steps=static_max_decode_steps,
        eos_token_id=(99, ),
        padding_token_id=-1,
        rng=rng,
        mesh=None,
        sampling_metadata=None,
        collect_expert_indices=True,
    )

    assert int(final_state.step_counter) == 3
    assert all_expert_indices is not None
    assert all_expert_indices.shape == (3, 2, 2, 4)

    expected_experts = np.zeros((3, 2, 2, 4), dtype=np.int32)
    expected_experts[1] = 1
    expected_experts[2] = 2

    assert np.array_equal(all_expert_indices, expected_experts)
    assert logprobs_tensors is None


def test_continue_decode_no_exit_on_eos():
    batch_size = 2
    max_decode_steps = 5
    static_max_decode_steps = 5

    init_tokens = jnp.array([10, 20], dtype=jnp.int32)
    active_mask = jnp.array([True, True], dtype=jnp.bool_)

    attn_metadata = AttentionMetadata(
        input_positions=jnp.array([0, 0], dtype=jnp.int32),
        block_tables=jnp.zeros((2, 16), dtype=jnp.int32),
        seq_lens=jnp.array([1, 1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0], dtype=jnp.int32),
        mamba_state_indices=None,
    )

    init_state = TpuSamplingState(
        current_tokens=init_tokens,
        active_mask=active_mask,
        attn_metadata=attn_metadata,
        step_counter=jnp.array(0, dtype=jnp.int32),
    )

    kv_caches = [jnp.zeros((2, 10))]

    def mock_model_fn(state, kv_caches, current_tokens, attn_metadata, *args):
        hidden_states = attn_metadata.input_positions.astype(jnp.float32)[:,
                                                                          None,
                                                                          None]
        return kv_caches, hidden_states, None, None

    def mock_compute_logits_fn(state, hidden_states, _):
        pos = hidden_states[:, 0, 0]
        logits = jnp.zeros((batch_size, 100))
        logits = logits.at[:, 0].set(pos)
        return logits

    def mock_sample_fn(rng, mesh, logits, sampling_metadata):
        pos = logits[:, 0].astype(jnp.int32)
        token_table = jnp.array(
            [
                [42, 43],  # pos 0
                [44, 99],  # pos 1 (req 1 hits EOS)
                [99, 50],  # pos 2 (req 0 hits EOS)
                [60, 61],  # pos 3
                [70, 71],  # pos 4
            ],
            dtype=jnp.int32,
        )
        batch_idx = jnp.arange(batch_size)
        next_tokens = token_table[pos, batch_idx]
        return next_tokens, None

    rng = jax.random.PRNGKey(0)

    (
        token_buffer,
        final_kv_caches,
        final_state,
        current_rng,
        all_expert_indices,
        logprobs_tensors,
    ) = continue_decode(
        state={},
        model_fn=mock_model_fn,
        compute_logits_fn=mock_compute_logits_fn,
        sample_fn=mock_sample_fn,
        init_state=init_state,
        kv_caches=kv_caches,
        max_decode_steps=max_decode_steps,
        static_max_decode_steps=static_max_decode_steps,
        eos_token_id=(99, ),
        padding_token_id=-1,
        rng=rng,
        mesh=None,
        sampling_metadata=None,
        continue_decode_eos_check_interval=-1,
    )

    # Verify loop ran all 5 steps despite EOS hit
    assert int(final_state.step_counter) == 5

    # Expected tokens:
    # Step 0: [42, 43]
    # Step 1: [44, 99] (req 1 hits EOS)
    # Step 2: [99, -1] (req 0 hits EOS; req 1 inactive -> -1)
    # Step 3: [-1, -1] (both inactive)
    # Step 4: [-1, -1] (both inactive)
    expected_tokens = np.array(
        [
            [42, 43],
            [44, 99],
            [99, -1],
            [-1, -1],
            [-1, -1],
        ],
        dtype=np.int32,
    )
    assert np.array_equal(token_buffer, expected_tokens)
    assert np.array_equal(final_state.active_mask, [False, False])


def test_continue_decode_exit_on_eos_interval():
    batch_size = 2
    max_decode_steps = 10
    static_max_decode_steps = 10

    init_tokens = jnp.array([10, 20], dtype=jnp.int32)
    active_mask = jnp.array([True, True], dtype=jnp.bool_)

    attn_metadata = AttentionMetadata(
        input_positions=jnp.array([0, 0], dtype=jnp.int32),
        block_tables=jnp.zeros((2, 16), dtype=jnp.int32),
        seq_lens=jnp.array([1, 1], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0], dtype=jnp.int32),
        mamba_state_indices=None,
    )

    init_state = TpuSamplingState(
        current_tokens=init_tokens,
        active_mask=active_mask,
        attn_metadata=attn_metadata,
        step_counter=jnp.array(0, dtype=jnp.int32),
    )

    kv_caches = [jnp.zeros((2, 10))]

    def mock_model_fn(state, kv_caches, current_tokens, attn_metadata, *args):
        hidden_states = attn_metadata.input_positions.astype(jnp.float32)[:,
                                                                          None,
                                                                          None]
        return kv_caches, hidden_states, None, None

    def mock_compute_logits_fn(state, hidden_states, _):
        pos = hidden_states[:, 0, 0]
        logits = jnp.zeros((batch_size, 100))
        logits = logits.at[:, 0].set(pos)
        return logits

    def mock_sample_fn(rng, mesh, logits, sampling_metadata):
        pos = logits[:, 0].astype(jnp.int32)
        # Token table: step 0 (pos 0) produces 99 (EOS) for req 1
        token_table = jnp.array(
            [
                [42, 99],  # pos 0: req 1 hits EOS
                [43, -1],  # pos 1
                [44, -1],  # pos 2
                [45, -1],  # pos 3
                [46, -1],  # pos 4
            ],
            dtype=jnp.int32,
        )
        batch_idx = jnp.arange(batch_size)
        next_tokens = token_table[pos, batch_idx]
        return next_tokens, None

    rng = jax.random.PRNGKey(0)

    (
        token_buffer,
        final_kv_caches,
        final_state,
        current_rng,
        all_expert_indices,
    ) = continue_decode(
        state={},
        model_fn=mock_model_fn,
        compute_logits_fn=mock_compute_logits_fn,
        sample_fn=mock_sample_fn,
        init_state=init_state,
        kv_caches=kv_caches,
        max_decode_steps=max_decode_steps,
        static_max_decode_steps=static_max_decode_steps,
        eos_token_id=(99, ),
        padding_token_id=-1,
        rng=rng,
        mesh=None,
        sampling_metadata=None,
        continue_decode_eos_check_interval=3,  # Only check exit every 3 steps
    )

    # EOS hit at step 0. Step checks at i=1, i=2 do not exit. Step check at i=3 (3 % 3 == 0) exits.
    assert int(final_state.step_counter) == 3
