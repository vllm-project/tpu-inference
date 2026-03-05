# Copyright 2025 Google LLC
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

from tpu_inference.spec_decode.jax.dflash import DFlashProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    return jax.sharding.Mesh(devices, axis_names=("model", ))


def test_sample_block_draft_tokens_uses_target_model_logits():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2

    call_record = {}
    # Use a JAX array as dummy state (JAX tracing requires array-like args)
    target_state = jnp.array(0)

    def fake_compute_logits_fn(state, hidden_states, lora_metadata):
        call_record["shape"] = hidden_states.shape
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]], dtype=jnp.float32)

    proposer.compute_logits_fn = fake_compute_logits_fn

    # hidden_states layout: [context_token, draft_token_0, draft_token_1, ...]
    # _sample_block_draft_tokens slices [1:1+num_speculative_tokens]
    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    draft_token_ids = proposer._sample_block_draft_tokens(
        target_state, hidden_states)

    np.testing.assert_array_equal(np.asarray(draft_token_ids),
                                  np.array([1, 0], dtype=np.int32))
    assert call_record["shape"] == (2, 8)


def test_sample_block_draft_tokens_returns_1d_int_ids():
    proposer = object.__new__(DFlashProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = 2

    proposer.compute_logits_fn = lambda _state, _hidden, _lora: jnp.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    # 1 context + 2 draft positions
    hidden_states = jnp.ones((3, 4), dtype=jnp.bfloat16)
    draft_token_ids = proposer._sample_block_draft_tokens(
        jnp.array(0), hidden_states)

    assert draft_token_ids.ndim == 1
    assert draft_token_ids.shape == (2, )
    assert jnp.issubdtype(draft_token_ids.dtype, jnp.integer)
