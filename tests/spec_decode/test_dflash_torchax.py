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

from tpu_inference.spec_decode.torchax.dflash import DFlashTorchaxProposer


def _make_single_device_mesh() -> jax.sharding.Mesh:
    devices = np.array(jax.devices()[:1])
    return jax.sharding.Mesh(devices, axis_names=("model", ))


def _make_proposer(num_speculative_tokens=2, block_size=3):
    proposer = object.__new__(DFlashTorchaxProposer)
    proposer.mesh = _make_single_device_mesh()
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.block_size = block_size
    proposer.mask_token_id = 0
    proposer.max_model_len = 128
    proposer._raw_hidden_dim = 16
    proposer._ctx_len = 0
    proposer._prev_seq_len = 0
    proposer._ctx_buf = jnp.zeros((128, 16), dtype=jnp.bfloat16)
    return proposer


def test_build_noise_block_shape_and_first_token():
    proposer = _make_proposer(num_speculative_tokens=2, block_size=3)
    seq_len_arr = jnp.array([10], dtype=jnp.int32)
    next_token_ids = jnp.array([42], dtype=jnp.int32)

    noise_ids, noise_positions = proposer._build_noise_block(
        seq_len_arr, next_token_ids, 0, 3)

    assert noise_ids.shape == (3, )
    assert noise_positions.shape == (3, )
    np.testing.assert_array_equal(np.asarray(noise_ids),
                                  np.array([42, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(noise_positions),
                                  np.array([10, 11, 12], dtype=np.int32))


def test_sample_block_draft_tokens_shape_and_dtype():
    proposer = _make_proposer(num_speculative_tokens=2, block_size=3)

    def fake_compute_logits(params, hidden_states, embed_weight):
        return jnp.array([[0.0, 2.0, 1.0], [4.0, 1.0, 0.0]], dtype=jnp.float32)

    proposer._compute_logits_fn = fake_compute_logits

    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    draft_ids = proposer._sample_block_draft_tokens({}, hidden_states,
                                                    jnp.zeros((3, 8)))

    assert draft_ids.ndim == 1
    assert draft_ids.shape == (2, )
    assert jnp.issubdtype(draft_ids.dtype, jnp.integer)
    np.testing.assert_array_equal(np.asarray(draft_ids),
                                  np.array([1, 0], dtype=np.int32))


def test_context_buffer_incremental_update():
    proposer = _make_proposer()
    assert proposer._ctx_len == 0

    raw = jnp.ones((5, 16), dtype=jnp.bfloat16) * 0.5
    proposer._prev_seq_len = 0
    seq_len = 5
    num_new = seq_len - proposer._ctx_len
    assert num_new == 5
    end = min(proposer._ctx_len + num_new, proposer.max_model_len)
    n_copy = end - proposer._ctx_len
    from jax import lax
    new_raw = raw[:n_copy].astype(jnp.bfloat16)
    proposer._ctx_buf = lax.dynamic_update_slice(proposer._ctx_buf, new_raw,
                                                 (proposer._ctx_len, 0))
    proposer._ctx_len = end

    assert proposer._ctx_len == 5
    np.testing.assert_allclose(np.asarray(proposer._ctx_buf[0, 0]),
                               0.5,
                               atol=0.01)
    np.testing.assert_allclose(np.asarray(proposer._ctx_buf[5, 0]),
                               0.0,
                               atol=0.01)


def test_context_crop_on_rejection():
    proposer = _make_proposer()
    proposer._ctx_len = 10
    proposer._prev_seq_len = 10

    seq_len = 7
    if proposer._prev_seq_len > 0 and seq_len < proposer._ctx_len:
        proposer._ctx_len = seq_len
    proposer._prev_seq_len = seq_len

    assert proposer._ctx_len == 7
    assert proposer._prev_seq_len == 7
