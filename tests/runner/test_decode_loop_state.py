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
"""Per-step state advancement of the fused decode loop.

`has_initial_state` must be promoted for a row after it runs a step: a
request that enters the loop with num_computed_tokens == 0 (a one-token
prompt whose first computed chunk IS the loop's first step) carries flag 0
in the loop-invariant metadata template, and if the flag never advances,
every recurrent (KDA/mamba) layer re-zeroes that row's state on EVERY loop
step. The fed-back token still advances, so greedy decode settles into a
strict two-token cycle — observed end-to-end before the fix. Padding rows
must keep flag 0 (the null block never holds real state).
"""
import jax.numpy as jnp
import numpy as np

from tpu_inference.runner.decode_loop import _update_loop_state

EOS = (2, )
PAD = 0


def _step(next_tokens, active, hi, dp_size=1, pad_len=0, seq_lens=None):
    n = len(next_tokens)
    if seq_lens is None:
        seq_lens = jnp.ones((n + pad_len, ), jnp.int32)
    return _update_loop_state(
        jnp.asarray(next_tokens, jnp.int32),
        jnp.asarray(active, bool),
        jnp.ones((n, ), jnp.int32),
        seq_lens,
        jnp.asarray(hi, jnp.int32),
        eos_token_id=EOS,
        padding_token_id=PAD,
        dp_size=dp_size,
        pad_len=pad_len,
    )


def test_active_row_with_zero_flag_is_promoted_after_one_step():
    (_, _, _, _, new_hi, _, _) = _step(next_tokens=[5], active=[True], hi=[0])
    np.testing.assert_array_equal(np.asarray(new_hi), [1])


def test_padding_rows_never_claim_state():
    # 2 real rows (one entering with flag 0) + 2 padding rows, padded the
    # same way seq_lens is padded.
    (_, _, _, _, new_hi, _, _) = _step(next_tokens=[5, 7],
                                       active=[True, True],
                                       hi=[0, 1, 0, 0],
                                       pad_len=2,
                                       seq_lens=jnp.ones((4, ), jnp.int32))
    np.testing.assert_array_equal(np.asarray(new_hi), [1, 1, 0, 0])


def test_eos_deactivated_row_keeps_its_flag():
    # Row hits EOS this step: it stops advancing but its state exists.
    (mask, _, _, _, new_hi, _, hit) = _step(next_tokens=[2],
                                            active=[True],
                                            hi=[1])
    assert not bool(np.asarray(mask)[0])
    assert bool(np.asarray(hit))
    np.testing.assert_array_equal(np.asarray(new_hi), [1])
