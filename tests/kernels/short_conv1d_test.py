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

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from tpu_inference.kernels.causal_conv1d import causal_conv1d
from tpu_inference.kernels.experimental.short_conv1d import short_conv1d


def _make_inputs(
    *,
    lengths: list[int],
    H: int = 2,
    max_reqs: int | None = None,
    has_initial_state: list[bool] | None = None,
):
    rng = np.random.default_rng(123)
    D, W = 128, 4
    total = len(lengths)
    num_tokens = sum(lengths)
    max_reqs = max(max_reqs or total, total)
    n_states = max_reqs + 5

    cu_seqlens = np.full((max_reqs + 1, ), num_tokens, dtype=np.int32)
    cu_seqlens[:total + 1] = np.concatenate([[0], np.cumsum(lengths)])

    state_indices = np.zeros((max_reqs, ), dtype=np.int32)
    slots = np.array([3, 1, 5, 2, 7, 4, 6, 8], dtype=np.int32)
    state_indices[:] = slots[np.arange(max_reqs) % slots.size] % n_states

    has_init = np.ones((max_reqs, ), dtype=np.int32)
    if has_initial_state is not None:
        has_init[:total] = np.array(has_initial_state, dtype=np.int32)

    x = (rng.normal(size=(num_tokens, H, D)) * 0.2).astype(np.float32)
    weight = (rng.normal(size=(W, H, D)) * 0.2).astype(np.float32)
    conv_state = (rng.normal(size=(n_states, W - 1, H, D)) * 0.2).astype(
        np.float32)
    return x, weight, conv_state, cu_seqlens, state_indices, has_init


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ShortConv1dTest(jtu.JaxTestCase):

    @parameterized.named_parameters(
        dict(
            testcase_name="decode_only",
            lengths=[1, 1, 1, 1, 1, 1, 1, 1],
            decode_end=8,
            has_initial_state=None,
            max_reqs=None,
            H=2,
            decoding_block_n=None,
            prefill_block_n=None,
        ),
        dict(
            testcase_name="decode_only_partial_tile",
            lengths=[1] * 10,
            decode_end=10,
            has_initial_state=None,
            max_reqs=None,
            H=2,
            decoding_block_n=None,
            prefill_block_n=None,
        ),
        dict(
            testcase_name="prefill_only_with_fresh_slots",
            lengths=[5, 7, 3],
            decode_end=0,
            has_initial_state=[True, False, True],
            max_reqs=None,
            H=2,
            decoding_block_n=None,
            prefill_block_n=None,
        ),
        dict(
            testcase_name="prefill_multi_block_forced_tile",
            lengths=[20],
            decode_end=0,
            has_initial_state=[False],
            max_reqs=None,
            H=2,
            decoding_block_n=None,
            prefill_block_n=8,
        ),
        dict(
            testcase_name="mixed_large_token_count_small_req_count",
            lengths=[1, 1, 35, 35],
            decode_end=2,
            has_initial_state=[True, True, False, True],
            max_reqs=16,
            H=2,
            decoding_block_n=None,
            prefill_block_n=None,
        ),
        dict(
            testcase_name="mixed_qwen_head_shape",
            lengths=[1, 1, 5, 7],
            decode_end=2,
            has_initial_state=[True, True, False, True],
            max_reqs=16,
            H=96,
            decoding_block_n=None,
            prefill_block_n=None,
        ),
    )
    def test_matches_legacy_causal_conv1d(
        self,
        lengths,
        decode_end,
        has_initial_state,
        max_reqs,
        H,
        decoding_block_n,
        prefill_block_n,
    ):
        x, weight, conv_state, cu_seqlens, state_indices, has_init = _make_inputs(
            lengths=lengths,
            H=H,
            max_reqs=max_reqs,
            has_initial_state=has_initial_state,
        )
        H, D = x.shape[1:]
        W = weight.shape[0]
        total = len(lengths)

        out, new_conv_state = short_conv1d(
            jnp.array(x, dtype=jnp.bfloat16),
            jnp.array(weight, dtype=jnp.bfloat16),
            jnp.array(conv_state, dtype=jnp.bfloat16),
            jnp.array(cu_seqlens),
            jnp.array(state_indices),
            jnp.array([decode_end, total], dtype=jnp.int32),
            jnp.array(has_init),
            decoding_block_n=decoding_block_n,
            prefill_block_n=prefill_block_n,
        )

        flat_weight = weight.transpose(1, 2, 0).reshape(H * D, 1, W)
        ref_out, ref_conv_state = causal_conv1d.ragged_causal_conv1d(
            jnp.array(x.reshape(x.shape[0], H * D), dtype=jnp.bfloat16),
            jnp.array(conv_state.reshape(conv_state.shape[0], W - 1, H * D),
                      dtype=jnp.bfloat16),
            jnp.array(flat_weight, dtype=jnp.bfloat16),
            None,
            jnp.array(cu_seqlens),
            jnp.array(state_indices),
            jnp.array([decode_end, decode_end, total], dtype=jnp.int32),
            jnp.array(has_init.astype(bool)),
            kernel_size=W,
        )

        self.assertAllClose(out,
                            ref_out.reshape(x.shape),
                            atol=4e-2,
                            rtol=4e-2,
                            check_dtypes=False)
        self.assertAllClose(new_conv_state,
                            ref_conv_state.reshape(conv_state.shape),
                            atol=4e-2,
                            rtol=4e-2,
                            check_dtypes=False)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
