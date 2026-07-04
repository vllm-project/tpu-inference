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
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.causal_conv1d import causal_conv1d


def reference_causal_conv1d(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    kernel_size: int,
    has_initial_state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    num_tokens = x.shape[0]
    num_seqs = state_indices.shape[0]
    sequences = jnp.arange(num_seqs)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    row_to_seq_idx = jnp.repeat(sequences, query_lens)

    real_num_seqs = int(distribution[2])
    real_num_tokens = int(query_start_loc[real_num_seqs])

    new_conv_state = jnp.copy(conv_state)
    out_list = []

    for row in range(num_tokens):
        if row >= real_num_tokens:
            out_list.append(jnp.zeros_like(x[0]))
            continue

        start_row = row - kernel_size + 1
        s_idx = int(row_to_seq_idx[row])
        seq_start = int(query_start_loc[s_idx])
        token_step = row - seq_start
        has_init = bool(has_initial_state[s_idx])

        # In speculative decoding (2D state_indices), read from the initial state (Slot 0)
        # and write to the corresponding history slot for the current token step.
        if state_indices.ndim == 2:
            state_idx_read = int(state_indices[s_idx, 0])
            state_idx_write = int(state_indices[s_idx, token_step + 1])
        else:
            state_idx_read = int(state_indices[s_idx])
            state_idx_write = int(state_indices[s_idx])

        row_out = jnp.zeros_like(x[0], dtype=jnp.float32)
        window_vals = []

        for k in range(kernel_size):
            idx = start_row + k

            if idx < seq_start:
                state_offset = idx - (seq_start - kernel_size + 1)
                if has_init:
                    val = conv_state[state_idx_read, state_offset]
                else:
                    val = jnp.zeros_like(x[0])
            else:
                val = x[idx]

            window_vals.append(val)
            row_out += val.astype(jnp.float32) * conv_weight[:, 0, k].astype(
                jnp.float32)

        if conv_bias is not None:
            row_out += conv_bias.astype(jnp.float32)

        out_list.append(row_out.astype(x.dtype))

        # Update the state at every token step to build the history
        update = jnp.stack(window_vals[1:], axis=0)
        new_conv_state = new_conv_state.at[state_idx_write].set(update)

    out = jnp.stack(out_list, axis=0)
    return out, new_conv_state


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class CausalConv1dTest(jtu.JaxTestCase):

    def _run_verification(
        self,
        lengths: list[int],
        q_loc: list[int],
        distribution: list[int],
        dim: int,
        kernel_size: int,
        has_bias: bool,
        dtype: jnp.dtype,
    ):
        num_tokens = q_loc[-1]
        max_reqs = len(lengths)
        state_indices = jnp.arange(1, max_reqs + 1)
        q_loc = jnp.array(q_loc)
        distribution = jnp.array(distribution, dtype=jnp.int32)

        key = jax.random.key(0)

        x = jax.random.normal(key, (num_tokens, dim), dtype=dtype)
        conv_state = jax.random.normal(
            key,
            (max_reqs + 1, kernel_size - 1, dim),
            dtype=dtype,
        )
        conv_weight = jax.random.normal(
            key,
            (dim, 1, kernel_size),
            dtype=dtype,
        )
        conv_bias = (jax.random.normal(key, (dim, ), dtype=dtype)
                     if has_bias else None)
        seq_lens = jnp.array(lengths)
        has_initial_state = seq_lens > (q_loc[1:] - q_loc[:-1])

        out_ref, new_conv_state_ref = reference_causal_conv1d(
            x=x,
            conv_state=conv_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            kernel_size=kernel_size,
            has_initial_state=has_initial_state,
        )

        out, new_conv_state = causal_conv1d.ragged_causal_conv1d(
            x=jnp.copy(x),
            conv_state=jnp.copy(conv_state),
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            kernel_size=kernel_size,
            has_initial_state=has_initial_state,
        )

        self.assertArraysAllClose(out, out_ref, rtol=1e-2, atol=1e-2)
        self.assertArraysAllClose(new_conv_state,
                                  new_conv_state_ref,
                                  rtol=1e-2,
                                  atol=1e-2)

    @parameterized.product(
        batch_config=[
            ([256], [0, 128], [0, 0, 1]),
        ],
        dim=[128, 512, 1024],
        kernel_size=[4],
        has_bias=[True, False],
        dtype=[jnp.bfloat16, jnp.float32],
    )
    def test_causal_conv1d(self, batch_config, dim, kernel_size, has_bias,
                           dtype):
        lengths, q_loc, distribution = batch_config
        self._run_verification(lengths, q_loc, distribution, dim, kernel_size,
                               has_bias, dtype)

    @parameterized.named_parameters(
        dict(
            testcase_name="multiple_sequences_equal_length",
            lengths=[64, 64, 64, 64],
            q_loc=[0, 32, 64, 96, 128],
            distribution=[0, 0, 4],
        ),
        dict(
            testcase_name="multiple_sequences_varying_length",
            lengths=[200, 100],
            q_loc=[0, 128, 192],
            distribution=[0, 0, 2],
        ),
        dict(
            testcase_name="empty_sequence",
            lengths=[100, 10, 50],
            q_loc=[0, 64, 64, 96],
            distribution=[0, 0, 3],
        ),
        dict(
            testcase_name="single_token_decode",
            lengths=[10] * 16,
            q_loc=list(range(17)),
            distribution=[16, 16, 16],
        ),
        dict(
            testcase_name="highly_ragged_varying_lengths",
            lengths=[150, 20, 100, 30],
            q_loc=[0, 128, 135, 199, 214],
            distribution=[0, 0, 4],
        ),
        dict(
            testcase_name="brand_new_prefill",
            lengths=[128, 64],
            q_loc=[0, 128, 192],
            distribution=[0, 0, 2],
        ),
    )
    def test_causal_conv1d_edge_cases(self, lengths, q_loc, distribution):
        self._run_verification(
            lengths,
            q_loc,
            distribution,
            dim=512,
            kernel_size=4,
            has_bias=True,
            dtype=jnp.bfloat16,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
