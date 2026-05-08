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
"""Correctness tests for recurrent scan kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.gdn.recurrent_scan_v2 import recurrent_scan
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref

jax.config.parse_flags_with_absl()


@pytest.mark.skip(reason="Need jax 0.10.0")
@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RecurrentScanKernelTest(jtu.JaxTestCase):

    def _test_recurrent_scan(self, decode_lengths, prefill_lengths,
                             chunk_size):
        dtype = jnp.bfloat16
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8

        MAX_TOKENS = 8192
        MAX_REQS = 512

        actual_num_tokens = decode_lengths + sum(prefill_lengths)
        actual_max_reqs = decode_lengths + len(prefill_lengths)

        q_loc = jnp.cumsum(
            jnp.array([0] + [1] * decode_lengths + prefill_lengths,
                      dtype=jnp.int32))
        q_loc = jnp.pad(
            q_loc,
            (0, MAX_REQS + 1 - len(q_loc)),
            mode="constant",
            constant_values=1,
        )

        rngs = iter(jax.random.split(jax.random.key(0), 15))

        available_indices = jax.random.permutation(
            next(rngs), jnp.arange(1, MAX_REQS, dtype=jnp.int32))
        valid_state_indices = available_indices[:actual_max_reqs]
        state_indices = jnp.pad(
            valid_state_indices,
            (0, MAX_REQS - actual_max_reqs),
            mode="constant",
            constant_values=0,
        )

        query = jax.random.normal(next(rngs), (MAX_TOKENS, n_kq * kq_head_dim),
                                  dtype=dtype)
        key = jax.random.normal(next(rngs), (MAX_TOKENS, n_kq * kq_head_dim),
                                dtype=dtype)
        value = jax.random.normal(next(rngs), (MAX_TOKENS, n_v * v_head_dim),
                                  dtype=dtype)
        b = jax.random.normal(next(rngs), (MAX_TOKENS, n_v), dtype=dtype)
        a = jax.random.normal(next(rngs), (MAX_TOKENS, n_v), dtype=dtype)

        # Pad inputs at the end to avoid out-of-bounds DMA reads
        query = jnp.pad(query, ((0, chunk_size), (0, 0)))
        key = jnp.pad(key, ((0, chunk_size), (0, 0)))
        value = jnp.pad(value, ((0, chunk_size), (0, 0)))
        b = jnp.pad(b, ((0, chunk_size), (0, 0)))
        a = jnp.pad(a, ((0, chunk_size), (0, 0)))

        recurrent_state = jnp.zeros((MAX_REQS, n_v, kq_head_dim, v_head_dim),
                                    dtype=jnp.float32)

        num_decodes = decode_lengths
        decode_state = jax.random.normal(
            next(rngs),
            (num_decodes, n_v, kq_head_dim, v_head_dim),
            dtype=jnp.float32,
        )
        prefill_state = jnp.zeros(
            (actual_max_reqs - num_decodes, n_v, kq_head_dim, v_head_dim),
            dtype=jnp.float32,
        )
        valid_recurrent_state = jnp.concatenate([decode_state, prefill_state],
                                                axis=0)
        recurrent_state = recurrent_state.at[valid_state_indices].set(
            valid_recurrent_state)

        A_log = jax.random.normal(next(rngs), (n_v, ), dtype=dtype)
        dt_bias = jax.random.normal(jax.random.key(0), (n_v, ), dtype=dtype)

        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)

        distribution = jnp.array(
            [decode_lengths, actual_max_reqs, actual_max_reqs],
            dtype=jnp.int32)

        # ── Reference (ragged_gated_delta_rule_ref) ──
        has_initial_state = jnp.zeros((MAX_REQS, ), dtype=bool)
        has_initial_state = has_initial_state.at[:num_decodes].set(True)

        dummy_state = jnp.zeros((1, n_v, kq_head_dim, v_head_dim),
                                dtype=jnp.float32)
        recurrent_state_ref = jnp.concatenate([dummy_state, recurrent_state],
                                              axis=0)

        ref_state, ref_o = ragged_gated_delta_rule_ref(
            mixed_qkv.astype(jnp.float32),
            b.astype(jnp.float32),
            a.astype(jnp.float32),
            recurrent_state_ref,
            A_log[None, None, :],
            dt_bias[None, None, :],
            q_loc,
            state_indices + 1,
            distribution,
            has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
        )

        # ── Kernel ──
        pallas_state, pallas_o = recurrent_scan(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            chunk_size=chunk_size,
            BT=chunk_size,
            use_qk_norm_in_gdn=True,
        )

        # ── Compare ──
        self.assertAllClose(
            pallas_o[:actual_num_tokens],
            ref_o[:actual_num_tokens],
            atol=5e-2,
            rtol=5e-2,
            check_dtypes=False,
        )
        self.assertAllClose(
            pallas_state[valid_state_indices],
            ref_state[valid_state_indices + 1],
            atol=5e-2,
            rtol=5e-2,
            check_dtypes=False,
        )

    @parameterized.parameters(
        (0, [128, 256]),
        (0, [64, 128]),
        (13, [8, 16, 24]),
        (8, []),
        (10, [9, 15]),
    )
    def test_basic(self, decode_N, mixed_seqlens):
        self._test_recurrent_scan(decode_N, mixed_seqlens, 64)

    @parameterized.parameters(
        (8, []),
        (10, [9, 15]),
    )
    def test_gqa(self, decode_N, mixed_seqlens):
        self._test_recurrent_scan(decode_N, mixed_seqlens, 64)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
