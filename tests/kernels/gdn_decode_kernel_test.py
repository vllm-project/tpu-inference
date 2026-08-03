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
"""Correctness tests for fused GDN kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from tests.test_utils import poison_tpu_memory
from tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref
from tpu_inference.kernels.gdn.v2.gdn_decode_kernel import \
    ragged_gated_delta_rule_decode_only

jax.config.parse_flags_with_absl()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    rng,
    decode_N,
    H_qk,
    H_v,
    K,
    V,
    dtype=jnp.bfloat16,
    max_num_req=None,
):
    """Build inputs for fused_gdn tests."""
    all_seqlens = [1] * decode_N
    N = len(all_seqlens)
    T = sum(all_seqlens)
    cu_seqlens = np.cumsum([0] + all_seqlens).astype(np.int32)

    if max_num_req is not None:
        padded_cu = np.full(max_num_req + 1, T, dtype=np.int32)
        padded_cu[:len(cu_seqlens)] = cu_seqlens
        cu_seqlens = padded_cu

    q = rng.randn(T, H_qk, K).astype(np.float32)
    k = rng.randn(T, H_qk, K).astype(np.float32)
    v = rng.randn(T, H_v, V).astype(np.float32)
    a = rng.randn(T, H_v).astype(np.float32)
    b = rng.randn(T, H_v).astype(np.float32)
    A_log = rng.randn(H_v).astype(np.float32)

    h0_N = max_num_req if max_num_req is not None else N
    h0 = rng.randn(h0_N, H_v, K, V).astype(np.float32)
    state_indices = np.arange(h0_N, dtype=np.int32)

    if dtype != np.float32:
        q, k, v, a, b = (jnp.array(x, dtype=dtype) for x in [q, k, v, a, b])
    else:
        q, k, v, a, b = (jnp.array(x) for x in [q, k, v, a, b])

    return (
        q,
        k,
        v,
        a,
        b,
        jnp.array(A_log),
        jnp.array(h0),
        jnp.array(cu_seqlens),
        jnp.array(state_indices),
        N,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class FusedGdnKernelTest(parameterized.TestCase):

    def _test_fused_gdn(
        self,
        decode_N,
        H_qk,
        H_v,
        K,
        V,
        *,
        max_num_req=None,
        use_dt_bias=False,
        atol=1e-2,
    ):
        rng = np.random.RandomState(42)
        q, k, v, a, b, A_log, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng,
            decode_N,
            H_qk,
            H_v,
            K,
            V,
            max_num_req=max_num_req,
        )
        T = q.shape[0]

        dt_bias = (jnp.array(rng.randn(H_v).astype(np.float32))
                   if use_dt_bias else jnp.zeros(H_v, dtype=jnp.float32))

        max_num_req_padded = state_indices.shape[0]
        has_initial_state = jnp.ones((max_num_req_padded, ), dtype=jnp.bool_)

        # ── Reference (ragged_gated_delta_rule_ref) ──
        mixed_qkv = jnp.concatenate(
            [q.reshape(T, -1),
             k.reshape(T, -1),
             v.reshape(T, -1)],
            axis=-1,
        )
        distribution_ref = jnp.array([decode_N, N, N], dtype=jnp.int32)

        ref_state, ref_o = ragged_gated_delta_rule_ref(
            mixed_qkv.astype(jnp.float32),
            b.astype(jnp.float32),
            a.astype(jnp.float32),
            h0.astype(jnp.float32),
            A_log[None, None, :],  # (1,1,H_v) to match curr_a rank in ref
            dt_bias[None, None, :],  # (1,1,H_v) to match curr_a rank in ref
            cu_seqlens,
            state_indices,
            distribution_ref,
            has_initial_state,
            n_kq=H_qk,
            n_v=H_v,
            d_k=K,
            d_v=V,
        )
        ref_o = ref_o.reshape(T, H_v, V)

        # ── Kernel ──
        pallas_state, pallas_o = ragged_gated_delta_rule_decode_only(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=h0,
            A_log=A_log,
            dt_bias=dt_bias if use_dt_bias else None,
            query_start_loc=cu_seqlens,
            state_indices=state_indices,
            distribution=distribution_ref,
            has_initial_state=has_initial_state,
            n_kq=H_qk,
            n_v=H_v,
            d_k=K,
            d_v=V,
            apply_silu=True,
        )
        pallas_o = pallas_o.reshape(T, H_v, V)

        # ── Compare ──
        np.testing.assert_allclose(np.array(pallas_o),
                                   np.array(ref_o),
                                   atol=atol,
                                   rtol=atol)
        np.testing.assert_allclose(
            np.array(pallas_state[:N]),
            np.array(ref_state[:N]),
            atol=atol,
            rtol=atol,
        )

    # ── Distribution forward (decode only) ──

    @parameterized.parameters(
        (13, ),
        (8, ),
        (1, ),
    )
    def test_basic(self, decode_N):
        self._test_fused_gdn(decode_N, 2, 2, 128, 128)

    # ── Padded max_num_req ──

    @parameterized.parameters(
        (3, 5),
        (8, 8),
        (1, 4),
    )
    def test_padded_max_num_req(self, decode_N, extra_pad):
        self._test_fused_gdn(
            decode_N,
            2,
            2,
            128,
            128,
            max_num_req=decode_N + extra_pad,
        )

    # ── GQA (H_v > H_qk) ──

    @parameterized.parameters(
        (5, ),
        (8, ),
        (1, ),
    )
    def test_gqa(self, decode_N):
        self._test_fused_gdn(decode_N, 2, 8, 128, 128)

    def test_uninitialized_memory_robustness(self):
        poison_tpu_memory()
        rng = np.random.RandomState(42)
        q, k, v, a, b, A_log, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng, 3, 2, 2, 128, 128)
        has_initial_state = jnp.ones((h0.shape[0], ), dtype=jnp.bool_)
        mixed_qkv = jnp.concatenate(
            [q.reshape(3, -1),
             k.reshape(3, -1),
             v.reshape(3, -1)],
            axis=-1,
        )
        distribution = jnp.array([3, N, N], dtype=jnp.int32)

        new_state, output = ragged_gated_delta_rule_decode_only(
            mixed_qkv,
            b,
            a,
            h0,
            A_log,
            None,
            cu_seqlens,
            state_indices,
            distribution,
            has_initial_state,
            n_kq=2,
            n_v=2,
            d_k=128,
            d_v=128,
            apply_silu=True,
        )
        self.assertFalse(jnp.any(jnp.isnan(output)))
        self.assertFalse(jnp.any(jnp.isnan(new_state)))

    def test_security_isolation(self):
        rng = np.random.RandomState(42)
        q, k, v, a, b, A_log, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng, 3, 2, 2, 128, 128)
        has_initial_state = jnp.ones((h0.shape[0], ), dtype=jnp.bool_)
        mixed_qkv = jnp.concatenate(
            [q.reshape(3, -1),
             k.reshape(3, -1),
             v.reshape(3, -1)],
            axis=-1,
        )
        distribution = jnp.array([3, N, N], dtype=jnp.int32)

        # Baseline clean output
        new_state_clean, output_clean = ragged_gated_delta_rule_decode_only(
            jnp.array(mixed_qkv),
            b,
            a,
            jnp.array(h0),
            A_log,
            None,
            cu_seqlens,
            state_indices,
            distribution,
            has_initial_state,
            n_kq=2,
            n_v=2,
            d_k=128,
            d_v=128,
            apply_silu=True,
        )

        # Inject NaNs to a malicious sequence's state
        h0_malicious = jnp.array(h0).at[1].set(jnp.nan)
        new_state_malicious, output_malicious = ragged_gated_delta_rule_decode_only(
            mixed_qkv=jnp.array(mixed_qkv),
            b=b,
            a=a,
            recurrent_state=h0_malicious,
            A_log=A_log,
            dt_bias=None,
            query_start_loc=cu_seqlens,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            n_kq=2,
            n_v=2,
            d_k=128,
            d_v=128,
            apply_silu=True,
        )

        np.testing.assert_allclose(
            np.array(output_malicious[:1]),
            np.array(output_clean[:1]),
            atol=0,
            rtol=0,
        )
        np.testing.assert_allclose(
            np.array(new_state_malicious[:1]),
            np.array(new_state_clean[:1]),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    absltest.main()
