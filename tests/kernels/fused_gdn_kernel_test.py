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
from jax._src import test_util as jtu

from tpu_inference.kernels.gdn import fused_gdn
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import \
    ragged_gated_delta_rule as ragged_gated_delta_rule_ref

jax.config.parse_flags_with_absl()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    rng,
    decode_N,
    mixed_seqlens,
    H_qk,
    H_v,
    K,
    V,
    dtype=jnp.bfloat16,
    max_num_req=None,
):
    """Build inputs for fused_gdn tests."""
    all_seqlens = [1] * decode_N + list(mixed_seqlens)
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


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class FusedGdnKernelTest(jtu.JaxTestCase):

    def _test_fused_gdn(
        self,
        decode_N,
        mixed_seqlens,
        H_qk,
        H_v,
        K,
        V,
        *,
        max_num_req=None,
        use_dt_bias=False,
        lower_bound=None,
        atol=1e-2,
    ):
        rng = np.random.RandomState(42)
        q, k, v, a, b, A_log, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng,
            decode_N,
            mixed_seqlens,
            H_qk,
            H_v,
            K,
            V,
            max_num_req=max_num_req,
        )
        T = q.shape[0]

        dt_bias = (jnp.array(rng.randn(H_v).astype(np.float32))
                   if use_dt_bias else jnp.zeros(H_v, dtype=jnp.float32))

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
            n_kq=H_qk,
            n_v=H_v,
            d_k=K,
            d_v=V,
        )
        ref_o = ref_o.reshape(T, H_v, V)

        # ── Kernel ──
        pallas_o, pallas_state = fused_gdn(
            q,
            k,
            v,
            cu_seqlens,
            a,  # [T, H_v] — broadcast to [T, H_v, K] inside fused_gdn
            h0,
            state_indices,
            b=b,
            distribution=jnp.array([decode_N, N], dtype=jnp.int32),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias if use_dt_bias else None,
            lower_bound=lower_bound,
        )

        # ── Compare ──
        self.assertAllClose(pallas_o,
                            ref_o,
                            atol=atol,
                            rtol=atol,
                            check_dtypes=False)
        self.assertAllClose(
            pallas_state[:N],
            ref_state[:N],
            atol=atol,
            rtol=atol,
            check_dtypes=False,
        )

    # ── Distribution forward (decode / mixed) ──

    @parameterized.parameters(
        (13, [8, 16, 24]),
        (8, []),
        (0, [9, 15]),
        (3, [9, 15]),
    )
    def test_basic(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(decode_N, mixed_seqlens, 2, 2, 128, 128)

    # ── Padded max_num_req ──

    @parameterized.parameters(
        (3, [9, 15], 5),
        (8, [], 8),
        (0, [9, 15], 4),
    )
    def test_padded_max_num_req(self, decode_N, mixed_seqlens, extra_pad):
        actual_N = decode_N + len(mixed_seqlens)
        self._test_fused_gdn(
            decode_N,
            mixed_seqlens,
            2,
            2,
            128,
            128,
            max_num_req=actual_N + extra_pad,
        )

    # ── GQA (H_v > H_qk) ──

    @parameterized.parameters(
        (5, [9, 15]),
        (8, []),
        (0, [9, 15]),
    )
    def test_gqa(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(decode_N, mixed_seqlens, 2, 8, 128, 128)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
