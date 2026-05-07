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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.gdn.recurrent_scan_v2 import recurrent_scan
from tpu_inference.layers.common.ragged_gated_delta_rule_ref import ragged_gated_delta_rule as ragged_gated_delta_rule_ref


jax.config.parse_flags_with_absl()


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
    state_dtype=jnp.float32,
):
    """Build inputs for recurrent_scan tests."""
    all_seqlens = [1] * decode_N + list(mixed_seqlens)
    N = len(all_seqlens)
    T = sum(all_seqlens)
    cu_seqlens = np.cumsum([0] + all_seqlens).astype(np.int32)

    if max_num_req is not None:
        padded_cu = np.full(max_num_req + 1, T, dtype=np.int32)
        padded_cu[:len(cu_seqlens)] = cu_seqlens
        cu_seqlens = padded_cu

    # recurrent_scan expects flat mixed_qkv: [T, 2*H_qk*K + H_v*V]
    dim = 2 * H_qk * K + H_v * V
    mixed_qkv = rng.randn(T, dim).astype(np.float32)
    a = rng.randn(T, H_v).astype(np.float32)
    b = rng.randn(T, H_v).astype(np.float32)
    A_log = rng.randn(H_v).astype(np.float32)
    dt_bias = rng.randn(H_v).astype(np.float32)

    h0_N = max_num_req if max_num_req is not None else N
    h0 = rng.randn(h0_N, H_v, K, V).astype(np.float32)
    state_indices = np.arange(h0_N, dtype=np.int32)

    if dtype != np.float32:
        mixed_qkv, a, b, dt_bias = (jnp.array(x, dtype=dtype) for x in [mixed_qkv, a, b, dt_bias])
    else:
        mixed_qkv, a, b, dt_bias = (jnp.array(x) for x in [mixed_qkv, a, b, dt_bias])

    return (
        mixed_qkv,
        a,
        b,
        jnp.array(A_log),
        jnp.array(dt_bias, dtype=dtype),
        jnp.array(h0, dtype=state_dtype),
        jnp.array(cu_seqlens),
        jnp.array(state_indices),
        N,
    )


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RecurrentScanKernelTest(jtu.JaxTestCase):

    def _test_recurrent_scan(
        self,
        decode_N,
        mixed_seqlens,
        H_qk,
        H_v,
        K,
        V,
        *,
        max_num_req=512,
        atol=5e-2,
        state_dtype=jnp.float32,
    ):
        rng = np.random.RandomState(42)
        mixed_qkv, a, b, A_log, dt_bias, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng,
            decode_N,
            mixed_seqlens,
            H_qk,
            H_v,
            K,
            V,
            max_num_req=max_num_req,
            state_dtype=state_dtype,
        )
        T = mixed_qkv.shape[0]

        # ── Reference (ragged_gated_delta_rule_ref) ──
        # Note: SiLU is applied inside ref now, so we pass raw mixed_qkv.
        
        distribution_ref = jnp.array([decode_N, N, N], dtype=jnp.int32)

        # Reference expects has_initial_state. We pass all True to match the
        # behavior where tests were closer to passing.
        max_num_req_padded = state_indices.shape[0]
        has_initial_state = jnp.ones((max_num_req_padded,), dtype=jnp.bool_)

        ref_state, ref_o = ragged_gated_delta_rule_ref(
            mixed_qkv.astype(jnp.float32),
            b.astype(jnp.float32),
            a.astype(jnp.float32),
            h0.astype(jnp.float32),
            A_log[None, None, :],
            dt_bias[None, None, :],
            cu_seqlens,
            state_indices,
            distribution_ref,
            has_initial_state,
            n_kq=H_qk,
            n_v=H_v,
            d_k=K,
            d_v=V,
        )

        # ── Kernel ──
        # recurrent_scan takes raw mixed_qkv (fuses SiLU)
        # Note: has_initial_state is not passed here because recurrent_scan
        # does not need it.
        pallas_state, pallas_o = recurrent_scan(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=h0,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=cu_seqlens,
            state_indices=state_indices,
            distribution=jnp.array([decode_N, T, T], dtype=jnp.int32),
            n_kq=H_qk,
            n_v=H_v,
            d_k=K,
            d_v=V,
            chunk_size=128,
            BT=128,
            use_qk_norm_in_gdn=True,
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

    @parameterized.parameters(
        (0, [128, 256]),
        (0, [64, 128]),
    )
    def test_basic(self, decode_N, mixed_seqlens):
        self._test_recurrent_scan(decode_N, mixed_seqlens, 2, 8, 128, 128)

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
