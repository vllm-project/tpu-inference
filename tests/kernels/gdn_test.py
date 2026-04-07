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

jax.config.parse_flags_with_absl()

# ---------------------------------------------------------------------------
# JAX reference implementation
# ---------------------------------------------------------------------------


def naive_recurrent_gdn(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """Pure JAX reference GDN forward pass."""
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = [x.astype(jnp.float32) for x in [q, k, v, g, beta]]
    q = q * scale

    if cu_seqlens is not None:
        assert B == 1
        cu_seqlens_np = np.asarray(cu_seqlens, dtype=np.int32)
        N = len(cu_seqlens_np) - 1
        o = jnp.zeros((1, T, H, V), dtype=jnp.float32)
        final_states = []
        t_indices = jnp.arange(T)

        for n in range(N):
            bos = int(cu_seqlens_np[n])
            eos = int(cu_seqlens_np[n + 1])
            S_init = jnp.zeros((H, K, V), dtype=jnp.float32)
            if initial_state is not None:
                S_init = S_init + initial_state[n]

            def step_fn(S, t, _bos=bos, _eos=eos):
                active = (t >= _bos) & (t < _eos)
                q_t, k_t, v_t = q[0, t], k[0, t], v[0, t]
                g_t, b_t = g[0, t], beta[0, t]
                S_new = S * jnp.exp(g_t[..., None])
                residual = v_t - jnp.sum(k_t[..., None] * S_new, axis=-2)
                S_new = S_new + (b_t[..., None] *
                                 k_t)[..., None] * residual[:, None, :]
                o_t = jnp.sum(q_t[..., None] * S_new, axis=-2)
                S = jnp.where(active, S_new, S)
                o_t = jnp.where(active, o_t, jnp.zeros_like(o_t))
                return S, o_t

            S_final_n, o_seq = jax.lax.scan(step_fn, S_init, t_indices)
            o = o.at[0].set(o[0] + o_seq)
            if output_final_state:
                final_states.append(S_final_n)

        S_final = jnp.stack(final_states,
                            axis=0) if output_final_state else None
        return o.astype(dtype), S_final

    S_init = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if initial_state is not None:
        S_init = S_init + initial_state

    q_s = jnp.transpose(q, (1, 0, 2, 3))
    k_s = jnp.transpose(k, (1, 0, 2, 3))
    v_s = jnp.transpose(v, (1, 0, 2, 3))
    g_s = jnp.transpose(g, (1, 0, 2, 3))
    b_s = jnp.transpose(beta, (1, 0, 2))

    def step_fn(S, inputs):
        q_i, k_i, v_i, g_i, b_i = inputs
        S = S * jnp.exp(g_i[..., None])
        residual = v_i - jnp.sum(k_i[..., None] * S, axis=-2)
        S = S + (b_i[..., None] * k_i)[..., None] * residual[:, :, None, :]
        o_i = jnp.sum(q_i[..., None] * S, axis=-2)
        return S, o_i

    S_final, o_scan = jax.lax.scan(step_fn, S_init, (q_s, k_s, v_s, g_s, b_s))
    o = jnp.transpose(o_scan, (1, 0, 2, 3))
    if not output_final_state:
        S_final = None
    return o.astype(dtype), S_final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x, axis=-1):
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / np.maximum(norm, 1e-12)


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

    q = rng.randn(1, T, H_qk, K).astype(np.float32) * 0.1
    k = rng.randn(1, T, H_qk, K).astype(np.float32) * 0.1
    v = rng.randn(1, T, H_v, V).astype(np.float32) * 0.1
    g = -np.log1p(np.exp(-rng.randn(1, T, H_v, K).astype(np.float32)))
    beta = (1.0 / (1.0 + np.exp(-rng.randn(1, T, H_v)))).astype(np.float32)
    h0_N = max_num_req if max_num_req is not None else N
    h0 = rng.randn(h0_N, H_v, K, V).astype(np.float32) * 0.01
    state_indices = np.arange(h0_N, dtype=np.int32)

    if dtype != np.float32:
        q, k, v, g, beta = (jnp.array(x, dtype=dtype)
                            for x in [q, k, v, g, beta])
    else:
        q, k, v, g, beta = (jnp.array(x) for x in [q, k, v, g, beta])

    return (
        q,
        k,
        v,
        g,
        beta,
        jnp.array(h0),
        jnp.array(cu_seqlens),
        jnp.array(state_indices),
        N,
    )


def _ref_gdn(q, k, v, g, beta, h0, cu_seqlens, H_qk=None, **kwargs):
    """Call naive_recurrent_gdn with float32 casts and optional GQA repeat."""
    q_ref = q.astype(jnp.float32)
    k_ref = k.astype(jnp.float32)
    if H_qk is not None:
        H_v = v.shape[2]
        repeat_factor = H_v // H_qk
        if repeat_factor > 1:
            q_ref = jnp.repeat(q_ref, repeat_factor, axis=2)
            k_ref = jnp.repeat(k_ref, repeat_factor, axis=2)
    return naive_recurrent_gdn(
        q_ref,
        k_ref,
        v.astype(jnp.float32),
        g=g.astype(jnp.float32),
        beta=beta.astype(jnp.float32),
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        **kwargs,
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
        use_l2norm=False,
        use_gate_in_kernel=False,
        use_dt_bias=False,
        lower_bound=None,
        atol=1e-2,
    ):
        rng = np.random.RandomState(42)
        q, k, v, g, beta, h0, cu_seqlens, state_indices, N = _make_inputs(
            rng,
            decode_N,
            mixed_seqlens,
            H_qk,
            H_v,
            K,
            V,
            max_num_req=max_num_req,
        )

        # Prepare gate / A_log / dt_bias for gate-in-kernel mode
        A_log = None
        dt_bias_arr = None
        if use_gate_in_kernel:
            g = jnp.array(
                rng.randn(*g.shape).astype(np.float32) * 0.5,
                dtype=jnp.bfloat16,
            )
            A_log = jnp.array(rng.randn(H_v).astype(np.float32) * 0.5)
            dt_bias_arr = (jnp.array(
                rng.randn(H_v, K).astype(np.float32) *
                0.1) if use_dt_bias else None)

        # ── Reference ──
        if use_gate_in_kernel:
            g_f32 = g.astype(jnp.float32)
            a = jnp.exp(A_log)
            if use_dt_bias:
                g_f32 = g_f32 + dt_bias_arr[None, None, :, :]
            if lower_bound is not None:
                gk = lower_bound / (1.0 +
                                    jnp.exp(-(a[None, None, :, None] * g_f32)))
            else:
                gk = -a[None, None, :, None] * jnp.log(1.0 + jnp.exp(g_f32))
            ref_g = gk
        else:
            ref_g = g

        if use_l2norm:
            q_norm = _l2_normalize(np.array(q.astype(jnp.float32)))
            k_norm = _l2_normalize(np.array(k.astype(jnp.float32)))
            q_ref, k_ref = jnp.array(q_norm), jnp.array(k_norm)
        else:
            q_ref, k_ref = q, k

        # GQA repeat for reference
        if H_v > H_qk:
            repeat_factor = H_v // H_qk
            q_ref = jnp.repeat(q_ref.astype(jnp.float32),
                               repeat_factor,
                               axis=2)
            k_ref = jnp.repeat(k_ref.astype(jnp.float32),
                               repeat_factor,
                               axis=2)

        ref_h0 = h0[:N] if max_num_req is not None else h0
        ref_cu = cu_seqlens[:N + 1] if max_num_req is not None else cu_seqlens

        ref_o, ref_ht = naive_recurrent_gdn(
            q_ref.astype(jnp.float32),
            k_ref.astype(jnp.float32),
            v.astype(jnp.float32),
            g=ref_g.astype(jnp.float32),
            beta=beta.astype(jnp.float32),
            initial_state=ref_h0,
            cu_seqlens=ref_cu,
            output_final_state=True,
        )

        # ── Kernel ──
        pallas_o, pallas_state = fused_gdn(
            q,
            k,
            v,
            cu_seqlens,
            g,
            h0,
            state_indices,
            beta=beta,
            distribution=jnp.array([decode_N, N], dtype=jnp.int32),
            use_qk_l2norm_in_kernel=use_l2norm,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias_arr,
            lower_bound=lower_bound,
        )

        # ── Compare ──
        self.assertAllClose(pallas_o,
                            ref_o,
                            atol=atol,
                            rtol=atol,
                            check_dtypes=False)
        self.assertAllClose(pallas_state[:N],
                            ref_ht,
                            atol=atol,
                            rtol=atol,
                            check_dtypes=False)

    # ── Distribution forward (decode / mixed) ──

    @parameterized.parameters(
        (13, [8, 16, 24]),
        (8, []),
        (0, [9, 15]),
    )
    def test_basic(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(decode_N, mixed_seqlens, 2, 2, 128, 128)

    # ── L2 normalization ──

    def test_l2norm(self):
        self._test_fused_gdn(5, [9, 15], 2, 2, 128, 128, use_l2norm=True)

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

    # ── Gate in kernel ──

    @parameterized.parameters(
        (3, [9, 15]),
        (8, []),
        (0, [9, 15]),
    )
    def test_gate_in_kernel(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(
            decode_N,
            mixed_seqlens,
            2,
            2,
            128,
            128,
            use_gate_in_kernel=True,
        )

    # ── GQA (H_v > H_qk) ──

    @parameterized.parameters(
        (5, [9, 15]),
        (8, []),
        (0, [9, 15]),
    )
    def test_gqa(self, decode_N, mixed_seqlens):
        self._test_fused_gdn(decode_N, mixed_seqlens, 2, 8, 128, 128)

    # ── GQA + gate in kernel + L2 norm ──

    def test_gqa_gate_l2norm(self):
        self._test_fused_gdn(
            5,
            [9, 15],
            2,
            8,
            128,
            128,
            use_l2norm=True,
            use_gate_in_kernel=True,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
