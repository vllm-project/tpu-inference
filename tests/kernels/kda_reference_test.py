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
"""Validation for the chunked-JAX KDA reference.

Oracle: an fp64 numpy implementation of the definitional KDA recurrence
(arXiv:2510.26692, eq. S_t = (I - beta k k^T) Diag(a) S_{t-1} + beta k v^T),
including q/k L2-norm, sigmoid(beta) and both gate forms, mirroring the
reference implementations in HF moonshotai/Kimi-K3 and
flash-linear-attention's fla.ops.kda.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.kda.reference.ragged_kda_chunked import (
    ragged_kda_decode_only, ragged_kda_mixed_prefill)

LOWER_BOUND = -5.0


def _softplus(x):
    return np.logaddexp(0.0, x)


def _l2norm_np(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def _gate_np(g_raw, A_log, dt_bias, lower_bound):
    h = A_log.shape[0]
    x = g_raw + dt_bias.reshape(h, -1)
    a = np.exp(A_log)[:, None]
    if lower_bound is None:
        return -a * _softplus(x)
    return lower_bound / (1.0 + np.exp(-a * x))


def naive_kda_fp64(q,
                   k,
                   v,
                   g_raw,
                   beta_raw,
                   A_log,
                   dt_bias,
                   lower_bound,
                   h0=None):
    """Definitional per-token recurrence, fp64. Shapes: q/k [T,H,K],
    v [T,H,V], g_raw [T,H,K], beta_raw [T,H]. Returns (o [T,H,V],
    S [H,K,V])."""
    q = _l2norm_np(q.astype(np.float64))
    k = _l2norm_np(k.astype(np.float64))
    v = v.astype(np.float64)
    scale = q.shape[-1]**-0.5
    beta = 1.0 / (1.0 + np.exp(-beta_raw.astype(np.float64)))
    g = _gate_np(g_raw.astype(np.float64), A_log.astype(np.float64),
                 dt_bias.astype(np.float64), lower_bound)
    T, H, K = q.shape
    V = v.shape[-1]
    S = np.zeros((H, K, V)) if h0 is None else h0.astype(np.float64).copy()
    o = np.zeros((T, H, V))
    for t in range(T):
        S = S * np.exp(g[t])[..., None]
        k_state = np.einsum('hk,hkv->hv', k[t], S)
        v_new = beta[t][..., None] * (v[t] - k_state)
        S = S + k[t][..., None] * v_new[..., None, :]
        o[t] = np.einsum('hk,hkv->hv', q[t] * scale, S)
    return o, S


def _rand_inputs(rng, T, H, K, V):
    return (rng.standard_normal((T, H, K), dtype=np.float32),
            rng.standard_normal((T, H, K), dtype=np.float32),
            rng.standard_normal((T, H, V), dtype=np.float32),
            rng.standard_normal((T, H, K), dtype=np.float32),
            rng.standard_normal((T, H), dtype=np.float32))


def _params(rng, H, K):
    A_log = np.log(rng.uniform(1, 16, size=H)).astype(np.float32)
    dt_bias = rng.uniform(-0.5, 0.5, size=H * K).astype(np.float32)
    return A_log, dt_bias


def _run_prefill(seqs,
                 A_log,
                 dt_bias,
                 lower_bound,
                 num_slots=None,
                 init_states=None,
                 has_initial_state=None,
                 chunk_size=64):
    """Helper: run ragged_kda_mixed_prefill over a list of per-seq input
    tuples; returns (per-seq outputs [T,H,V], final states [H,K,V])."""
    H, K = seqs[0][0].shape[1], seqs[0][0].shape[2]
    V = seqs[0][2].shape[2]
    n = len(seqs)
    num_slots = num_slots or (n + 1)

    def cat(i):
        return jnp.asarray(np.concatenate([s[i] for s in seqs], axis=0))

    lens = [s[0].shape[0] for s in seqs]
    qsl = jnp.asarray(np.cumsum([0] + lens).astype(np.int32))
    state = jnp.zeros((num_slots, H, K, V), dtype=jnp.float32)
    if init_states is not None:
        state = state.at[1:n + 1].set(jnp.asarray(np.stack(init_states)))
    state_indices = jnp.arange(1, n + 1, dtype=jnp.int32)
    distribution = jnp.array([0, 0, n], dtype=jnp.int32)
    his = None if has_initial_state is None else jnp.asarray(has_initial_state)
    new_state, out = ragged_kda_mixed_prefill(cat(0),
                                              cat(1),
                                              cat(2),
                                              cat(4),
                                              cat(3),
                                              jnp.asarray(A_log),
                                              jnp.asarray(dt_bias),
                                              qsl,
                                              state,
                                              state_indices,
                                              distribution,
                                              has_initial_state=his,
                                              chunk_size=chunk_size,
                                              gate_lower_bound=lower_bound)
    out = np.asarray(out)
    outs, ofs = [], 0
    for L in lens:
        outs.append(out[ofs:ofs + L].reshape(L, H, V))
        ofs += L
    return outs, [np.asarray(new_state[i]) for i in range(1, n + 1)]


@pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None],
                         ids=["sigmoid", "softplus"])
def test_fp64_oracle_ragged_batch(lower_bound):
    """Gate 1: fp64-oracle parity on a random ragged batch (mixed lengths,
    incl. non-chunk-multiples)."""
    rng = np.random.default_rng(0)
    H, K, V = 4, 128, 128
    A_log, dt_bias = _params(rng, H, K)
    lens = [1, 37, 64, 100, 129]
    seqs = [_rand_inputs(rng, T, H, K, V) for T in lens]
    outs, states = _run_prefill(seqs, A_log, dt_bias, lower_bound)
    for (q, k, v, g, b), o_jax, s_jax in zip(seqs, outs, states):
        o_ref, s_ref = naive_kda_fp64(q, k, v, g, b, A_log, dt_bias,
                                      lower_bound)
        np.testing.assert_allclose(o_jax, o_ref, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(s_jax, s_ref, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("regime", ["floor", "one"])
@pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None],
                         ids=["sigmoid", "softplus"])
def test_adversarial_gate_regimes(lower_bound, regime):
    """Gate 5: decay pinned near the lower bound / near 1."""
    rng = np.random.default_rng(1)
    T, H, K, V = 96, 4, 64, 64
    q, k, v, g, b = _rand_inputs(rng, T, H, K, V)
    g = (6.0 + 2 * rng.random(
        (T, H, K)) if regime == "floor" else -8.0 - 2 * rng.random(
            (T, H, K))).astype(np.float32)
    A_log, dt_bias = _params(rng, H, K)
    outs, states = _run_prefill([(q, k, v, g, b)], A_log, dt_bias, lower_bound)
    o_ref, s_ref = naive_kda_fp64(q, k, v, g, b, A_log, dt_bias, lower_bound)
    np.testing.assert_allclose(outs[0], o_ref, atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(states[0], s_ref, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("split", [37, 64, 100])
def test_chunk_boundary_invariance(split):
    """Gate 2: one-shot(T) == prefill(split) + continuation(T-split) via
    has_initial_state, for arbitrary and chunk-aligned split points."""
    rng = np.random.default_rng(2)
    T, H, K, V = 137, 4, 64, 64
    q, k, v, g, b = _rand_inputs(rng, T, H, K, V)
    A_log, dt_bias = _params(rng, H, K)
    full_o, full_s = _run_prefill([(q, k, v, g, b)], A_log, dt_bias,
                                  LOWER_BOUND)
    part1 = (q[:split], k[:split], v[:split], g[:split], b[:split])
    o1, s1 = _run_prefill([part1], A_log, dt_bias, LOWER_BOUND)
    part2 = (q[split:], k[split:], v[split:], g[split:], b[split:])
    o2, s2 = _run_prefill([part2],
                          A_log,
                          dt_bias,
                          LOWER_BOUND,
                          init_states=s1,
                          has_initial_state=np.array([True]))
    o_cat = np.concatenate([o1[0], o2[0]], axis=0)
    np.testing.assert_allclose(o_cat, full_o[0], atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(s2[0], full_s[0], atol=2e-5, rtol=2e-4)


def test_prefill_equals_decode_steps():
    """Gate 3: prefill(T) == T decode-only steps threading the state."""
    rng = np.random.default_rng(3)
    T, H, K, V = 40, 4, 64, 64
    q, k, v, g, b = _rand_inputs(rng, T, H, K, V)
    A_log, dt_bias = _params(rng, H, K)
    pre_o, pre_s = _run_prefill([(q, k, v, g, b)], A_log, dt_bias, LOWER_BOUND)

    state = jnp.zeros((2, H, K, V), dtype=jnp.float32)
    state_indices = jnp.array([1], dtype=jnp.int32)
    distribution = jnp.array([1, 0, 1], dtype=jnp.int32)
    qsl = jnp.array([0, 1], dtype=jnp.int32)
    dec_o = []
    for t in range(T):
        state, out = ragged_kda_decode_only(jnp.asarray(q[t:t + 1]),
                                            jnp.asarray(k[t:t + 1]),
                                            jnp.asarray(v[t:t + 1]),
                                            jnp.asarray(b[t:t + 1]),
                                            jnp.asarray(g[t:t + 1]),
                                            state,
                                            jnp.asarray(A_log),
                                            jnp.asarray(dt_bias),
                                            qsl,
                                            state_indices,
                                            distribution,
                                            gate_lower_bound=LOWER_BOUND)
        dec_o.append(np.asarray(out).reshape(H, V))
    np.testing.assert_allclose(np.stack(dec_o), pre_o[0], atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(state[1]),
                               pre_s[0],
                               atol=2e-5,
                               rtol=2e-4)


def test_gdn_reduction_property():
    """Gate 4: with per-channel decay collapsed to a scalar per head
    (identical raw gate on every channel, softplus form, per-head dt_bias
    replicated), KDA must reproduce the GDN scalar-decay reference."""
    if jax.default_backend() == "cpu":
        pytest.skip("GDN reference needs the Pallas triangle solver (TPU)")
    from tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_chunked import \
        ragged_gated_delta_rule_mixed_prefill  # noqa: E501
    rng = np.random.default_rng(4)
    T, H, K, V = 100, 4, 128, 128
    q, k, v, _, b = _rand_inputs(rng, T, H, K, V)
    a_scalar = rng.standard_normal((T, H), dtype=np.float32)
    A_log = np.log(rng.uniform(1, 16, size=H)).astype(np.float32)
    dt_bias_h = rng.uniform(-0.5, 0.5, size=H).astype(np.float32)

    qsl = jnp.array([0, T], dtype=jnp.int32)
    state = jnp.zeros((2, H, K, V), dtype=jnp.float32)
    sidx = jnp.array([1], dtype=jnp.int32)
    dist = jnp.array([0, 0, 1], dtype=jnp.int32)

    gdn_state, gdn_out = ragged_gated_delta_rule_mixed_prefill(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(b),
        jnp.asarray(a_scalar),
        jnp.asarray(A_log),
        jnp.asarray(dt_bias_h),
        qsl,
        state,
        sidx,
        dist,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32)

    g_pc = np.repeat(a_scalar[..., None], K, axis=-1)
    dt_bias_pc = np.repeat(dt_bias_h, K)
    kda_outs, kda_states = _run_prefill([(q, k, v, g_pc, b)], A_log,
                                        dt_bias_pc, None)
    np.testing.assert_allclose(kda_outs[0].reshape(T, -1),
                               np.asarray(gdn_out),
                               atol=5e-5,
                               rtol=5e-4)
    np.testing.assert_allclose(kda_states[0],
                               np.asarray(gdn_state[1]),
                               atol=5e-5,
                               rtol=5e-4)


GOLDEN = "/mnt/disks/persist/models_cuiq/k3-tiny/kda_op_goldens.npz"


@pytest.mark.parametrize("case", [
    "base_sig", "base_sp", "long_sig", "long_sp", "decay_floor_sig",
    "decay_one_sig", "decay_floor_sp", "decay_one_sp", "small_sig",
    "initstate_sig"
])
def test_cross_impl_golden_parity(case):
    """Gate 6 (op-level): parity vs the fla-derived fp64 op goldens —
    an independent implementation of the same contract."""
    import os
    if not os.path.exists(GOLDEN):
        pytest.skip(f"golden file not present: {GOLDEN}")
    z = np.load(GOLDEN)
    lower_bound = float(z["lower_bound"]) if case.endswith("_sig") else None
    B = z[f"{case}.q"].shape[0]
    for i in range(B):

        def pick(n):
            return z[f"{case}.{n}"][i].astype(np.float32)

        h0 = (z[f"{case}.h0"][i].astype(np.float32)
              if f"{case}.h0" in z.files else None)
        seqs = [(pick("q"), pick("k"), pick("v"), pick("g_raw"),
                 pick("beta_raw"))]
        outs, states = _run_prefill(
            seqs,
            z[f"{case}.A_log"].astype(np.float32),
            z[f"{case}.dt_bias"].astype(np.float32),
            lower_bound,
            init_states=[h0] if h0 is not None else None,
            has_initial_state=np.array([True]) if h0 is not None else None)
        np.testing.assert_allclose(outs[0],
                                   z[f"{case}.o_fp64"][i],
                                   atol=3e-5,
                                   rtol=3e-4)
        np.testing.assert_allclose(states[0],
                                   z[f"{case}.state_fp64"][i],
                                   atol=3e-5,
                                   rtol=3e-4)
