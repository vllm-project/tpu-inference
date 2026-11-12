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
"""Generator for the KDA op-level golden fixtures used by kda_reference_test.

Produces the ``.npz`` files consumed by the cross-implementation parity tests
in ``tests/kernels/kda_reference_test.py``:

  * ``--smoke``: the small (<1 MB) fixture checked in next to the test as
    ``tests/kernels/kda_smoke_goldens.npz``, always exercised in CI by
    ``test_cross_impl_smoke_golden_parity``.
  * default (full): the 10-case fixture pointed at by the ``K3_KDA_GOLDENS``
    env var (opt-in gate ``test_cross_impl_golden_parity``); too large to
    check in, so it is regenerated on demand with this script.

Oracle
------
The golden outputs are computed with the definitional per-token KDA recurrence
(arXiv:2510.26692) as published in flash-linear-attention ``fla-core==0.5.2``
(https://github.com/fla-org/flash-linear-attention, MIT license):

  * recurrence: ``fla/ops/kda/naive.py::naive_recurrent_kda``. If ``fla`` is
    importable it is used directly; otherwise this file's line-for-line torch
    transcription of that function (``_naive_recurrent_kda``) is used -- the
    two are numerically identical, so the fixture does not depend on which
    path ran.
  * input transforms (always computed in-file, matching fla-core 0.5.2):
    q/k L2 norm with ``rstd = 1/sqrt(sum(x^2) + 1e-6)`` in fp32
    (``fla/modules/l2norm.py``), ``beta = sigmoid(beta_raw)``, and the two
    gate forms from ``fla/ops/kda/gate.py`` -- also mirrored by the HF
    reference ``modeling_kimi_linear.py`` on the moonshotai/Kimi-K3 and
    moonshotai/Kimi-Linear-48B-A3B-Instruct model cards:
      lower_bound None: ``g = -exp(A_log) * softplus(g_raw + dt_bias)``
      lower_bound L:    ``g = L * sigmoid(exp(A_log) * (g_raw + dt_bias))``

Each case is evaluated in fp64 and fp32; the fp32-vs-fp64 delta printed per
case is what calibrated the tolerances used in the parity tests.

Regeneration (CPU-only, no accelerator needed)
----------------------------------------------
  pip install "torch>=2.0" numpy            # CPU wheels are sufficient
  pip install fla-core==0.5.2               # optional: use fla's recurrence
                                            # directly instead of the
                                            # transcription below
  python tests/kernels/kda_goldens_generator.py --smoke \
      --out tests/kernels/kda_smoke_goldens.npz
  python tests/kernels/kda_goldens_generator.py --out kda_op_goldens.npz

All randomness comes from a single ``torch.Generator`` seeded with
``SEED = 20260728`` and consumed in case order, so output is deterministic.
"""

import argparse

import numpy as np
import torch

SEED = 20260728
LOWER_BOUND = -5.0

# (name, B, T, H, K, V, gate_form, gate_regime). Cases with "initstate" in
# the name also draw a random initial state h0.
FULL_CASES = [
    ("base_sig", 2, 48, 8, 128, 128, "sigmoid", "normal"),
    ("base_sp", 2, 48, 8, 128, 128, "softplus", "normal"),
    ("long_sig", 1, 256, 4, 128, 128, "sigmoid", "normal"),
    ("long_sp", 1, 256, 4, 128, 128, "softplus", "normal"),
    ("decay_floor_sig", 1, 64, 4, 128, 128, "sigmoid", "floor"),
    ("decay_one_sig", 1, 64, 4, 128, 128, "sigmoid", "one"),
    ("decay_floor_sp", 1, 64, 4, 128, 128, "softplus", "floor"),
    ("decay_one_sp", 1, 64, 4, 128, 128, "softplus", "one"),
    ("small_sig", 1, 16, 2, 32, 32, "sigmoid", "normal"),
    ("initstate_sig", 2, 32, 4, 64, 64, "sigmoid", "normal"),
]

# Checked-in smoke subset: both gate forms plus the initial-state path, at
# shapes small enough to keep the fixture well under 1 MB.
SMOKE_CASES = [
    ("smoke_sig", 1, 24, 2, 32, 32, "sigmoid", "normal"),
    ("smoke_sp", 1, 24, 2, 32, 32, "softplus", "normal"),
    ("smoke_initstate_sig", 1, 16, 2, 32, 32, "sigmoid", "normal"),
]


def _naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    """Transcription of fla-core 0.5.2 fla/ops/kda/naive.py::

    ``naive_recurrent_kda`` (MIT license, fla-org/flash-linear-attention),
    restricted to H == HV (no grouped value heads; none of the cases here
    use them). Shapes: q/k [B,T,H,K], v [B,T,H,V], g [B,T,H,K] (log-space
    per-channel decay), beta [B,T,H]. Returns (o [B,T,H,V], S [B,H,K,V]).
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    q, k, v, g, beta = (x.to(torch.float) for x in (q, k, v, g, beta))
    q = q * scale
    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        # fla runs the recurrence in fp32 regardless of input dtype; the
        # original applies ``S += initial_state`` in-place, which downcasts.
        S = S + initial_state.to(S.dtype)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:,
                                                                           i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('bhk,bhv->bhkv', b_i[..., None] * k_i, v_i -
                             (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('bhk,bhkv->bhv', q_i, S)
    return o.to(dtype), S


def _get_recurrence():
    try:
        from fla.ops.kda.naive import naive_recurrent_kda  # noqa: F401

        def run(q, k, v, g, beta, initial_state):
            o, s = naive_recurrent_kda(q=q,
                                       k=k,
                                       v=v,
                                       g=g,
                                       beta=beta,
                                       initial_state=initial_state,
                                       output_final_state=True)
            return o, s

        return run, "fla-core"
    except ImportError:

        def run(q, k, v, g, beta, initial_state):
            return _naive_recurrent_kda(q,
                                        k,
                                        v,
                                        g,
                                        beta,
                                        initial_state=initial_state)

        return run, "in-file transcription"


def _l2norm(x, eps=1e-6):
    # fla/modules/l2norm.py: rstd = 1/sqrt(sum(x^2) + eps), fp32.
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(
        x.dtype)


def _kda_gate(g_raw, A_log, dt_bias, form):
    # fla/ops/kda/gate.py (naive_kda_lowerbound_gate / naive_kda_gate).
    H = g_raw.shape[-2]
    gf = g_raw.float() + dt_bias.float().view(H, -1)
    if form == "sigmoid":
        return LOWER_BOUND * torch.sigmoid(A_log.float().view(H, 1).exp() * gf)
    return -A_log.float().view(H, 1).exp() * torch.nn.functional.softplus(gf)


def run_case(recurrence, name, B, T, H, K, V, form, regime, gen, store):
    q = torch.randn(B, T, H, K, generator=gen, dtype=torch.float64)
    k = torch.randn(B, T, H, K, generator=gen, dtype=torch.float64)
    v = torch.randn(B, T, H, V, generator=gen, dtype=torch.float64)
    if regime == "floor":
        # Decay pinned toward the lower bound: large positive g_raw.
        g_raw = 6.0 + 2.0 * torch.rand(
            B, T, H, K, generator=gen, dtype=torch.float64)
    elif regime == "one":
        # Decay ~1 (no forgetting): g_raw strongly negative.
        g_raw = -8.0 - 2.0 * torch.rand(
            B, T, H, K, generator=gen, dtype=torch.float64)
    else:
        g_raw = torch.randn(B, T, H, K, generator=gen, dtype=torch.float64)
    beta_raw = torch.randn(B, T, H, generator=gen, dtype=torch.float64)
    A_log = torch.log(
        torch.empty(H, dtype=torch.float64).uniform_(1, 16, generator=gen))
    dt_bias = torch.empty(H * K, dtype=torch.float64).uniform_(-0.5,
                                                               0.5,
                                                               generator=gen)
    h0 = None
    if "initstate" in name:
        h0 = 0.5 * torch.randn(B, H, K, V, generator=gen, dtype=torch.float64)

    def compute(dtype):

        def cast(t):
            return None if t is None else t.to(dtype)

        qq, kk = _l2norm(cast(q)), _l2norm(cast(k))
        beta = torch.sigmoid(cast(beta_raw))
        g = _kda_gate(cast(g_raw), cast(A_log), cast(dt_bias), form)
        return recurrence(qq, kk, cast(v), g.to(dtype), beta, cast(h0))

    o64, s64 = compute(torch.float64)
    o32, s32 = compute(torch.float32)
    for key, t in [("q", q), ("k", k), ("v", v), ("g_raw", g_raw),
                   ("beta_raw", beta_raw), ("A_log", A_log),
                   ("dt_bias", dt_bias), ("h0", h0), ("o_fp64", o64),
                   ("state_fp64", s64), ("o_fp32", o32), ("state_fp32", s32)]:
        if t is not None:
            store[f"{name}.{key}"] = t.numpy()
    do = (o64 - o32.to(torch.float64)).abs()
    ds = (s64 - s32.to(torch.float64)).abs()
    rel = do / (o64.abs() + 1e-9)
    print(f"{name:20} fp32-vs-fp64: o max_abs={do.max():.3e} "
          f"max_rel={rel.max():.3e}  state max_abs={ds.max():.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--smoke",
                        action="store_true",
                        help="generate the small checked-in smoke fixture "
                        "instead of the full K3_KDA_GOLDENS fixture")
    args = parser.parse_args()

    recurrence, oracle = _get_recurrence()
    print(f"recurrence oracle: {oracle}")
    gen = torch.Generator().manual_seed(SEED)
    store = {"lower_bound": np.array(LOWER_BOUND)}
    for case in (SMOKE_CASES if args.smoke else FULL_CASES):
        run_case(recurrence, *case, gen, store)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out}: {len(store)} tensors")


if __name__ == "__main__":
    main()
