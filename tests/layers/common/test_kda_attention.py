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
"""KDA sublayer validation against HF-reference goldens.

Set ``K3_TINY_CKPT`` / ``K3_TINY_SOFTPLUS_CKPT`` to K3-tiny checkpoint dirs
that contain ``goldens_run1.npz`` (produced by the golden harness from the
HF reference implementation in moonshotai/Kimi-K3); tests skip when unset.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.kda_attention import (
    KDAParams, KDAState, kda_a_log_from_checkpoint, kda_attention)

CKPTS = {
    "sigmoid_fullrank": os.environ.get("K3_TINY_CKPT", ""),
    "softplus_lowrank": os.environ.get("K3_TINY_SOFTPLUS_CKPT", ""),
}
KDA_LAYERS = [0, 1, 2, 4, 5, 6]  # 0-indexed KDA layers in K3-tiny
H, D, W = 8, 128, 4
EPS = 1e-5


def _load(ckpt, layer):
    from safetensors import safe_open
    t = {}
    with safe_open(os.path.join(ckpt, "model.safetensors"), "np") as f:
        pre = f"model.layers.{layer}.self_attn."
        for k in f.keys():
            if k.startswith(pre):
                t[k[len(pre):]] = f.get_tensor(k).astype(np.float32)
    full_rank = "g_proj.weight" in t
    params = KDAParams(
        q_proj=jnp.asarray(t["q_proj.weight"]),
        k_proj=jnp.asarray(t["k_proj.weight"]),
        v_proj=jnp.asarray(t["v_proj.weight"]),
        q_conv_weight=jnp.asarray(t["q_conv1d.weight"]),
        k_conv_weight=jnp.asarray(t["k_conv1d.weight"]),
        v_conv_weight=jnp.asarray(t["v_conv1d.weight"]),
        # The two released checkpoints store this per-head vector differently
        # (Kimi-K3 zero-pads it out to head_dim, Kimi-Linear-48B keeps it
        # 4-D) and the tiny fixtures mirror each, so read it through the same
        # helper the model's weight loader uses instead of assuming a layout.
        A_log=jnp.asarray(kda_a_log_from_checkpoint(t["A_log"], H)),
        dt_bias=jnp.asarray(t["dt_bias"]),
        f_a_proj=jnp.asarray(t["f_a_proj.weight"]),
        f_b_proj=jnp.asarray(t["f_b_proj.weight"]),
        b_proj=jnp.asarray(t["b_proj.weight"]),
        o_norm_weight=jnp.asarray(t["o_norm.weight"]),
        o_proj=jnp.asarray(t["o_proj.weight"]),
        g_proj=jnp.asarray(t["g_proj.weight"]) if full_rank else None,
        g_a_proj=None if full_rank else jnp.asarray(t["g_a_proj.weight"]),
        g_b_proj=None if full_rank else jnp.asarray(t["g_b_proj.weight"]),
    )
    lower_bound = -5.0 if full_rank else None  # matches the two variants
    return params, lower_bound


def _zero_state(slots=2):
    return KDAState(
        conv_q=jnp.zeros((slots, W - 1, H * D), jnp.float32),
        conv_k=jnp.zeros((slots, W - 1, H * D), jnp.float32),
        conv_v=jnp.zeros((slots, W - 1, H * D), jnp.float32),
        recurrent=jnp.zeros((slots, H, D, D), jnp.float32),
    )


def _run(x,
         params,
         lower_bound,
         state,
         n_seqs=1,
         lens=None,
         decode=False,
         has_init=False,
         mesh=None):
    lens = lens or [x.shape[0]]
    qsl = jnp.asarray(np.cumsum([0] + lens).astype(np.int32))
    dist = (jnp.array([n_seqs, n_seqs, n_seqs], jnp.int32)
            if decode else jnp.array([0, 0, n_seqs], jnp.int32))
    return kda_attention(jnp.asarray(x),
                         params,
                         state,
                         state_indices=jnp.arange(1,
                                                  n_seqs + 1,
                                                  dtype=jnp.int32),
                         query_start_loc=qsl,
                         distribution=dist,
                         has_initial_state=jnp.full((n_seqs, ), has_init),
                         num_heads=H,
                         head_dim=D,
                         kernel_size=W,
                         gate_lower_bound=lower_bound,
                         rms_norm_eps=EPS,
                         mesh=mesh)


def _maxerr(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


@pytest.mark.parametrize("variant", list(CKPTS))
@pytest.mark.parametrize("tag", ["p48", "p64"])
@pytest.mark.parametrize("layer", KDA_LAYERS)
def test_sublayer_prefill_golden_parity(variant, tag, layer):
    """Gate a (prefill): layer out matches HF reference self_attn out."""
    ckpt = CKPTS[variant]
    if not ckpt:
        pytest.skip("checkpoint env var unset")
    z = np.load(os.path.join(ckpt, "goldens_run1.npz"))
    params, lb = _load(ckpt, layer)
    xin = z[f"{tag}.layers.{layer}.self_attn.in"][0]  # [T, hidden]
    ref = z[f"{tag}.layers.{layer}.self_attn.out"][0]
    _, out = _run(xin, params, lb, _zero_state())
    err = _maxerr(out, ref)
    print(f"[observed] {variant} {tag} L{layer} prefill max_abs={err:.3e}")
    np.testing.assert_allclose(np.asarray(out), ref, atol=5e-4, rtol=5e-3)


@pytest.mark.parametrize("variant", list(CKPTS))
@pytest.mark.parametrize("layer", [0, 4])
def test_sublayer_decode_golden_parity(variant, layer):
    """Gate a (decode): prefill T=32 then 8 decode steps — per-step outs,
    conv caches, and recurrent states match the HF reference."""
    ckpt = CKPTS[variant]
    if not ckpt:
        pytest.skip("checkpoint env var unset")
    z = np.load(os.path.join(ckpt, "goldens_run1.npz"))
    params, lb = _load(ckpt, layer)

    xin = z[f"d32.pre.layers.{layer}.self_attn.in"][0]
    state = _zero_state()
    state, _ = _run(xin, params, lb, state)

    # Post-prefill state parity. HF/fla conv cache is [1, dim, W] holding the
    # last W raw conv inputs (newest at -1); ours is [slots, W-1, dim] holding
    # the last W-1 (the window needed for the next token).
    errs = {}
    errs["S"] = _maxerr(state.recurrent[1],
                        z[f"d32.layers.{layer}.recurrent_state"][0])
    for name, mine in (("q", state.conv_q), ("k", state.conv_k),
                       ("v", state.conv_v)):
        ref_cache = z[f"d32.layers.{layer}.conv_{name}"][0]  # [dim, W]
        errs[f"conv_{name}"] = _maxerr(mine[1], ref_cache[:, 1:].T)
    print(f"[observed] {variant} L{layer} post-prefill " +
          " ".join(f"{k}={v:.3e}" for k, v in errs.items()))
    assert errs["S"] < 5e-4
    for name in ("q", "k", "v"):
        assert errs[f"conv_{name}"] < 5e-5

    step_errs = []
    for s in range(8):
        xs = z[f"d32.step{s}.layers.{layer}.self_attn.in"][0]
        ref = z[f"d32.step{s}.layers.{layer}.self_attn.out"][0]
        state, out = _run(xs, params, lb, state, decode=True, has_init=True)
        step_errs.append(_maxerr(out, ref))
        np.testing.assert_allclose(np.asarray(out), ref, atol=5e-4, rtol=5e-3)
    err_final = _maxerr(state.recurrent[1],
                        z[f"d32.final.layers.{layer}.recurrent_state"][0])
    print(f"[observed] {variant} L{layer} decode step max_abs="
          f"{max(step_errs):.3e} final_S={err_final:.3e}")
    assert err_final < 5e-4


@pytest.mark.parametrize("variant", list(CKPTS))
def test_conv_cache_roundtrip(variant):
    """Gate b: prefill(T) + n decode steps == one-shot prefill(T+n),
    crossing a chunk boundary (T=62, n=6)."""
    ckpt = CKPTS[variant]
    if not ckpt:
        pytest.skip("checkpoint env var unset")
    params, lb = _load(ckpt, KDA_LAYERS[0])
    rng = np.random.default_rng(7)
    T, n = 62, 6
    x = (0.05 * rng.standard_normal((T + n, H * D))).astype(np.float32)

    state_full = _zero_state()
    state_full, out_full = _run(x, params, lb, state_full)

    state = _zero_state()
    state, out_pre = _run(x[:T], params, lb, state)
    outs = [np.asarray(out_pre)]
    for t in range(T, T + n):
        state, out_t = _run(x[t:t + 1],
                            params,
                            lb,
                            state,
                            decode=True,
                            has_init=True)
        outs.append(np.asarray(out_t))
    out_cat = np.concatenate(outs, axis=0)
    err_o = _maxerr(out_cat, out_full)
    err_s = _maxerr(state.recurrent[1], state_full.recurrent[1])
    err_c = max(_maxerr(state.conv_q[1], state_full.conv_q[1]),
                _maxerr(state.conv_k[1], state_full.conv_k[1]),
                _maxerr(state.conv_v[1], state_full.conv_v[1]))
    print(f"[observed] {variant} roundtrip out={err_o:.3e} S={err_s:.3e} "
          f"conv={err_c:.3e}")
    np.testing.assert_allclose(out_cat,
                               np.asarray(out_full),
                               atol=5e-5,
                               rtol=5e-4)
    assert err_s < 5e-5 and err_c < 5e-6


@pytest.mark.parametrize("variant", list(CKPTS))
def test_mixed_batch_equals_separate(variant):
    """Gate c: one mixed ragged call (two prefills) == two separate calls."""
    ckpt = CKPTS[variant]
    if not ckpt:
        pytest.skip("checkpoint env var unset")
    params, lb = _load(ckpt, KDA_LAYERS[1])
    rng = np.random.default_rng(8)
    la, lc = 50, 17
    xa = (0.05 * rng.standard_normal((la, H * D))).astype(np.float32)
    xc = (0.05 * rng.standard_normal((lc, H * D))).astype(np.float32)

    state_m = _zero_state(slots=3)
    state_m, out_m = _run(np.concatenate([xa, xc]),
                          params,
                          lb,
                          state_m,
                          n_seqs=2,
                          lens=[la, lc])

    sa, oa = _run(xa, params, lb, _zero_state())
    sc, oc = _run(xc, params, lb, _zero_state())
    err_o = max(_maxerr(out_m[:la], oa), _maxerr(out_m[la:], oc))
    err_s = max(_maxerr(state_m.recurrent[1], sa.recurrent[1]),
                _maxerr(state_m.recurrent[2], sc.recurrent[1]))
    print(f"[observed] {variant} mixed-vs-separate out={err_o:.3e} "
          f"S={err_s:.3e}")
    assert err_o < 5e-5 and err_s < 5e-5


def _head_mesh(head_ways):
    """A mesh whose only nontrivial axis is one of the ATTN_HEAD axes."""
    import jax
    from jax.sharding import Mesh

    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
    if len(jax.devices()) < head_ways:
        pytest.skip(f"needs >= {head_ways} devices")
    shape = tuple(head_ways if n == "model" else 1 for n in MESH_AXIS_NAMES)
    return Mesh(np.array(jax.devices()[:head_ways]).reshape(shape),
                axis_names=MESH_AXIS_NAMES)


@pytest.mark.parametrize("variant", list(CKPTS))
@pytest.mark.parametrize("decode", [False, True])
def test_head_sharding_matches_the_unsharded_layer(variant, decode):
    """Splitting the head axis must not change the answer.

    KDA is head-parallel -- convs are per channel, the gate is per
    (head, channel), and the recurrent scan keeps one independent state per
    head -- so sharding heads should be exactly equivalent, not merely close.
    Both legs use the same weights and the same inputs; the only difference is
    that one runs the state-carrying core inside a `shard_map` that splits the
    head axis. Outputs *and* the written-back caches are compared, because a
    head-axis mix-up shows up in the state before it shows up in the output.
    """
    import jax
    ckpt = CKPTS[variant]
    if not ckpt:
        pytest.skip("checkpoint env var unset")
    params, lb = _load(ckpt, KDA_LAYERS[0])

    rng = np.random.default_rng(0)
    T = 8 if decode else 32
    n_seqs = T if decode else 1
    x = rng.standard_normal((T, params.q_proj.shape[1])).astype(np.float32)
    lens = [1] * n_seqs if decode else [T]

    ref_state, ref_out = _run(x,
                              params,
                              lb,
                              _zero_state(slots=n_seqs + 1),
                              n_seqs=n_seqs,
                              lens=lens,
                              decode=decode)
    mesh = _head_mesh(4)
    assert H % 4 == 0, "test needs the head count to divide the head ways"
    with jax.set_mesh(mesh):
        got_state, got_out = _run(x,
                                  params,
                                  lb,
                                  _zero_state(slots=n_seqs + 1),
                                  n_seqs=n_seqs,
                                  lens=lens,
                                  decode=decode,
                                  mesh=mesh)

    assert float(np.abs(np.asarray(ref_out)).max()) > 0, (
        "reference output is all zeros; the comparison would be vacuous")
    np.testing.assert_allclose(np.asarray(got_out),
                               np.asarray(ref_out),
                               rtol=2e-5,
                               atol=2e-5)
    for name in ("recurrent", "conv_q", "conv_k", "conv_v"):
        np.testing.assert_allclose(np.asarray(getattr(got_state, name)),
                                   np.asarray(getattr(ref_state, name)),
                                   rtol=2e-5,
                                   atol=2e-5,
                                   err_msg=f"{name} differs when heads split")


def test_head_sharding_refuses_to_tear_a_head():
    """`num_heads * head_dim` can divide when `num_heads` does not, so the
    head count is checked explicitly. `shard_map` would also reject this mesh
    (via `A_log`, which is shaped `[num_heads]`), but only at the first
    forward and with a message listing several arguments; this fails early
    and names the head count."""
    ckpt = CKPTS["sigmoid_fullrank"]
    if not ckpt:
        pytest.skip("checkpoint env var unset")
    params, lb = _load(ckpt, KDA_LAYERS[0])
    assert (H * D) % 16 == 0 and H % 16 != 0, "premise of this test"
    with pytest.raises(ValueError, match="do not divide"):
        _run(np.zeros((4, params.q_proj.shape[1]), np.float32),
             params,
             lb,
             _zero_state(),
             mesh=_head_mesh(16))
