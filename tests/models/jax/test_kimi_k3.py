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
"""Kimi-Linear / Kimi-K3 model parity against HF-reference goldens.

The config/AttnRes tests run anywhere. The parity tests need a K3-tiny
checkpoint directory (``config.json`` + ``model.safetensors`` +
``goldens_run1.npz``, produced by the golden harness from the HF reference in
``moonshotai/Kimi-K3``) and a TPU for the MLA kernel; point
``K3_TINY_CKPT`` / ``K3_TINY_SOFTPLUS_CKPT`` at them, otherwise they skip.

The two checkpoints cover both configurations of every family-level flag that
K3-tiny can express: ``K3_TINY_CKPT`` is the Kimi-K3 form (sigmoid
lower-bound KDA gate, full-rank output gate) and ``K3_TINY_SOFTPLUS_CKPT`` is
the Kimi-Linear-48B form (softplus gate, low-rank ``g_a``/``g_b``).

Everything is compared at ``jax.default_matmul_precision("highest")``: the
goldens come from a torch fp32 CPU forward, while TPU einsums default to a
reduced-precision fp32 algorithm that costs ~3 orders of magnitude of
agreement in a 9-layer stack. These tests check the model's math, not the
matmul default.
"""

import dataclasses
import json
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.kda_attention import KDAState
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.models.jax.kimi_k3 import (KimiConfig,
                                              KimiLinearForCausalLM,
                                              apply_attn_res)
from tpu_inference.utils import align_to

CKPTS = {
    "sigmoid_fullrank": os.environ.get("K3_TINY_CKPT", ""),
    "softplus_lowrank": os.environ.get("K3_TINY_SOFTPLUS_CKPT", ""),
}

# Gates. The observed maxima are ~1.5e-4 on activations of magnitude ~5 and
# ~1e-4 on logits of magnitude ~3, i.e. ~3e-5 relative.
ACT_ATOL = 2e-3
LOGIT_ATOL = 2e-3

NUM_BLOCKS, BLOCK_SIZE = 8, 64


def _skip_if_missing(ckpt):
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip("K3-tiny checkpoint not available (set K3_TINY_CKPT / "
                    "K3_TINY_SOFTPLUS_CKPT)")


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _kimi_config(ckpt) -> KimiConfig:
    with open(os.path.join(ckpt, "config.json")) as f:
        return KimiConfig(types.SimpleNamespace(**json.load(f)))


def _vllm_config_double(ckpt):
    """The slice of VllmConfig the model reads, without an engine."""
    with open(os.path.join(ckpt, "config.json")) as f:
        hf = types.SimpleNamespace(**json.load(f))
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=hf,
            dtype=jnp.float32,
            get_vocab_size=lambda: hf.vocab_size),
        quant_config=UnquantizedConfig({}),
        cache_config=types.SimpleNamespace(cache_dtype="auto"))


def _checkpoint_weights(ckpt):
    from safetensors import safe_open
    f = safe_open(os.path.join(ckpt, "model.safetensors"), "pt")
    for name in f.keys():
        yield name, f.get_tensor(name)


def _build_loaded_model(ckpt, mesh):
    with jax.set_mesh(mesh):
        model = KimiLinearForCausalLM(_vllm_config_double(ckpt),
                                      jax.random.PRNGKey(0), mesh)
        loaded = model.load_weights(_checkpoint_weights(ckpt))
    return model, loaded


def _fresh_caches(config):
    """Manually allocated per-layer state (hybrid cache wiring is Phase 3)."""
    import tpu_inference.kernels.mla.v1.kernel as mla_kernel
    hp = config.kda_num_heads * config.kda_head_dim
    w = config.kda_conv_kernel_size
    # The MLA cache pads the latent and the shared ("rope") dims to 128 each,
    # so its width is the sum of the aligned dims, not of the raw ones.
    kv_dim = align_to(config.kv_lora_rank, 128) + align_to(
        config.qk_rope_head_dim, 128)
    caches = []
    for i in range(config.num_hidden_layers):
        if config.is_kda_layer(i):
            # Slot 0 is the null block; the tests use slot 1.
            # Allocated as a plain tuple, the way the KV-cache manager
            # hands a `MambaSpec` layer's slot to the model.
            caches.append(
                tuple(
                    KDAState(conv_q=jnp.zeros((4, w - 1, hp), jnp.float32),
                             conv_k=jnp.zeros((4, w - 1, hp), jnp.float32),
                             conv_v=jnp.zeros((4, w - 1, hp), jnp.float32),
                             recurrent=jnp.zeros(
                                 (4, config.kda_num_heads, config.kda_head_dim,
                                  config.kda_head_dim), jnp.float32))))
        else:
            caches.append(
                jnp.zeros(
                    mla_kernel.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE,
                                                  kv_dim, jnp.float32),
                    jnp.float32))
    return caches


def _metadata(num_tokens, positions, seq_len, *, decode):
    return AttentionMetadata(
        input_positions=jnp.asarray(positions, jnp.int32),
        block_tables=jnp.arange(NUM_BLOCKS, dtype=jnp.int32),
        seq_lens=jnp.array([seq_len], jnp.int32),
        query_start_loc=jnp.array([0, num_tokens], jnp.int32),
        # (decode_end, prefill_end, mixed_end); decode-only is
        # distribution[0] == distribution[2].
        request_distribution=jnp.array([1, 1, 1] if decode else [0, 0, 1],
                                       jnp.int32),
        mamba_state_indices=jnp.array([1], jnp.int32),
        # `seq_len > num_tokens` means the request already has state in its
        # slot -- what the runner publishes as `has_initial_state`.
        has_initial_state=jnp.array([int(seq_len > num_tokens)], jnp.int32),
        padded_num_reqs=1)


@nnx.jit
def _forward(model, caches, input_ids, md):
    caches, hidden, _, _ = model(caches, input_ids, md)
    return caches, hidden, model.compute_logits(hidden)


# --------------------------------------------------------------------------
# Config / structure (no checkpoint required)
# --------------------------------------------------------------------------


def _config_from(**overrides):
    base = dict(hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=4,
                vocab_size=32,
                rms_norm_eps=1e-5,
                hidden_act="silu",
                num_attention_heads=2,
                num_key_value_heads=2,
                kv_lora_rank=4,
                qk_nope_head_dim=4,
                qk_rope_head_dim=2,
                v_head_dim=4)
    base.update(overrides)
    return KimiConfig(types.SimpleNamespace(**base))


def test_config_defaults_to_kimi_linear_48b_behaviour():
    """Every K3-only feature is off when its config field is absent."""
    c = _config_from()
    assert not c.use_attn_residuals
    assert c.routed_expert_hidden_size is None
    assert c.moe_hidden_size == c.hidden_size
    assert c.kda_gate_lower_bound is None  # softplus decay form
    assert not c.kda_use_full_rank_gate  # low-rank g_a/g_b
    assert not c.mla_use_output_gate
    assert c.q_lora_rank is None


def test_kda_layers_are_one_indexed_in_the_config():
    c = _config_from(linear_attn_config={"kda_layers": [1, 2, 4]})
    assert [i for i in range(4) if c.is_kda_layer(i)] == [0, 1, 3]


def test_moe_and_dense_layer_split():
    c = _config_from(num_experts=8, first_k_dense_replace=1)
    assert not c.is_moe_layer(0)
    assert all(c.is_moe_layer(i) for i in range(1, 4))


def test_attn_res_checkpoint_layers():
    c = _config_from(attn_res_block_size=2, num_hidden_layers=6)
    assert [i for i in range(6) if c.is_attn_res_checkpoint(i)] == [0, 2, 4]


def _apply_attn_res_numpy(prefix_sum, blocks, proj_w, norm_w, eps):
    """fp64 transcription of modeling_kimi_linear.py::_apply_attn_res."""
    v = np.concatenate([
        np.asarray(blocks, np.float64).transpose(1, 0, 2),
        prefix_sum[:, None, :].astype(np.float64)
    ],
                       axis=1)
    k = v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps)
    score_weight = norm_w.astype(np.float64) * proj_w.reshape(-1).astype(
        np.float64)
    scores = (k * score_weight).sum(-1)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    return np.einsum('tb,tbd->td', probs, v)


def test_apply_attn_res_matches_reference():
    rng = np.random.default_rng(0)
    t, d, nb = 7, 16, 3
    prefix_sum = rng.standard_normal((t, d), dtype=np.float32)
    blocks = [rng.standard_normal((t, d), dtype=np.float32) for _ in range(nb)]
    proj_w = rng.standard_normal((d, 1), dtype=np.float32)
    norm_w = rng.standard_normal((d, ), dtype=np.float32)

    got = np.asarray(
        apply_attn_res(jnp.asarray(prefix_sum),
                       [jnp.asarray(b) for b in blocks], jnp.asarray(proj_w),
                       jnp.asarray(norm_w), 1e-5))
    ref = _apply_attn_res_numpy(prefix_sum, blocks, proj_w, norm_w, 1e-5)
    err = np.abs(got - ref).max()
    print(f"[observed] apply_attn_res max_abs={err:.3e}")
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=1e-4)


def test_apply_attn_res_is_a_convex_mix_of_its_candidates():
    """Anti-vacuity: the output must lie inside the candidates' hull, so a
    mix that silently ignored `block_residual` would be visible."""
    rng = np.random.default_rng(1)
    t, d = 5, 8
    prefix_sum = jnp.asarray(rng.standard_normal((t, d), dtype=np.float32))
    blocks = [
        jnp.asarray(rng.standard_normal((t, d), dtype=np.float32))
        for _ in range(2)
    ]
    proj_w = jnp.asarray(rng.standard_normal((d, 1), dtype=np.float32))
    norm_w = jnp.asarray(rng.standard_normal((d, ), dtype=np.float32))
    out = np.asarray(apply_attn_res(prefix_sum, blocks, proj_w, norm_w, 1e-5))
    stack = np.stack([np.asarray(b)
                      for b in blocks] + [np.asarray(prefix_sum)])
    assert (out <= stack.max(0) + 1e-5).all()
    assert (out >= stack.min(0) - 1e-5).all()
    # ...and it is not simply the prefix sum passed through.
    assert np.abs(out - np.asarray(prefix_sum)).max() > 1e-3


def test_moe_layer_split_respects_first_k_dense_replace():
    c = _config_from(num_experts=8, first_k_dense_replace=2)
    assert [i for i in range(4) if c.is_moe_layer(i)] == [2, 3]


# --------------------------------------------------------------------------
# Kimi-Linear-48B-shaped config: the paths K3-tiny cannot reach
# --------------------------------------------------------------------------

# Both tiny checkpoints have attention residuals, a latent MoE, situ, and
# routed_scaling_factor 1.0, so the golden tests above never exercise the
# plain residual stream, the full-width MoE, or the routed scaling. This is
# the Kimi-Linear-48B shape (attn_res / routed_expert_hidden_size / situ /
# q_lora_rank all absent) at tiny dimensions.
_48B_SHAPED = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    vocab_size=64,
    hidden_act="silu",
    num_attention_heads=2,
    num_key_value_heads=2,
    q_lora_rank=None,
    kv_lora_rank=16,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    mla_use_nope=True,
    num_experts=8,
    num_experts_per_token=2,
    moe_intermediate_size=16,
    num_shared_experts=1,
    first_k_dense_replace=1,
    routed_scaling_factor=2.446,
    linear_attn_config={
        # 1-indexed: layer 0 is MLA (+ dense MLP), layer 1 KDA.
        "kda_layers": [2],
        "num_heads": 2,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
    })


def _build_48b_shaped(mesh, **overrides):
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig
    from tpu_inference.models.jax.kimi_k3 import KimiLinearModel
    config = _config_from(**{**_48B_SHAPED, **overrides})
    with jax.set_mesh(mesh):
        model = KimiLinearModel(config=config,
                                rngs=nnx.Rngs(0),
                                mesh=mesh,
                                dtype=jnp.float32,
                                quant_config=UnquantizedConfig({}),
                                random_init=True)
    return config, model


def test_48b_shaped_model_omits_k3_only_parameters(mesh):
    """No attention-residual or latent-MoE weights when the flags are absent."""
    config, model = _build_48b_shaped(mesh)
    names = {n for n, _ in model.named_parameters()}
    assert not config.use_attn_residuals
    assert not [n for n in names if "_res_proj" in n or "_res_norm" in n]
    assert not [n for n in names if "routed_expert_" in n]
    # q_lora_rank=None selects the single fused q_proj.
    assert any(n.endswith("layers.0.self_attn.q_proj.weight") for n in names)
    assert not [n for n in names if "q_a_proj" in n or "q_b_proj" in n]
    # ...while the KDA layer keeps the low-rank output gate.
    assert any("layers.1.self_attn.g_a_proj" in n for n in names)
    assert not [n for n in names if "layers.1.self_attn.g_proj" in n]


def test_48b_shaped_moe_runs_experts_at_full_hidden_width(mesh):
    config, model = _build_48b_shaped(mesh)
    moe = model.layers[1].block_sparse_moe
    assert not moe.use_latent_moe
    assert moe.experts.hidden_size == config.hidden_size


@nnx.jit
def _moe_block_fwd(model, x):
    return model.layers[1].block_sparse_moe(x)


def test_routed_scaling_factor_scales_the_routed_contribution(mesh):
    """The reference applies it to the top-k weights inside the gate, so the
    routed path -- and only the routed path -- scales with it."""
    x = jax.random.normal(jax.random.PRNGKey(0),
                          (4, _48B_SHAPED["hidden_size"]),
                          dtype=jnp.float32)

    def routed_only(factor):
        _, model = _build_48b_shaped(mesh,
                                     routed_scaling_factor=factor,
                                     num_shared_experts=None)
        with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
            return np.asarray(_moe_block_fwd(model, x), np.float32)

    one = routed_only(1.0)
    scaled = routed_only(2.446)
    err = np.abs(scaled - one * 2.446).max()
    print(f"[observed] routed_scaling_factor linearity max_abs={err:.3e}")
    np.testing.assert_allclose(scaled, one * 2.446, rtol=2e-5, atol=2e-5)
    # Anti-vacuity: the scaling actually changed the output.
    assert np.abs(scaled - one).max() > 1e-3


@nnx.jit
def _plain_residual_probe(model, caches, x, md):
    """The layer's own output alongside an explicit x + attn + mlp."""
    layer = model.layers[1]  # KDA layer: no MLA kernel needed
    _, got, block_residual = layer(x,
                                   kv_cache=caches[1],
                                   attention_metadata=md,
                                   block_residual=[])
    _, attn_out = layer.self_attn(layer.input_layernorm(x), caches[1], md)
    h = x + attn_out
    expect = h + layer._mlp()(layer.post_attention_layernorm(h))
    return got, expect, block_residual


def test_48b_shaped_decoder_uses_the_plain_residual_stream(mesh):
    """Without attention residuals the layer must be x + attn + mlp."""
    config, model = _build_48b_shaped(mesh)
    md = _metadata(4, np.arange(4), 4, decode=False)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, config.hidden_size),
                          dtype=jnp.float32) * 0.1

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        got, expect, block_residual = _plain_residual_probe(
            model, _fresh_caches(config), x, md)

    assert block_residual == []
    err = np.abs(np.asarray(got, np.float32) -
                 np.asarray(expect, np.float32)).max()
    print(f"[observed] plain residual stream max_abs={err:.3e}")
    np.testing.assert_allclose(np.asarray(got), np.asarray(expect), atol=1e-5)


def test_kda_layer_rejects_missing_has_initial_state(mesh):
    """The state-resumption flag comes from the runner, not from a local
    derivation off `query_start_loc` -- which would be wrong under attention
    DP. If the field is absent the layer must say so rather than guess."""
    config, model = _build_48b_shaped(mesh)
    md = dataclasses.replace(_metadata(4, np.arange(4), 4, decode=False),
                             has_initial_state=None)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, config.hidden_size),
                          dtype=jnp.float32) * 0.1
    kda_layer = model.layers[1].self_attn
    caches = _fresh_caches(config)

    with jax.set_mesh(mesh):
        with pytest.raises(ValueError, match="has_initial_state is None"):
            kda_layer(x, caches[1], md)


# --------------------------------------------------------------------------
# Golden parity (needs a checkpoint + TPU)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(CKPTS))
def test_every_checkpoint_tensor_is_loaded(variant, mesh):
    """The loader must consume the whole checkpoint, not a subset."""
    from safetensors import safe_open
    ckpt = CKPTS[variant]
    _skip_if_missing(ckpt)
    model, loaded = _build_loaded_model(ckpt, mesh)

    with safe_open(os.path.join(ckpt, "model.safetensors"), "np") as f:
        ckpt_names = set(f.keys())
    model_names = {n for n, _ in model.named_parameters()}
    # Routed experts are fused into `kernel_*` params, and the absorbed-MLA
    # k_up/v_up are derived from kv_b_proj during load; neither has a 1:1
    # checkpoint name.
    unloaded = {
        n
        for n in model_names - set(loaded)
        if "_up_proj" not in n and ".experts." not in n
    }
    assert not unloaded, f"[kimi-k3] params never loaded: {sorted(unloaded)}"

    # ...and no checkpoint tensor is silently dropped. The two exempt groups
    # are consumed by a module-level `load_weights` that reports no name:
    # the routed experts (fused into the `kernel_*` params) and kv_b_proj
    # (split into k_up_proj / v_up_proj).
    unconsumed = {
        n
        for n in ckpt_names - set(loaded)
        if ".experts." not in n and not n.endswith("kv_b_proj.weight")
    }
    assert not unconsumed, (
        f"[kimi-k3] checkpoint tensors ignored by the loader: "
        f"{sorted(unconsumed)}")


@pytest.mark.parametrize("variant", sorted(CKPTS))
@pytest.mark.parametrize("tag,seq_len", [("p48", 48), ("p64", 64)])
def test_prefill_logits_match_goldens(variant, tag, seq_len, mesh):
    ckpt = CKPTS[variant]
    _skip_if_missing(ckpt)
    g = np.load(os.path.join(ckpt, "goldens_run1.npz"))
    model, _ = _build_loaded_model(ckpt, mesh)

    input_ids = jnp.asarray(g[f"{tag}.input_ids"][0], jnp.int32)
    md = _metadata(seq_len, np.arange(seq_len), seq_len, decode=False)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        _, hidden, logits = _forward(model, _fresh_caches(model.config),
                                     input_ids, md)

    ref_hidden = g[f"{tag}.final_norm.out"][0]
    ref_logits = g[f"{tag}.logits"][0]
    got_hidden = np.asarray(hidden, np.float32)
    got_logits = np.asarray(logits, np.float32)
    print(f"[observed] {variant} {tag} hidden max_abs="
          f"{np.abs(got_hidden - ref_hidden).max():.3e} logits max_abs="
          f"{np.abs(got_logits - ref_logits).max():.3e} "
          f"(logit scale {np.abs(ref_logits).max():.3e})")
    np.testing.assert_allclose(got_hidden, ref_hidden, atol=ACT_ATOL)
    np.testing.assert_allclose(got_logits, ref_logits, atol=LOGIT_ATOL)
    np.testing.assert_array_equal(got_logits.argmax(-1), ref_logits.argmax(-1))


@pytest.mark.parametrize("variant", sorted(CKPTS))
def test_greedy_decode_reproduces_golden_tokens(variant, mesh):
    """Prefill 32 tokens, then 8 greedy steps through the recurrent state."""
    ckpt = CKPTS[variant]
    _skip_if_missing(ckpt)
    g = np.load(os.path.join(ckpt, "goldens_run1.npz"))
    model, _ = _build_loaded_model(ckpt, mesh)
    config = model.config

    input_ids = jnp.asarray(g["d32.input_ids"][0], jnp.int32)
    prompt_len = input_ids.shape[0]
    worst_logit_err = 0.0
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        caches, _, logits = _forward(
            model, _fresh_caches(config), input_ids,
            _metadata(prompt_len,
                      np.arange(prompt_len),
                      prompt_len,
                      decode=False))
        ref = g["d32.prefill_logits"][0]
        got = np.asarray(logits, np.float32)
        worst_logit_err = max(worst_logit_err, np.abs(got - ref).max())
        np.testing.assert_allclose(got, ref, atol=LOGIT_ATOL)

        # The recurrent state carried into decode must match the reference's.
        worst_state_err = 0.0
        for i in range(config.num_hidden_layers):
            if not config.is_kda_layer(i):
                continue
            ref_state = g[f"d32.layers.{i}.recurrent_state"][0]
            got_state = np.asarray(
                KDAState(*caches[i]).recurrent[1], np.float32)
            worst_state_err = max(worst_state_err,
                                  np.abs(got_state - ref_state).max())
            np.testing.assert_allclose(got_state, ref_state, atol=ACT_ATOL)

        tokens = [int(np.asarray(logits)[-1].argmax())]
        for step in range(8):
            position = prompt_len + step
            caches, _, logits = _forward(
                model, caches, jnp.asarray([tokens[-1]], jnp.int32),
                _metadata(1, [position], position + 1, decode=True))
            got = np.asarray(logits, np.float32)
            ref = g[f"d32.step{step}.logits"][0]
            worst_logit_err = max(worst_logit_err, np.abs(got - ref).max())
            tokens.append(int(got[-1].argmax()))

    print(f"[observed] {variant} d32 worst logit max_abs={worst_logit_err:.3e}"
          f" worst recurrent-state max_abs={worst_state_err:.3e}")
    np.testing.assert_array_equal(np.array(tokens), g["d32.greedy_tokens"][0])


@pytest.mark.parametrize("variant", sorted(CKPTS))
def test_per_layer_parity(variant, mesh):
    """Per-site parity through the whole stack.

    Checks every mix point the AttnRes threading touches, so a residual-stream
    bug is localized to a layer rather than only showing up in the logits.
    """
    ckpt = CKPTS[variant]
    _skip_if_missing(ckpt)
    g = np.load(os.path.join(ckpt, "goldens_run1.npz"))
    model, _ = _build_loaded_model(ckpt, mesh)
    config = model.config

    @nnx.jit
    def capture(model, caches, input_ids, md):
        out = {}
        m = model.model
        x = m.embed_tokens(input_ids)
        block_residual = []
        for i, layer in enumerate(m.layers):
            prefix_sum = x
            hidden = x
            if block_residual:
                hidden = layer._mix(layer.self_attention_res_proj,
                                    layer.self_attention_res_norm, prefix_sum,
                                    block_residual)
            out[f"{i}.input_layernorm.in"] = hidden
            if m.config.is_attn_res_checkpoint(i):
                block_residual = list(block_residual) + [prefix_sum]
                prefix_sum = None
            hidden = layer.input_layernorm(hidden)
            out[f"{i}.self_attn.in"] = hidden
            caches[i], attn_out = layer.self_attn(hidden, caches[i], md)
            out[f"{i}.self_attn.out"] = attn_out
            prefix_sum = (attn_out if prefix_sum is None else prefix_sum +
                          attn_out)
            hidden = layer._mix(layer.mlp_res_proj, layer.mlp_res_norm,
                                prefix_sum, block_residual)
            out[f"{i}.post_attention_layernorm.in"] = hidden
            hidden = layer.post_attention_layernorm(hidden)
            mlp_name = layer.mlp_attr
            out[f"{i}.{mlp_name}.in"] = hidden
            mlp_out = layer._mlp()(hidden)
            out[f"{i}.{mlp_name}.out"] = mlp_out
            prefix_sum = prefix_sum + mlp_out
            out[f"{i}.out"] = prefix_sum
            x = prefix_sum
        return out

    seq_len = 48
    input_ids = jnp.asarray(g["p48.input_ids"][0], jnp.int32)
    md = _metadata(seq_len, np.arange(seq_len), seq_len, decode=False)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        captured = capture(model, _fresh_caches(config), input_ids, md)

    worst, worst_key = 0.0, ""
    for key, value in captured.items():
        golden_key = f"p48.layers.{key}"
        assert golden_key in g, f"[kimi-k3] no golden for {golden_key}"
        err = np.abs(np.asarray(value, np.float32) - g[golden_key][0]).max()
        if err > worst:
            worst, worst_key = err, golden_key
        assert err < ACT_ATOL, f"[kimi-k3] {golden_key} off by {err:.3e}"
    print(f"[observed] {variant} per-layer worst max_abs={worst:.3e} "
          f"at {worst_key} ({len(captured)} sites)")
