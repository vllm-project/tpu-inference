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
"""Every tensor of the RELEASED Kimi-K3 checkpoint, against the parameters the
model actually builds at the real config.

Why this exists: the K3-tiny fixtures were written to the shapes our model
expects, so they agree with it by construction and cannot catch a checkpoint
that disagrees. This test takes the other side -- the shapes below are
transcribed from the released artifacts, not from this repo:

  * ``REAL_K3_TEXT_CONFIG``: ``text_config`` of ``moonshotai/Kimi-K3``'s
    ``config.json``.
  * ``RELEASED_TENSORS``: ``{name pattern: (shape, tensor count)}`` reduced
    from ``HfApi.get_safetensors_metadata("moonshotai/Kimi-K3")`` (497,052 text
    tensors over 96 shards, after the wrapper prefix is stripped and the 168
    vision/projector tensors are dropped) by replacing every ``.<digits>.``
    with ``.N.``. The counts are part of the assertion: they pin the layer
    classification (69 KDA / 24 MLA / 92 MoE / 1 dense) and the expert count,
    not just the widths.

The model is built with ``nnx.eval_shape``, so the full 93-layer / 896-expert /
163,840-vocab parameter tree costs no memory and needs no accelerator.
"""

import re
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh

from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax import JaxModule
from tpu_inference.models.jax.kimi_k3 import (KimiDeltaAttention,
                                              KimiLinearForCausalLM,
                                              attach_default_weight_loaders,
                                              load_kda_a_log)

# `full_attn_layers` is every 4th layer plus a second one at the end (1-indexed
# in the config); the remaining 69 are KDA.
_FULL_ATTN_LAYERS = list(range(4, 93, 4)) + [93]

REAL_K3_TEXT_CONFIG = {
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "num_hidden_layers": 93,
    "vocab_size": 163840,
    "rms_norm_eps": 1e-05,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "moe_intermediate_size": 3072,
    "num_shared_experts": 2,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "num_expert_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "use_grouped_topk": True,
    "routed_scaling_factor": 1.0,
    "routed_expert_hidden_size": 3584,
    "latent_moe_use_norm": True,
    "attn_res_block_size": 12,
    "tie_word_embeddings": False,
    "linear_attn_config": {
        "full_attn_layers": _FULL_ATTN_LAYERS,
        "kda_layers":
        [i for i in range(1, 94) if i not in set(_FULL_ATTN_LAYERS)],
        "head_dim": 128,
        "num_heads": 96,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
    },
}

# {checkpoint name pattern: (shape, number of tensors)}.
RELEASED_TENSORS = {
    "model.embed_tokens.weight": ([163840, 7168], 1),
    "lm_head.weight": ([163840, 7168], 1),
    "model.norm.weight": ([7168], 1),
    "model.output_attn_res_norm.weight": ([7168], 1),
    "model.output_attn_res_proj.weight": ([1, 7168], 1),
    "model.layers.N.input_layernorm.weight": ([7168], 93),
    "model.layers.N.post_attention_layernorm.weight": ([7168], 93),
    "model.layers.N.self_attention_res_norm.weight": ([7168], 93),
    "model.layers.N.self_attention_res_proj.weight": ([1, 7168], 93),
    "model.layers.N.mlp_res_norm.weight": ([7168], 93),
    "model.layers.N.mlp_res_proj.weight": ([1, 7168], 93),
    # Dense MLP, layer 0 only (`first_k_dense_replace=1`).
    "model.layers.N.mlp.gate_proj.weight": ([33792, 7168], 1),
    "model.layers.N.mlp.up_proj.weight": ([33792, 7168], 1),
    "model.layers.N.mlp.down_proj.weight": ([7168, 33792], 1),
    # KDA layers (69).
    "model.layers.N.self_attn.q_proj.weight": ([12288, 7168], 69),
    "model.layers.N.self_attn.k_proj.weight": ([12288, 7168], 69),
    "model.layers.N.self_attn.v_proj.weight": ([12288, 7168], 69),
    "model.layers.N.self_attn.q_conv1d.weight": ([12288, 1, 4], 69),
    "model.layers.N.self_attn.k_conv1d.weight": ([12288, 1, 4], 69),
    "model.layers.N.self_attn.v_conv1d.weight": ([12288, 1, 4], 69),
    # 96 heads, but stored in a head_dim-wide buffer -- see
    # `test_a_log_is_stored_one_scalar_per_head_zero_padded`.
    "model.layers.N.self_attn.A_log": ([128], 69),
    "model.layers.N.self_attn.dt_bias": ([12288], 69),
    "model.layers.N.self_attn.f_a_proj.weight": ([128, 7168], 69),
    "model.layers.N.self_attn.f_b_proj.weight": ([12288, 128], 69),
    "model.layers.N.self_attn.b_proj.weight": ([96, 7168], 69),
    "model.layers.N.self_attn.o_norm.weight": ([128], 69),
    # MLA layers (24).
    "model.layers.N.self_attn.q_a_proj.weight": ([1536, 7168], 24),
    "model.layers.N.self_attn.q_a_layernorm.weight": ([1536], 24),
    "model.layers.N.self_attn.q_b_proj.weight": ([18432, 1536], 24),
    "model.layers.N.self_attn.kv_a_proj_with_mqa.weight": ([576, 7168], 24),
    "model.layers.N.self_attn.kv_a_layernorm.weight": ([512], 24),
    "model.layers.N.self_attn.kv_b_proj.weight": ([24576, 512], 24),
    # Output gate and o_proj exist on both attention kinds (93).
    "model.layers.N.self_attn.g_proj.weight": ([12288, 7168], 93),
    "model.layers.N.self_attn.o_proj.weight": ([7168, 12288], 93),
    # MoE layers (92).
    "model.layers.N.block_sparse_moe.gate.weight": ([896, 7168], 92),
    "model.layers.N.block_sparse_moe.gate.e_score_correction_bias":
    ([896], 92),
    "model.layers.N.block_sparse_moe.routed_expert_down_proj.weight":
    ([3584, 7168], 92),
    "model.layers.N.block_sparse_moe.routed_expert_up_proj.weight":
    ([7168, 3584], 92),
    "model.layers.N.block_sparse_moe.routed_expert_norm.weight": ([3584], 92),
    "model.layers.N.block_sparse_moe.shared_experts.gate_proj.weight":
    ([6144, 7168], 92),
    "model.layers.N.block_sparse_moe.shared_experts.up_proj.weight":
    ([6144, 7168], 92),
    "model.layers.N.block_sparse_moe.shared_experts.down_proj.weight":
    ([7168, 6144], 92),
    # Routed experts, MXFP4: 2 fp4 codes per byte, one E8M0 scale per 32.
    # 82,432 = 92 MoE layers x 896 experts.
    "model.layers.N.block_sparse_moe.experts.N.w1.weight_packed":
    ([3072, 1792], 82432),
    "model.layers.N.block_sparse_moe.experts.N.w1.weight_scale": ([3072, 112],
                                                                  82432),
    "model.layers.N.block_sparse_moe.experts.N.w2.weight_packed":
    ([3584, 1536], 82432),
    "model.layers.N.block_sparse_moe.experts.N.w2.weight_scale": ([3584,
                                                                   96], 82432),
    "model.layers.N.block_sparse_moe.experts.N.w3.weight_packed":
    ([3072, 1792], 82432),
    "model.layers.N.block_sparse_moe.experts.N.w3.weight_scale": ([3072, 112],
                                                                  82432),
}

# The routed experts are stacked on a leading expert axis in this model, so one
# fused parameter answers for all 896 of a layer's checkpoint tensors.
_EXPERT_PARAM = {
    "w1": "kernel_gating_EDF",
    "w2": "kernel_down_proj_EFD",
    "w3": "kernel_up_proj_EDF",
}


@pytest.fixture(scope="module")
def real_config_params():
    """``{parameter name: (shape, dtype)}`` for the full real Kimi-K3 config.

    ``nnx.eval_shape`` builds the parameter tree abstractly, so this is the
    real 93-layer / 896-expert / 163,840-vocab model at no memory cost.
    """
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig
    hf = types.SimpleNamespace(**REAL_K3_TEXT_CONFIG)
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=hf,
            dtype=jnp.bfloat16,
            get_vocab_size=lambda: REAL_K3_TEXT_CONFIG["vocab_size"]),
        quant_config=UnquantizedConfig({}),
        cache_config=types.SimpleNamespace(cache_dtype="auto"))
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    mesh = Mesh(devices, axis_names=MESH_AXIS_NAMES)
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: KimiLinearForCausalLM(
            vllm_config, jax.random.PRNGKey(0), mesh))
    return {
        n: (tuple(p.get_value().shape), p.get_value().dtype)
        for n, p in model.named_parameters()
    }


@pytest.fixture(scope="module")
def real_config_param_shapes(real_config_params):
    """``{parameter name pattern: shape}``, reduced the same way the released
    tensor names are. Every parameter sharing a pattern must share a shape --
    otherwise one released shape could not answer for all of them."""
    by_pattern = {}
    for name, (shape, _dtype) in real_config_params.items():
        seen = by_pattern.setdefault(_pattern(name), shape)
        assert seen == shape, (
            f"{_pattern(name)} covers parameters of differing shapes: "
            f"{seen} and {shape}")
    return by_pattern


def _pattern(name):
    return re.sub(r"\.\d+\.", ".N.", name)


def _wanted_checkpoint_shape(pattern, param_shape):
    """The checkpoint shape this parameter can be loaded from.

    Inverts what ``kimi_k3.attach_default_weight_loaders`` declares: the
    embedding and the depthwise conv keep their checkpoint layout, ``A_log``
    is read out of a flat buffer, everything 2-D is stored transposed
    (``[out, in]``) and everything 1-D is stored as is.
    """
    if pattern.endswith(("embed_tokens.weight", "conv1d.weight")):
        return param_shape
    if pattern.endswith("self_attn.A_log"):
        # One scalar per head, stored in a head_dim-wide zero-padded buffer.
        # The released width alone says nothing about the head count, so hold
        # the parameter to the config's `num_heads` and only then accept the
        # wider buffer it is stored in.
        lac = REAL_K3_TEXT_CONFIG["linear_attn_config"]
        if param_shape != (lac["num_heads"], ):
            return param_shape
        return (lac["head_dim"], )
    if len(param_shape) == 2:
        return param_shape[::-1]
    return param_shape


def _resolve(pattern, param_shapes):
    """(model parameter pattern, checkpoint shape it accepts) for a released
    tensor pattern. The shape is ``None`` when the model has no such
    parameter, so the coverage test can report it instead of raising."""
    if ".experts.N.w" in pattern:
        head, tail = pattern.split(".experts.N.")
        w, kind = tail.split(".")
        name = f"{head}.experts.{_EXPERT_PARAM[w]}"
        if name not in param_shapes:
            return name, None
        # Drop the stacked expert axis, then un-transpose to [out, in].
        out_dim, in_dim = param_shapes[name][1:][::-1]
        if kind == "weight_packed":
            return name, (out_dim, in_dim // 2)
        return name, (out_dim, in_dim // 32)
    if pattern not in param_shapes:
        return pattern, None
    return pattern, _wanted_checkpoint_shape(pattern, param_shapes[pattern])


def _released_vs_model(param_shapes):
    """[(pattern, checkpoint shape, shape the model accepts)] for mismatches."""
    bad = []
    for pattern, (shape, _count) in RELEASED_TENSORS.items():
        _name, wanted = _resolve(pattern, param_shapes)
        if wanted is None:
            bad.append((pattern, tuple(shape), "no such parameter"))
        elif tuple(shape) != tuple(wanted):
            bad.append((pattern, tuple(shape), tuple(wanted)))
    return bad


def test_model_has_a_parameter_for_every_released_tensor(
        real_config_param_shapes):
    """No released tensor lands on a parameter that does not exist."""
    missing = [
        p for p in RELEASED_TENSORS
        if _resolve(p, real_config_param_shapes)[1] is None
    ]
    assert not missing, f"released tensors with no parameter: {missing}"


def test_every_released_tensor_matches_the_parameter_shape(
        real_config_param_shapes):
    """The whole-checkpoint load, checked without a multi-host pod.

    ``A_log`` is the tensor a full-checkpoint serve first failed on
    (checkpoint ``[128]`` vs a ``[96]`` parameter); this asserts all 48
    patterns at once so the next one does not have to be found the same way.
    """
    bad = _released_vs_model(real_config_param_shapes)
    assert not bad, "released checkpoint disagrees with the model:\n" + "\n".join(
        f"  {p}: checkpoint {list(c)} vs model accepts {m}" for p, c, m in bad)


def test_the_shape_check_is_not_vacuous(real_config_param_shapes):
    """Perturbing one parameter must be caught -- otherwise the test above
    proves nothing about the shapes it enumerates."""
    for pattern in (
            "model.layers.N.self_attn.A_log",
            "model.layers.N.self_attn.q_proj.weight",
            "model.layers.N.block_sparse_moe.experts.kernel_gating_EDF"):
        perturbed = dict(real_config_param_shapes)
        shape = list(perturbed[pattern])
        shape[-1] += 1
        perturbed[pattern] = tuple(shape)
        assert _released_vs_model(perturbed), (
            f"perturbing {pattern} went undetected")


def test_released_tensor_counts_match_the_layer_classification(
        real_config_params, real_config_param_shapes):
    """The per-pattern tensor counts pin the layer stack, not just the widths:
    69 KDA + 24 MLA layers, 92 MoE + 1 dense, 896 experts each."""
    counts = {}
    for name in real_config_params:
        counts[_pattern(name)] = counts.get(_pattern(name), 0) + 1
    experts_per_layer = REAL_K3_TEXT_CONFIG["num_experts"]
    for pattern, (_shape, released_count) in RELEASED_TENSORS.items():
        param_pattern, _ = _resolve(pattern, real_config_param_shapes)
        built = counts[param_pattern]
        expected = (released_count // experts_per_layer
                    if ".experts.N.w" in pattern else released_count)
        assert built == expected, (
            f"{pattern}: checkpoint has {released_count} tensors "
            f"(=> {expected} parameters) but the model builds {built}")


def test_every_parameter_is_filled_by_a_released_tensor(
        real_config_param_shapes):
    """The other direction: nothing the model builds is left uninitialized."""
    filled = {
        _resolve(p, real_config_param_shapes)[0]
        for p in RELEASED_TENSORS
    }
    unfilled = sorted(set(real_config_param_shapes) - filled)
    assert not unfilled, f"parameters no released tensor fills: {unfilled[:20]}"


# --------------------------------------------------------------------------
# A_log: one scalar per head, stored padded -- the tensor a full-checkpoint
# serve first failed to load.
# --------------------------------------------------------------------------


def _a_log_param(num_heads=4):
    return nnx.Param(jnp.zeros((num_heads, ), jnp.float32))


def test_a_log_is_stored_one_scalar_per_head_zero_padded():
    """Kimi-K3's layout: a head_dim-wide buffer, `num_heads` values then zeros.

    Measured over the released checkpoint: every one of the 69 KDA layers
    stores `[128]` with entries 0..95 non-zero and 96..127 exactly 0.0, at
    `num_heads=96` / `head_dim=128`.
    """
    values = torch.tensor([0.5, -1.25, 2.0, 0.125], dtype=torch.float32)
    padded = torch.zeros(128, dtype=torch.float32)
    padded[:4] = values
    param = _a_log_param()
    load_kda_a_log(param, padded, param_name="test.A_log")
    np.testing.assert_array_equal(np.asarray(param.get_value()),
                                  values.numpy())


def test_a_log_accepts_the_kimi_linear_48b_layout():
    """Kimi-Linear-48B stores the same vector as `[1, 1, num_heads, 1]`."""
    values = torch.tensor([0.5, -1.25, 2.0, 0.125], dtype=torch.float32)
    param = _a_log_param()
    load_kda_a_log(param, values.reshape(1, 1, 4, 1), param_name="test.A_log")
    np.testing.assert_array_equal(np.asarray(param.get_value()),
                                  values.numpy())


def test_a_log_refuses_to_truncate_a_non_zero_tail():
    """A non-zero tail is not padding, so dropping it would silently change
    every head's decay -- the failure mode this loader exists to prevent."""
    padded = torch.zeros(128, dtype=torch.float32)
    padded[:4] = torch.tensor([0.5, -1.25, 2.0, 0.125])
    padded[7] = 0.75
    with pytest.raises(ValueError, match="not zero padding"):
        load_kda_a_log(_a_log_param(), padded, param_name="test.A_log")


def test_a_log_rejects_a_checkpoint_with_too_few_heads():
    with pytest.raises(ValueError, match="cannot fill it"):
        load_kda_a_log(_a_log_param(num_heads=8),
                       torch.zeros(4, dtype=torch.float32),
                       param_name="test.A_log")


def test_the_attached_loader_reads_the_released_a_log_layout():
    """End-to-end through `attach_default_weight_loaders`, not by calling
    `load_kda_a_log` directly.

    The shape table above compares the released shapes against the parameters,
    but it does not run the loader; only this does. Drop the `A_log` entry from
    `_LOADER_FN_OVERRIDES` and the generic loader takes the tensor instead and
    raises `Shape mismatch in non-vocab layer ... torch (128,) vs jax (96,)` --
    the failure the first full-checkpoint serve hit.
    """

    class _DecoderLayer(JaxModule):
        """Minimal stand-in for the decoder layer, so the KDA parameters carry
        the `self_attn.` path the loader overrides are keyed on."""

        def __init__(self, kda):
            self.self_attn = kda

    num_heads, head_dim, hidden = 4, 8, 16
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    mesh = Mesh(devices, axis_names=MESH_AXIS_NAMES)
    values = torch.tensor([0.5, -1.25, 2.0, 0.125], dtype=torch.float32)
    padded = torch.zeros(head_dim, dtype=torch.float32)
    padded[:num_heads] = values
    with jax.set_mesh(mesh):
        layer = _DecoderLayer(
            KimiDeltaAttention(hidden_size=hidden,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               conv_kernel_size=4,
                               rms_norm_eps=1e-5,
                               dtype=jnp.bfloat16,
                               use_full_rank_gate=True,
                               mesh=mesh,
                               gate_lower_bound=-5.0,
                               rngs=nnx.Rngs(params=0)))
        attach_default_weight_loaders(layer)
        names = dict(layer.named_parameters())
        assert "self_attn.A_log" in names, sorted(names)
        param = names["self_attn.A_log"]
        param.weight_loader(param, padded)
    np.testing.assert_array_equal(np.asarray(param.get_value()),
                                  values.numpy())
