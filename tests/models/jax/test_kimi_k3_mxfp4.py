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
"""Kimi routed experts stored as compressed-tensors ``mxfp4-pack-quantized``.

The fixture pair these tests need is a bf16 checkpoint and an MXFP4 twin whose
expert weights were drawn *on* the fp4 grid, so ``dequantize(mxfp4) == bf16``
holds exactly and any difference between the two models is a loader bug rather
than quantization error. Point ``K3_TINY_CKPT`` and ``K3_TINY_MXFP4_CKPT`` at
them, otherwise these tests skip.

What is gated here:
  * the checkpoint's fp4 codes and E8M0 scales survive loading unchanged (the
    model is quantization-aware-trained, so re-deriving scales would move the
    weights off the values training converged on);
  * expert weights stay 4-bit on device;
  * a layer the config's target list claims is quantized but the checkpoint
    stores as plain bf16 still loads (the config ignore list is not a
    trustworthy statement of what is compressed -- the checkpoint is).
"""

import json
import os
import types
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from safetensors import safe_open
from vllm.model_executor.layers.quantization.compressed_tensors.utils import \
    should_ignore_layer

from tpu_inference.layers.common.quantization import (e8m0_to_fp32,
                                                      u8_unpack_e2m1)
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig
from tpu_inference.layers.jax.quantization.mxfp4 import \
    CompressedTensorsMxfp4MoEMethod
from tpu_inference.layers.jax.quantization.unquantized import UnquantizedConfig
from tpu_inference.models.jax.kimi_k3 import KimiLinearForCausalLM

BF16_CKPT = os.environ.get("K3_TINY_CKPT", "")
MXFP4_CKPT = os.environ.get("K3_TINY_MXFP4_CKPT", "")

# A quantized expert projection, and the layer it lives in. Layer 1 is the
# first MoE layer (layer 0 is the dense MLP).
MOE_LAYER = 1

# Module paths the config's target/ignore lists claim are quantized while the
# checkpoint stores them as plain bf16 -- the Kimi-K3 hazard this loader has to
# resolve from the checkpoint rather than the config.
_AMBIGUOUS_PREFIXES = (
    f"model.layers.{MOE_LAYER}.block_sparse_moe.routed_expert_down_proj",
    f"model.layers.{MOE_LAYER}.block_sparse_moe.routed_expert_up_proj",
    f"model.layers.{MOE_LAYER}.mlp_res_proj",
)


def _skip_if_missing():
    for ckpt, var in ((BF16_CKPT, "K3_TINY_CKPT"), (MXFP4_CKPT,
                                                    "K3_TINY_MXFP4_CKPT")):
        if not ckpt or not os.path.isdir(ckpt):
            pytest.skip(f"K3-tiny checkpoint not available (set {var})")


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _hf_config(ckpt):
    with open(os.path.join(ckpt, "config.json")) as f:
        return types.SimpleNamespace(**json.load(f))


def _vllm_config_double(ckpt, *, quantized):
    """The slice of VllmConfig the model reads, without an engine."""
    hf = _hf_config(ckpt)
    if quantized:
        quant_config = CompressedTensorsConfig(hf.quantization_config,
                                               model_name_or_path=ckpt)
    else:
        quant_config = UnquantizedConfig({})
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=hf,
            dtype=jnp.float32,
            get_vocab_size=lambda: hf.vocab_size),
        quant_config=quant_config,
        cache_config=types.SimpleNamespace(cache_dtype="auto"))


def _checkpoint_weights(ckpt):
    f = safe_open(os.path.join(ckpt, "model.safetensors"), "pt")
    for name in f.keys():
        yield name, f.get_tensor(name)


def _build_loaded_model(ckpt, mesh, *, quantized):
    with jax.set_mesh(mesh):
        model = KimiLinearForCausalLM(
            _vllm_config_double(ckpt, quantized=quantized),
            jax.random.PRNGKey(0), mesh)
        model.load_weights(_checkpoint_weights(ckpt))
    return model


@pytest.fixture(scope="module")
def mxfp4_model(mesh):
    _skip_if_missing()
    return _build_loaded_model(MXFP4_CKPT, mesh, quantized=True)


def _moe_block(model, layer_idx=MOE_LAYER):
    return model.model.layers[layer_idx].block_sparse_moe


def _checkpoint_tensor(ckpt, name):
    with safe_open(os.path.join(ckpt, "model.safetensors"),
                   framework="numpy") as f:
        return jnp.asarray(f.get_tensor(name))


def _expert_prefix(layer_idx=MOE_LAYER):
    return f"model.layers.{layer_idx}.block_sparse_moe.experts"


def test_expert_weights_stay_four_bit_on_device(mxfp4_model):
    """4-bit residency: the decode must not expand the experts to bf16."""
    experts = _moe_block(mxfp4_model).experts
    for name in ("kernel_gating_EDF", "kernel_up_proj_EDF",
                 "kernel_down_proj_EFD"):
        param = getattr(experts, name)
        assert param.value.dtype == jnp.float4_e2m1fn, (
            f"{name} was materialized as {param.value.dtype}, expected "
            "float4_e2m1fn")
        scale = getattr(experts, f"{name}_weight_scale")
        assert scale.value.dtype == jnp.float32


def test_expert_codes_are_the_checkpoint_nibbles(mxfp4_model):
    """Lossless values: every loaded fp4 code equals the checkpoint's."""
    experts = _moe_block(mxfp4_model).experts
    num_experts = experts.num_local_experts
    for suffix, param_name in (("w1", "kernel_gating_EDF"),
                               ("w3", "kernel_up_proj_EDF"),
                               ("w2", "kernel_down_proj_EFD")):
        loaded = getattr(experts, param_name).value
        for expert_id in (0, num_experts - 1):
            packed = _checkpoint_tensor(
                MXFP4_CKPT,
                f"{_expert_prefix()}.{expert_id}.{suffix}.weight_packed")
            # Checkpoint is [out, in] with `in` packed; the kernel is (E,in,out).
            expected = jnp.swapaxes(u8_unpack_e2m1(packed), 0, 1)
            got = loaded[expert_id]
            assert got.shape == expected.shape
            assert jnp.array_equal(got.astype(
                jnp.float32), expected.astype(
                    jnp.float32)), (f"expert {expert_id} {suffix} codes "
                                    "differ from the checkpoint")


def test_checkpoint_scales_are_passed_through_exactly(mxfp4_model):
    """Lossless scales: no requantization, no recomputed exponents.

    The loaded scale must re-encode to the checkpoint's E8M0 bytes exactly.
    """
    experts = _moe_block(mxfp4_model).experts
    num_experts = experts.num_local_experts
    for suffix, param_name in (("w1", "kernel_gating_EDF"),
                               ("w3", "kernel_up_proj_EDF"),
                               ("w2", "kernel_down_proj_EFD")):
        loaded = getattr(experts, f"{param_name}_weight_scale").value
        for expert_id in (0, num_experts - 1):
            raw_u8 = _checkpoint_tensor(
                MXFP4_CKPT,
                f"{_expert_prefix()}.{expert_id}.{suffix}.weight_scale")
            expected = jnp.swapaxes(e8m0_to_fp32(raw_u8), 0, 1)
            got = loaded[expert_id]
            assert got.shape == expected.shape
            assert jnp.array_equal(got, expected), (
                f"expert {expert_id} {suffix} scales are not the checkpoint's")
            # Every scale is an exact power of two, i.e. nothing was averaged
            # or re-fit over a larger block.
            _, exponent = jnp.frexp(got)
            assert jnp.array_equal(
                jnp.ldexp(jnp.full_like(got, 0.5), exponent), got)


def test_scale_group_size_matches_the_checkpoint(mxfp4_model):
    """One scale per 32 input elements, as `group_size` in the config says."""
    experts = _moe_block(mxfp4_model).experts
    group_size = _hf_config(MXFP4_CKPT).quantization_config["config_groups"][
        "group_0"]["weights"]["group_size"]
    for param_name in ("kernel_gating_EDF", "kernel_up_proj_EDF",
                       "kernel_down_proj_EFD"):
        weight = getattr(experts, param_name).value
        scale = getattr(experts, f"{param_name}_weight_scale").value
        assert weight.shape[1] == scale.shape[1] * group_size
        assert weight.shape[2] == scale.shape[2]


def test_uncompressed_targets_are_read_off_the_checkpoint_not_the_config():
    """Trust the checkpoint's tensor names, not the config's ignore list.

    K3 targets every ``Linear`` and its ignore list does not mention the MoE
    latent projections or the attention-residual projections, yet the
    checkpoint stores those as plain bf16. A config-only decision calls them
    quantized and the load then fails looking for a `weight_packed` that isn't
    there.
    """
    _skip_if_missing()
    hf = _hf_config(MXFP4_CKPT)
    config = CompressedTensorsConfig(hf.quantization_config,
                                     model_name_or_path=MXFP4_CKPT)
    assert config._packed_modules is not None, (
        "checkpoint scan failed; the rest of this test would pass vacuously")

    # Every Linear is a target, so the config alone offers no way out.
    assert "Linear" in config._target_scheme_map

    for prefix in _AMBIGUOUS_PREFIXES:
        # The hazard is real: nothing in the ignore list covers these...
        assert not should_ignore_layer(prefix,
                                       ignore=config._ct.ignore,
                                       fused_mapping=MappingProxyType({}))
        # ...but the checkpoint does not store them compressed.
        assert not config._is_packed_in_checkpoint(prefix)

    # The experts, which the checkpoint really does compress, are selected.
    assert config._is_packed_in_checkpoint(_expert_prefix())


def test_moe_layer_gets_the_mxfp4_method(mxfp4_model):
    assert isinstance(
        _moe_block(mxfp4_model).experts.quant_method,
        CompressedTensorsMxfp4MoEMethod)


def test_uncompressed_moe_falls_back_to_the_unquantized_method():
    """A checkpoint that targets the experts but stores them plain bf16."""
    _skip_if_missing()
    hf = _hf_config(MXFP4_CKPT)
    config = CompressedTensorsConfig(hf.quantization_config,
                                     model_name_or_path=BF16_CKPT)
    assert config._packed_modules == set()
    assert not config._is_packed_in_checkpoint(_expert_prefix())


def test_fused_backend_is_rejected(mesh):
    """The fused GMM backends drop the router bias and have no situ branch."""
    _skip_if_missing()
    from tpu_inference.layers.common.moe import MoEBackend

    layer = types.SimpleNamespace(moe_backend=MoEBackend.GMM_TP,
                                  prefix="experts")
    with pytest.raises(NotImplementedError, match="unfused MoE backends"):
        CompressedTensorsMxfp4MoEMethod(layer)


def _logits(model, mesh, ckpt):
    """Prefill logits for a fixed prompt, with hand-allocated state."""
    from tests.models.jax.test_kimi_k3 import (_forward, _fresh_caches,
                                               _kimi_config, _metadata)
    config = _kimi_config(ckpt)
    seq_len = 32
    rng = np.random.default_rng(0)
    input_ids = jnp.asarray(rng.integers(0, config.vocab_size, size=seq_len),
                            dtype=jnp.int32)
    caches = _fresh_caches(config)
    md = _metadata(seq_len,
                   jnp.arange(seq_len, dtype=jnp.int32),
                   seq_len,
                   decode=False)
    with jax.set_mesh(mesh):
        _, _, logits = _forward(model, caches, input_ids, md)
    return logits


def test_mxfp4_logits_equal_the_bf16_twin(mxfp4_model, mesh):
    """The fixture's fp4 codes dequantize exactly, so the models must agree."""
    bf16_model = _build_loaded_model(BF16_CKPT, mesh, quantized=False)
    ref = _logits(bf16_model, mesh, BF16_CKPT)
    got = _logits(mxfp4_model, mesh, MXFP4_CKPT)
    max_abs = float(jnp.max(jnp.abs(got - ref)))
    assert max_abs == 0.0, (
        f"MXFP4 logits differ from the bf16 twin by {max_abs} (both "
        "checkpoints hold the same values; any difference is a loader bug)")

    # Anti-vacuity: the comparison has to be able to fail. Bumping one expert
    # group's scale by a single E8M0 step must move the logits.
    experts = _moe_block(mxfp4_model).experts
    original = experts.kernel_gating_EDF_weight_scale.value
    experts.kernel_gating_EDF_weight_scale.value = original.at[0, 0,
                                                               0].multiply(2.0)
    try:
        perturbed = _logits(mxfp4_model, mesh, MXFP4_CKPT)
    finally:
        experts.kernel_gating_EDF_weight_scale.value = original
    assert float(jnp.max(jnp.abs(perturbed - ref))) > 0.0


def _routed_expert_output(experts, x, weights_TX, indices_TX, mesh):
    with jax.set_mesh(mesh):
        return np.asarray(
            experts.quant_method.apply_jax(experts,
                                           x,
                                           router_logits=(weights_TX,
                                                          indices_TX)))


def test_expert_parallel_gmm_matches_the_bf16_twin(mesh):
    """The sharded MEGABLX_GMM path is what the real model runs on.

    DENSE_MAT evaluates every expert on every device and only works while the
    experts are replicated; once they shard, K3 dispatches through the
    grouped-matmul kernel, which consumes the fp4 codes and their per-group
    scales directly rather than a dequantized copy.

    Driven at the MoE layer rather than the whole model: the two must differ
    only in quantization, and a whole-model comparison across meshes cannot
    give that (the fixture's hand-allocated caches make expert-parallel and
    replicated whole-model runs disagree by ~1.7 in logits for *unquantized*
    weights too, so it would not isolate the variable).
    """
    _skip_if_missing()
    from tests.models.jax.test_kimi_k3 import _expert_parallel_mesh
    ep_mesh = _expert_parallel_mesh(2)

    bf16_experts = _moe_block(
        _build_loaded_model(BF16_CKPT, ep_mesh, quantized=False)).experts
    mxfp4_experts = _moe_block(
        _build_loaded_model(MXFP4_CKPT, ep_mesh, quantized=True)).experts
    for experts in (bf16_experts, mxfp4_experts):
        assert experts.moe_backend.value == "megablox_gmm"

    rng = np.random.default_rng(0)
    num_tokens = 16
    x = jnp.asarray(rng.standard_normal(
        (num_tokens, bf16_experts.hidden_size)),
                    dtype=jnp.float32)
    top_k = bf16_experts.top_k
    indices_TX = jnp.asarray(rng.integers(0,
                                          bf16_experts.num_local_experts,
                                          size=(num_tokens, top_k)),
                             dtype=jnp.int32)
    weights_TX = jnp.full((num_tokens, top_k), 1.0 / top_k, dtype=jnp.float32)

    ref = _routed_expert_output(bf16_experts, x, weights_TX, indices_TX,
                                ep_mesh)
    got = _routed_expert_output(mxfp4_experts, x, weights_TX, indices_TX,
                                ep_mesh)

    assert not np.isnan(got).any()
    max_abs = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref)))
    assert max_abs < 5e-3 * max(scale, 1.0), (
        f"expert-parallel MXFP4 expert output differs from the bf16 twin by "
        f"{max_abs} (output scale {scale})")
