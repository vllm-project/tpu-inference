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
"""One real Kimi-K3 MoE layer's experts, loaded from the real checkpoint.

The tiny fixture pins the loader's *logic*; this pins it against the bytes the
released checkpoint actually ships -- real expert count and widths, real E8M0
scale distribution, the `language_model.` prefix the multimodal wrapper adds.

Point ``K3_REAL_SHARD`` at a directory holding the shard that contains a whole
MoE layer (in the released 96-shard checkpoint, layer 1 lives entirely in
`model-00002-of-000096.safetensors`) plus that checkpoint's `config.json`,
otherwise these tests skip.

The reference is deliberately independent of the loader: nibbles are pulled
out of the uint8 words with numpy shifts and masks and mapped through the
E2M1 value table by hand, and the scales are read as raw exponents.
"""

import json
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from safetensors import safe_open

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.quantization.compressed_tensors import \
    CompressedTensorsConfig

REAL_SHARD_DIR = os.environ.get("K3_REAL_SHARD", "")
LAYER = int(os.environ.get("K3_REAL_SHARD_LAYER", "1"))
# Loading every expert of a real layer is ~15 GB of packed weights; the byte
# comparisons run over all of them, the forward-parity check over this many.
NUM_EXPERTS_FOR_FORWARD = int(os.environ.get("K3_REAL_FORWARD_EXPERTS", "8"))

# The eight magnitudes an e2m1 code can denote, indexed by its low 3 bits.
E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           dtype=np.float64)

PROJECTIONS = (("w1", "kernel_gating_EDF"), ("w3", "kernel_up_proj_EDF"),
               ("w2", "kernel_down_proj_EFD"))


def _skip_if_missing():
    if not REAL_SHARD_DIR or not os.path.isdir(REAL_SHARD_DIR):
        pytest.skip("real Kimi-K3 shard not available (set K3_REAL_SHARD)")


def _shard_path():
    shards = [
        f for f in sorted(os.listdir(REAL_SHARD_DIR))
        if f.endswith(".safetensors")
    ]
    if len(shards) != 1:
        pytest.skip(f"expected exactly one shard in {REAL_SHARD_DIR}, "
                    f"found {len(shards)}")
    return os.path.join(REAL_SHARD_DIR, shards[0])


def _text_config():
    with open(os.path.join(REAL_SHARD_DIR, "config.json")) as f:
        config = json.load(f)
    return config.get("text_config", config), config


def _reference_dequantize(packed: np.ndarray,
                          scale_u8: np.ndarray) -> np.ndarray:
    """Independent fp64 dequantization of one `[out, in]` expert projection."""
    low = packed & 0x0F
    high = packed >> 4
    # compressed-tensors packs the even-indexed value in the low nibble.
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    magnitude = E2M1_MAGNITUDES[codes & 0x07]
    sign = np.where(codes & 0x08, -1.0, 1.0)
    values = sign * magnitude
    # E8M0: the stored byte is the exponent biased by 127.
    scale = np.ldexp(1.0, scale_u8.astype(np.int32) - 127).astype(np.float64)
    group = values.shape[1] // scale.shape[1]
    return (values.reshape(values.shape[0], scale.shape[1], group) *
            scale[:, :, None]).reshape(values.shape)


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, axis_names=MESH_AXIS_NAMES)


@pytest.fixture(scope="module")
def expert_prefix():
    _skip_if_missing()
    with safe_open(_shard_path(), framework="numpy") as f:
        names = [
            n for n in f.keys() if f".layers.{LAYER}." in n
            and ".experts." in n and n.endswith(".weight_packed")
        ]
    if not names:
        pytest.skip(f"shard holds no packed experts for layer {LAYER}")
    # ".../experts.<id>.<proj>.weight_packed" -> ".../experts"
    return names[0].rsplit(".", 3)[0]


@pytest.fixture(scope="module")
def loaded_experts(mesh, expert_prefix):
    """A real-width KimiRoutedExperts holding the checkpoint's own bytes."""
    from flax import nnx
    from jax.sharding import PartitionSpec as P

    from tpu_inference.models.jax.kimi_k3 import KimiRoutedExperts

    # Spelled out rather than taken from `ShardingAxisName`: that resolves
    # lazily against whichever mesh layout the process configured first, and
    # this test builds its mesh directly instead of going through the model.
    expert_axis = ("attn_dp_expert", "expert")
    data_axis = "data"

    text_config, full_config = _text_config()
    quant_config = CompressedTensorsConfig(
        full_config.get("quantization_config")
        or text_config["quantization_config"],
        model_name_or_path=REAL_SHARD_DIR)

    with safe_open(_shard_path(), framework="pt") as f:
        names = sorted(n for n in f.keys()
                       if n.startswith(expert_prefix + "."))
        num_experts = 1 + max(
            int(n[len(expert_prefix) + 1:].split(".")[0]) for n in names)
        assert num_experts == text_config["num_experts"], (
            f"shard holds {num_experts} experts, config says "
            f"{text_config['num_experts']}")

        with jax.set_mesh(mesh):
            experts = KimiRoutedExperts(
                dtype=jnp.bfloat16,
                num_local_experts=num_experts,
                apply_expert_weight_before_computation=False,
                expert_axis_name=expert_axis,
                num_expert_parallelism=1,
                hidden_size=text_config["routed_expert_hidden_size"],
                intermediate_size_moe=text_config["moe_intermediate_size"],
                num_experts_per_tok=text_config["num_experts_per_token"],
                mesh=mesh,
                hidden_act=text_config["hidden_act"],
                rngs=nnx.Rngs(0),
                random_init=False,
                quant_config=quant_config,
                activation_ffw_td=P(data_axis, None),
                activation_ffw_ted=P(data_axis, None, None),
                edf_sharding=P(expert_axis, None, None),
                efd_sharding=P(expert_axis, None, None),
                moe_backend=MoEBackend.DENSE_MAT,
                qwix_quantized_weight_dtype=None,
                prefix=expert_prefix,
                # Only `num_experts_per_tok` is read during construction; the
                # tests below drive the expert kernels directly, not routing.
                router=SimpleNamespace(
                    num_experts_per_tok=text_config["num_experts_per_token"]),
                scoring_func="sigmoid",
                renormalize=True)
            experts.load_weights((n, f.get_tensor(n)) for n in names)
            assert experts.quant_method.process_weights_after_loading(experts)
    return experts


def _checkpoint(name):
    with safe_open(_shard_path(), framework="numpy") as f:
        return f.get_tensor(name)


def test_real_layer_scales_are_byte_identical(loaded_experts, expert_prefix):
    """QAT fidelity: the loaded scales re-encode to the checkpoint's bytes.

    Runs over every expert in the layer, so a scale recomputed anywhere -- for
    one projection, one expert, one group -- fails.
    """
    checked = 0
    for suffix, param_name in PROJECTIONS:
        loaded = np.asarray(
            getattr(loaded_experts, f"{param_name}_weight_scale").value)
        for expert_id in range(loaded_experts.num_local_experts):
            raw = _checkpoint(
                f"{expert_prefix}.{expert_id}.{suffix}.weight_scale")
            _, exponent = np.frexp(loaded[expert_id].T)
            reencoded = (exponent - 1 + 127).astype(np.uint8)
            assert np.array_equal(reencoded, raw), (
                f"expert {expert_id} {suffix}: loaded scales do not re-encode "
                "to the checkpoint's E8M0 bytes")
            checked += 1
    assert checked == 3 * loaded_experts.num_local_experts


def test_real_layer_codes_match_an_independent_decode(loaded_experts,
                                                      expert_prefix):
    """Lossless values, checked against a hand-written nibble decode."""
    for suffix, param_name in PROJECTIONS:
        loaded = getattr(loaded_experts, param_name).value
        for expert_id in (0, loaded_experts.num_local_experts - 1):
            packed = _checkpoint(
                f"{expert_prefix}.{expert_id}.{suffix}.weight_packed")
            scale = _checkpoint(
                f"{expert_prefix}.{expert_id}.{suffix}.weight_scale")
            expected = _reference_dequantize(packed, scale).T
            scales = np.asarray(
                getattr(loaded_experts,
                        f"{param_name}_weight_scale").value)[expert_id]
            group = expected.shape[0] // scales.shape[0]
            got = (np.asarray(loaded[expert_id], dtype=np.float64).reshape(
                scales.shape[0], group, -1) * scales[:, None, :]).reshape(
                    expected.shape)
            assert np.array_equal(got, expected), (
                f"expert {expert_id} {suffix}: dequantized weights differ "
                "from the independent reference")


def test_real_layer_forward_matches_a_cpu_dequant_reference(
        loaded_experts, expert_prefix, mesh):
    """The device path's expert matmul, against an fp64 CPU reference."""
    from tpu_inference.layers.common.process_weights.moe_weights import \
        UnfusedMoEWeights
    from tpu_inference.layers.jax.moe.dense_moe import \
        dequantize_unfused_moe_weights

    n = min(NUM_EXPERTS_FOR_FORWARD, loaded_experts.num_local_experts)
    latent = loaded_experts.hidden_size
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, latent)).astype(np.float32)

    weights = UnfusedMoEWeights(
        w1_weight=loaded_experts.kernel_gating_EDF.value[:n],
        w1_weight_scale=loaded_experts.kernel_gating_EDF_weight_scale.
        value[:n],
        w1_bias=None,
        w2_weight=loaded_experts.kernel_up_proj_EDF.value[:n],
        w2_weight_scale=loaded_experts.kernel_up_proj_EDF_weight_scale.
        value[:n],
        w2_bias=None,
        w3_weight=loaded_experts.kernel_down_proj_EFD.value[:n],
        w3_weight_scale=loaded_experts.kernel_down_proj_EFD_weight_scale.
        value[:n],
        w3_bias=None,
    )
    with jax.set_mesh(mesh):
        expanded = dequantize_unfused_moe_weights(weights, jnp.float32)
        gate = np.asarray(
            jnp.einsum("TD,EDF->TEF", jnp.asarray(x), expanded.w1_weight))

    reference = np.empty_like(gate, dtype=np.float64)
    for expert_id in range(n):
        packed = _checkpoint(f"{expert_prefix}.{expert_id}.w1.weight_packed")
        scale = _checkpoint(f"{expert_prefix}.{expert_id}.w1.weight_scale")
        w = _reference_dequantize(packed, scale)  # [F, D]
        reference[:, expert_id, :] = x.astype(np.float64) @ w.T

    max_abs = float(np.max(np.abs(gate.astype(np.float64) - reference)))
    scale_of = float(np.max(np.abs(reference)))
    assert max_abs < 1e-3 * max(scale_of, 1.0), (
        f"real-layer gate projection differs from the fp64 reference by "
        f"{max_abs} (output scale {scale_of})")
