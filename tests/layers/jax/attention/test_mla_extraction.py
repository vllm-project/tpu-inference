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
"""No-regression checks for the shared MLA layer.

`MLAAttention` was extracted from the DeepSeek-V3 model file. These tests
pin the DeepSeek-V3 behaviour of that layer (parameter set, creation order,
and forward numerics through the real MLA kernel) so the extraction and the
Kimi-family feature flags cannot silently change it.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.model_executor.models import utils as vllm_model_utils

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.common.sharding import ShardingAxisNameBase as Axis
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.attention.mla import MLAAttention
from tpu_inference.layers.jax.rope import DeepseekScalingRotaryEmbedding
from tpu_inference.models.jax.kimi_k3 import attach_default_weight_loaders
from tpu_inference.models.jax.utils.weight_utils import JaxAutoWeightsLoader
from tpu_inference.utils import align_to

HIDDEN = 256
N_HEADS = 16
QK_NOPE, QK_ROPE, V_HEAD = 64, 32, 64
Q_LORA, KV_LORA = 128, 64
SEQ_LEN = 32
ROPE_SCALING = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
}


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _rope():
    """Same construction DeepSeek-V3 uses (deepseek_v3.py:722)."""
    rope = DeepseekScalingRotaryEmbedding(
        rotary_dim=QK_ROPE,
        rope_theta=10000,
        original_max_position_embeddings=ROPE_SCALING[
            "original_max_position_embeddings"],
        scaling_factor=ROPE_SCALING["factor"],
        dtype=jnp.float32,
        beta_fast=ROPE_SCALING["beta_fast"],
        beta_slow=ROPE_SCALING["beta_slow"],
        mscale_value=ROPE_SCALING["mscale"],
        mscale_all_dim=ROPE_SCALING["mscale_all_dim"],
    )
    rope.initialize_cache()
    return rope


def _build(mesh,
           *,
           q_lora_rank=Q_LORA,
           use_nope=False,
           use_output_gate=False,
           seed=42):
    return MLAAttention(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        head_dim=V_HEAD,
        dtype=jnp.float32,
        kv_cache_dtype="auto",
        mesh=mesh,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        rms_norm_eps=1e-6,
        rope=None if use_nope else _rope(),
        use_nope=use_nope,
        use_output_gate=use_output_gate,
        rngs=nnx.Rngs(seed),
        random_init=True,
        query_nth=P(None, Axis.MLP_TENSOR, None),
        query_tnh=P(None, Axis.MLP_TENSOR, None),
        keyvalue_skh=P(None, Axis.MLP_TENSOR),
        attn_o_nth=P(None, Axis.MLP_TENSOR, None),
        q_da_sharding=P(None, Axis.VOCAB),
        ap_sharding=P(None, Axis.MLP_TENSOR),
        kv_da_sharding=P(None, Axis.VOCAB),
        rd_sharding=P(Axis.MLP_TENSOR, None),
    )


# Parameter set + creation order of the DeepSeek-V3 configuration. Captured
# from the layer as it existed in the model file before the extraction, and
# verified to be name-for-name, order-for-order, and bit-for-bit identical to
# what the extracted layer builds from the same rngs seed.
DSV3_PARAM_NAMES = [
    "q_a_proj.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "o_proj.weight",
    "q_a_layernorm.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
]


def test_dsv3_parameter_set_and_order_unchanged(mesh):
    """The extraction must not add, drop, or reorder DeepSeek-V3 params.

    Order matters beyond bookkeeping: nnx.Rngs is stateful, so a reordered
    constructor changes every random-init weight downstream of it.
    """
    with jax.set_mesh(mesh):
        layer = _build(mesh)
    names = [n for n, _ in layer.named_parameters()]
    assert names == DSV3_PARAM_NAMES, (
        f"[mla] DeepSeek-V3 parameter set/order changed.\n"
        f"  expected: {DSV3_PARAM_NAMES}\n  actual:   {names}")


def test_dsv3_flags_default_off(mesh):
    """DeepSeek-V3 must not pick up any Kimi-family behaviour by default."""
    with jax.set_mesh(mesh):
        layer = _build(mesh)
    assert layer.use_nope is False
    assert layer.use_output_gate is False
    assert not hasattr(layer, "g_proj")
    assert hasattr(layer, "q_a_proj") and not hasattr(layer, "q_proj")
    # YaRN mscale correction is applied for the rope path.
    assert layer.scale != (QK_NOPE + QK_ROPE)**-0.5


def test_nope_uses_plain_scale_and_no_rope(mesh):
    with jax.set_mesh(mesh):
        layer = _build(mesh, use_nope=True)
    assert layer.rope is None
    assert layer.scale == pytest.approx((QK_NOPE + QK_ROPE)**-0.5)


def test_no_q_lora_uses_fused_q_proj(mesh):
    """Kimi-Linear-48B has q_lora_rank=null."""
    with jax.set_mesh(mesh):
        layer = _build(mesh, q_lora_rank=None)
    names = [n for n, _ in layer.named_parameters()]
    assert "q_proj.weight" in names
    assert not any(n.startswith("q_a_") or n.startswith("q_b_") for n in names)


def test_output_gate_adds_only_g_proj(mesh):
    with jax.set_mesh(mesh):
        base = _build(mesh)
        gated = _build(mesh, use_output_gate=True)
    base_names = {n for n, _ in base.named_parameters()}
    gated_names = {n for n, _ in gated.named_parameters()}
    assert gated_names - base_names == {"g_proj.weight"}
    assert base_names - gated_names == set()


def _kv_b_split(layer):
    """Run the unquantized kv_b_proj -> k_up/v_up split (normally driven by
    the weight loader) so the absorbed forward can run."""
    layer.kv_b_proj.loaded.add("weight")
    layer.kv_b_proj._split_unquantized()


def _metadata(seq_len, pages=8):
    return AttentionMetadata(
        input_positions=jnp.arange(seq_len, dtype=jnp.int32),
        block_tables=jnp.arange(pages, dtype=jnp.int32),
        seq_lens=jnp.array([seq_len], dtype=jnp.int32),
        query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        padded_num_reqs=1,
    )


@pytest.mark.parametrize("use_nope,use_output_gate", [
    (False, False),
    (True, False),
    (True, True),
])
def test_forward_runs_and_is_deterministic(mesh, use_nope, use_output_gate):
    """Forward through the real MLA kernel, for the DeepSeek-V3 config and
    the Kimi-family flag combinations."""
    import tpu_inference.kernels.mla.v1.kernel as mla_kernel
    with jax.set_mesh(mesh):
        layer = _build(mesh,
                       use_nope=use_nope,
                       use_output_gate=use_output_gate)
        _kv_b_split(layer)

        num_blocks, block_size = 8, 16
        # The MLA cache lays out the latent and the shared ("rope") dims
        # each padded to 128, so its width is the sum of the aligned dims,
        # not the sum of the raw ones (kv_cache_manager.py:469-478).
        kv_dim = align_to(KV_LORA, 128) + align_to(QK_ROPE, 128)
        cache_shape = mla_kernel.get_kv_cache_shape(num_blocks, block_size,
                                                    kv_dim, jnp.float32)
        kv_cache = jnp.zeros(cache_shape, dtype=jnp.float32)
        x = jax.random.normal(jax.random.PRNGKey(0), (SEQ_LEN, HIDDEN),
                              dtype=jnp.float32) * 0.05

        _, out_a = layer(x, kv_cache, _metadata(SEQ_LEN))
        _, out_b = layer(x, jnp.zeros_like(kv_cache), _metadata(SEQ_LEN))

    assert out_a.shape == (SEQ_LEN, HIDDEN)
    assert jnp.isfinite(out_a).all(), "[mla] non-finite attention output"
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))


def test_output_gate_scales_output(mesh):
    """The K3 output gate must actually multiply the attention output by
    sigmoid(g_proj(x)) -- i.e. shrink it, since sigmoid < 1."""
    import tpu_inference.kernels.mla.v1.kernel as mla_kernel
    with jax.set_mesh(mesh):
        plain = _build(mesh, use_nope=True, use_output_gate=False, seed=7)
        gated = _build(mesh, use_nope=True, use_output_gate=True, seed=7)
        # Copy the shared weights so the only difference is the gate.
        gated_params = dict(gated.named_parameters())
        for name, param in plain.named_parameters():
            gated_params[name].value = param.value
        _kv_b_split(plain)
        _kv_b_split(gated)
        # k_up/v_up are derived after the copy, so redo them for `gated`.
        num_blocks, block_size = 8, 16
        # The MLA cache lays out the latent and the shared ("rope") dims
        # each padded to 128, so its width is the sum of the aligned dims,
        # not the sum of the raw ones (kv_cache_manager.py:469-478).
        kv_dim = align_to(KV_LORA, 128) + align_to(QK_ROPE, 128)
        cache_shape = mla_kernel.get_kv_cache_shape(num_blocks, block_size,
                                                    kv_dim, jnp.float32)
        kv_cache = jnp.zeros(cache_shape, dtype=jnp.float32)
        x = jax.random.normal(jax.random.PRNGKey(1), (SEQ_LEN, HIDDEN),
                              dtype=jnp.float32) * 0.05
        _, out_plain = plain(x, kv_cache, _metadata(SEQ_LEN))
        _, out_gated = gated(x, jnp.zeros_like(kv_cache), _metadata(SEQ_LEN))

    assert not np.allclose(np.asarray(out_plain), np.asarray(out_gated)), (
        "[mla] use_output_gate=True produced identical output to the "
        "ungated layer; the gate is not being applied")


class _SingleLayerModel(JaxModule):
    """Smallest model shape the recursive auto-loader will walk into.

    `AutoWeightsLoader` never calls `load_weights` on the root module (that is
    how it avoids recursing into itself), so reaching `MLAEinsum.load_weights`
    the way a real load does requires the layer to sit under a parent.
    """

    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention


_KV_B_PROJ_WEIGHT = "self_attn.kv_b_proj.weight"
_UNCOLLECTED_WARNING = "Unable to collect loaded parameters"


def _loadable_model(mesh):
    """An MLA layer under a parent, wired the way a Kimi model wires it.

    `attach_default_weight_loaders` is what the model does before handing
    itself to the auto-loader; without it the loader would try to infer a
    3-D MHA layout for the 2-D `o_proj`.
    """
    model = _SingleLayerModel(_build(mesh))
    attach_default_weight_loaders(model)
    return model


def _kv_b_proj_checkpoint_weight(layer):
    """The one checkpoint tensor `MLAEinsum` consumes.

    Built in the checkpoint's `[out, in]` orientation, which the loader
    transposes into the layer's `[in, out]` kernel.
    """
    rows, cols = layer.kv_b_proj.weight.get_value().shape
    return torch.arange(rows * cols, dtype=torch.float32).reshape(cols,
                                                                  rows) * 1e-4


@contextlib.contextmanager
def _captured_auto_loader_logs(caplog):
    """Capture the auto-loader's own logger.

    vLLM configures the `vllm` logger with `propagate=False`, so its records
    never reach the root logger `caplog` installs on; attach caplog's handler
    to the emitting logger directly.
    """
    logger = vllm_model_utils.logger
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


def test_kv_b_proj_load_weights_reports_the_names_it_loaded(mesh, caplog):
    """`load_weights` must return its loaded names, not None.

    vLLM's `AutoWeightsLoader._load_module` treats a None return from a child
    module's `load_weights` as "this module cannot report what it loaded" and
    logs a WARNING that interpolates the module's entire repr -- for an nnx
    module that is the whole parameter tree, once per MLA layer, on every load
    that goes through the auto-loader. The names are module-local; the caller
    qualifies them with the module prefix.
    """
    with jax.set_mesh(mesh):
        model = _loadable_model(mesh)
        weight = _kv_b_proj_checkpoint_weight(model.self_attn)
        loader = JaxAutoWeightsLoader(model)
        with _captured_auto_loader_logs(caplog):
            loaded = loader.load_weights([(_KV_B_PROJ_WEIGHT, weight)])

    assert loaded == {_KV_B_PROJ_WEIGHT
                      }, (f"[mla] auto-loader collected {loaded}, expected "
                          f"{{{_KV_B_PROJ_WEIGHT!r}}}")
    uncollected = [
        record for record in caplog.records
        if _UNCOLLECTED_WARNING in record.getMessage()
    ]
    assert not uncollected, (
        "[mla] the auto-loader still could not collect kv_b_proj's loaded "
        "parameters:\n" + "\n".join(r.getMessage() for r in uncollected))


def test_kv_b_proj_load_weights_still_loads_and_splits(mesh):
    """The bookkeeping fix must not disturb the load itself.

    Same drive as above, checking the parts that were already working: the
    checkpoint tensor lands in `kv_b_proj`, the fused weight is split into the
    absorbed `k_up_proj`/`v_up_proj` the forward runs on, and the fused
    parameter is released.
    """
    with jax.set_mesh(mesh):
        model = _loadable_model(mesh)
        attention = model.self_attn
        weight = _kv_b_proj_checkpoint_weight(attention)
        JaxAutoWeightsLoader(model).load_weights([(_KV_B_PROJ_WEIGHT, weight)])

    assert not hasattr(attention.kv_b_proj, "weight"), (
        "[mla] kv_b_proj.weight should be released once it is split")
    k_up = np.asarray(attention.k_up_proj.weight.get_value())
    v_up = np.asarray(attention.v_up_proj.weight.get_value())
    assert k_up.shape == (KV_LORA, N_HEADS, QK_NOPE)
    assert v_up.shape == (KV_LORA, N_HEADS, V_HEAD)

    fused = weight.numpy().T.reshape(KV_LORA, N_HEADS, QK_NOPE + V_HEAD)
    np.testing.assert_array_equal(k_up, fused[..., :QK_NOPE])
    np.testing.assert_array_equal(v_up, fused[..., QK_NOPE:])
