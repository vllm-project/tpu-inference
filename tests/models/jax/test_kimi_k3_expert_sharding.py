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
"""Routed-expert sharding: the expert axis and the tensor axis compose.

Kimi-K3's routed experts are ~1.4 TB and its non-expert stack ~113 GB. Split
only along the expert axis, a fixed device budget has to choose between them:
devices spent on ``model`` (which the non-expert stack needs) are devices not
spent on the expert axis, so the experts end up replicated. Splitting ``E``
over the expert axis *and* ``F`` over ``model`` removes the choice.

The tests below cover the three things that can go wrong independently: the
declared specs, the guard that refuses a mesh which would replicate the
experts anyway, and the numerics of the composed layout (the down projection
now contracts a sharded dimension, so its result is a partial sum that has to
be reduced).
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisNameBase)
from tpu_inference.models.jax.kimi_k3 import (experts_are_block_quantized,
                                              routed_expert_sharding_ways,
                                              routed_expert_shardings,
                                              validate_routed_expert_sharding)

E_AXES = ShardingAxisNameBase.ATTN_DATA_EXPERT
F_AXIS = ShardingAxisNameBase.MODEL


def _mesh(**axis_sizes):
    """A mesh with the named axes set and every other axis at 1."""
    need = int(np.prod(list(axis_sizes.values())))
    if len(jax.devices()) < need:
        pytest.skip(f"needs >= {need} devices, have {len(jax.devices())}")
    shape = tuple(axis_sizes.get(name, 1) for name in MESH_AXIS_NAMES)
    return Mesh(np.array(jax.devices()[:need]).reshape(shape),
                axis_names=MESH_AXIS_NAMES)


def _config(num_experts=32, moe_intermediate_size=384, moe_hidden_size=512):
    return types.SimpleNamespace(num_experts=num_experts,
                                 moe_intermediate_size=moe_intermediate_size,
                                 moe_hidden_size=moe_hidden_size)


# --------------------------------------------------------------------------
# The specs themselves
# --------------------------------------------------------------------------


def test_both_axes_are_sharded():
    """The point of the change: E and F are both split, over disjoint axes.

    This is the assertion that fails if F goes back to being replicated.
    """
    edf, efd = routed_expert_shardings()
    assert edf == P(E_AXES, None, F_AXIS)
    assert efd == P(E_AXES, F_AXIS, None)
    # Disjoint, or the two splits would collide instead of multiplying.
    assert F_AXIS not in E_AXES


def test_the_contracted_dimension_decides_the_reduction():
    """`sparse_moe_distributed_fwd` reduces over ``<spec>[1]`` of each kernel
    spec. That has to line up with which contraction is split:

      * gate/up contract D, replicated in `edf` -> spec[1] is None, no psum;
      * down contracts F, split in `efd` -> spec[1] is the tensor axis, psum.

    Getting this backwards is silently wrong rather than loud, so it is
    asserted directly.
    """
    edf, efd = routed_expert_shardings()
    assert edf[1] is None, ("gate/up contract a replicated D; a reduction "
                            "here would double-count")
    assert efd[1] == F_AXIS, ("down contracts a split F; without this the "
                              "partial sums are never added")


def test_sharding_ways_multiply():
    mesh = _mesh(attn_dp_expert=2, model=4)
    assert routed_expert_sharding_ways(mesh) == (2, 4)


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def test_per_device_expert_share_drops_by_the_product():
    """The memory claim, as arithmetic: composing the two axes divides the
    per-device expert bytes by their product, not by either one alone."""
    cfg = _config()
    total = cfg.num_experts * 3 * cfg.moe_hidden_size * cfg.moe_intermediate_size

    e_only, f_only = routed_expert_sharding_ways(_mesh(attn_dp_expert=8))
    assert (e_only, f_only) == (8, 1)
    composed_e, composed_f = routed_expert_sharding_ways(
        _mesh(attn_dp_expert=2, model=4))

    # Same 8 devices either way, but only the composed layout can also give
    # `model` to the non-expert stack.
    assert e_only * f_only == composed_e * composed_f == 8
    assert total // (composed_e * composed_f) < total


def test_replicated_experts_are_refused():
    """A multi-device mesh with neither axis split: the failure that showed up
    as an out-of-memory part-way through loading."""
    mesh = _mesh(attn_dp=4)
    with pytest.raises(ValueError, match="would be replicated"):
        validate_routed_expert_sharding(_config(), mesh, block_quantized=False)


def test_single_device_replication_is_allowed():
    """The same condition is normal on one device, and must not raise."""
    ways = validate_routed_expert_sharding(_config(),
                                           _mesh(),
                                           block_quantized=False)
    assert ways == (1, 1)


def test_indivisible_expert_count_is_refused():
    mesh = _mesh(attn_dp_expert=4)
    with pytest.raises(ValueError, match="num_experts=6 is not divisible"):
        validate_routed_expert_sharding(_config(num_experts=6),
                                        mesh,
                                        block_quantized=False)


def test_indivisible_intermediate_size_is_refused():
    mesh = _mesh(model=4)
    with pytest.raises(ValueError, match="moe_intermediate_size=6 is not"):
        validate_routed_expert_sharding(_config(moe_intermediate_size=6),
                                        mesh,
                                        block_quantized=False)


def test_unquantized_config_is_not_treated_as_block_quantized():
    """The unquantized path supplies an `UnquantizedConfig`, not `None`, and
    its weights carry no block scales -- so the tiling rule must not apply to
    it. Treating "quant_config is not None" as "quantized" wrongly refused
    meshes that had no F sharding at all."""
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig
    assert not experts_are_block_quantized(None)
    assert not experts_are_block_quantized(UnquantizedConfig({}))


def test_real_k3_dimensions_divide_on_the_target_mesh():
    """The 32-device layout this change exists to enable: E over 4, F over 8."""
    mesh = _mesh(attn_dp_expert=4, model=8)
    cfg = _config(num_experts=896,
                  moe_intermediate_size=3072,
                  moe_hidden_size=3584)
    assert validate_routed_expert_sharding(cfg, mesh,
                                           block_quantized=True) == (4, 8)


# --------------------------------------------------------------------------
# Numerics: the down projection now contracts a split dimension
# --------------------------------------------------------------------------


def _routed_experts(ckpt, mesh, *, quantized):
    from tests.models.jax.test_kimi_k3_mxfp4 import (_build_loaded_model,
                                                     _moe_block)
    return _moe_block(_build_loaded_model(ckpt, mesh,
                                          quantized=quantized)).experts


def _apply(experts, x, weights_TX, indices_TX, mesh):
    with jax.set_mesh(mesh):
        return np.asarray(
            experts.quant_method.apply_jax(experts,
                                           x,
                                           router_logits=(weights_TX,
                                                          indices_TX)))


def _expert_inputs(experts, num_tokens=16, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((num_tokens, experts.hidden_size)),
                    dtype=jnp.float32)
    indices_TX = jnp.asarray(rng.integers(0,
                                          experts.num_local_experts,
                                          size=(num_tokens, experts.top_k)),
                             dtype=jnp.int32)
    weights_TX = jnp.full((num_tokens, experts.top_k),
                          1.0 / experts.top_k,
                          dtype=jnp.float32)
    return x, weights_TX, indices_TX


@pytest.fixture(scope="module")
def wide_f_checkpoints(tmp_path_factory):
    """K3-tiny with a wider ``moe_intermediate_size``.

    The shipped fixture has F=384, whose only splits are 192 and 96 -- neither
    a multiple of 128, which the block-quantized grouped matmul requires (see
    `test_indivisible_per_shard_intermediate_is_refused`). F=1024 splits into
    512 and 256, so the composed layout can actually be exercised. Built here
    rather than shipped: it is random weights compared against themselves, so
    it needs no goldens.
    """
    import json

    from tests.models.jax.utils.k3_tiny_generator import (TINY_TEXT_CONFIG,
                                                          build_state_dicts,
                                                          write_checkpoint)
    cfg = json.loads(json.dumps(TINY_TEXT_CONFIG))
    cfg["moe_intermediate_size"] = 1024
    cfg["num_experts"] = 8
    bf16, mxfp4 = build_state_dicts(42, cfg=cfg)
    root = tmp_path_factory.mktemp("k3-wide-f")
    write_checkpoint(str(root / "bf16"), bf16, False, None, cfg=cfg)
    write_checkpoint(str(root / "mxfp4"), mxfp4, True, None, cfg=cfg)
    return {"bf16": str(root / "bf16"), "mxfp4": str(root / "mxfp4")}


@pytest.mark.parametrize("ckpt_kind", ["bf16", "mxfp4"])
def test_composed_sharding_matches_expert_only_sharding(
        ckpt_kind, wide_f_checkpoints):
    """Splitting F as well as E must not change the answer.

    Held against the expert-only layout rather than a single device: same
    backend (MEGABLX_GMM), same expert-axis width, same weights. The only
    difference is that F is split 4 ways, which makes the down projection
    contract a sharded dimension -- so if its partial sums were not reduced,
    or reduced over the wrong axis, this is where it shows.
    """
    quantized = ckpt_kind == "mxfp4"
    ckpt = wide_f_checkpoints[ckpt_kind]

    expert_only = _mesh(attn_dp_expert=2)
    composed = _mesh(attn_dp_expert=2, model=4)
    assert routed_expert_sharding_ways(expert_only) == (2, 1)
    assert routed_expert_sharding_ways(composed) == (2, 4)

    ref_experts = _routed_experts(ckpt, expert_only, quantized=quantized)
    new_experts = _routed_experts(ckpt, composed, quantized=quantized)
    for experts in (ref_experts, new_experts):
        assert experts.moe_backend.value == "megablox_gmm"

    # The composed layout really did split F on device, not just in the spec.
    down = new_experts.kernel_down_proj_EFD.value
    assert down.sharding.spec == P(E_AXES, F_AXIS, None)
    assert down.addressable_shards[0].data.shape[1] == down.shape[1] // 4

    x, weights_TX, indices_TX = _expert_inputs(ref_experts)
    ref = _apply(ref_experts, x, weights_TX, indices_TX, expert_only)
    got = _apply(new_experts, x, weights_TX, indices_TX, composed)

    scale = float(np.abs(ref).max())
    assert scale > 0, "reference output is all zeros; the comparison is vacuous"
    assert not np.isnan(got).any(), "composed layout produced NaNs"
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6 * scale)


def test_indivisible_per_shard_intermediate_is_refused():
    """F can divide evenly and still leave a per-shard width the quantized
    grouped matmul mishandles.

    Measured: F=384 split 4 ways gives 96, and the block-quantized path then
    returns 9 NaNs out of 8192 output entries while the rest stay correct to
    ~1e-8 -- a silent corruption, so it is refused up front. The same width is
    clean for bf16 weights, hence the `block_quantized` gate.
    """
    mesh = _mesh(attn_dp_expert=2, model=4)
    cfg = _config(moe_intermediate_size=384)  # 384/4 = 96, 96 % 128 != 0
    validate_routed_expert_sharding(cfg, mesh, block_quantized=False)
    with pytest.raises(ValueError, match="not a multiple of 128"):
        validate_routed_expert_sharding(cfg, mesh, block_quantized=True)


def test_real_k3_per_shard_intermediate_is_a_multiple_of_128():
    """K3's F=3072 over the intended 8-way tensor axis gives 384."""
    mesh = _mesh(attn_dp_expert=4, model=8)
    cfg = _config(num_experts=896,
                  moe_intermediate_size=3072,
                  moe_hidden_size=3584)
    assert validate_routed_expert_sharding(cfg, mesh,
                                           block_quantized=True) == (4, 8)
    assert (3072 // 8) % 128 == 0
