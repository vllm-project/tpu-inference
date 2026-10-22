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
"""MoEBackend.FUSED_EP_MOE_V2: EP MoE layers through tokamax's fused kernel.

tokamax's fused_ep_moe_v2 (tokamax/_src/ops/experimental/fused_moe) runs a
whole expert-parallel MoE layer in one Pallas kernel. It gathers each shard's
token rows into VMEM by index, runs the shard's experts with their whole
weights in VMEM, and DMAs each result row, in fp8, to the device that owns the
token; a small kernel there adds the top-k rows. There is no dense
[tokens, hidden] partial sum and no reduce-scatter, and a shard's work grows
with its own rows.

The kernel takes a 1-D mesh. It gets one over the same devices in the same
order, so the tokens, gates and weights stay where they are. The weights use
the GMM_EP layout (w13 = [gate | up], sharded over the expert axes).
"""

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import tpu_inference.envs as envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_EP_AXIS = "fused_ep_moe_v2"
_LAYOUTS: dict = {}


def _axes(spec) -> tuple[str, ...]:
    if spec is None:
        return ()
    return (spec, ) if isinstance(spec, str) else tuple(spec)


def ep_layout(mesh: Mesh):
    """(1-D mesh, token replica groups, None), or (None, None, reason).

    The 1-D mesh lists mesh.devices in their own order. That is the EP rank
    order when every axis with more than one device is an expert axis, in mesh
    order. Devices with the same attention-data coordinates hold the same token
    rows; each replica group lists their EP ranks, and the kernel splits those
    rows between them.
    """
    if mesh in _LAYOUTS:
        return _LAYOUTS[mesh]
    names = mesh.axis_names
    used = [a for a in names if mesh.shape[a] > 1]
    expert = [a for a in _axes(ShardingAxisName.EXPERT) if a in names]
    if [a for a in expert if mesh.shape[a] > 1] != used:
        layout = (None, None, f"the expert axes {tuple(expert)} don't span "
                  f"the mesh {dict(mesh.shape)} in mesh order")
    else:
        attn = [
            names.index(a) for a in _axes(ShardingAxisName.ATTN_DATA)
            if a in names
        ]
        groups: dict[tuple, list[int]] = {}
        for rank, (coords, _) in enumerate(np.ndenumerate(mesh.devices)):
            groups.setdefault(tuple(coords[i] for i in attn), []).append(rank)
        mesh_1d = Mesh(mesh.devices.reshape(-1), (_EP_AXIS, ))
        layout = (mesh_1d, tuple(tuple(g) for g in groups.values()), None)
    _LAYOUTS[mesh] = layout
    return layout


def unsupported_reason(
        mesh: Mesh, w1: jax.Array, w1_scale: jax.Array | None, activation: str,
        scoring_fn: str,
        hash_based_topk_indices: jax.Array | None) -> str | None:
    """Why the kernel can't run this layer, or None if it can."""
    if w1.dtype == jnp.float8_e4m3fn:
        if w1_scale is None or w1_scale.shape[1] != 1:
            return "its fp8 path takes one weight scale per output channel"
    elif w1.dtype != jnp.bfloat16 or w1_scale is not None:
        return f"it takes fp8 or unscaled bf16 weights, not {w1.dtype}"
    padded_hidden = w1.shape[1]
    if padded_hidden % 128 or padded_hidden > 64 * 128:
        return f"hidden size {padded_hidden} isn't a multiple of 128 up to 8192"
    if activation != "silu":
        return f"only silu is wired up here, not {activation}"
    if scoring_fn not in ("softmax", "sigmoid", "sqrtsoftplus"):
        return f"it doesn't score with {scoring_fn}"
    if hash_based_topk_indices is not None:
        return "it routes by top-k only, not by hash"
    _, _, reason = ep_layout(mesh)
    return reason


def _move_rows(x: jax.Array, mesh: Mesh, src, dst) -> jax.Array:
    """Re-label which devices' blocks make up x's rows; no data moves.

    Each device keeps its local block. With src=ATTN_DATA and dst=EXPERT, the
    rows become every EP rank's own copy of its DP rank's rows, back to back;
    the reverse declares the replicas (equal, after the kernel's combine) as
    one.
    """
    rest = (None, ) * (x.ndim - 1)
    return jax.shard_map(lambda local: local,
                         mesh=mesh,
                         in_specs=P(src, *rest),
                         out_specs=P(dst, *rest),
                         check_vma=False)(x)


def fused_ep_moe_v2_func(hidden_states: jax.Array,
                         w1: jax.Array,
                         w2: jax.Array,
                         w1_scale: jax.Array | None,
                         w2_scale: jax.Array | None,
                         w1_bias: jax.Array | None,
                         w2_bias: jax.Array | None,
                         gating_output: jax.Array,
                         *,
                         topk: int,
                         renormalize: bool,
                         mesh: Mesh,
                         scoring_fn: str,
                         score_bias: jax.Array | None = None) -> jax.Array:
    """One MoE layer through tokamax's fused_ep_moe_v2.

    hidden_states [num_tokens, hidden] and gating_output [num_tokens, experts]
    come in sharded over the attention-data axes; w1 [E, H, 2I], w2 [E, I, H]
    and their per-channel scales [E, 1, 1, N] (fp8 only) over the expert axes.
    Returns [num_tokens, hidden], sharded like hidden_states. The layer must
    pass unsupported_reason().
    """
    from tokamax._src.ops.experimental.fused_moe import host as tk_host
    from tokamax._src.ops.experimental.fused_moe import layer as tk_layer

    mesh_1d, groups, _ = ep_layout(mesh)
    num_tokens, hidden_size = hidden_states.shape
    num_experts, padded_hidden, _ = w1.shape
    if padded_hidden != hidden_size:
        hidden_states = jnp.pad(hidden_states,
                                ((0, 0), (0, padded_hidden - hidden_size)))
    fp8 = w1.dtype == jnp.float8_e4m3fn
    logger.info_once(
        "[MoE]: fused_ep_moe_v2 over %d devices, %d experts each, token rows "
        "split over replica groups of %d; %d-row tiles, %s routing plan, %s "
        "weights.", mesh.devices.size, num_experts // mesh.devices.size,
        len(groups[0]), envs.FUSED_EP_MOE_V2_CAPACITY,
        "sharded" if envs.FUSED_EP_MOE_V2_SHARDED_PLAN else "replicated",
        "fp8" if fp8 else "bf16")
    attn, expert = ShardingAxisName.ATTN_DATA, ShardingAxisName.EXPERT
    x = _move_rows(hidden_states, mesh, attn, expert)
    gating = _move_rows(gating_output, mesh, attn, expert)
    # The kernel's shard_map is over the 1-D mesh, which has to be the context
    # mesh while it traces.
    with jax.sharding.use_abstract_mesh(mesh_1d.abstract_mesh):
        out = tk_layer.fused_ep_moe_v2(
            x,
            w1,
            w2,
            # [E, 1, 1, N] per output channel -> [E, N], the kernel's layout.
            None if w1_scale is None else w1_scale.reshape(num_experts, -1),
            None if w2_scale is None else w2_scale.reshape(num_experts, -1),
            gating,
            w1_bias,
            w2_bias,
            topk=topk,
            renormalize=renormalize,
            mesh=mesh_1d,
            capacity=envs.FUSED_EP_MOE_V2_CAPACITY,
            weight_format=(tk_host.WeightFormat.FP8
                           if fp8 else tk_host.WeightFormat.BF16),
            act_fn="silu",
            sharded_plan=envs.FUSED_EP_MOE_V2_SHARDED_PLAN,
            scoring_fn=scoring_fn,
            score_bias=score_bias,
            token_replica_groups=groups,
        )
    out = _move_rows(out, mesh, expert, attn)
    return out[:num_tokens, :hidden_size]
