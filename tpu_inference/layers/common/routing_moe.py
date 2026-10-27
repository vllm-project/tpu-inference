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
"""MoEBackend.ROUTING_MOE: runs EP MoE layers with tokamax's routing_moe.

Only the EP path is wired up and tested for now; TP support (the kernel's
tp_axis) is a TODO.

tokamax's routing_moe (tokamax/_src/ops/experimental/routing_moe) all-gathers
the tokens over the EP axis; each device then gathers its experts' token rows
by index, runs gate/up and down, and sends each result row straight to the
device that owns the token, which sums the token's top-k rows weighted by the
routing weights. Unlike the GMM path, no device builds a full [tokens, hidden]
partial sum and there is no reduce-scatter.

The kernel expects a 1-D mesh, so it gets one built from the same devices in
the same order; the weights stay where they are, in the GMM_EP layout
(w13 = [gate | up], sharded over the expert axes). Devices with the same
attention-data coordinates hold the same token rows; each of them passes the
kernel its own share of those rows, so every token is routed once. The
experts are picked by the GMM path's routing, fused_moe_gmm.select_experts.
"""

import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import tpu_inference.envs as envs
from tpu_inference.layers.common.fused_moe_gmm import select_experts
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import as_axes
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

_EP_AXIS = "routing_moe"
_LAYOUTS: dict = {}


def ep_layout(mesh: Mesh):
    """Flatten mesh into the 1-D mesh the kernel takes.

    Returns (1-D mesh, replica axes, None), or (None, None, reason) if the
    kernel can't use this mesh.

    Flattening keeps the device order. That matches the EP rank order only if
    every mesh axis with more than one device is an expert axis, in mesh order.
    The replica axes are the used axes that aren't attention-data axes: devices
    that differ only along them hold the same token rows.
    """
    expert_axes = as_axes(ShardingAxisName.EXPERT)
    attn_axes = as_axes(ShardingAxisName.ATTN_DATA)
    # The axis names can change after import (2D sharding, tests), so they are
    # part of the key.
    key = (mesh, expert_axes, attn_axes)
    if key in _LAYOUTS:
        return _LAYOUTS[key]
    names = mesh.axis_names
    used = [a for a in names if mesh.shape[a] > 1]
    expert = [a for a in expert_axes if a in names]
    if [a for a in expert if mesh.shape[a] > 1] != used:
        layout = (None, None, f"the expert axes {tuple(expert)} don't span "
                  f"the mesh {dict(mesh.shape)} in mesh order")
    else:
        replica = tuple(a for a in used if a not in attn_axes)
        mesh_1d = Mesh(mesh.devices.reshape(-1), (_EP_AXIS, ))
        layout = (mesh_1d, replica, None)
    _LAYOUTS[key] = layout
    return layout


def unsupported_reason(
        mesh: Mesh, dtype: jnp.dtype, w1: jax.Array, w2: jax.Array,
        w1_scale: jax.Array | None, w2_scale: jax.Array | None,
        w1_bias: jax.Array | None, w2_bias: jax.Array | None, activation: str,
        scoring_fn: str,
        hash_based_topk_indices: jax.Array | None) -> str | None:
    """Why the kernel can't run this layer, or None if it can.

    dtype is the dtype of the layer's hidden states.
    """
    # The kernel lays token rows out in bf16 256-column chunks, hence the bf16
    # and multiple-of-256 requirements below.
    if dtype != jnp.bfloat16:
        return f"it takes bf16 hidden states, not {dtype}"
    if w1.dtype != w2.dtype:
        return f"its w13 and w2 dtypes differ ({w1.dtype}, {w2.dtype})"
    if w1.dtype == jnp.float8_e4m3fn:
        if any(s is None or s.shape[1] != 1 for s in (w1_scale, w2_scale)):
            return "its fp8 path takes one weight scale per output channel"
    elif w1.dtype != jnp.bfloat16:
        return f"it takes fp8 or bf16 weights, not {w1.dtype}"
    elif w1_scale is not None or w2_scale is not None:
        return "its bf16 path takes unscaled weights"
    if w1_bias is not None or w2_bias is not None:
        return "it doesn't take biases"
    padded_hidden = w1.shape[1]
    if padded_hidden % 256:
        return f"hidden size {padded_hidden} isn't a multiple of 256"
    # GMM EP pads w13's intermediate size to a multiple of 128 but leaves w2's
    # as is, and the kernel feeds all of w13's output to w2.
    if w1.shape[2] != 2 * w2.shape[1] or w2.shape[1] % 256:
        return (f"its intermediate sizes (w13 {w1.shape[2] // 2}, w2 "
                f"{w2.shape[1]}) aren't one multiple of 256")
    if w1.shape[0] % mesh.devices.size:
        return (f"{w1.shape[0]} experts don't split evenly over "
                f"{mesh.devices.size} devices")
    if activation != "silu":
        return f"only silu is wired up here, not {activation}"
    if scoring_fn not in ("softmax", "sigmoid", "sqrtsoftplus"):
        return f"select_experts doesn't score with {scoring_fn}"
    if hash_based_topk_indices is not None:
        return "hash routing isn't wired up here"
    if envs.MOE_APPROX_TOPK:
        return "MOE_APPROX_TOPK isn't wired up here"
    if envs.FORCE_MOE_RANDOM_ROUTING:
        return "FORCE_MOE_RANDOM_ROUTING is implemented on the GMM path only"
    _, _, reason = ep_layout(mesh)
    return reason


def _to_ep_rows(x: jax.Array, mesh: Mesh, replica: tuple[str,
                                                         ...]) -> jax.Array:
    """Give each EP rank its own share of its attention-data rows.

    Each device keeps 1/R of its local rows, R being the number of devices
    that hold them, picked by its index along the replica axes; the shares
    become one block of rows per EP rank. The local rows are first padded to a
    multiple of R by repeating the last row, so the padding is routed like a
    real token rather than all to the first experts.
    """
    num_replicas = get_mesh_shape_product(mesh, list(replica))
    rest = (None, ) * (x.ndim - 1)

    def keep_share(local):
        if num_replicas == 1:
            return local
        pad = -local.shape[0] % num_replicas
        if pad:
            local = jnp.pad(local, ((0, pad), ) + ((0, 0), ) * (x.ndim - 1),
                            mode="edge")
        size = local.shape[0] // num_replicas
        return jax.lax.dynamic_slice_in_dim(local,
                                            jax.lax.axis_index(replica) * size,
                                            size)

    return jax.shard_map(keep_share,
                         mesh=mesh,
                         in_specs=P(ShardingAxisName.ATTN_DATA, *rest),
                         out_specs=P(ShardingAxisName.EXPERT, *rest),
                         check_vma=False)(x)


def _from_ep_rows(out: jax.Array, mesh: Mesh, replica: tuple[str, ...],
                  num_local_rows: int) -> jax.Array:
    """Undo _to_ep_rows: the replicas all-gather their shares and drop the
    padding."""
    rest = (None, ) * (out.ndim - 1)

    def gather_shares(local):
        if replica:
            local = jax.lax.all_gather(local, replica, axis=0, tiled=True)
        return local[:num_local_rows]

    return jax.shard_map(gather_shares,
                         mesh=mesh,
                         in_specs=P(ShardingAxisName.EXPERT, *rest),
                         out_specs=P(ShardingAxisName.ATTN_DATA, *rest),
                         check_vma=False)(out)


def routing_moe_func(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array | None,
        w2_scale: jax.Array | None,
        gating_output: jax.Array,
        *,
        topk: int,
        renormalize: bool,
        mesh: Mesh,
        activation: str,
        scoring_fn: str,
        expert_score_correction_bias: jax.Array | None = None) -> jax.Array:
    """Run one MoE layer with tokamax's routing_moe.

    hidden_states [num_tokens, hidden] and gating_output [num_tokens, experts]
    are sharded over the attention-data axes. w1 [E, H, 2I], w2 [E, I, H] and,
    for fp8, their per-channel scales [E, 1, 1, N] are sharded over the expert
    axes. The routing arguments are as for select_experts. Returns
    [num_tokens, hidden], sharded like hidden_states. Assumes
    unsupported_reason() returned None for this layer.
    """
    from tokamax._src.ops.experimental.routing_moe.layer import \
        modular as tk_modular
    from tokamax._src.ops.experimental.routing_moe.layer import plan as tk_plan

    mesh_1d, replica, _ = ep_layout(mesh)
    num_tokens, hidden_size = hidden_states.shape
    num_experts, padded_hidden, _ = w1.shape
    # An expert id outside the weights would leave its rows unsent, and the
    # remote scatter would wait for them forever.
    assert gating_output.shape == (num_tokens, num_experts)
    if padded_hidden != hidden_size:
        hidden_states = jnp.pad(hidden_states,
                                ((0, 0), (0, padded_hidden - hidden_size)))
    logger.info_once(
        "[MoE]: routing_moe over %d devices, %d experts each, token rows "
        "split over replica axes %s; %s weights.", mesh.devices.size,
        num_experts // mesh.devices.size, replica,
        "fp8" if w1.dtype == jnp.float8_e4m3fn else "bf16")
    num_replicas = get_mesh_shape_product(mesh, list(replica))
    num_local_rows = num_tokens * num_replicas // mesh.devices.size
    x = _to_ep_rows(hidden_states, mesh, replica)
    # Route after the split, so that each device routes only its own share.
    topk_weights, topk_indices = select_experts(
        _to_ep_rows(gating_output, mesh, replica),
        topk,
        renormalize,
        scoring_fn,
        expert_score_correction_bias=expert_score_correction_bias)
    # tokamax's shard_map runs over the 1-D mesh, so make it the current mesh
    # while tracing.
    with jax.sharding.use_abstract_mesh(mesh_1d.abstract_mesh):
        out = tk_modular.routing_moe(
            x,
            topk_weights,
            topk_indices,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            mesh=mesh_1d,
            activation=activation,
            # TODO: TP support; only the EP path is tested for now.
            tp_axis=None,
            ep_axis=_EP_AXIS,
            use_sp=True,
            # Each output row goes straight to the device that owns its token.
            moe_plan=tk_plan.RoutingMoEPlan(remote_scatter=True),
        )
    out = _from_ep_rows(out, mesh, replica, num_local_rows)
    return out[:, :hidden_size]
