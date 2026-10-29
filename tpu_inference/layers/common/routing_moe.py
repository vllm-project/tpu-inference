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

The kernel runs over the mesh's expert axes (ShardingAxisName.EXPERT) as its EP
axis, and the weights stay where they are, in the GMM_EP layout
(w13 = [gate | up], sharded over the expert axes). Devices with the same
attention-data coordinates hold the same token rows; each of them passes the
kernel its own share of those rows, so every token is routed once. The
experts are picked by moe_router.select_experts, as on the GMM path.

ROUTING_MOE_REMOTE_SCATTER=0 selects tokamax's local mode instead of the remote
scatter above: each device sums its experts' rows into a partial result for
every token, and a reduce-scatter combines the partial results.
"""

import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import tpu_inference.envs as envs
from tpu_inference.layers.common.moe_router import select_experts
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import as_axes
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)


def _layout(mesh: Mesh) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The kernel's EP axes and the replica axes for this mesh.

    The EP axes are the mesh's expert axes. The replica axes are the used axes
    that aren't attention-data axes: devices that differ only along them hold
    the same token rows.
    """
    names = mesh.axis_names
    ep_axes = tuple(a for a in as_axes(ShardingAxisName.EXPERT) if a in names)
    attn_axes = as_axes(ShardingAxisName.ATTN_DATA)
    replica = tuple(a for a in names
                    if mesh.shape[a] > 1 and a not in attn_axes)
    return ep_axes, replica


def unsupported_reason(mesh: Mesh) -> str | None:
    """Why this integration can't run the layer, or None if it can.

    Only what the integration itself doesn't handle; select_experts and the
    kernel raise for what they don't support.
    """
    ep_axes, _ = _layout(mesh)
    used = [a for a in mesh.axis_names if mesh.shape[a] > 1]
    if any(a not in ep_axes for a in used):
        return (f"its expert axes {ep_axes} don't cover the mesh "
                f"{dict(mesh.shape)}")
    if envs.FORCE_MOE_RANDOM_ROUTING:
        return "FORCE_MOE_RANDOM_ROUTING is implemented on the GMM path only"
    return None


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
        w1_bias: jax.Array | None = None,
        w2_bias: jax.Array | None = None,
        hash_based_topk_indices: jax.Array | None = None,
        expert_score_correction_bias: jax.Array | None = None) -> jax.Array:
    """Run one MoE layer with tokamax's routing_moe.

    hidden_states [num_tokens, hidden] and gating_output [num_tokens, experts]
    are sharded over the attention-data axes. w1 [E, H, 2I], w2 [E, I, H] and,
    for fp8, their scales [E, blocks, 1, N] are sharded over the expert axes;
    the biases go to the kernel as they are. hash_based_topk_indices
    [num_tokens, topk] is sharded like hidden_states, and the routing
    arguments are as for select_experts. Returns [num_tokens, hidden], sharded
    like hidden_states. Assumes unsupported_reason() returned None for this
    layer.
    """
    from tokamax._src.ops.experimental.routing_moe.layer import \
        modular as tk_modular
    from tokamax._src.ops.experimental.routing_moe.layer import plan as tk_plan

    ep_axes, replica = _layout(mesh)
    num_tokens, hidden_size = hidden_states.shape
    num_experts, padded_hidden, _ = w1.shape
    # An expert id outside the weights would leave its rows unsent, and with
    # remote scatter the kernel would wait for them forever.
    assert gating_output.shape == (num_tokens, num_experts)
    if padded_hidden != hidden_size:
        hidden_states = jnp.pad(hidden_states,
                                ((0, 0), (0, padded_hidden - hidden_size)))
    logger.info_once(
        "[MoE]: routing_moe over %d devices, %d experts each, token rows "
        "split over replica axes %s; %s weights, %s scatter.",
        mesh.devices.size, num_experts // mesh.devices.size, replica,
        "fp8" if w1.dtype == jnp.float8_e4m3fn else "bf16",
        "remote" if envs.ROUTING_MOE_REMOTE_SCATTER else "local")
    num_replicas = get_mesh_shape_product(mesh, list(replica))
    num_local_rows = num_tokens * num_replicas // mesh.devices.size
    x = _to_ep_rows(hidden_states, mesh, replica)
    # Route after the split, so that each device routes only its own share.
    if hash_based_topk_indices is not None:
        hash_based_topk_indices = _to_ep_rows(hash_based_topk_indices, mesh,
                                              replica)
    topk_weights, topk_indices = select_experts(
        _to_ep_rows(gating_output, mesh, replica),
        topk,
        renormalize,
        scoring_fn,
        hash_based_topk_indices=hash_based_topk_indices,
        expert_score_correction_bias=expert_score_correction_bias)
    out = tk_modular.routing_moe(
        x,
        topk_weights,
        topk_indices,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        mesh=mesh,
        activation=activation,
        # TODO: TP support; only the EP path is tested for now.
        tp_axis=None,
        # Several mesh axes acting as one EP axis, ranked in this order, as
        # P(ShardingAxisName.EXPERT) shards the weights.
        ep_axis=ep_axes,
        use_sp=True,
        moe_plan=tk_plan.RoutingMoEPlan(
            remote_scatter=envs.ROUTING_MOE_REMOTE_SCATTER),
    )
    out = _from_ep_rows(out, mesh, replica, num_local_rows)
    return out[:, :hidden_size]
