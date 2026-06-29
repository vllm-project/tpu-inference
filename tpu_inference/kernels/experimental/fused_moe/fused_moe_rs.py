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
"""EP MoE with ICI reduce-scatter fused into the ``gmm_fused_rs`` kernel.

A single Pallas call performs: gather -> GMM1 -> activation -> GMM2 -> ICI
reduce-scatter. Only the nodedup path is provided.
"""

import functools
import os

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import get_mesh_shape_product

from .gmm_fused_rs_nodedup import _select_fused_rs_block_sizes
from .gmm_fused_rs_nodedup import gmm_v2_fused_rs as gmm_v2_fused_rs_nodedup
from .gmm_v2_gather_scatter import _recover_quant_block_size

EXPERT = ShardingAxisName.EXPERT
MLP_DATA = ShardingAxisName.MLP_DATA


def _flatten_partition_axes(*axis_specs):
    axes = ()
    for spec in axis_specs:
        if spec is None:
            continue
        if isinstance(spec, tuple):
            axes += _flatten_partition_axes(*spec)
        else:
            axes += (spec, )
    return axes[0] if len(axes) == 1 else axes


def combine_partition_axes(*axis_specs):
    axes = _flatten_partition_axes(*axis_specs)
    if axes is None or isinstance(axes, str):
        return axes
    combined = ()
    for a in axes:
        if a not in combined:
            combined += (a, )
    return combined[0] if len(combined) == 1 else combined


def get_moe_expert_axis(mesh, default_axis=EXPERT):
    """Runtime FFN expert axis spanning the physical EP mesh axes."""
    axes = tuple(a for a in ("attn_dp", "attn_dp_expert", "expert", "model")
                 if a in mesh.shape)
    return _flatten_partition_axes(*axes) if axes else default_axis


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    if scoring_fn == "softmax":
        return jax.nn.softmax(x, axis=-1)
    if scoring_fn == "sigmoid":
        return jax.nn.sigmoid(x)
    raise NotImplementedError(f"unsupported scoring function: {scoring_fn}")


def enabled_tpu_sp() -> bool:
    """Sequence-parallel MoE. Defaults ON; set ``TPU_MOE_ENABLE_SP=0`` for OFF."""
    return os.environ.get("TPU_MOE_ENABLE_SP", "1") != "0"


def _routing_and_topk(
    gating_output,
    scoring_fn,
    topk,
    renormalize,
    dtype,
    mesh,
    *,
    expert_bias: jax.Array | None = None,
    route_scale: float = 1.0,
    router_output_multiplier: float | None = None,
    router_score_division_eps: float | None = None,
):
    if router_output_multiplier is not None:
        gating_output = gating_output * router_output_multiplier

    scores = apply_scoring_fn(scoring_fn, gating_output)
    scores = jax.lax.with_sharding_constraint(
        scores, NamedSharding(mesh, P(MLP_DATA, None)))

    if expert_bias is not None:
        # Bias shifts selection but not the final weights.
        scores_for_topk = scores + expert_bias
        _, topk_indices = jax.lax.top_k(scores_for_topk, k=topk)
        topk_weights = jnp.take_along_axis(scores, topk_indices, axis=1)
    else:
        topk_weights, topk_indices = jax.lax.top_k(scores, k=topk)

    if renormalize:
        denom = topk_weights.sum(axis=-1, keepdims=True)
        if router_score_division_eps is not None:
            denom = denom + router_score_division_eps
        topk_weights = topk_weights / denom

    topk_weights = (topk_weights * route_scale).astype(dtype)
    return topk_weights, topk_indices


# Per-row scalar-prefetch arrays scale as 4 * size_m bytes each and must fit the
# kernel's SMEM budget; guard at the empirical overflow threshold.
_FUSED_RS_MAX_SAFE_SIZE_M = 262144


def _assert_fused_rs_smem_safe(size_m: int) -> None:
    assert size_m < _FUSED_RS_MAX_SAFE_SIZE_M, (
        f"gmm_v2_fused_rs SMEM OOM at size_m={size_m} "
        f"(>= {_FUSED_RS_MAX_SAFE_SIZE_M}). Reduce tokens * top_k or shard more."
    )


_FP8_OUTPUT_COMM_ENV_VALUES = frozenset(
    ("fp8", "float8_e4m3fn", "jnp.float8_e4m3fn"))


def _enable_fp8_output_comm_from_env() -> bool:
    value = os.environ.get("TPU_MOE_FP8_OUTPUT_COMM", "")
    return value.lower() in _FP8_OUTPUT_COMM_ENV_VALUES


def _all_gather_token_hidden(
    token_hidden: jax.Array,
    *,
    axis_name,
    fp8_enabled: bool,
) -> jax.Array:
    """All-gather token shards, optionally quantizing the payload to FP8."""
    if not fp8_enabled:
        return jax.lax.all_gather(token_hidden,
                                  axis_name=axis_name,
                                  axis=0,
                                  tiled=True)

    with jax.named_scope("moe_fp8_post_gather"):
        out_dtype = token_hidden.dtype
        token_hidden_f32 = token_hidden.astype(jnp.float32)
        fp8_max = jnp.array(jnp.finfo(jnp.float8_e4m3fn).max,
                            dtype=jnp.float32)
        absmax = jnp.max(jnp.abs(token_hidden_f32), axis=-1, keepdims=True)
        scale = jnp.maximum(absmax, jnp.array(1e-6,
                                              dtype=jnp.float32)) / fp8_max
        token_hidden_fp8 = jnp.clip(token_hidden_f32 / scale, -fp8_max,
                                    fp8_max).astype(jnp.float8_e4m3fn)
        gathered_fp8 = jax.lax.all_gather(token_hidden_fp8,
                                          axis_name=axis_name,
                                          axis=0,
                                          tiled=True)
        gathered_scale = jax.lax.all_gather(scale,
                                            axis_name=axis_name,
                                            axis=0,
                                            tiled=True)
        return (gathered_fp8.astype(jnp.float32) *
                gathered_scale).astype(out_dtype)


def _compute_rs_routing(topk_indices, *, num_experts, topk):
    """Inline routing: lhs_indices, group_sizes, output_indices, topk_slot_indices.

    Uses integer arithmetic and one-hot+sum (not gathers / bincount) to keep the
    computation cheap and fusable. dtype stays int32 for scalar-prefetch refs.
    """
    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
    group_sizes = jnp.sum(
        (topk_indices_flat[:, None] == expert_ids[None, :]).astype(jnp.int32),
        axis=0,
    )
    lhs_indices = topk_argsort_indices // topk
    topk_slot_indices = topk_argsort_indices % topk
    output_indices = lhs_indices
    return lhs_indices, group_sizes, output_indices, topk_slot_indices


def moe_gmm_local_rs_nodedup(
    hidden_states_local: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    w1_global_scale: jax.Array | None,
    w2_global_scale: jax.Array | None,
    group_offset: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    post_expert_norm_weight_input: jax.Array | None = None,
    *,
    activation: str,
    topk: int,
    ep_size: int,
    ep_axis_name=EXPERT,
    has_post_norm: bool = False,
    sp_enabled: bool = True,
    fp8_post_gather: bool = False,
) -> jax.Array:
    """Per-chip MoE body: ICI direct-write per row, then weighted top_k reduce."""
    num_tokens = hidden_states_local.shape[0]
    hidden_size = w2.shape[-1]
    chunk_size = num_tokens // ep_size
    num_experts = w1.shape[0] * ep_size  # global num_experts
    num_local_experts = w1.shape[0]
    my_id = jax.lax.axis_index(ep_axis_name)

    # Routing inlined here so it fuses with the kernel pipeline.
    lhs_indices, group_sizes, output_indices, topk_slot_indices = _compute_rs_routing(
        topk_indices, num_experts=num_experts, topk=topk)
    size_m = lhs_indices.shape[0]
    _assert_fused_rs_smem_safe(size_m)

    # Local row range [local_start, local_end) for this chip's experts.
    go_val = group_offset[0]
    expert_idx = jnp.arange(num_experts, dtype=jnp.int32)
    local_start = jnp.sum(jnp.where(expert_idx < go_val, group_sizes, 0))
    local_end = local_start + jnp.sum(
        jnp.where(
            jnp.logical_and(expert_idx >= go_val, expert_idx
                            < go_val + num_local_experts),
            group_sizes,
            0,
        ))

    # Rows from other chips destined for me (dest == my_id and not local).
    send_dest_chips = output_indices // chunk_size
    rows = jnp.arange(size_m, dtype=jnp.int32)
    row_is_mine = jnp.logical_and(rows >= local_start, rows < local_end)
    to_me_remote = jnp.logical_and(send_dest_chips == my_id,
                                   jnp.logical_not(row_is_mine))
    my_recv_count = jnp.sum(jnp.where(to_me_remote, 1, 0))
    total_recv_count = jnp.array([my_recv_count], dtype=jnp.int32)

    # tile_m here MUST match the value the kernel selects internally, otherwise
    # max_num_gm under-counts and the kernel's final gather DMA is left unawaited.
    block_sizes = _select_fused_rs_block_sizes(
        size_m=size_m,
        size_k1=w1.shape[1],
        size_n1=w1.shape[2],
        size_k2=w2.shape[1],
        size_n2=w2.shape[2],
        size_group=num_local_experts,
        size_lhs_group=group_sizes.shape[0],
        ep_size=ep_size,
        out_dtype=hidden_states_local.dtype,
        w1_dtype=w1.dtype,
        w2_dtype=w2.dtype,
        is_quantized=w1_scale is not None,
        quant_block_size=(_recover_quant_block_size(
            w1.shape[1], w1_scale.shape[1]) if w1_scale is not None else None),
        act_fn=activation,
        fp8_direct_write=fp8_post_gather,
    )
    tile_m = block_sizes.tile_m
    max_num_gm = jnp.array(num_experts + (size_m + tile_m - 1) // tile_m - 1,
                           dtype=jnp.int32)

    out_buf = gmm_v2_fused_rs_nodedup(
        hidden_states_local,
        w1,
        w2,
        group_sizes,
        lhs_indices,
        output_indices,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        act_fn=activation,
        output_size=num_tokens,
        group_offset=group_offset,
        topk_indices=topk_slot_indices,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        max_num_gm=max_num_gm,
        total_recv_count=total_recv_count,
        top_k=topk,
        fp8_direct_write=fp8_post_gather,
    )

    # SP off: topk_weights is replicated; slice this chip's shard to match out_3d.
    local_topk_weights = (
        topk_weights if sp_enabled else jax.lax.dynamic_slice_in_dim(
            topk_weights, my_id * chunk_size, chunk_size, axis=0))
    out_3d = out_buf.reshape(chunk_size, topk, hidden_size)
    post_expert_norm_weight = post_expert_norm_weight_input if has_post_norm else None
    if post_expert_norm_weight is not None:
        norm_size = post_expert_norm_weight.shape[0]  # unpadded hidden_size
        pnw_raw = post_expert_norm_weight.astype(jnp.float32) + 1.0
        if hidden_size > norm_size:
            # Zero padded columns so they don't affect variance.
            col_idx = jnp.arange(hidden_size, dtype=jnp.int32)
            valid_mask = (col_idx < norm_size)[None, None, :]
            out_f32 = out_3d.astype(jnp.float32) * valid_mask
            pnw = jnp.concatenate([
                pnw_raw,
                jnp.zeros(hidden_size - norm_size, dtype=jnp.float32)
            ])
        else:
            out_f32 = out_3d.astype(jnp.float32)
            pnw = pnw_raw
        out_f32 = out_f32 * pnw[None, None, :]
        var = jnp.sum(out_f32**2, axis=-1, keepdims=True) / norm_size
        out_3d = (out_f32 * jax.lax.rsqrt(var + 1e-8)).astype(out_3d.dtype)

    token_hidden = jnp.sum(out_3d * local_topk_weights[:, :, None], axis=1)

    if sp_enabled:
        # Kernel reduce-scatter is the SP exit; output stays token-sharded.
        return token_hidden
    # SP off: gather the per-chip token shard back to the replicated batch.
    return _all_gather_token_hidden(
        token_hidden,
        axis_name=ep_axis_name,
        fp8_enabled=fp8_post_gather,
    )


def expert_parallel_gmm_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    *,
    activation: str,
    topk: int,
    mesh: Mesh,
    post_expert_norm_weight: jax.Array | None = None,
    w1_global_scale: jax.Array | None = None,
    w2_global_scale: jax.Array | None = None,
    fp8_post_gather: bool = False,
) -> jax.Array:
    """shard_map driver: routing runs per-chip so it fuses with the kernel."""
    expert_axis = get_moe_expert_axis(mesh, EXPERT)
    ep_size = get_mesh_shape_product(mesh, expert_axis)
    ep_p_spec = P(expert_axis)
    data_p_spec = P(MLP_DATA)
    # SP off: hidden replicated in/out with an explicit all-gather in the body.
    sp_enabled = enabled_tpu_sp()
    hidden_in_spec = data_p_spec if sp_enabled else P()
    moe_out_spec = (P(combine_partition_axes(MLP_DATA, expert_axis))
                    if sp_enabled else P())
    fp8_post_gather = ((not sp_enabled) and w1_scale is not None
                       and w2_scale is not None and
                       (fp8_post_gather or _enable_fp8_output_comm_from_env()))
    # SP off: replicate routing tensors too; the body slices its local chunk.
    topk_w_spec = (P(combine_partition_axes(MLP_DATA, expert_axis), None)
                   if sp_enabled else P())
    topk_i_spec = P(MLP_DATA, None) if sp_enabled else P()
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jnp.arange(0, num_experts, num_experts_per_shard)

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec
    w1_gs_spec = None if w1_global_scale is None else ep_p_spec
    w2_gs_spec = None if w2_global_scale is None else ep_p_spec

    _has_pn_rs = post_expert_norm_weight is not None
    _pnw_rs = (post_expert_norm_weight if post_expert_norm_weight is not None
               else jnp.zeros((1, ), jnp.bfloat16))
    result = jax.shard_map(
        functools.partial(
            moe_gmm_local_rs_nodedup,
            activation=activation,
            topk=topk,
            ep_size=ep_size,
            ep_axis_name=expert_axis,
            has_post_norm=_has_pn_rs,
            sp_enabled=sp_enabled,
            fp8_post_gather=fp8_post_gather,
        ),
        mesh=mesh,
        in_specs=(
            hidden_in_spec,
            ep_p_spec,  # w1
            w1_scale_spec,
            w1_bias_spec,
            ep_p_spec,  # w2
            w2_scale_spec,
            w2_bias_spec,
            w1_gs_spec,
            w2_gs_spec,
            ep_p_spec,  # group_offset
            topk_w_spec,
            topk_i_spec,
            P(),  # post_expert_norm_weight
        ),
        out_specs=moe_out_spec,
        check_vma=False,
    )(
        hidden_states,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        w1_global_scale,
        w2_global_scale,
        group_offset,
        topk_weights,
        topk_indices,
        _pnw_rs,
    )

    return result


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "renormalize",
        "mesh",
        "activation",
        "scoring_fn",
        "fp8_post_gather",
    ),
)
def fused_moe_func_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array | None,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    activation: str,
    scoring_fn: str,
    post_expert_norm_weight: jax.Array | None = None,
    topk_weights: jax.Array | None = None,
    topk_indices: jax.Array | None = None,
    fp8_post_gather: bool = False,
) -> jax.Array:
    """EP MoE with ICI reduce-scatter fused in kernel (gmm_fused_rs).

    Uses caller-supplied ``topk_weights``/``topk_indices`` when both are given;
    otherwise computes top-k from ``gating_output``. Then runs the fused kernel
    (gather -> GMM1 -> act -> GMM2 -> ICI reduce-scatter) and reduces over top_k.
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype

    assert (num_tokens * topk) % 16 == 0
    if topk_weights is not None and topk_indices is not None:
        # Honor pre-computed routing; do not recompute from gating_output.
        topk_weights = jax.lax.with_sharding_constraint(
            topk_weights, NamedSharding(mesh, P(MLP_DATA, None)))
        topk_indices = jax.lax.with_sharding_constraint(
            topk_indices, NamedSharding(mesh, P(MLP_DATA, None)))
    else:
        assert gating_output is not None, (
            "fused_moe_func_rs: either pre-computed topk_weights+topk_indices "
            "or gating_output must be provided.")
        assert gating_output.shape == (num_tokens, global_num_experts)
        topk_weights, topk_indices = _routing_and_topk(gating_output,
                                                       scoring_fn, topk,
                                                       renormalize, dtype,
                                                       mesh)

    # Pad hidden_states to w1's K dimension if needed.
    if padded_hidden_size != hidden_size:
        hidden_states = jnp.pad(hidden_states,
                                ((0, 0),
                                 (0, padded_hidden_size - hidden_size)))

    result = expert_parallel_gmm_rs(
        hidden_states,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        topk_weights,
        topk_indices,
        activation=activation,
        topk=topk,
        mesh=mesh,
        post_expert_norm_weight=post_expert_norm_weight,
        fp8_post_gather=fp8_post_gather,
    )

    return result[:num_tokens, :hidden_size]


__all__ = [
    "fused_moe_func_rs",
    "expert_parallel_gmm_rs",
    "moe_gmm_local_rs_nodedup",
    "_compute_rs_routing",
    "_FUSED_RS_MAX_SAFE_SIZE_M",
    "_assert_fused_rs_smem_safe",
]
