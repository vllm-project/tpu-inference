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
"""Fused GDN kernel wrapper — dispatch and public API."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.gdn.fused_gdn_decode_kernel import \
    fused_decoding_gdn
from tpu_inference.kernels.gdn.fused_gdn_recurrent_kernel import \
    fused_recurrent_gdn


def _dispatch_with_distribution(
    q,
    k,
    v,
    cu_seqlens,
    g,
    initial_state,
    state_indices,
    b,
    has_initial_state,
    *,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    lower_bound,
    distribution,
):
    """Dispatch to decode and recurrent kernels following the RPA pattern.

    Both kernels update the state cache in-place via ``input_output_aliases``.
    The decode kernel runs first, then its updated state and output are
    chained to the recurrent kernel.

    ``has_initial_state`` is consumed only by the recurrent kernel: decode
    tokens always have a valid prior state (a request must finish prefill
    before it can decode), so masking is unnecessary on the decode path.
    """
    # ── Decode kernel → updates state in-place ──
    # NOTE: pass `initial_state` through as-is. The kernels upcast to fp32
    # on VMEM load for compute precision; HBM storage stays at the array's
    # dtype. An `astype(jnp.float32)` here would materialize an fp32 copy
    # of a bf16 state and undo the storage win before the kernel runs.
    o_d, state_1 = fused_decoding_gdn(
        q,
        k,
        v,
        g.astype(jnp.float32),
        initial_state,
        state_indices,
        distribution,
        b,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    # ── Recurrent kernel → updates state in-place ──
    o_r, state_2 = fused_recurrent_gdn(
        q,
        k,
        o_d,
        cu_seqlens,
        g.astype(jnp.float32),
        state_1,
        state_indices,
        b,
        has_initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o_r, state_2


# ── Public API ──


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "lower_bound",
    ],
    donate_argnames=["v", "initial_state"],
)
def fused_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    cu_seqlens: jax.Array,  # [max_num_req+1] int32
    g: jax.Array,  # [T, H_v, K] or [T, H_v]
    initial_state: jax.Array,  # [num_states, H_v, K, V]
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None = None,  # [T, H_v] or None
    has_initial_state: jax.Array | None = None,  # [max_num_req] bool
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,  # [H_v] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Fused recurrent GDN forward pass.

    Supports GQA: ``H_v`` (value heads from ``v``) can be a multiple of
    ``H_qk`` (query/key heads from ``q``/``k``).  The kernel repeats
    q/k internally.

    Args:
        q: Queries ``[T, H_qk, K]``.
        k: Keys ``[T, H_qk, K]``.
        v: Values ``[T, H_v, V]``.
        cu_seqlens: Cumulative sequence lengths ``[max_num_req+1]``.
        g: Gating ``[T, H_v, K]`` or ``[T, H_v]`` (broadcast to K).
        initial_state: State cache ``[num_states, H_v, K, V]``.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        b: Raw betas ``[T, H_v]`` (sigmoid applied inside kernel).
            ``None`` means beta=1 (no beta gating).
        has_initial_state: Boolean tensor of shape ``[max_num_req]``.
            ``True`` when the request's recurrent slot already holds a
            valid prior state (chunked-prefill continuation, prefix-cache
            hit, or running decode). ``False`` for brand-new prefills,
            whose slot is zeroed inside the recurrent kernel before the
            update so stale data from a previous tenant doesn't leak.
            ``None`` (default) is treated as all-True, preserving the
            pre-fix behaviour for callers that don't manage slot reuse.
        scale: Scale factor.  Default ``K ** -0.5``.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        A_log: Per-head log gate ``[H_v]`` float32.
        dt_bias: Per-head bias ``[H_v]`` float32. Optional.
            Broadcast to ``[H_v, num_lanes]`` internally.
        lower_bound: If set, use sigmoid gate instead of softplus.

    Returns:
        ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]`` with final states
        written back at the corresponding ``state_indices`` positions.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]

    # Broadcast g from [T, H_v] to [T, H_v, K] if needed.
    if g.shape == (T, H_v):
        g = jnp.broadcast_to(g[..., None], (T, H_v, K))
    elif g.shape != (T, H_v, K):
        raise ValueError(
            f"g shape {g.shape} must be [{T}, {H_v}, {K}] or [{T}, {H_v}]")

    # Validate pre-broadcast inputs.
    if b is not None and b.shape != (T, H_v):
        raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}]")
    if A_log is not None and A_log.shape != (H_v, ):
        raise ValueError(f"A_log shape {A_log.shape} must be [{H_v}]")
    if dt_bias is not None and dt_bias.shape != (H_v, ):
        raise ValueError(f"dt_bias shape {dt_bias.shape} must be [{H_v}]")

    cu_seqlens = cu_seqlens.astype(jnp.int32)
    state_indices = state_indices.astype(jnp.int32)

    if scale is None:
        scale = K**-0.5
    num_lanes = pltpu.get_tpu_info().num_lanes
    if b is not None:
        b = jnp.broadcast_to(b[:, :, None],
                             (T, H_v, num_lanes))  # [T, H_v, num_lanes]
    if dt_bias is not None:
        dt_bias = jnp.broadcast_to(dt_bias[:, None], (H_v, num_lanes)).astype(
            jnp.float32)  # [H_v, num_lanes]
    distribution = distribution.astype(jnp.int32)

    if A_log is not None:
        A_log = jnp.broadcast_to(A_log[:, None], (H_v, num_lanes)).astype(
            jnp.float32)  # [H_v, num_lanes]

    # Public contract is Boolean (matching the chunked / ref impls);
    # cast to int32 here for SMEM compatibility — the recurrent kernel
    # checks `has_init == 0` to decide whether to zero h0. Default to
    # all-True (no masking), matching the pre-fix behaviour.
    max_num_req = state_indices.shape[0]
    if has_initial_state is None:
        has_initial_state = jnp.ones((max_num_req, ), dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    o, state = _dispatch_with_distribution(
        q,
        k,
        v,
        cu_seqlens,
        g,
        initial_state,
        state_indices,
        b,
        has_initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o, state


def ragged_gated_delta_rule(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state=None,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
    """Adapter matching the ragged_gated_delta_rule_{ref,chunked} interface.

    Internally reshapes inputs and delegates to :func:`fused_gdn`.

    Args:
        mixed_qkv: ``(num_tokens, 2*n_kq*d_k + n_v*d_v)`` post-conv/silu.
        b: ``(num_tokens, n_v)`` — raw beta (sigmoid applied in kernel).
        a: ``(num_tokens, n_v)`` — raw alpha (gate transform in kernel).
        recurrent_state: ``(num_states, n_v, d_k, d_v)``.
        A_log: ``(n_v,)`` float32.
        dt_bias: ``(n_v,)`` float32.
        query_start_loc: ``(num_seqs+1,)`` int32.
        state_indices: ``(num_seqs,)`` int32.
        distribution: ``(3,)`` int32 — ``(decode_end, prefill_end, mixed_end)``.
        has_initial_state: Optional Boolean tensor of shape
            ``(max_reqs,)``. ``True`` when the request's slot already
            holds a valid prior recurrent state; ``False`` for brand-new
            prefills (the recurrent kernel zeros h0 for those slots so
            stale data from a reused mamba slot doesn't leak). ``None``
            (default) is treated as all-True, preserving the
            pre-PR-#2408 behaviour. Pass it when you want the same
            stale-slot guard the chunked and ref impls already enforce.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key dimension.
        d_v: Value dimension.

    Returns:
        ``(updated_recurrent_state, output)`` where
        *updated_recurrent_state* is ``(num_states, n_v, d_k, d_v)`` and
        *output* is ``(num_tokens, n_v*d_v)``.
    """
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k

    q = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
    k = mixed_qkv[..., key_dim:key_dim * 2].reshape(num_tokens, n_kq, d_k)
    v = mixed_qkv[..., key_dim * 2:].reshape(num_tokens, n_v, d_v)

    g = a

    # (decode_end, prefill_end, mixed_end) → (decode_end, total)
    fused_distribution = jnp.stack([distribution[0], distribution[2]])

    output, new_recurrent_state = fused_gdn(
        q,
        k,
        v,
        cu_seqlens=query_start_loc,
        g=g,
        initial_state=recurrent_state,
        state_indices=state_indices,
        distribution=fused_distribution,
        b=b,
        has_initial_state=has_initial_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    output = output.reshape(num_tokens, n_v * d_v)
    return new_recurrent_state, output
