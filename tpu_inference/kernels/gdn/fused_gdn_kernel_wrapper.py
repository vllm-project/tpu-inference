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
"""Fused GDN kernel wrapper — dispatch and public API.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.gdn.fused_decoding_gdn_kernel import \
    fused_decoding_gdn
from tpu_inference.kernels.gdn.fused_recurrent_gdn_kernel import \
    fused_recurrent_gdn


def _dispatch_with_distribution(
    q,
    k,
    v,
    cu_seqlens,
    g,
    initial_state,
    state_indices,
    beta,
    *,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    a_log_input,
    dt_bias,
    lower_bound,
    distribution,
):
    """Dispatch to decode and recurrent kernels following the RPA pattern.

    Both kernels update the state cache in-place via ``input_output_aliases``.
    The decode kernel runs first, then its updated state and output are
    chained to the recurrent kernel.
    """
    # Strip batch dimension — kernels operate on 3D tensors.
    q_3d, k_3d, v_3d, g_3d, beta_3d = q[0], k[0], v[0], g[0], beta[0]

    # ── Decode kernel → updates state in-place ──
    o_d, state_1 = fused_decoding_gdn(
        q_3d,
        k_3d,
        v_3d,
        g_3d.astype(jnp.float32),
        initial_state.astype(jnp.float32),
        state_indices,
        distribution,
        beta_3d,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        a_log_input=a_log_input,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )

    # ── Recurrent kernel → updates state in-place ──
    o_r, state_2 = fused_recurrent_gdn(
        q_3d,
        k_3d,
        o_d,
        cu_seqlens,
        g_3d.astype(jnp.float32),
        state_1,
        state_indices,
        beta_3d,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        a_log_input=a_log_input,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o_r[None], state_2


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
    q: jax.Array,  # [1, T, H_qk, K]
    k: jax.Array,  # [1, T, H_qk, K]
    v: jax.Array,  # [1, T, H_v, V]
    cu_seqlens: jax.Array,  # [max_num_req+1] int32
    g: jax.Array,  # [1, T, H_v, K] or [1, T, H_v]
    initial_state: jax.Array,  # [num_states, H_v, K, V]
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    beta: jax.Array | None = None,  # [1, T, H_v] or None
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,  # [H_v] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v, K] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Fused recurrent GDN forward pass.

    Supports GQA: ``H_v`` (value heads from ``v``) can be a multiple of
    ``H_qk`` (query/key heads from ``q``/``k``).  The kernel repeats
    q/k internally.

    Args:
        q: Queries ``[1, T, H_qk, K]``.
        k: Keys ``[1, T, H_qk, K]``.
        v: Values ``[1, T, H_v, V]``.
        cu_seqlens: Cumulative sequence lengths ``[max_num_req+1]``.
        g: Gating ``[1, T, H_v, K]`` or ``[1, T, H_v]`` (broadcast to K).
        initial_state: State cache ``[num_states, H_v, K, V]``.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        beta: Betas ``[1, T, H_v]``.  Default ``None`` (ones).
        scale: Scale factor.  Default ``K ** -0.5``.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        A_log: Per-head log gate ``[H_v]`` float32.
        dt_bias: Per-head-key bias ``[H_v, K]`` float32. Optional.
        lower_bound: If set, use sigmoid gate instead of softplus.

    Returns:
        ``(o, updated_state)`` — *o* is ``[1, T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]`` with final states
        written back at the corresponding ``state_indices`` positions.
    """
    B, T, H_qk, K = q.shape
    H_v = v.shape[2]
    V = v.shape[3]

    if B != 1:
        raise ValueError(f"B must be 1 (got {B}) with cu_seqlens")
    if k.shape != (B, T, H_qk, K):
        raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of H_qk={H_qk}")
    if v.shape != (B, T, H_v, V):
        raise ValueError(f"v shape {v.shape} must be [{B}, {T}, {H_v}, {V}]")
    if g.shape == (B, T, H_v):
        g = jnp.broadcast_to(g[..., None], (B, T, H_v, K))
    elif g.shape != (B, T, H_v, K):
        raise ValueError(
            f"g shape {g.shape} must be [{B}, {T}, {H_v}, {K}] or [{B}, {T}, {H_v}]"
        )
    if initial_state.shape[1:] != (H_v, K, V):
        raise ValueError(
            f"initial_state trailing dims must be ({H_v}, {K}, {V})")
    max_num_req = len(cu_seqlens) - 1
    if state_indices.shape != (max_num_req, ):
        raise ValueError(
            f"state_indices shape {state_indices.shape} must be ({max_num_req},)"
        )
    if beta is not None and beta.shape != (B, T, H_v):
        raise ValueError(f"beta shape {beta.shape} must be [{B}, {T}, {H_v}]")

    cu_seqlens = cu_seqlens.astype(jnp.int32)
    state_indices = state_indices.astype(jnp.int32)

    if scale is None:
        scale = K**-0.5
    if beta is None:
        beta = jnp.ones((B, T, H_v), dtype=q.dtype)
    beta = jnp.broadcast_to(beta[..., None], (B, T, H_v, V))
    distribution = distribution.astype(jnp.int32)

    # Pad A_log's H_v dim to num_lanes for TPU DMA alignment.
    if A_log is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        H_padded = ((H_v + num_lanes - 1) // num_lanes) * num_lanes
        a_log_input = jnp.pad(A_log, (0, H_padded - H_v)).reshape(1, H_padded)
    else:
        a_log_input = None

    o, state = _dispatch_with_distribution(
        q,
        k,
        v,
        cu_seqlens,
        g,
        initial_state,
        state_indices,
        beta,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        a_log_input=a_log_input,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o, state
