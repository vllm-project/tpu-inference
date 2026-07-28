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
"""KDA (Kimi Delta Attention) sublayer for the Kimi-Linear / Kimi-K3 family.

Full sublayer per the HF reference (`moonshotai/Kimi-K3`
``modeling_kimi_linear.py::KimiDeltaAttention``):

    q = SiLU(ShortConv4(q_proj(x)));  same for k;  v = SiLU(ShortConv4(v_proj(x)))
    g_raw = f_b_proj(f_a_proj(x));    beta_raw = b_proj(x)
    o     = KDA_core(q, k, v, g_raw, beta_raw)      # kernels/kda/reference
    gate  = g_proj(x)  (full-rank, K3)  |  g_b_proj(g_a_proj(x))  (low-rank, 48B)
    out   = o_proj( RMSNorm_headwise(o) * sigmoid(gate) )

State per slot: THREE separate conv windows (q/k/v, each
``(num_blocks, kernel_size-1, proj)``, model dtype) plus the recurrent state
``(num_blocks, H, K, V)`` (fp32), all slot-indexed with the same mamba-slot
machinery as GDN (``ragged_conv1d`` handles gathering/scattering and the
has_initial_state masking).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tpu_inference.kernels.kda.reference.ragged_kda_chunked import (
    ragged_kda_decode_only, ragged_kda_mixed_prefill)
from tpu_inference.layers.common.ragged_conv1d_jax import ragged_conv1d


class KDAParams(NamedTuple):
    """KDA sublayer weights, torch ``[out, in]`` layout (as checkpointed).

    ``g_proj`` is set for the full-rank output gate (Kimi-K3,
    ``use_full_rank_gate=true``); ``g_a_proj``/``g_b_proj`` for the low-rank
    gate (Kimi-Linear-48B). Exactly one of the two forms must be present.
    """
    q_proj: jax.Array  # [H*K, hidden]
    k_proj: jax.Array  # [H*K, hidden]
    v_proj: jax.Array  # [H*V, hidden]
    q_conv_weight: jax.Array  # [H*K, 1, W]
    k_conv_weight: jax.Array  # [H*K, 1, W]
    v_conv_weight: jax.Array  # [H*V, 1, W]
    A_log: jax.Array  # [H]
    dt_bias: jax.Array  # [H*K]
    f_a_proj: jax.Array  # [K, hidden]
    f_b_proj: jax.Array  # [H*K, K]
    b_proj: jax.Array  # [H, hidden]
    o_norm_weight: jax.Array  # [V]
    o_proj: jax.Array  # [hidden, H*V]
    g_proj: jax.Array | None = None  # [H*V, hidden]
    g_a_proj: jax.Array | None = None  # [K, hidden]
    g_b_proj: jax.Array | None = None  # [H*V, K]


class KDAState(NamedTuple):
    """Slot-indexed KDA caches (block 0 is the null block)."""
    conv_q: jax.Array  # [num_blocks, W-1, H*K], model dtype
    conv_k: jax.Array  # [num_blocks, W-1, H*K], model dtype
    conv_v: jax.Array  # [num_blocks, W-1, H*V], model dtype
    recurrent: jax.Array  # [num_blocks, H, K, V], fp32


def _linear(x: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.dot(x, w.T, precision=jax.lax.Precision.HIGHEST)


def _gated_rmsnorm(o: jax.Array, gate: jax.Array, weight: jax.Array,
                   eps: float) -> jax.Array:
    """Headwise RMSNorm then sigmoid-gate multiply, fp32 internals
    (fla ``FusedRMSNormGated(activation='sigmoid')`` semantics)."""
    of = o.astype(jnp.float32)
    y = of * jax.lax.rsqrt((of * of).mean(-1, keepdims=True) + eps)
    y = y * weight.astype(jnp.float32)
    return (y * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(o.dtype)


def kda_attention(
    x: jax.Array,
    params: KDAParams,
    state: KDAState,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: int = 4,
    gate_lower_bound: float | None = None,
    rms_norm_eps: float = 1e-5,
    chunk_size: int = 64,
) -> tuple[KDAState, jax.Array]:
    """Runs the KDA sublayer over a ragged token batch.

    Args:
      x: Post-layernorm hidden states, ``[num_tokens, hidden]``.
      params/state: See :class:`KDAParams` / :class:`KDAState`.
      state_indices / query_start_loc / distribution / has_initial_state:
        Standard mamba-slot ragged-batch metadata (same semantics as the GDN
        path; ``distribution[0] == distribution[2]`` selects decode-only).
      num_heads/head_dim: H and K (= V for KDA).
      gate_lower_bound: None -> softplus decay form; float -> sigmoid
        lower-bound form (K3 uses -5.0).

    Returns ``(new_state, out [num_tokens, hidden])``.
    """
    assert (params.g_proj is None) != (
        params.g_a_proj
        is None), "exactly one of g_proj / (g_a_proj, g_b_proj) must be set"

    q = _linear(x, params.q_proj)
    k = _linear(x, params.k_proj)
    v = _linear(x, params.v_proj)

    q, new_conv_q = ragged_conv1d(q,
                                  state.conv_q,
                                  params.q_conv_weight,
                                  None,
                                  query_start_loc,
                                  state_indices,
                                  distribution,
                                  has_initial_state,
                                  kernel_size=kernel_size)
    k, new_conv_k = ragged_conv1d(k,
                                  state.conv_k,
                                  params.k_conv_weight,
                                  None,
                                  query_start_loc,
                                  state_indices,
                                  distribution,
                                  has_initial_state,
                                  kernel_size=kernel_size)
    v, new_conv_v = ragged_conv1d(v,
                                  state.conv_v,
                                  params.v_conv_weight,
                                  None,
                                  query_start_loc,
                                  state_indices,
                                  distribution,
                                  has_initial_state,
                                  kernel_size=kernel_size)
    q = jax.nn.silu(q.astype(jnp.float32))
    k = jax.nn.silu(k.astype(jnp.float32))
    v = jax.nn.silu(v.astype(jnp.float32))

    num_tokens = x.shape[0]
    q = q.reshape(num_tokens, num_heads, head_dim)
    k = k.reshape(num_tokens, num_heads, head_dim)
    v = v.reshape(num_tokens, num_heads, head_dim)

    g_raw = _linear(_linear(x, params.f_a_proj), params.f_b_proj)
    g_raw = g_raw.reshape(num_tokens, num_heads, head_dim)
    beta_raw = _linear(x, params.b_proj)  # [T, H]

    is_decode_only = distribution[0] == distribution[2]

    def decode_branch(_):
        return ragged_kda_decode_only(q,
                                      k,
                                      v,
                                      beta_raw,
                                      g_raw,
                                      state.recurrent,
                                      params.A_log,
                                      params.dt_bias,
                                      query_start_loc,
                                      state_indices,
                                      distribution,
                                      gate_lower_bound=gate_lower_bound)

    def mixed_branch(_):
        return ragged_kda_mixed_prefill(q,
                                        k,
                                        v,
                                        beta_raw,
                                        g_raw,
                                        params.A_log,
                                        params.dt_bias,
                                        query_start_loc,
                                        state.recurrent,
                                        state_indices,
                                        distribution,
                                        has_initial_state=has_initial_state,
                                        chunk_size=chunk_size,
                                        gate_lower_bound=gate_lower_bound)

    new_recurrent, o = jax.lax.cond(is_decode_only, decode_branch,
                                    mixed_branch, None)
    o = o.reshape(num_tokens, num_heads, head_dim)

    if params.g_proj is not None:
        gate = _linear(x, params.g_proj)
    else:
        gate = _linear(_linear(x, params.g_a_proj), params.g_b_proj)
    gate = gate.reshape(num_tokens, num_heads, head_dim)

    o = _gated_rmsnorm(o, gate, params.o_norm_weight, rms_norm_eps)
    out = _linear(o.reshape(num_tokens, num_heads * head_dim), params.o_proj)

    new_state = KDAState(conv_q=new_conv_q,
                         conv_k=new_conv_k,
                         conv_v=new_conv_v,
                         recurrent=new_recurrent)
    return new_state, out
