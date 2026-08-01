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

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.kda.reference.ragged_kda_chunked import (
    ragged_kda_decode_only, ragged_kda_mixed_prefill)
from tpu_inference.layers.common.ragged_conv1d_jax import ragged_conv1d
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  rank_local_slot_indices)


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


def kda_a_log_from_checkpoint(stored, num_heads: int, name: str = "A_log"):
    """Read the per-head KDA decay vector out of a checkpoint tensor.

    KDA's decay is per channel, but ``A_log`` is not the per-channel part: the
    channel structure comes from ``f_b_proj(f_a_proj(x)) + dt_bias``, and
    ``exp(A_log[h])`` is a single per-head scale applied to all of head ``h``'s
    channels. The reference implementations agree on this:

      * ``modeling_kimi_linear.py::KimiDeltaAttention`` allocates ``A_log``
        with ``num_heads`` entries.
      * vLLM's KDA gate kernel indexes it by head and broadcasts over the head
        dimension -- ``b_a = exp(A_log[i_h])`` with ``i_h`` the head program id
        and the channel offset applied only to ``raw_g``/``dt_bias``
        (``vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py``,
        ``_kda_gate_beta_fwd_kernel``).

    Two storage layouts occur across the released checkpoints:

      * ``moonshotai/Kimi-Linear-48B-A3B-Instruct``: ``[1, 1, num_heads, 1]``.
      * ``moonshotai/Kimi-K3``: a flat buffer ``head_dim`` entries wide whose
        leading ``num_heads`` entries are the parameter and whose tail is zero
        padding. Measured over the released checkpoint, every one of the 69 KDA
        layers stores ``[128]`` with entries 0..95 non-zero and 96..127 exactly
        0.0, for ``num_heads=96`` / ``head_dim=128``. (Per-head-dim tensors in
        the same layers -- ``o_norm.weight``, also ``[128]`` -- are dense, so
        the zero tail is padding and not a value.)

    Flattening and keeping the leading ``num_heads`` entries reads both. The
    padding is required to be zero rather than assumed: a checkpoint whose tail
    is non-zero holds something other than padding, and truncating it silently
    would corrupt the decay of every head.

    Written against the array protocol rather than a specific array library so
    that anything reading a Kimi checkpoint directly -- the weight loader, the
    KDA sublayer testbeds -- applies one convention instead of its own.
    """
    flat = stored.reshape(-1)
    count = flat.shape[0]
    if count == num_heads:
        return flat
    if count < num_heads:
        raise ValueError(
            f"[kimi] '{name}': checkpoint stores {count} KDA decay values but "
            f"the model has {num_heads} heads; A_log holds one scalar per "
            f"head, so the checkpoint cannot fill it.")
    tail = flat[num_heads:]
    if int((tail != 0).sum()):
        raise ValueError(
            f"[kimi] '{name}': checkpoint stores {count} KDA decay values for "
            f"{num_heads} heads and the trailing {count - num_heads} are not "
            f"zero padding (max |value| {float(abs(tail).max())}). A_log is "
            f"per head; refusing to drop values that may be a different "
            f"parameterization.")
    return flat[:num_heads]


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


def _kda_ragged_core(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta_raw: jax.Array,
    g_raw: jax.Array,
    conv_q: jax.Array,
    conv_k: jax.Array,
    conv_v: jax.Array,
    recurrent: jax.Array,
    q_conv_weight: jax.Array,
    k_conv_weight: jax.Array,
    v_conv_weight: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: int,
    gate_lower_bound: float | None,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The state-carrying part of KDA: convs, recurrent scan, state writeback.

    Everything here is indexed by the ragged-batch metadata, so it is what
    :func:`kda_attention` runs inside a ``shard_map``: under attention DP the
    metadata is a per-rank concatenation and only makes sense one rank at a
    time. The pure per-token projections stay outside.

    Returns ``(new_conv_q, new_conv_k, new_conv_v, new_recurrent,
    o [num_tokens, H, K])``.
    """
    # Global slot ids -> ids within this rank's shard of the state pool.
    state_indices = rank_local_slot_indices(state_indices, conv_q.shape[0])

    q, new_conv_q = ragged_conv1d(q,
                                  conv_q,
                                  q_conv_weight,
                                  None,
                                  query_start_loc,
                                  state_indices,
                                  distribution,
                                  has_initial_state,
                                  kernel_size=kernel_size)
    k, new_conv_k = ragged_conv1d(k,
                                  conv_k,
                                  k_conv_weight,
                                  None,
                                  query_start_loc,
                                  state_indices,
                                  distribution,
                                  has_initial_state,
                                  kernel_size=kernel_size)
    v, new_conv_v = ragged_conv1d(v,
                                  conv_v,
                                  v_conv_weight,
                                  None,
                                  query_start_loc,
                                  state_indices,
                                  distribution,
                                  has_initial_state,
                                  kernel_size=kernel_size)
    q = jax.nn.silu(q.astype(jnp.float32))
    k = jax.nn.silu(k.astype(jnp.float32))
    v = jax.nn.silu(v.astype(jnp.float32))

    num_tokens = q.shape[0]
    q = q.reshape(num_tokens, num_heads, head_dim)
    k = k.reshape(num_tokens, num_heads, head_dim)
    v = v.reshape(num_tokens, num_heads, head_dim)
    g_raw = g_raw.reshape(num_tokens, num_heads, head_dim)

    is_decode_only = distribution[0] == distribution[2]

    def decode_branch(_):
        return ragged_kda_decode_only(q,
                                      k,
                                      v,
                                      beta_raw,
                                      g_raw,
                                      recurrent,
                                      A_log,
                                      dt_bias,
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
                                        A_log,
                                        dt_bias,
                                        query_start_loc,
                                        recurrent,
                                        state_indices,
                                        distribution,
                                        has_initial_state=has_initial_state,
                                        chunk_size=chunk_size,
                                        gate_lower_bound=gate_lower_bound)

    new_recurrent, o = jax.lax.cond(is_decode_only, decode_branch,
                                    mixed_branch, None)
    return (new_conv_q, new_conv_k, new_conv_v, new_recurrent,
            o.reshape(num_tokens, num_heads, head_dim))


def _core_specs():
    """shard_map specs for `_kda_ragged_core`'s arguments, in order.

    Resolved per call, not at import: `ShardingAxisName` is a lazy proxy whose
    backing class is chosen from the runtime sharding configuration.

    Per-token arrays and the state pool split over ATTN_DATA; the ragged
    metadata splits the same way (that is what makes each rank's view
    self-consistent). The KDA parameters are replicated (see the sharding note
    on `KimiDeltaAttention`), so the head / channel dimensions stay whole
    inside the shard; when those parameters get head-sharded, the `None`s on
    the state and per-token arrays' trailing dims become ATTN_HEAD.
    """
    data = ShardingAxisName.ATTN_DATA
    in_specs = (
        P(data, None),  # q
        P(data, None),  # k
        P(data, None),  # v
        P(data, None),  # beta_raw
        P(data, None),  # g_raw
        P(data, None, None),  # conv_q
        P(data, None, None),  # conv_k
        P(data, None, None),  # conv_v
        P(data, None, None, None),  # recurrent
        P(),  # q_conv_weight
        P(),  # k_conv_weight
        P(),  # v_conv_weight
        P(),  # A_log
        P(),  # dt_bias
        P(data),  # query_start_loc
        P(data),  # state_indices
        P(data),  # distribution
        P(data),  # has_initial_state
    )
    out_specs = (
        P(data, None, None),  # new_conv_q
        P(data, None, None),  # new_conv_k
        P(data, None, None),  # new_conv_v
        P(data, None, None, None),  # new_recurrent
        P(data, None, None),  # o
    )
    return in_specs, out_specs


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
    mesh: jax.sharding.Mesh | None = None,
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
      mesh: When given, the state-carrying part runs inside a ``shard_map``
        over ``ShardingAxisName.ATTN_DATA``, exactly as
        ``run_jax_gdn_attention`` does. This is REQUIRED under attention DP:
        the runner packs `query_start_loc` as a per-rank concatenation of
        `(reqs_per_rank + 1)`-entry blocks, so `query_start_loc[1:] -
        query_start_loc[:-1]` is only a per-request query-length vector inside
        one rank's slice — evaluated on the packed global array it is one
        entry per rank too long and mixes offsets across ranks. `None` skips
        the shard_map and is for single-rank callers (unit tests).

    Returns ``(new_state, out [num_tokens, hidden])``.
    """
    assert (params.g_proj is None) != (
        params.g_a_proj
        is None), "exactly one of g_proj / (g_a_proj, g_b_proj) must be set"

    num_tokens = x.shape[0]
    q = _linear(x, params.q_proj)
    k = _linear(x, params.k_proj)
    v = _linear(x, params.v_proj)
    g_raw = _linear(_linear(x, params.f_a_proj), params.f_b_proj)
    beta_raw = _linear(x, params.b_proj)  # [T, H]

    core = functools.partial(_kda_ragged_core,
                             num_heads=num_heads,
                             head_dim=head_dim,
                             kernel_size=kernel_size,
                             gate_lower_bound=gate_lower_bound,
                             chunk_size=chunk_size)
    if mesh is not None:
        in_specs, out_specs = _core_specs()
        core = jax.shard_map(core,
                             mesh=mesh,
                             in_specs=in_specs,
                             out_specs=out_specs,
                             check_vma=False)

    new_conv_q, new_conv_k, new_conv_v, new_recurrent, o = core(
        q, k, v, beta_raw, g_raw, state.conv_q, state.conv_k, state.conv_v,
        state.recurrent, params.q_conv_weight, params.k_conv_weight,
        params.v_conv_weight, params.A_log, params.dt_bias, query_start_loc,
        state_indices, distribution, has_initial_state)

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
