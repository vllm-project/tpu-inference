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
"""JAX Mamba2 SSD for the vLLM Mamba2 mixer on TPU.

Mamba2 is a linear-recurrent + conv1d sequence mixer (like GDN). This module
provides the SSD core: a ragged token-by-token scan that handles a mixed
prefill+decode batch in one compiled graph (mirrors ragged_gated_delta_rule),
plus the per-shard conv1d + SSD body and its shard_map wrapper.

Per-token recurrence (selective_state_update semantics):
    dt    = softplus(dt + dt_bias)
    dA    = exp(dt * A),  A = -exp(A_log) (already negated in the layer)
    state = dA * state + dt * (x ⊗ B)
    y     = <state, C> + D * x
B/C are grouped (n_groups < n_heads) and expanded per head.
"""
import functools

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.ragged_conv1d_jax import ragged_conv1d
from tpu_inference.layers.common.sharding import ShardingAxisName


def ragged_mamba2_ssd(
    hidden_states,  # (num_tokens, H, P) post-conv x
    B,  # (num_tokens, n_groups, N)
    C,  # (num_tokens, n_groups, N)
    dt,  # (num_tokens, H)
    recurrent_state,  # (num_blocks, H, P, N) f32 paged cache
    A,  # (H,) f32, already -exp(A_log)
    dt_bias,  # (H,) f32
    D,  # (H,) f32
    query_start_loc,  # (max_reqs + 1,) i32
    state_indices,  # (max_reqs,) i32
    distribution,  # (3,) i32 (decode_end, prefill_end, mixed_end)
    has_initial_state,  # (max_reqs,) bool
    *,
    n_groups: int,
):
    """Ragged Mamba2 SSD recurrence over a mixed prefill+decode batch.

    Token-by-token scan: handles prefill (state builds across a sequence) and
    decode (single token) identically, indexed into the paged ssm_state cache by
    `state_indices`. Mirrors `ragged_gated_delta_rule`.
    """
    num_tokens, H, P_dim = hidden_states.shape
    heads_per_group = H // n_groups
    A = A.astype(jnp.float32)
    max_reqs = state_indices.shape[0]
    token_idx = jnp.arange(num_tokens)

    num_valid_seqs = distribution[2]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    last_valid_loc = query_start_loc[num_valid_seqs]
    eff_qsl = jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)
    req_indices = jnp.clip(
        jnp.sum(token_idx[:, None] >= eff_qsl[None, :], axis=1) - 1, 0,
        max_reqs - 1)
    valid_mask = token_idx < last_valid_loc

    # Zero the carry state for brand-new prefills (no prior context), once up
    # front, so the scan body can stay a simple per-token read of the cache.
    gathered = recurrent_state[state_indices]
    recurrent_state = recurrent_state.at[state_indices].set(
        jnp.where(has_initial_state[:, None, None, None], gathered,
                  jnp.zeros_like(gathered)))

    def expand(bc_t):  # (n_groups, N) -> (H, N)
        return jnp.repeat(bc_t, heads_per_group, axis=0)

    def scan_fn(state_all, xs):
        x_t, B_t, C_t, dt_t, req_i, is_valid = xs
        si = state_indices[req_i]
        st = state_all[si]  # (H, P, N) f32

        x32 = x_t.astype(jnp.float32)
        Be = expand(B_t).astype(jnp.float32)
        Ce = expand(C_t).astype(jnp.float32)
        dt_sp = jax.nn.softplus(dt_t.astype(jnp.float32) + dt_bias)
        dA = jnp.exp(dt_sp * A)

        dBx = dt_sp[:, None, None] * x32[:, :, None] * Be[:, None, :]
        new_st = dA[:, None, None] * st + dBx
        y = jnp.einsum("hpn,hn->hp", new_st, Ce) + D[:, None] * x32

        state_all = jnp.where(
            is_valid, state_all.at[si].set(new_st.astype(state_all.dtype)),
            state_all)
        return state_all, y.astype(x_t.dtype)

    xs = (hidden_states, B, C, dt, req_indices, valid_mask)
    new_state, y = jax.lax.scan(scan_fn, recurrent_state, xs)
    return new_state, y  # (num_blocks, H, P, N), (num_tokens, H, P)


def _run_local(x_l, B_l, C_l, dt, conv_state, ssm_state, cw_x, cw_B, cw_C, cb_x,
               cb_B, cb_C, A, dt_bias, D, query_start_loc, state_indices,
               distribution, seq_lens, *, n_groups, n_heads, head_dim, ssm_n,
               kernel_size):
    """Per-shard conv1d + SSD on a TP shard.

    x/B/C arrive as SEPARATE per-shard tensors (x head-sharded, B/C group-
    sharded) so this shard's heads and groups correspond; we re-concatenate them
    into this shard's [x_r | B_r | C_r] for the conv. See `mamba2_core_tpu` for
    why the concatenated [x|B|C] cannot be sharded as one P(ATTN_HEAD) split.
    """
    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    mixed_xBC = jnp.concatenate([x_l, B_l, C_l], axis=-1)
    conv_weight = jnp.concatenate([cw_x, cw_B, cw_C], axis=0)
    conv_bias = (jnp.concatenate([cb_x, cb_B, cb_C], axis=0)
                 if cb_x is not None else None)

    out_xBC, new_conv_state = ragged_conv1d(mixed_xBC,
                                            conv_state,
                                            conv_weight,
                                            conv_bias,
                                            query_start_loc,
                                            state_indices,
                                            distribution,
                                            has_initial_state,
                                            kernel_size=kernel_size)
    # vLLM's causal_conv1d_fn applies activation="silu" to the conv output;
    # ragged_conv1d does not, so apply it here (GDN does the same separately).
    out_xBC = jax.nn.silu(out_xBC)

    inter = n_heads * head_dim
    groups_ssm = n_groups * ssm_n
    x = out_xBC[:, :inter].reshape(-1, n_heads, head_dim)
    B = out_xBC[:, inter:inter + groups_ssm].reshape(-1, n_groups, ssm_n)
    C = out_xBC[:, inter + groups_ssm:].reshape(-1, n_groups, ssm_n)

    new_ssm, y = ragged_mamba2_ssd(x,
                                   B,
                                   C,
                                   dt,
                                   ssm_state,
                                   A,
                                   dt_bias,
                                   D,
                                   query_start_loc,
                                   state_indices,
                                   distribution,
                                   has_initial_state,
                                   n_groups=n_groups)
    y = y.reshape(-1, n_heads * head_dim)
    return (new_conv_state, new_ssm), y


def run_jax_mamba2(x_l, B_l, C_l, j_dt, conv_state, ssm_state, cw_x, cw_B, cw_C,
                   cb_x, cb_B, cb_C, j_A, j_dt_bias, j_D, query_start_loc,
                   state_indices, distribution, seq_lens, *, n_groups, n_heads,
                   head_dim, ssm_n, kernel_size, mesh):
    """shard_map wrapper over `_run_local` (ATTN_DATA x ATTN_HEAD)."""
    H = P(ShardingAxisName.ATTN_HEAD)
    DH = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD)
    D_ = P(ShardingAxisName.ATTN_DATA)
    cb_spec = H if cb_x is not None else None
    in_specs = (
        DH,
        DH,
        DH,  # x_l, B_l, C_l
        DH,  # dt
        P(ShardingAxisName.ATTN_DATA, None,
          ShardingAxisName.ATTN_HEAD),  # conv_state
        P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None,
          None),  # ssm_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # cw_x
        P(ShardingAxisName.ATTN_HEAD, None, None),  # cw_B
        P(ShardingAxisName.ATTN_HEAD, None, None),  # cw_C
        cb_spec,
        cb_spec,
        cb_spec,  # cb_x/B/C
        H,
        H,
        H,  # A, dt_bias, D
        D_,
        D_,
        D_,
        D_,  # qsl, state_indices, distribution, seq_lens
    )
    out_specs = (
        (P(ShardingAxisName.ATTN_DATA, None, ShardingAxisName.ATTN_HEAD),
         P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None, None)),
        DH,
    )
    local = functools.partial(_run_local,
                              n_groups=n_groups,
                              n_heads=n_heads,
                              head_dim=head_dim,
                              ssm_n=ssm_n,
                              kernel_size=kernel_size)
    return jax.shard_map(local,
                         mesh=mesh,
                         in_specs=in_specs,
                         out_specs=out_specs,
                         check_vma=False)(x_l, B_l, C_l, j_dt, conv_state,
                                          ssm_state, cw_x, cw_B, cw_C, cb_x,
                                          cb_B, cb_C, j_A, j_dt_bias, j_D,
                                          query_start_loc, state_indices,
                                          distribution, seq_lens)
