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
"""Wrapper for ragged gated delta rule implementations."""

import dataclasses
import enum

import jax
import jax.numpy as jnp

from tpu_inference.kernels.gdn import triangle_solver
from tpu_inference.kernels.gdn.fused_gdn_kernel_wrapper import \
    ragged_gated_delta_rule as fused_impl
from tpu_inference.kernels.gdn.recurrent_scan_v2 import recurrent_scan
from tpu_inference.layers.common import \
    ragged_gated_delta_rule_chunked as jax_impl


@dataclasses.dataclass(frozen=True)
class RaggedGatedDeltaRuleConfig:
    prefill_impl: str = 'jax'
    decode_impl: str = 'jax'
    use_qk_norm_in_gdn: bool = True


class RaggedGatedDeltaRuleImpl(enum.Enum):
    """Implementation options for the ragged gated delta rule."""
    REF = 'ref'
    CHUNKED_JAX_PD = 'chunked_jax_pd'
    CHUNKED_KERNEL_PD = 'chunked_kernel_pd'
    CHUNKED_KERNEL_P_JAX_D = 'chunked_kernel_p_jax_d'
    CHUNKED_KERNEL_P_RECURRENT_KERNEL_D = 'chunked_kernel_p_recurrent_kernel_d'
    RECURRENT_KERNEL_PD = 'recurrent_kernel_pd'

    @property
    def prefill_impl(self) -> str:
        if self in (
                RaggedGatedDeltaRuleImpl.REF,
                RaggedGatedDeltaRuleImpl.CHUNKED_JAX_PD,
        ):
            return 'jax'
        elif self == RaggedGatedDeltaRuleImpl.RECURRENT_KERNEL_PD:
            return 'fused'
        else:
            return 'recurrent_scan_v2'

    @property
    def decode_impl(self) -> str:
        if self in (
                RaggedGatedDeltaRuleImpl.REF,
                RaggedGatedDeltaRuleImpl.CHUNKED_JAX_PD,
                RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_P_JAX_D,
        ):
            return 'jax'
        else:
            return 'fused'

    def to_config(self) -> RaggedGatedDeltaRuleConfig:
        return RaggedGatedDeltaRuleConfig(
            prefill_impl=self.prefill_impl,
            decode_impl=self.decode_impl,
        )


@jax.jit(
    # because , recurrent_scan_v2 call pltpu.get_tpu_info().num_lanes
    donate_argnames=('recurrent_state', ),
    static_argnames=(
        'config',
        'chunk_size',
        'triangle_solver_impl',
        'n_kq',
        'n_v',
        'd_k',
        'd_v',
    ),
)
@jax.named_scope('ragged_gated_delta_rule_wrapper')
def ragged_gated_delta_rule_wrapper(
    config: RaggedGatedDeltaRuleConfig,
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray,
    *,
    chunk_size: int,
    triangle_solver_impl: triangle_solver.TriangleSolverImpl = triangle_solver.
    TriangleSolverImpl.GAUSSIAN,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies the gated delta rule over ragged seq lengths using various implementations.

  This function separates mixed QKV, handles repeating for multi-query attention
  if needed, and routes to either the decode-only or mixed-prefill branch
  depending on sequence lengths and config.

  Args:
    config: Configuration for ragged gated delta rule.
    mixed_qkv: Mixed query, key, value tensor.
    b: b tensor (for beta).
    a: a tensor (for g).
    recurrent_state: Recurrent state tensor of shape `(num_blocks, n_v, d_k,
      d_v)`. `num_blocks` is always equal or larger than `max_seqs + 1`. The
      first block is a null_block and only used for padded / invalid tokens.
    A_log: A_log tensor.
    dt_bias: dt_bias tensor.
    query_start_loc: Start locations of sequences.
    state_indices: Indices mapping sequences to recurrent state slots.
    distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
      mixed_end)`.
    has_initial_state: Boolean tensor of shape `(max_reqs,)`. ``True`` when the
      request has prior recurrent state to use (chunked-prefill continuation or
      prefix-cache hit). ``False`` for brand-new prefills, in which case the
      gathered recurrent state is treated as zeros — matching GPU's
      `initial_state[~has_initial_state, ...] = 0` in
      `gdn_linear_attn._forward_core`.
    chunk_size: Chunk size for padding.
    triangle_solver_impl: Which triangle solver implementation to use.
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key/query dimension.
    d_v: Value dimension.

  Returns:
    A tuple containing:
      - updated_recurrent_state: Updated recurrent state tensor of shape
        `(num_blocks, n_v, d_k, d_v)`.
      - output: Output tensor.
  """
    is_decode_only = distribution[0] == distribution[2]

    def decode_only_branch(_):
        impl = config.decode_impl
        if impl == 'fused':
            qkv_in = jax.nn.silu(mixed_qkv)
            new_state, output = fused_impl(
                mixed_qkv=qkv_in,
                b=b,
                a=a,
                recurrent_state=recurrent_state,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                distribution=distribution,
                has_initial_state=has_initial_state,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
            )
            return new_state, output
        elif impl == 'jax':
            qkv_in = jax.nn.silu(mixed_qkv)
            num_tokens = qkv_in.shape[0]
            key_dim = n_kq * d_k
            query = qkv_in[..., :key_dim]
            key = qkv_in[..., key_dim:key_dim * 2]
            value = qkv_in[..., key_dim * 2:]
            q_reshaped = query.reshape(num_tokens, n_kq, d_k)
            k_reshaped = key.reshape(num_tokens, n_kq, d_k)
            v_reshaped = value.reshape(num_tokens, n_v, d_v)
            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q_reshaped = jnp.repeat(q_reshaped, repeat_factor, axis=1)
                k_reshaped = jnp.repeat(k_reshaped, repeat_factor, axis=1)
            b_reshaped = b.reshape(num_tokens, n_v)
            a_reshaped = a.reshape(num_tokens, n_v)

            new_state, output = jax_impl.ragged_gated_delta_rule_decode_only(
                query=q_reshaped,
                key=k_reshaped,
                value=v_reshaped,
                b_reshaped=b_reshaped,
                a_reshaped=a_reshaped,
                recurrent_state=recurrent_state,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                distribution=distribution,
                use_qk_norm_in_gdn=config.use_qk_norm_in_gdn,
            )
            return new_state, output.astype(mixed_qkv.dtype)
        else:
            raise ValueError(f'Unknown decode_impl: {impl}')

    def mixed_prefill_branch(_):
        impl = config.prefill_impl
        if impl == 'jax':
            qkv_in = jax.nn.silu(mixed_qkv)
            num_tokens = qkv_in.shape[0]
            key_dim = n_kq * d_k
            query = qkv_in[..., :key_dim]
            key = qkv_in[..., key_dim:key_dim * 2]
            value = qkv_in[..., key_dim * 2:]
            q_reshaped = query.reshape(num_tokens, n_kq, d_k)
            k_reshaped = key.reshape(num_tokens, n_kq, d_k)
            v_reshaped = value.reshape(num_tokens, n_v, d_v)
            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q_reshaped = jnp.repeat(q_reshaped, repeat_factor, axis=1)
                k_reshaped = jnp.repeat(k_reshaped, repeat_factor, axis=1)
            b_reshaped = b.reshape(num_tokens, n_v)
            a_reshaped = a.reshape(num_tokens, n_v)
            return jax_impl.ragged_gated_delta_rule_mixed_prefill(
                query=q_reshaped,
                key=k_reshaped,
                value=v_reshaped,
                b_reshaped=b_reshaped,
                a_reshaped=a_reshaped,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                recurrent_state=recurrent_state,
                state_indices=state_indices,
                distribution=distribution,
                chunk_size=chunk_size,
                use_qk_norm_in_gdn=config.use_qk_norm_in_gdn,
                triangle_solver_impl=triangle_solver_impl,
                has_initial_state=has_initial_state,
            )
        elif impl == 'recurrent_scan_v2':
            return recurrent_scan(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                recurrent_state=recurrent_state,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                distribution=distribution,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                chunk_size=chunk_size,
                BT=chunk_size,
                use_qk_norm_in_gdn=config.use_qk_norm_in_gdn,
                has_initial_state=has_initial_state,
            )
        elif impl == 'fused':
            qkv_in = jax.nn.silu(mixed_qkv)
            return fused_impl(
                mixed_qkv=qkv_in,
                b=b,
                a=a,
                recurrent_state=recurrent_state,
                A_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                distribution=distribution,
                has_initial_state=has_initial_state,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
            )
        else:
            raise ValueError(f'Unknown prefill_impl: {impl}')

    return jax.lax.cond(is_decode_only,
                        decode_only_branch,
                        mixed_prefill_branch,
                        operand=None)
