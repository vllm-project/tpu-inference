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
"""
Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl

"""
import functools
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.vllm.ops.ragged_conv1d_jax import ragged_conv1d
from tpu_inference.layers.vllm.ops.ragged_gated_delta_rule_jax import \
    ragged_gated_delta_rule
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def run_jax_gdn_attention_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the local JAX GDN attention mechanism with combined QKV tensors.

    Args:
        mixed_qkv: Combined QKV tensor of shape `(num_tokens, dim)`.
        b: B tensor of shape `(num_tokens, n_v)`.
        a: A tensor of shape `(num_tokens, n_v)`.
        conv_state: Combined convolutional state of shape `(max_reqs, kernel_size
          - 1, dim)`.
        recurrent_state: Recurrent state of shape `(max_reqs, n_v, d_k, d_v)`.
        conv_weight: Combined convolutional weight of shape `(dim, 1,
          kernel_size)`.
        conv_bias: Optional combined convolutional bias of shape `(dim,)`.
        A_log: Log of A parameter of shape `(n_v,)`.
        dt_bias: Delta T bias of shape `(n_v,)`.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.

    Returns:
        A tuple containing the new states and the output.
        - A tuple of (new_conv_state, new_recurrent_state).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    # Ensure query_start_loc is monotonically increasing to handle padded slots
    query_start_loc = jnp.maximum.accumulate(query_start_loc)

    out_mixed_qkv, new_conv_state = ragged_conv1d(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        kernel_size,
    )

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    new_recurrent_state, output = ragged_gated_delta_rule(
        out_mixed_qkv,
        b,
        a,
        recurrent_state,
        A_log,
        dt_bias,
        query_start_loc,
        state_indices,
        n_kq,
        n_v,
        d_k,
        d_v,
    )

    return (new_conv_state, new_recurrent_state), output


def run_jax_gdn_attention(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    mesh: jax.sharding.Mesh,
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs the Jax GDN attention mechanism.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        conv_state: Convolutional state tensor of shape `(max_reqs, kernel_size -
          1, dim)`.
        recurrent_state: Recurrent state tensor of shape `(max_reqs, n_v, d_k,
          d_v)`.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.

    Returns:
        A tuple containing the new states and the output.
        - A tuple of (new_conv_state, new_recurrent_state).
          - new_conv_state: `(max_reqs, kernel_size - 1, dim)`
          - new_recurrent_state: `(max_reqs, n_v, d_k, d_v)`
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    in_specs = (
        P(None, ShardingAxisName.ATTN_HEAD),  # j_mixed_qkv
        P(None, ShardingAxisName.ATTN_HEAD),  # j_b
        P(None, ShardingAxisName.ATTN_HEAD),  # j_a
        P(None, None, ShardingAxisName.ATTN_HEAD),  # conv_state
        P(None, ShardingAxisName.ATTN_HEAD, None, None),  # recurrent_state
        P(ShardingAxisName.ATTN_HEAD, None, None),  # j_conv_weight
        P(ShardingAxisName.ATTN_HEAD)
        if j_conv_bias is not None else None,  # j_conv_bias
        P(ShardingAxisName.ATTN_HEAD),  # j_A_log
        P(ShardingAxisName.ATTN_HEAD),  # j_dt_bias
        P(),  # query_start_loc
        P(),  # state_indices
    )

    out_specs = (
        (
            P(None, None, ShardingAxisName.ATTN_HEAD),  # new_conv_state
            P(None, ShardingAxisName.ATTN_HEAD, None,
              None),  # new_recurrent_state
        ),
        P(None, ShardingAxisName.ATTN_HEAD),  # output
    )

    tp_size = mesh.shape[ShardingAxisName.ATTN_HEAD]

    p_run_jax_gdn_attention_local = functools.partial(
        run_jax_gdn_attention_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )

    (new_conv_state, new_recurrent_state), output = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        conv_state,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
    )

    return (new_conv_state, new_recurrent_state), output


def gdn_attention_core_tpu(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
    mesh: jax.sharding.Mesh,
) -> None:
    """
    This acts as main bridge between PyTorch and JAX for the GDN core attention.
    Uses a robust, token-by-token scan to inherently handle any mix of
    ragged prefill and decode sequences without dynamic shape compilation errors.

    Some key details:
    1. Cache Mapping: We'll read vLLM's  `block_tables` and `query_start_loc`
       and translate them into static index arrays (`req_indices` and `state_indices`).
    2. JAX Scan: We use `jax.lax.scan` to perform a robust, token-by-token loop over
       the flat inputs. This allows us to handle ANY mix of prefill and decode tokens
       in a single compiled XLA graph.
    3. Conditional Updates: The `valid_mask` ensures that padded dummy tokens
       (used to keep the tensor shape static) do not corrupt the recurrent state
       in the cache.
    """
    fc = get_forward_context()
    attn_metadata = fc.attn_metadata[layer_name]

    layer_module = fc.no_compile_layers[layer_name]
    vllm_context = get_vllm_model_wrapper_context()

    n_kq = layer_module.num_k_heads
    n_v = layer_module.num_v_heads
    d_k = layer_module.head_k_dim
    d_v = layer_module.head_v_dim
    kernel_size = layer_module.conv_kernel_size

    j_mixed_qkv = jax_view(mixed_qkv)  # [num_tokens, dim]
    j_b = jax_view(b)
    j_a = jax_view(a)

    j_conv_weight = jax_view(layer_module.conv1d.weight)
    j_conv_bias = jax_view(layer_module.conv1d.bias
                           ) if layer_module.conv1d.bias is not None else None
    j_A_log = jax_view(layer_module.A_log)
    j_dt_bias = jax_view(layer_module.dt_bias)

    # The j_mixed_qkv and j_conv_weight are not in an interleaved layout.
    # E.g. they are in [Q Q | K K | V V] layout. We need [Q K | Q K | Q K] layout.
    # Use reorder_concatenated_tensor_for_sharding to reorder into correct layout
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    tp_size = mesh.shape[ShardingAxisName.ATTN_HEAD]
    j_mixed_qkv = reorder_concatenated_tensor_for_sharding(
        j_mixed_qkv, [key_dim, key_dim, value_dim], tp_size, -1)
    j_conv_weight = reorder_concatenated_tensor_for_sharding(
        j_conv_weight, [key_dim, key_dim, value_dim], tp_size, 0)

    layer_idx = vllm_context.layer_name_to_kvcache_index[layer_name]
    conv_state, recurrent_state = vllm_context.kv_caches[layer_idx]

    # Map physical cache blocks
    flat_block_tables = jax_view(attn_metadata.block_tables)
    max_reqs = attn_metadata.seq_lens.shape[0]
    max_blocks_per_req = flat_block_tables.shape[0] // max_reqs
    block_tables_2d = jnp.reshape(flat_block_tables,
                                  (max_reqs, max_blocks_per_req))
    state_indices = block_tables_2d[:, 0].astype(jnp.int32)

    # Map tokens to their respective requests
    q_loc = jax_view(attn_metadata.query_start_loc)

    (new_conv_state,
     new_recurrent_state), j_output = run_jax_gdn_attention(j_mixed_qkv,
                                                            j_b,
                                                            j_a,
                                                            conv_state,
                                                            recurrent_state,
                                                            j_conv_weight,
                                                            j_conv_bias,
                                                            j_A_log,
                                                            j_dt_bias,
                                                            state_indices,
                                                            q_loc,
                                                            n_kq,
                                                            n_v,
                                                            d_k,
                                                            d_v,
                                                            kernel_size,
                                                            mesh=mesh)

    vllm_context.kv_caches[layer_idx] = (new_conv_state, new_recurrent_state)

    j_output_flat = j_output.reshape(core_attn_out.shape)
    core_attn_out.copy_(torch_view(j_output_flat))


def gdn_in_proj_tpu(
    hidden_states: torch.Tensor,
    qkvz_size: int,
    ba_size: int,
    prefix: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch OP replacement for gdn_in_proj.
    Executes the underlying linear layers (in_proj_qkvz and in_proj_ba) directly
    to mimic the old/non-kernel execution path that was reverted in:
    https://github.com/vllm-project/vllm/pull/36795

    Args:
        hidden_states: Tensor of shape (num_tokens, hidden_size)
        qkvz_size: int, unused but needed for signature
        ba_size: int, unused but needed for signature
        prefix: str
    Returns:
        mixed_qkvz: Tensor of shape (num_tokens, qkvz_size)
        ba: Tensor of shape (num_tokens, ba_size)
    """
    fc = get_forward_context()
    # The 'prefix' argument perfectly matches the key used to register the module
    layer_module = fc.no_compile_layers[prefix]

    # Run the original projections instead of the fused C++ kernel
    mixed_qkvz, _ = layer_module.in_proj_qkvz(hidden_states)
    ba, _ = layer_module.in_proj_ba(hidden_states)

    return mixed_qkvz, ba


def apply_gated_delta_net_torch_ops_patch(mesh: jax.sharding.Mesh) -> None:
    """
    This is a patch to inject the `gdn_attention_core` op so the
    Torch/GPU  kernel is bypassed in favor of the TPU kernel
    here:
    https://github.com/vllm-project/vllm/blob/697e4ff3528c72806a4d00ed9b7581332b9efd43/vllm/model_executor/models/qwen3_next.py#L671

    """
    try:
        import vllm.model_executor.models.qwen3_next  # noqa: F401
    except ImportError:
        pass

    # Ensure the op exists in the namespace, which initializes the OpOverloadPacket
    if hasattr(torch.ops, "vllm") and hasattr(torch.ops.vllm,
                                              "gdn_attention_core"):
        # dummy call to ensure the op is registered
        torch.ops.vllm.gdn_attention_core = functools.partial(
            gdn_attention_core_tpu, mesh=mesh)

    if hasattr(torch.ops.vllm, "gdn_in_proj"):
        # dummy call to ensure the op is registered
        torch.ops.vllm.gdn_in_proj = gdn_in_proj_tpu
