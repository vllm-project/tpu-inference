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

import jax
import jax.numpy as jnp
import torch
# NOTE: we don't specify this in our requirements.txt but it should be coming
# from upstream vLLM
from einops import rearrange
from torchax.interop import jax_view, torch_view
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn_linear_attn import \
    GatedDeltaNetAttention

from tpu_inference import envs
from tpu_inference.layers.common.gdn_attention import (
    GdnAttentionConfig, RaggedGatedDeltaRuleImpl, run_jax_gdn_attention)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product


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
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    j_mixed_qkv = reorder_concatenated_tensor_for_sharding(
        j_mixed_qkv, [key_dim, key_dim, value_dim], tp_size, -1)
    j_conv_weight = reorder_concatenated_tensor_for_sharding(
        j_conv_weight, [key_dim, key_dim, value_dim], tp_size, 0)

    layer_idx = vllm_context.layer_name_to_kvcache_index[layer_name]
    conv_state, recurrent_state = vllm_context.kv_caches[layer_idx]
    state_len = conv_state.shape[1]
    if state_len > kernel_size - 1:
        conv_state_in = conv_state[:, :kernel_size - 1, :]
    else:
        conv_state_in = conv_state

    # Index mamba state by the per-request slot id from
    # `InputBatch.mamba_state_indices_cpu`, not by `block_tables[:, 0]`
    # (vLLM's GPU convention). Two reasons:
    #
    #  1. `_maybe_set_compact_mamba_num_blocks_override` caps the mamba
    #     pool at `max_num_seqs + 1` while the attention pool is much
    #     larger; using `block_tables[:, 0]` (a value in the attention
    #     range) would walk off the end of the mamba arrays.
    #  2. When vLLM's input batch runs `condense` to compact the persistent
    #     batch (https://github.com/vllm-project/vllm/blob/de3da0b/vllm/v1/worker/gpu_input_batch.py#L662 — moves
    #     requests into lower-index slots after earlier ones finish), the
    #     slot id moves with the request so the kernel still reads/writes
    #     the slot that holds this request's real state.
    state_indices = jax_view(attn_metadata.mamba_state_indices).astype(
        jnp.int32)

    # Map tokens to their respective requests
    q_loc = jax_view(attn_metadata.query_start_loc)
    distribution = jax_view(attn_metadata.request_distribution)
    j_seq_lens = jax_view(attn_metadata.seq_lens)

    config = GdnAttentionConfig(
        ragged_gated_delta_rule_impl=RaggedGatedDeltaRuleImpl(
            envs.RAGGED_GATED_DELTA_RULE_IMPL))

    (new_conv_state_extracted,
     new_recurrent_state), j_output = run_jax_gdn_attention(j_mixed_qkv,
                                                            j_b,
                                                            j_a,
                                                            conv_state_in,
                                                            recurrent_state,
                                                            j_conv_weight,
                                                            j_conv_bias,
                                                            j_A_log,
                                                            j_dt_bias,
                                                            state_indices,
                                                            q_loc,
                                                            distribution,
                                                            j_seq_lens,
                                                            n_kq,
                                                            n_v,
                                                            d_k,
                                                            d_v,
                                                            kernel_size,
                                                            mesh=mesh,
                                                            config=config)
    if state_len > kernel_size - 1:
        remaining_old_state = conv_state[:, kernel_size - 1:, :]
        new_conv_state = jnp.concatenate(
            [new_conv_state_extracted, remaining_old_state], axis=1)
    else:
        new_conv_state = new_conv_state_extracted

    vllm_context.kv_caches[layer_idx] = (new_conv_state, new_recurrent_state)

    j_output_flat = j_output.reshape(core_attn_out.shape)
    core_attn_out.copy_(torch_view(j_output_flat))


@GatedDeltaNetAttention.register_oot
class VllmGatedDeltaNetAttention(GatedDeltaNetAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Implements the exact same logic as in vLLM (https://github.com/vllm-project/vllm/blob/9c81f35/vllm/model_executor/layers/mamba/gdn_linear_attn.py#L508)
        but omits the reshape in Part 3 for z/core_attn_out that is causing an unnecessary all-gather.

        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        mesh = vllm_model_wrapper_context.mesh
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba)
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value))
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        gdn_attention_core_tpu(mixed_qkv,
                               b,
                               a,
                               core_attn_out,
                               self.prefix,
                               mesh=mesh)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)
