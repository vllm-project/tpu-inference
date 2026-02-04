# Copyright 2025 Google LLC
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

import math
import os
import re
from dataclasses import InitVar, dataclass
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flax.typing import PRNGKey, Sharding
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Float
from torchax.ops.mappings import j2t_dtype
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.kernels.mla.v1.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes import \
    get_tuned_block_sizes
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import (quantize_kv,
                                                      u8_unpack_e2m1)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import (Embedder, FlaxUtils, LMhead,
                                             RMSNorm)
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.moe.utils import (get_expert_parallelism,
                                                select_moe_backend)
from tpu_inference.layers.jax.quantization.unquantized import UnquantizedConfig
from tpu_inference.layers.jax.rope import DeepseekScalingRotaryEmbedding
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (BaseWeightLoader,
                                                         get_param,
                                                         print_param_info)

KVCache = Tuple[jax.Array, jax.Array]

logger = init_logger(__name__)

# A map from JAX dtype to the corresponding PyTorch integer dtype for raw memory viewing.
DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
}

modeling_flax_utils = FlaxUtils()


@dataclass(kw_only=True)
class DeepseekV3BaseAttention(nnx.Module):
    """
    Base class containing shared logic for DeepSeek Attention mechanisms.
    Handles initialization of common layers and defines skeleton forward pass.
    """
    # Core configuration
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: dict[str, Any]
    dtype: jnp.dtype
    kv_cache_dtype: str
    mesh: Mesh

    # Attention-specific configuration
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float

    # Sharding
    rd_sharding: Sharding = ()
    q_da_sharding: Sharding = ()
    ap_sharding: Sharding = ()
    kv_da_sharding: Sharding = ()
    activation_attention_td: Sharding = ()
    activation_q_td: Sharding = ()
    query_tnh: P = P()
    keyvalue_skh: P = P()
    attn_o_tnh: P = P()
    activation_attention_out_td: Sharding = ()

    # Weight initialization
    random_init: bool = False
    rope_mscale_all_dim: float = 1.0

    # RNG for weight initialization
    rngs: InitVar[nnx.Rngs]

    # Scales for Q/KV quantization (per-tensor)
    _q_scale: float = 1
    _k_scale: float = 1
    _v_scale: float = 1

    def __post_init__(self, rngs: nnx.Rngs):
        self.N = self.num_attention_heads
        self.K = self.num_key_value_heads
        self.D = self.hidden_size
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.rope_scaling["factor"] <= 1.0:
            yarn_mscale = 1.0
        else:
            yarn_mscale = 0.1 * self.rope_mscale_all_dim * math.log(
                self.rope_scaling["factor"]) + 1.0
        self.scale = self.qk_head_dim**-0.5 * yarn_mscale**2

        self.rope = DeepseekScalingRotaryEmbedding(
            rotary_dim=self.qk_rope_head_dim,
            rope_theta=self.rope_theta,
            original_max_position_embeddings=self.
            rope_scaling["original_max_position_embeddings"],
            scaling_factor=self.rope_scaling["factor"],
            dtype=self.dtype,
            beta_fast=self.rope_scaling["beta_fast"],
            beta_slow=self.rope_scaling["beta_slow"],
            mscale_value=self.rope_scaling["mscale"],
            mscale_all_dim=self.rope_scaling["mscale_all_dim"],
        )

        self.kernel_q_down_proj_DA = create_param(rngs,
                                                  (self.D, self.q_lora_rank),
                                                  self.q_da_sharding,
                                                  self.dtype,
                                                  random_init=self.random_init)

        self.kernel_q_up_proj_AP = create_param(
            rngs, (self.q_lora_rank, self.N * self.qk_head_dim),
            self.ap_sharding,
            self.dtype,
            random_init=self.random_init)

        self.kernel_kv_down_proj_DA = create_param(
            rngs, (self.D, self.kv_lora_rank + self.qk_rope_head_dim),
            self.kv_da_sharding,
            self.dtype,
            random_init=self.random_init)

        self.kernel_o_proj_RD = create_param(
            rngs, (self.N * self.v_head_dim, self.D),
            self.rd_sharding,
            self.dtype,
            random_init=self.random_init)

        self.q_rms_norm = RMSNorm(dims=self.q_lora_rank,
                                  epsilon=self.rms_norm_eps,
                                  with_scale=True,
                                  dtype=self.dtype,
                                  random_init=self.random_init,
                                  rngs=rngs)

        self.kv_rms_norm = RMSNorm(dims=self.kv_lora_rank,
                                   epsilon=self.rms_norm_eps,
                                   with_scale=True,
                                   dtype=self.dtype,
                                   random_init=self.random_init,
                                   rngs=rngs)

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                self.kv_cache_dtype)

        self.setup_specific_layers(rngs)

    def setup_specific_layers(self, *args) -> None:
        pass

    def compute_q_projection(self, *args) -> jax.Array:
        raise NotImplementedError

    def compute_kv_projection(self, *args) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    def compute_attention(self, *args) -> Tuple[KVCache, jax.Array]:
        raise NotImplementedError

    def process_output(self, outputs_TNH) -> jax.Array:
        return outputs_TNH

    def __call__(
            self, x: jax.Array, is_prefill: bool, kv_cache: KVCache,
            attention_metadata: AttentionMetadata
    ) -> Tuple[KVCache, jax.Array]:
        """Performs the forward pass of the attention module.  Expects that the
        child class has implemented the `compute_q_projection`, `compute_kv_projection`,
        and `compute_attention` methods.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """

        md = attention_metadata
        x = jnp.asarray(x, self.dtype)
        x_SD = nnx.with_sharding_constraint(x, self.activation_attention_td)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td)

        with jax.named_scope("q_proj"):
            q_data = self.compute_q_projection(x_q_TD, md.input_positions)

        with jax.named_scope("kv_proj"):
            kv_data = self.compute_kv_projection(x_SD, md.input_positions)

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.compute_attention(
                q_data, kv_data, is_prefill, kv_cache, md)

            outputs_TNH = self.process_output(outputs_TNH)

            if outputs_TNH.shape[-1] != self.v_head_dim:
                outputs_TNH = outputs_TNH[..., :self.v_head_dim]

            with jax.named_scope("o_proj"):
                outputs_TR = outputs_TNH.reshape(outputs_TNH.shape[0],
                                                 self.N * self.v_head_dim)
                o_TD = jnp.einsum("TR,RD -> TD", outputs_TR,
                                  self.kernel_o_proj_RD.value)
                o_TD = nnx.with_sharding_constraint(
                    o_TD, self.activation_attention_out_td)

            return new_kv_cache, o_TD


@dataclass(kw_only=True)
class DeepseekV3Attention(DeepseekV3BaseAttention):
    """Standard Multi-Head Attention (MHA) for DeepSeek models."""

    def setup_specific_layers(self, rngs: nnx.Rngs) -> None:
        self.kernel_kv_up_proj_AL = create_param(
            rngs,
            (self.kv_lora_rank, self.N *
             (self.qk_nope_head_dim + self.v_head_dim)),
            self.ap_sharding,
            self.dtype,
            random_init=self.random_init,
        )

    def compute_q_projection(self, x_q_TD: jax.Array,
                             input_positions: jax.Array) -> jax.Array:
        """
        Computes the query projection for MHA.

        Args:
            x_q_TD: The input tensor of shape `(tokens_query, d_model)`.
            input_positions: The input positions tensor of shape `(padded_total_num_scheduled_tokens,)`.

        Returns:
            The query tensor of shape `(tokens_query, num_query_heads, head_dim)`.
        """
        q_TA = jnp.einsum("TD,DA -> TA", x_q_TD,
                          self.kernel_q_down_proj_DA.value)
        q_TA = self.q_rms_norm(q_TA)
        q_TP = jnp.einsum("TA,AP -> TP", q_TA, self.kernel_q_up_proj_AP.value)
        q_TNH = q_TP.reshape(q_TA.shape[0], self.N, self.qk_head_dim)

        q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = self.rope.apply_rope(input_positions, q_rope_TNH)
        q_TNH = jnp.concatenate([q_nope_TNH, q_rope_TNH], axis=-1)

        return nnx.with_sharding_constraint(q_TNH, self.query_tnh)

    def compute_kv_projection(
            self, x_SD: jax.Array,
            input_positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the key-value projection for MHA.

        Args:
            x_SD: The input tensor of shape `(tokens_kv, d_model)`.
            input_positions: The input positions tensor of shape `(padded_total_num_scheduled_tokens,)`.

        Returns:
            Tuple of key-value tensors of shape `(tokens_kv, num_query_heads, d_model)`.
        """

        kv_SA = jnp.einsum("SD,DA -> SA", x_SD,
                           self.kernel_kv_down_proj_DA.value)

        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        k_rope_SNH = k_rope_SH[..., None, :]
        k_rope_SNH = self.rope.apply_rope(input_positions, k_rope_SNH)
        assert k_rope_SNH.shape[1] == 1

        k_rope_SNH = jnp.broadcast_to(
            k_rope_SNH, (k_rope_SNH.shape[0], self.N, self.qk_rope_head_dim))

        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_rms_norm(kv_SA)
        kv_SA = nnx.with_sharding_constraint(kv_SA, self.keyvalue_skh)

        kv_SL = jnp.einsum("SA,AL -> SL", kv_SA,
                           self.kernel_kv_up_proj_AL.value)
        kv_nope_SNH = kv_SL.reshape(kv_SA.shape[0], self.N,
                                    self.qk_nope_head_dim + self.v_head_dim)

        k_nope_SNH = kv_nope_SNH[..., :self.qk_nope_head_dim]
        v_SNH = kv_nope_SNH[..., self.qk_nope_head_dim:]

        k_SNH = jnp.concatenate([k_nope_SNH, k_rope_SNH], axis=-1)

        # Shard
        k_SNH = nnx.with_sharding_constraint(k_SNH, self.keyvalue_skh)
        v_SNH = nnx.with_sharding_constraint(v_SNH, self.keyvalue_skh)

        return (k_SNH, v_SNH)

    def compute_attention(self, q_data: jax.Array, kv_data: Tuple[jax.Array,
                                                                  jax.Array],
                          is_prefill: bool, kv_cache: KVCache,
                          md: AttentionMetadata) -> Tuple[jax.Array, KVCache]:
        """
        Computes self-attention for MHA.

        Args:
            q_data: The query tensor of shape `(tokens_query, num_query_heads, head_dim)`.
            kv_data: Tuple of key-value tensors of shape `(tokens_kv, num_query_heads, d_model)`.
            is_prefill: Whether the attention is for prefill or not.
            kv_cache: KVCache object.
            md: AttentionMetadata object.

        Returns:
            Tuple of output tensors of shape `(tokens_query, num_query_heads, head_dim)` and KVCache object.
        """

        q_TNH = q_data
        k_SNH, v_SNH = kv_data

        multiple_of_128 = ((self.qk_head_dim - 1) // 128 + 1) * 128
        q_TNH = jnp.pad(q_TNH, ((0, 0), (0, 0),
                                (0, multiple_of_128 - self.qk_head_dim)))
        k_SNH = jnp.pad(k_SNH, ((0, 0), (0, 0),
                                (0, multiple_of_128 - self.qk_head_dim)))
        v_SNH = jnp.pad(v_SNH, ((0, 0), (0, 0),
                                (0, multiple_of_128 - self.v_head_dim)))

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k_SNH, v_SNH = quantize_kv(self.kv_cache_quantized_dtype, k_SNH,
                                       v_SNH, k_scale, v_scale)

        def _ragged_paged_attention(q, k, v, cache, seq_lens, block_tables,
                                    starts, dist):
            return ragged_paged_attention(q,
                                          k,
                                          v,
                                          cache,
                                          seq_lens,
                                          block_tables,
                                          starts,
                                          dist,
                                          sm_scale=self.scale,
                                          q_scale=q_scale,
                                          k_scale=k_scale,
                                          v_scale=v_scale)

        in_specs = (
            self.query_tnh,  # q
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # v
            P(None, None, ShardingAxisName.ATTN_HEAD),  # kv_cache
            P(),  # md.seq_lens: Replicated
            P(),  # page_indices_flat: Replicated
            P(),  # query_start_loc: Replicated
            P(),  # distribution: Replicated
        )

        out_specs = (self.attn_o_tnh, P(None, None,
                                        ShardingAxisName.ATTN_HEAD))

        output_TNH, kv_cache = jax.jit(
            jax.shard_map(_ragged_paged_attention,
                          mesh=self.mesh,
                          in_specs=in_specs,
                          out_specs=out_specs,
                          check_vma=False))(q_TNH, k_SNH, v_SNH, kv_cache,
                                            md.seq_lens, md.block_tables,
                                            md.query_start_loc,
                                            md.request_distribution)

        return kv_cache, output_TNH


@dataclass(kw_only=True)
class DeepseekV3MLA(DeepseekV3BaseAttention):
    """Multi-Head Latent Attention (MLA) for DeepSeek V3."""
    anh_sharding: Sharding = ()

    def setup_specific_layers(self, rngs: nnx.Rngs) -> None:
        self.kernel_k_up_proj_ANH = create_param(
            rngs, (self.kv_lora_rank, self.N, self.qk_nope_head_dim),
            self.anh_sharding,
            self.dtype,
            random_init=self.random_init)

        self.kernel_v_up_proj_ANH = create_param(
            rngs, (self.kv_lora_rank, self.N, self.v_head_dim),
            self.anh_sharding,
            self.dtype,
            random_init=self.random_init)

    def compute_q_projection(
            self, x_q_TD: jax.Array,
            input_positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the query projection for MLA.

        Args:
            x_q_TD: The input tensor of shape `(tokens_query, d_model)`.
            input_positions: The input positions tensor of shape `(padded_total_num_scheduled_tokens,)`.

        Returns:
            A tuple of query tensor of shape `(tokens_query, num_query_heads, q_lora_rank)` and
            rope tensor of shape `(tokens_query, num_query_heads, head_dim)`.
        """
        q_TA = jnp.einsum("TD,DA -> TA", x_q_TD,
                          self.kernel_q_down_proj_DA.value)
        q_TA = self.q_rms_norm(q_TA)
        q_TP = jnp.einsum("TA,AP -> TP", q_TA, self.kernel_q_up_proj_AP.value)
        q_TNH = q_TP.reshape(q_TA.shape[0], self.N, self.qk_head_dim)

        q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = self.rope.apply_rope(input_positions, q_rope_TNH)

        q_TNA = jnp.einsum("TNH,ANH -> TNA", q_nope_TNH,
                           self.kernel_k_up_proj_ANH.value)

        q_TNA = nnx.with_sharding_constraint(q_TNA, self.query_tnh)
        return (q_TNA, q_rope_TNH)

    def compute_kv_projection(
            self, x_SD: jax.Array,
            input_positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the key-value projection for MLA.

        Args:
            x_SD: The input tensor of shape `(tokens_kv, d_model)`.
            input_positions: The input positions tensor of shape `(padded_total_num_scheduled_tokens,)`.

        Returns:
            A tuple of key-value tensor of shape `(tokens_kv, q_lora_rank)` and
            rope tensor of shape `(tokens_kv, head_dim)`.
        """
        kv_SA = jnp.einsum("SD,DA -> SA", x_SD,
                           self.kernel_kv_down_proj_DA.value)

        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        k_rope_SNH = k_rope_SH[..., None, :]
        k_rope_SNH = self.rope.apply_rope(input_positions, k_rope_SNH)
        assert k_rope_SNH.shape[1] == 1
        k_rope_SH = k_rope_SNH[:, 0, :]

        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_rms_norm(kv_SA)
        kv_SA = nnx.with_sharding_constraint(kv_SA, self.keyvalue_skh)

        return (kv_SA, k_rope_SH)

    def compute_attention(self, q_data: Tuple[jax.Array, jax.Array],
                          kv_data: Tuple[jax.Array, jax.Array],
                          is_prefill: bool, kv_cache: KVCache,
                          md: AttentionMetadata) -> Tuple[KVCache, jax.Array]:
        """
        Computes the attention for MLA.

        Args:
            q_data: A tuple of query tensor of shape `(tokens_query, num_query_heads, q_lora_rank)` and
                rope tensor of shape `(tokens_query, num_query_heads, head_dim)`.
            kv_data: A tuple of key-value tensor of shape `(tokens_kv, q_lora_rank)` and
                rope tensor of shape `(tokens_kv, head_dim)`.
            is_prefill: A boolean indicating whether to use prefill or not.
            kv_cache: The key-value cache.
            md: The attention metadata.

        Returns:
            A tuple of key-value cache and output tensor of shape `(tokens_query, num_query_heads, q_lora_rank)`.
        """

        q_TNA, q_rope_TNH = q_data
        k_SA, k_rope_SH = kv_data

        in_specs = (
            self.query_tnh,  # q
            self.query_tnh,  # q_rope
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # k_rope
            P(ShardingAxisName.MLP_TENSOR),  # kv_cache
            P(ShardingAxisName.ATTN_DATA),  # md.seq_lens: Replicated
            P(ShardingAxisName.ATTN_DATA),  # page_indices_flat: Replicated
            P(ShardingAxisName.ATTN_DATA),  # query_start_loc: Replicated
            P(ShardingAxisName.ATTN_DATA),  # distribution: Replicated
        )
        out_specs = (self.attn_o_tnh, P(ShardingAxisName.MLP_TENSOR))

        def _mla_ragged_paged_attention(q, q_rope, k, k_rope, cache, *args):
            max_num_tokens = q.shape[0]
            max_num_seqs = md.seq_lens.shape[0]
            pages_per_seq = md.block_tables.shape[0] // max_num_seqs

            bkv_p, bq_sz = get_tuned_block_sizes(q.dtype, cache.dtype,
                                                 self.num_attention_heads, 1,
                                                 self.qk_nope_head_dim,
                                                 cache.shape[1],
                                                 max_num_tokens, pages_per_seq)
            num_kv_pages_per_block = min(min(pages_per_seq, bkv_p), 4)
            num_queries_per_block = min(min(max_num_tokens, bq_sz), 4)

            out, new_cache = mla_ragged_paged_attention(
                q,
                q_rope,
                k,
                k_rope,
                cache,
                *args,
                sm_scale=self.scale,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block)
            return new_cache, out

        kv_cache, output_TNA = jax.jit(
            jax.shard_map(_mla_ragged_paged_attention,
                          mesh=self.mesh,
                          in_specs=in_specs,
                          out_specs=out_specs,
                          check_vma=False))(q_TNA, q_rope_TNH, k_SA, k_rope_SH,
                                            kv_cache, md.seq_lens,
                                            md.block_tables,
                                            md.query_start_loc,
                                            md.request_distribution)

        return kv_cache, output_TNA

    def process_output(self, outputs_TNA: jax.Array) -> jax.Array:
        """
        Processes output for MLA specifically.

        Args:
            outputs_TNH: The output tensor of shape `(tokens_query, num_query_heads, q_lora_rank)`.

        Returns:
            The processed output tensor of shape `(tokens_query, num_query_heads, head_dim)`.
        """

        # MLA Specific: Apply V-Up Projection after attention
        # Outputs from MLA kernel are in latent space (TNA), project to TNH
        outputs_TNH = jnp.einsum("TNA,ANH -> TNH", outputs_TNA,
                                 self.kernel_v_up_proj_ANH.value)
        return outputs_TNH


@dataclass(kw_only=True)
class DeepseekV3MLP(nnx.Module):
    """A Gated Feed-Forward Network (FFN) layer.

    This module consists of two linear projections (gating and up-projection),
    an element-wise multiplication of the activated gating projection and the
    up-projection, followed by a final downward projection.

    Attributes:
        sharding_cfg: The configuration for tensor sharding.
    """
    dtype: jnp.dtype
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    df_sharding: Sharding = ()
    fd_sharding: Sharding = ()
    activation_ffw_td: Sharding = ()
    random_init: bool = False

    rngs: InitVar[nnx.Rngs]

    def __call__(self, x_TD):
        """Performs the forward pass of the FFW layer.

        Args:
            x_TD: The input tensor of shape either `(sequence, d_model)`

        Returns:
            The output tensor of shape `(batch, sequence, d_model)`.
        """
        # TODO: refactor to use JaxEinsum
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("wi_0"):
            gating_TF = jnp.einsum('TD,DF -> TF', x_TD,
                                   self.kernel_gating_DF.value)
            activated_gating_TF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TF)
        with jax.named_scope("wi_1"):
            up_proj_TF = jnp.einsum('TD,DF -> TF', x_TD,
                                    self.kernel_up_proj_DF.value)
        fuse_TF = activated_gating_TF * up_proj_TF
        with jax.named_scope("wo"):
            output_TD = jnp.einsum('TF,FD -> TD', fuse_TF,
                                   self.kernel_down_proj_FD.value)

        return output_TD

    def __post_init__(self, rngs: nnx.Rngs):
        D = self.hidden_size
        F = self.intermediate_size

        # TODO: replace this with JaxEinsums
        self.kernel_gating_DF = create_param(rngs,
                                             shape=(D, F),
                                             dtype=self.dtype,
                                             sharding=self.df_sharding,
                                             random_init=self.random_init)
        self.kernel_up_proj_DF = create_param(rngs,
                                              shape=(D, F),
                                              dtype=self.dtype,
                                              sharding=self.df_sharding,
                                              random_init=self.random_init)
        self.kernel_down_proj_FD = create_param(rngs,
                                                shape=(F, D),
                                                dtype=self.dtype,
                                                sharding=self.fd_sharding,
                                                random_init=self.random_init)


@dataclass(kw_only=True)
class DeepseekV3MoE(nnx.Module):
    """
    Corresponds to vLLM's DeepseekV2MoE.
    Handles the routed and shared experts + the relevant forward pass.

    Reference here: https://github.com/vllm-project/vllm/blob/2b465570e6dd327e8422ef9c87e9b2b1454ceaed/vllm/model_executor/models/deepseek_v2.py#L223
    """
    experts: JaxMoE
    shared_experts: Optional[DeepseekV3MLP] = None

    routed_scaling_factor: float = 1.0

    def __call__(self, x_TD: jax.Array) -> jax.Array:
        # Compute Routed Experts
        final_hidden_states = self.experts(x_TD)

        # (Maybe) Compute Shared Experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x_TD)
            final_hidden_states += shared_output

        return final_hidden_states


@dataclass(kw_only=True)
class DeepseekV3DecoderLayer(nnx.Module):
    """
    Implementats the DecoderLayer for DeepseekV3.
    """
    layer_idx: int
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    self_attn: Union[DeepseekV3Attention, DeepseekV3MLA]

    # MLP can be either the Dense MLP (for first k layers) or DeepseekV2MoE
    # TODO: rename to mlp? custom_module seems needlessly confusing
    custom_module: nnx.Module | DeepseekV3MoE | DeepseekV3MLP

    def __call__(
        self, x_TD: jax.Array, is_prefill: bool, kv_cache: List[jax.Array],
        attention_metadata: AttentionMetadata
    ) -> Tuple[List[jax.Array], jax.Array]:

        # Run Self-Attention
        residual = x_TD
        hidden_states = self.input_layernorm(x_TD)
        new_cache, attn_output = self.self_attn(hidden_states, is_prefill,
                                                kv_cache, attention_metadata)
        hidden_states = residual + attn_output

        # Run MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        custom_module_output = self.custom_module(hidden_states)

        # Residual
        hidden_states = residual + custom_module_output

        return new_cache, hidden_states


@dataclass
class DeepSeekV3Router(nnx.Module):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.
    """

    hidden_size: int
    num_experts: int
    num_experts_per_tok: int
    n_groups: int
    topk_groups: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    dtype: jnp.dtype
    rngs: InitVar[nnx.Rngs]

    # Sharding Attributes
    activation_ffw_td: Sharding = ()
    ed_sharding: Sharding = ()
    e_sharding: Sharding = ()

    random_init: bool = False

    router_bias_dtype: jnp.dtype = jnp.float32

    moe_backend: MoEBackend = MoEBackend.DENSE_MAT

    def get_topk_indices(self, scores_TE: Float) -> Float:
        """Get the topk indices of the scores.

        Args:
            scores_TE: The scores to get the topk indices of. Shape (sequence, num_experts).

        Returns:
            The topk indices of the scores. Shape (sequence, num_experts_per_tok).
        """

        scores_TE = scores_TE + self.bias_E
        if self.n_groups > 1:
            experts_per_group = self.num_experts // self.n_groups
            group_scores_TGM = jnp.reshape(
                scores_TE, (-1, self.n_groups, experts_per_group))
            group_scores_TG2 = jax.lax.top_k(group_scores_TGM, k=2)[0]
            group_scores_TG = jnp.sum(group_scores_TG2, axis=-1)
            indices = jax.lax.top_k(group_scores_TG, k=self.topk_groups)[1]

            mask_TG = jnp.any(jnp.arange(
                self.n_groups)[:, None] == indices[..., None, :],
                              axis=-1)
            mask_TE = jnp.repeat(mask_TG,
                                 scores_TE.shape[-1] // mask_TG.shape[-1], -1)
            scores_TE = jnp.where(mask_TE, scores_TE, 0.0)

        indices_TX = jax.lax.top_k(scores_TE, k=self.num_experts_per_tok)[1]

        return indices_TX

    def __call__(self, x_TD: Float) -> Tuple[Float, Float]:
        """Routes tokens to top k experts.

        Args:
            x_TD: Input array of shape (sequence, d_model).

        Returns:
            A tuple containing:
                - weights: Normalized weights for selected experts, shape (sequence, num_experts_per_tok).
                - indices: Indices of selected experts, shape (sequence, num_experts_per_tok).
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)

        scores_TE = jnp.einsum("TD,DE -> TE", x_TD, self.kernel_DE.value)
        scores_TE = nnx.sigmoid(scores_TE)

        if self.moe_backend in MoEBackend.fused_moe_backends():
            return scores_TE

        original_scores_TE = scores_TE
        topk_indices_TX = self.get_topk_indices(scores_TE)
        weights_TX = jnp.take_along_axis(original_scores_TE,
                                         topk_indices_TX,
                                         axis=-1)

        if self.norm_topk_prob:
            weights_TX /= jnp.sum(weights_TX, axis=-1)[..., None] + 1e-20

        weights_TX *= self.routed_scaling_factor

        return weights_TX, topk_indices_TX

    def __post_init__(self, rngs: nnx.Rngs):
        """Generates the router kernel (weights and bias) for routing."""
        D = self.hidden_size
        E = self.num_experts
        # TODO: replace this with a JaxEinsum
        self.kernel_DE = create_param(rngs,
                                      shape=(D, E),
                                      dtype=self.dtype,
                                      sharding=self.ed_sharding,
                                      random_init=self.random_init)
        self.bias_E = create_param(rngs,
                                   shape=(E, ),
                                   dtype=self.router_bias_dtype,
                                   sharding=self.e_sharding,
                                   random_init=self.random_init)


@dataclass
class DeepSeekV3WeightLoader(BaseWeightLoader):

    def __init__(self,
                 vllm_config: VllmConfig,
                 num_layers,
                 hidden_size,
                 q_lora_rank,
                 kv_lora_rank,
                 attn_heads,
                 qk_nope_head_dim,
                 qk_rope_head_dim,
                 v_head_dim,
                 num_local_experts,
                 model_dtype,
                 moe_backend,
                 use_mla_kernel=False):
        super().__init__(vllm_config, framework="pt")
        self.num_layers = num_layers
        self.is_verbose = vllm_config.additional_config.get(
            "is_verbose", None) is not None
        self.num_routed_experts = num_local_experts
        self.attn_heads = attn_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.model_dtype = model_dtype
        self.use_mla_kernel = use_mla_kernel
        self.moe_backend = moe_backend

        self._transpose_map = {
            # dense mlp
            r"mlp\.down_proj": (1, 0),
            r"mlp\.gate_proj": (1, 0),
            r"mlp\.up_proj": (1, 0),
            # mla
            r"q_a_proj": (1, 0),
            r"q_b_proj": (1, 0),
            r"kv_a_proj_with_mqa": (1, 0),
            r"kv_b_proj": (1, 0),
            r"k_b_proj": (2, 0, 1),  # used for MLA kernel
            r"v_b_proj": (2, 0, 1),  # used for MLA kernel
            r"o_proj": (1, 0),
            # moe
            r"mlp\.gate\.weight": (1, 0),
            r"mlp\.experts\.\d+\.gate_proj": (0, 2, 1),
            r"mlp\.experts\.\d+\.down_proj": (0, 2, 1),
            r"mlp\.experts\.\d+\.up_proj": (0, 2, 1),
            r"mlp\.shared_experts\.down_proj": (1, 0),
            r"mlp\.shared_experts\.gate_proj": (1, 0),
            r"mlp\.shared_experts\.up_proj": (1, 0),
            # lm_head
            r"lm_head\.weight": (1, 0)
        }

        # Set the mappings from loaded parameter keys to standardized names.
        self._loaded_to_standardized_keys = {
            # encode & decode
            "model.embed_tokens.weight":
            "embedder.input_embedding_table_VD",
            "lm_head.weight":
            "lm_head.input_embedding_table_DV",
            # final norm
            "model.norm.weight":
            "final_norm.scale",
            # norm in transformer blocks
            "model.layers.*.input_layernorm.weight":
            "layers.*.input_layernorm.scale",
            "model.layers.*.post_attention_layernorm.weight":
            "layers.*.post_attention_layernorm.scale",
            # attention (MLA)
            "model.layers.*.self_attn.q_a_layernorm.weight":
            "layers.*.self_attn.q_rms_norm.scale",
            "model.layers.*.self_attn.kv_a_layernorm.weight":
            "layers.*.self_attn.kv_rms_norm.scale",
            "model.layers.*.self_attn.q_a_proj.weight":
            "layers.*.self_attn.kernel_q_down_proj_DA",
            "model.layers.*.self_attn.q_b_proj.weight":
            "layers.*.self_attn.kernel_q_up_proj_AP",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight":
            "layers.*.self_attn.kernel_kv_down_proj_DA",
            "model.layers.*.self_attn.kv_b_proj.weight":
            "layers.*.self_attn.kernel_kv_up_proj_AL",
            "model.layers.*.self_attn.o_proj.weight":
            "layers.*.self_attn.kernel_o_proj_RD",
            # Dense ffw
            "model.layers.*.mlp.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_DF",
            "model.layers.*.mlp.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_DF",
            "model.layers.*.mlp.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_FD",
            # MOE(routed experts) - Nested under .experts now
            "model.layers.*.mlp.gate.weight":
            "layers.*.custom_module.experts.router.kernel_DE",
            "model.layers.*.mlp.gate.e_score_correction_bias":
            "layers.*.custom_module.experts.router.bias_E",
            "model.layers.*.mlp.experts.*.gate_proj.weight":
            "layers.*.custom_module.experts.kernel_gating_EDF",
            "model.layers.*.mlp.experts.*.down_proj.weight":
            "layers.*.custom_module.experts.kernel_down_proj_EFD",
            "model.layers.*.mlp.experts.*.up_proj.weight":
            "layers.*.custom_module.experts.kernel_up_proj_EDF",
            "model.layers.*.mlp.experts.*.gate_upproj_fused.weight":
            "layers.*.custom_module.experts.kernel_gating_upproj_E2DF",
            # MOE(shared experts) - Nested under .shared_experts inside custom_module
            "model.layers.*.mlp.shared_experts.down_proj.weight":
            "layers.*.custom_module.shared_experts.kernel_down_proj_FD",
            "model.layers.*.mlp.shared_experts.gate_proj.weight":
            "layers.*.custom_module.shared_experts.kernel_gating_DF",
            "model.layers.*.mlp.shared_experts.up_proj.weight":
            "layers.*.custom_module.shared_experts.kernel_up_proj_DF",
        }

        if self.use_mla_kernel:
            self._loaded_to_standardized_keys.update({
                "model.layers.*.self_attn.k_b_proj.weight":
                "layers.*.self_attn.kernel_k_up_proj_ANH",
                "model.layers.*.self_attn.v_b_proj.weight":
                "layers.*.self_attn.kernel_v_up_proj_ANH",
            })
        # TODO (jacobplatin): we should not be hard-coding these
        self.scale_dtype, self.quant_dtype = jnp.bfloat16, jnp.float8_e4m3fn

        self.is_model_quantized = not vllm_config.additional_config.get(
            "skip_quantization", False)

        # TODO (jacobplatin): remove this once the JAX path refactor is done
        self.is_native_fp8_model = False

        if self.is_model_quantized:
            # NOTE: this is only needed for pre-quantized models when doing random weight loading
            # because the scales that Qwix configures by default don't necessarily match the
            # scales in practice
            # TODO (jacobplatin): remove or clean this up
            self.scale_shape_map_for_random_weight_loading = {
                # MoE experts (3D)
                "custom_module.experts.kernel_down_proj_EFD": (256, 1, 7168),
                "custom_module.experts.kernel_gating_EDF": (256, 28, 2048),
                "custom_module.experts.kernel_up_proj_EDF": (256, 28, 2048),
                # Shared experts (2D)
                "custom_module.shared_experts.kernel_down_proj_FD": (1, 7168),
                "custom_module.shared_experts.kernel_gating_DF": (28, 2048),
                "custom_module.shared_experts.kernel_up_proj_DF": (28, 2048),
                # Dense FFW (2D)
                "custom_module.kernel_gating_DF": (28, 18432),
                "custom_module.kernel_up_proj_DF": (28, 18432),
                "custom_module.kernel_down_proj_FD": (72, 7168),
                # Attention (3D for MLA, 2D for the rest)
                "self_attn.kernel_q_down_proj_DA": (28, 1536),
                "self_attn.kernel_q_up_proj_AP": (6, 24576),
                "self_attn.kernel_kv_down_proj_DA": (28, 576),
                "self_attn.kernel_kv_up_proj_AL": (2, 32768),
                "self_attn.kernel_o_proj_RD": (64, 7168),
                "self_attn.kernel_k_up_proj_ANH": (2, 128, 128),  # MLA
                "self_attn.kernel_v_up_proj_ANH": (2, 128, 128),  # MLA
            }

            # TODO (jacobplatin): remove this check eventually!
            assert self.quant_dtype == jnp.float8_e4m3fn, f"Expected quant_dtype to be float8_e4m3fn for DeepSeek but got {self.quant_dtype}"

            quantization_block_sizes = vllm_config.model_config.hf_config.quantization_config[
                "weight_block_size"]
            # TODO (jacobplatin): remove this once the JAX path refactor is done
            if quantization_block_sizes == [128, 128]:
                assert len(
                    quantization_block_sizes
                ) == 2, f"Expected only 2 quantization block sizes but got {quantization_block_sizes}"
                self.quantization_block_size_n = quantization_block_sizes[0]
                self.quantization_block_size_k = quantization_block_sizes[1]
                self.is_native_fp8_model = True

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            # extract layer number and replace it with *
            layer_num = re.search(r"layers\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
            # extract expert number if exists and replace it with *
            if "experts" in loaded_key and "shared_experts" not in loaded_key:
                layer_key = re.sub(r"experts\.\d+", "experts.*", layer_key)
            # get standardized key and replace * with layer number.
            mapped_key = self._loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self._loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def _transpose_params(self, param_key: str, param_tensor: jax.Array):
        for key, value in self._transpose_map.items():
            if re.search(key, param_key):
                return jnp.transpose(param_tensor, value)
        return param_tensor  # Base case / no-op

    def _process_moe_weights(self, loaded_name, loaded_weight, weights_dict):
        layer_num = re.search(r"layers\.(\d+)", loaded_name).group(1)
        expert_num_str = re.search(r"experts\.(\d+)", loaded_name).group(1)
        expert_idx = int(expert_num_str)

        if layer_num not in weights_dict:
            weights_dict[layer_num] = ([None] * self.num_routed_experts, 0)

        expert_list, count = weights_dict[layer_num]

        expert_list[expert_idx] = loaded_weight
        count += 1
        weights_dict[layer_num] = (expert_list, count)

        if count == self.num_routed_experts:
            stacked_weights = torch.stack(expert_list, axis=0)
            del weights_dict[layer_num]
            return stacked_weights
        return None

    def _load_individual_weight(self,
                                name,
                                weight,
                                model_params,
                                model_mesh,
                                scale=None) -> Tuple[int, int]:
        """
        Loads a single weight into the model.

        NOTE: if using the base quantized model, it is assumed that the Qwix abstract
        pass has been run and that the model weights are thus QArrays, which we
        will then load the weights/scales into.

        Args:
            name: The name of the weight.
            weight: The weight to load.
            model_params: The model parameters.
            model_mesh: The model mesh.
            scale: The scale of the weight (if using the pre-quantized model).

        Returns:
            Tuple[int, int]: The size (in bytes) for the given layer overall and per shard.
                NOTE: if using the pre-quantized model (with Qwix), we'll include the scale size as well.
        """
        mapped_name = self.map_loaded_to_standardized_name(name)
        base_model_weight = get_param(model_params, mapped_name)
        model_weight = base_model_weight.array.qvalue if hasattr(
            base_model_weight, "array") else base_model_weight
        sharding = base_model_weight.array.qvalue.sharding if hasattr(
            base_model_weight, "array") else base_model_weight.sharding

        # Convert weights from torch into numpy
        if weight.dtype == torch.uint8 and scale is not None:
            # Assume packed FP4 format when uint8 weights with scale provided
            weight_jax_u8 = jnp.array(weight.cpu().numpy())
            weight_np = u8_unpack_e2m1(weight_jax_u8)
            scale = scale.to(torch.float32).numpy().astype(self.scale_dtype)
        else:
            cast_type = model_weight.value.dtype
            # Special-case: FP4 values stored as FP8 for compatibility.
            # If the model expects float4_e2m1fn but the checkpoint provides FP8,
            # convert by numeric value (float32) then cast to float4.
            if cast_type == jnp.float4_e2m1fn and weight.dtype == torch.float8_e4m3fn:
                weight_np = jnp.array(weight.float().numpy()).astype(cast_type)
            else:
                torch_view_type = DTYPE_VIEW_MAP.get(jnp.dtype(cast_type))

                if torch_view_type:
                    # Avoid unnecessary upcasting and mem copy by viewing the tensor's
                    # raw data as integers before converting to a JAX array.
                    weight_np = jnp.array(
                        weight.view(torch_view_type).numpy()).view(cast_type)
                else:
                    raise ValueError(
                        f"Unsupported dtype for tensor conversion: {cast_type}"
                    )

            if scale is not None:
                scale = scale.to(torch.float32).numpy().astype(
                    self.scale_dtype)
        weight_np = self._transpose_params(name, weight_np)
        if scale is not None:
            scale = self._transpose_params(name, scale)
            # Ensure scale is broadcastable to weight_np by repeating per-axis.
            weight_shape = weight_np.shape
            scale_shape = scale.shape
            if len(weight_shape) == len(scale_shape):
                new_scale = scale
                # TODO (jacobplatin): remove once the refactor is complete
                if self.is_native_fp8_model:
                    for idx, (weight_dim, scale_dim) in enumerate(
                            zip(weight_shape, scale_shape)):
                        if weight_dim // self.quantization_block_size_n != scale_dim and weight_dim // scale_dim != 1:
                            old_scale_shape = scale.shape
                            scale = scale.repeat(
                                self.quantization_block_size_n,
                                axis=idx)[:, :weight_dim]
                            logger.warning(
                                f"Got a weight with shape {weight_shape} and scale with shape {old_scale_shape} "
                                f"where the scale_dim {scale_dim} does not match the weight_dim {weight_dim} "
                                f"multiplied by the quantization block size {self.quantization_block_size_n}. "
                                f"Repeating the scale to new shape {scale.shape} along axis {idx} with repeat size {self.quantization_block_size_n}."
                            )
                else:
                    for wdim, sdim in zip(weight_shape, scale_shape):
                        if (wdim % sdim != 0):
                            raise ValueError(
                                f"Weight dim {wdim} is not divisible by scale dim {sdim} for weight {name} with shape {weight_shape} and scale {scale_shape}!"
                            )
                    if scale_shape != new_scale.shape:
                        logger.warning(
                            f"Adjusted scale shape {scale_shape} to {new_scale.shape} to match weight {weight_shape}"
                        )
                    scale = new_scale
            else:
                raise ValueError(
                    f"Scale rank {scale_shape} does not match weight rank {weight_shape}"
                )
        if model_weight.value.shape != weight_np.shape:
            raise ValueError(
                f"Loaded shape for {name}: {weight_np.shape} "
                f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
            )

        def get_slice(index):
            return weight_np[index]

        def get_slice_scale(index):
            # ruff: noqa: F821
            return scale[index]

        sharded_array = jax.make_array_from_callback(
            weight_np.shape, NamedSharding(model_mesh, P(*sharding)),
            get_slice)

        if scale is not None:
            maybe_sharded_scale = scale
            # Since, by default, we'll use the same sharding as the weights, we might
            # encounter an error where the smaller/different sharding dim isn't divisible
            # by the parallel size
            # NOTE: Qwix expert dangyi@ mentioned that we don't need to worry about accuracy
            # impacts when sharing scales
            try:
                maybe_sharded_scale = jax.make_array_from_callback(
                    scale.shape, NamedSharding(model_mesh, P(*sharding)),
                    get_slice_scale)
            except ValueError:
                logger.warning(
                    f"Could not create sharded scale for {name} with shape {scale.shape} and sharding {sharding}, skipping sharding..."
                )
            assert base_model_weight.array.scale.value.dtype == maybe_sharded_scale.dtype, f"Expected dtype for model weight scale with name {mapped_name} and dtype ({base_model_weight.array.scale.value.dtype}) to match that of the incoming weight scale ({maybe_sharded_scale.dtype})"
            assert base_model_weight.array.qvalue.value.dtype == sharded_array.dtype, f"Expected dtype for model weight with name {mapped_name} and dtype ({base_model_weight.array.qvalue.value.dtype}) to match that of the incoming weight ({sharded_array.dtype})"
            base_model_weight.array.scale.value = maybe_sharded_scale
            base_model_weight.array.qvalue.value = sharded_array
        else:
            assert model_weight.value.dtype == sharded_array.dtype, f"Expected dtype for model weight with name {mapped_name} and dtype ({model_weight.value.dtype}) to match that of the incoming weight ({sharded_array.dtype})"
            model_weight.value = sharded_array

        model_weight_size_bytes = model_weight.nbytes / 1e9
        model_weight_local_size_bytes = model_weight.addressable_shards[
            0].data.nbytes / 1e9

        if scale is not None:
            model_weight_size_bytes += base_model_weight.array.scale.nbytes / 1e9
            model_weight_local_size_bytes += base_model_weight.array.scale.addressable_shards[
                0].data.nbytes / 1e9

        if self.is_verbose:
            logger.info(f"Memory usage after loading in {name}: "
                        f"hbm={utils.hbm_usage_gb(jax.local_devices())}Gb")
            print_param_info(model_weight, name)
            if scale is not None:
                print_param_info(base_model_weight.array.scale,
                                 "scale for " + name)

        del weight, scale
        return model_weight_size_bytes, model_weight_local_size_bytes

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        logger.warning(
            f"loaded_to_standardized_keys: {self._loaded_to_standardized_keys}"
        )
        cumulative_global_memory = 0
        cumulative_local_memory = 0
        mlp_experts_gate_proj_weights = {}
        mlp_experts_gate_proj_scales = {}
        mlp_experts_up_proj_weights = {}
        mlp_experts_up_proj_scales = {}
        mlp_experts_down_proj_weights = {}
        mlp_experts_down_proj_scales = {}
        stacked_tensors = {}
        quantized_weights = {}
        quantized_scales = {}
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.get_weights_iterator():
                # Skip if the model has fewer layers than original.
                if re.search(r"layers\.(\d+)", loaded_name):
                    layer_num = re.search(r"layers\.(\d+)",
                                          loaded_name).group(1)
                    if int(layer_num) >= self.num_layers:
                        del loaded_weight
                        continue
                if 'layers.61' in loaded_name:
                    # skip loading MTP module.
                    del loaded_weight
                    continue
                if re.search(r"experts\.(\d+)", loaded_name):
                    expert_num = re.search(r"experts\.(\d+)",
                                           loaded_name).group(1)
                    if int(expert_num) >= self.num_routed_experts:
                        del loaded_weight
                        continue
                # NOTE: loaded_weight will be a Torch tensor, so we need to convert it to the
                # equivalent jnp dtype
                # TODO (jacobplatin): refactor this so that we instead change / update `model_weights_generator`
                # instead of checking "weight_scale_inv" and assuming quantization method is fp8
                scale = None
                # Mixed quantization: accept both fp8 and packed fp4 (uint8) tensors
                allowed_quant_dtypes = {
                    j2t_dtype(self.quant_dtype.dtype), torch.uint8
                }
                if loaded_weight.dtype in allowed_quant_dtypes:
                    if self.is_model_quantized:
                        scale_name = loaded_name.replace(
                            ".weight", ".weight_scale_inv")
                        if scale_name in quantized_scales:
                            scale = quantized_scales[scale_name]
                            del quantized_scales[scale_name]
                        else:
                            quantized_weights[loaded_name] = loaded_weight
                            continue
                    else:
                        quantized_weights[loaded_name] = loaded_weight
                        continue

                if loaded_name.endswith(".weight_scale_inv"):
                    if self.is_model_quantized:
                        weight_name = loaded_name.replace(
                            ".weight_scale_inv", ".weight")
                        if weight_name in quantized_weights:
                            scale = loaded_weight
                            loaded_weight = quantized_weights[weight_name]
                            loaded_name = weight_name
                            del quantized_weights[weight_name]
                        else:
                            quantized_scales[loaded_name] = loaded_weight
                            continue
                    # In the case that we don't want to use the quantized weights,
                    # we'll dequantize the weights using the loaded scale on-the-fly
                    else:
                        # assuming weights are loaded before scales.
                        weight_name = loaded_name.replace(
                            ".weight_scale_inv", ".weight")
                        loaded_weight = weights_dequant_cpu(
                            quantized_weights[weight_name], loaded_weight,
                            self.model_dtype)
                        loaded_name = weight_name
                        del quantized_weights[weight_name]
                # concat mlp.experts weights
                stacked_scales = None
                stacked_weights = None
                if "mlp.experts" in loaded_name:
                    if "down_proj" in loaded_name:
                        proj_type = "down_proj"
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_down_proj_weights)
                        if scale is not None:
                            stacked_scales = self._process_moe_weights(
                                loaded_name, scale,
                                mlp_experts_down_proj_scales)
                    if "gate_proj" in loaded_name:
                        proj_type = "gate_proj"
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_gate_proj_weights)
                        if scale is not None:
                            stacked_scales = self._process_moe_weights(
                                loaded_name, scale,
                                mlp_experts_gate_proj_scales)
                    if "up_proj" in loaded_name:
                        proj_type = "up_proj"
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_up_proj_weights)
                        if scale is not None:
                            stacked_scales = self._process_moe_weights(
                                loaded_name, scale, mlp_experts_up_proj_scales)
                    if stacked_weights is not None:
                        stacked_tensors[layer_num +
                                        proj_type] = (stacked_weights,
                                                      stacked_scales)
                        if all(f"{layer_num}{p}" in stacked_tensors
                               for p in ["gate_proj", "up_proj", "down_proj"]):

                            gate_w, gate_s = stacked_tensors.pop(layer_num +
                                                                 "gate_proj")
                            up_w, up_s = stacked_tensors.pop(layer_num +
                                                             "up_proj")
                            down_w, down_s = stacked_tensors.pop(layer_num +
                                                                 "down_proj")

                            gate_name = loaded_name.replace(
                                proj_type, "gate_proj")
                            up_name = loaded_name.replace(proj_type, "up_proj")
                            down_name = loaded_name.replace(
                                proj_type, "down_proj")

                            weight_bytes, weight_shards = self._load_individual_weight(
                                gate_name,
                                gate_w,
                                model_params,
                                model_for_loading.mesh,
                                scale=gate_s)

                            weight_bytes_up, weight_shards_up = self._load_individual_weight(
                                up_name,
                                up_w,
                                model_params,
                                model_for_loading.mesh,
                                scale=up_s)
                            weight_bytes += weight_bytes_up
                            weight_shards += weight_shards_up

                            weight_bytes_down, weight_shards_down = self._load_individual_weight(
                                down_name,
                                down_w,
                                model_params,
                                model_for_loading.mesh,
                                scale=down_s)
                            weight_bytes += weight_bytes_down
                            weight_shards += weight_shards_down
                        else:
                            continue
                        if self.is_verbose:
                            cumulative_global_memory += weight_bytes
                            cumulative_local_memory += weight_shards
                            logger.info(
                                f"Cumulative global memory: {cumulative_global_memory} GB"
                            )
                            logger.info(
                                f"Cumulative local memory: {cumulative_local_memory} GB"
                            )
                    else:
                        continue
                else:
                    if self.use_mla_kernel and "kv_b_proj" in loaded_name:
                        # loaded_weight shape: (num_heads * (d_k + d_v), kv_lora_rank)
                        # scale shape: (num_heads * (d_k + d_v) / block_n, kv_lora_rank / block_k)
                        # Reshape to (num_heads, (d_k + d_v), kv_lora_rank) and split
                        weight_reshaped = loaded_weight.view(
                            self.attn_heads,
                            self.qk_nope_head_dim + self.v_head_dim,
                            self.kv_lora_rank)

                        k_weight = weight_reshaped[:, :self.
                                                   qk_nope_head_dim, :]
                        v_weight = weight_reshaped[:,
                                                   self.qk_nope_head_dim:, :]

                        loaded_weights_list = [k_weight, v_weight]
                        loaded_names = [
                            loaded_name.replace("kv_b_proj", "k_b_proj"),
                            loaded_name.replace("kv_b_proj", "v_b_proj")
                        ]

                        scales_list = [None, None]
                        if scale is not None:
                            # TODO (jacobplatin): remove once refactor happens
                            if self.is_native_fp8_model:
                                bn = self.quantization_block_size_n
                                bk = self.quantization_block_size_k
                                scale_reshaped = scale.view(
                                    self.attn_heads,
                                    (self.qk_nope_head_dim + self.v_head_dim)
                                    // bn, self.kv_lora_rank // bk)

                                k_scale = scale_reshaped[:, :self.
                                                         qk_nope_head_dim //
                                                         bn, :]
                                v_scale = scale_reshaped[:, self.
                                                         qk_nope_head_dim //
                                                         bn:, :]
                            else:
                                assert loaded_weight.shape[0] == scale.shape[0]
                                block_size_k = loaded_weight.shape[
                                    1] // scale.shape[1]
                                assert block_size_k > 0, f"Expected non-zero block size but got {block_size_k}!"
                                scale_reshaped = scale.view(
                                    self.attn_heads,
                                    (self.qk_nope_head_dim + self.v_head_dim),
                                    self.kv_lora_rank // block_size_k)

                                k_scale = scale_reshaped[:, :self.
                                                         qk_nope_head_dim, :]
                                v_scale = scale_reshaped[:, self.
                                                         qk_nope_head_dim:, :]
                            scales_list = [k_scale, v_scale]

                    else:
                        loaded_weights_list = [loaded_weight]
                        loaded_names = [loaded_name]
                        scales_list = [scale]

                    for loaded_name, loaded_weight, scale in zip(
                            loaded_names, loaded_weights_list, scales_list):

                        weight_bytes, weight_shards = self._load_individual_weight(
                            loaded_name,
                            loaded_weight,
                            model_params,
                            model_for_loading.mesh,
                            scale=scale)
                        if self.is_verbose:
                            cumulative_global_memory += weight_bytes
                            cumulative_local_memory += weight_shards
                            logger.info(
                                f"Cumulative global memory: {cumulative_global_memory} GB"
                            )
                            logger.info(
                                f"Cumulative local memory: {cumulative_local_memory} GB"
                            )

        del mlp_experts_gate_proj_weights
        del mlp_experts_up_proj_weights
        del mlp_experts_down_proj_weights
        del quantized_weights
        del quantized_scales
        # TODO: validate that all of the model_params were accounted for as well.
        nnx.update(model_for_loading, model_params)


@dataclass
class DeepSeekV3(nnx.Module):
    WeightLoader = DeepSeekV3WeightLoader

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: jax.Array,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)

        # NOTE: the default is 61
        num_layers: int = vllm_config.model_config.hf_config.num_hidden_layers
        num_local_experts: int = 256

        vocab_size: int = 129280
        hidden_size: int = 7168
        # NOTE: this dtype may be implicitly overriden if using to Qwix to load in the quantized weights
        dtype: jnp.dtype = jnp.bfloat16
        num_attention_heads: int = 128
        num_key_value_heads: int = 128
        ffw_intermediate_size: int = 18432
        moe_intermediate_size: int = 2048
        num_experts_per_token: int = 8
        n_group: int = 8
        interleave_moe_layer_step: int = 1  # Deepseek V3 has moe_layer_freq=1 in hf config.
        hidden_act: str = "silu"
        rms_norm_eps: float = 1e-06
        routed_scaling_factor: float = 2.5
        first_k_dense_replace: int = 3  # replace the first few MOE layers to dense layer.
        self.use_mla_kernel: bool = self.vllm_config.model_config.use_mla

        logger.info(f"Is using MLA kernel in DeepSeek: {self.use_mla_kernel}")

        num_shared_experts = 1
        rope_theta = 10000
        rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        }
        q_lora_rank = 1536
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128

        self.mesh = mesh

        # TODO (jacobplatin): this shouldn't be related to
        # the (DeepSeek) modelling code since it's really
        # MoE-specific, but because we do weight loading
        # here, we need to keep it for now.
        # TODO (jacobplatin): remove this in another PR
        edf_sharding = (None, ShardingAxisName.MODEL_1,
                        ShardingAxisName.MODEL_2)
        self.expert_axis_name = edf_sharding[0]
        self.num_expert_parallelism = get_expert_parallelism(
            self.expert_axis_name, self.mesh)
        self.use_ep = self.num_expert_parallelism > 1
        self.moe_backend = select_moe_backend(self.use_ep)

        # TODO (jacobplatin): temporary workaround for now before FP8 is fully ready for DeepSeek
        vllm_config.quant_config = UnquantizedConfig(
            vllm_config.model_config.hf_config.quantization_config)

        # TODO (jacobplatin): we will resolve this issue in a forthcoming PR that will refactor weight loading
        if vllm_config.load_config.load_format == "dummy" and self.moe_backend in MoEBackend.fused_moe_backends(
        ):
            raise ValueError(
                f"Random / dummy weights are not supported for {MoEBackend.fused_moe_backends()} backends right now."
            )

        self.weight_loader = self.WeightLoader(
            vllm_config=vllm_config,
            num_layers=num_layers,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            attn_heads=num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_local_experts=num_local_experts,
            model_dtype=dtype,
            moe_backend=self.moe_backend,
            use_mla_kernel=self.use_mla_kernel)

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 dtype=dtype,
                                 rngs=self.rng,
                                 vd_sharding=(ShardingAxisName.MLP_TENSOR,
                                              None))

        self.layers = []

        def _create_deepseek_attention(
        ) -> Union[DeepseekV3MLA, DeepseekV3Attention]:
            if self.use_mla_kernel:
                query_tnh_spec = P(ShardingAxisName.MLP_TENSOR, None, None)
                keyvalue_skh_spec = P(ShardingAxisName.MLP_TENSOR, None)
                attn_o_tnh_spec = P(ShardingAxisName.MLP_TENSOR, None, None)
                anh_sharding=(None, ShardingAxisName.MLP_TENSOR, None)
            else:
                query_tnh_spec=P(None, ShardingAxisName.MLP_TENSOR, None)
                keyvalue_skh_spec=P(None, ShardingAxisName.MLP_TENSOR, None)
                attn_o_tnh_spec=P(None, ShardingAxisName.MLP_TENSOR, None)
            rd_sharding=(ShardingAxisName.MLP_TENSOR, None)
            ap_sharding=(None, ShardingAxisName.MLP_TENSOR)
            q_da_sharding=(None, ShardingAxisName.VOCAB)
            kv_da_sharding=(None, ShardingAxisName.VOCAB)
            
            if self.vllm_config.additional_config.get(
                "replicate_attn_weights", False):
                rd_sharding=()
                ap_sharding=()
                q_da_sharding=()
                kv_da_sharding=()
                if self.use_mla_kernel:
                    anh_sharding=()


            attn_cls = None
            if self.use_mla_kernel:
                attn_cls = DeepseekV3MLA
            else:
                attn_cls = DeepseekV3Attention
                assert num_attention_heads == num_key_value_heads, "Expected same number of of attention heads and key value heads for MHA."

            kwargs = dict(
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                rms_norm_eps=rms_norm_eps,
                v_head_dim=v_head_dim,
                mesh=self.mesh,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=1
                if self.use_mla_kernel else num_key_value_heads,
                head_dim=v_head_dim,  # MLA uses v_head_dim as head_dim
                dtype=dtype,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                rngs=self.rng,
                activation_attention_td=(None, None),
                activation_q_td=(None, None),
                query_tnh=query_tnh_spec,
                keyvalue_skh=keyvalue_skh_spec,
                activation_attention_out_td=(None, None),
                attn_o_tnh=attn_o_tnh_spec,
                q_da_sharding=q_da_sharding,
                ap_sharding=ap_sharding,
                kv_da_sharding=kv_da_sharding,
                rd_sharding=rd_sharding)
            if self.use_mla_kernel:
                kwargs.update(anh_sharding=anh_sharding)

            return attn_cls(**kwargs)

        for i in range(num_layers):
            input_layernorm = RMSNorm(
                dims=hidden_size,
                epsilon=rms_norm_eps,
                with_scale=True,
                dtype=dtype,
                rngs=self.rng,
            )

            post_attention_layernorm = RMSNorm(
                dims=hidden_size,
                epsilon=rms_norm_eps,
                with_scale=True,
                dtype=dtype,
                rngs=self.rng,
            )

            # Logic to determine if this layer is Dense or MoE
            # * The first k layers are always dense.
            # * Subsequent layers are MoE if interleave_moe_layer_step conditions are met
            if i < first_k_dense_replace:
                is_moe_layer = False
            else:
                is_moe_layer = ((i + 1) % interleave_moe_layer_step == 0)

            if not is_moe_layer:
                # Dense Layer (used for first k layers or interleaved dense layers)
                mlp_layer = DeepseekV3MLP(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=ffw_intermediate_size,
                    rngs=self.rng,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                    df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                    fd_sharding=(ShardingAxisName.MLP_TENSOR, None))
            else:
                # MoE Layer
                moe_dtype = jnp.float8_e4m3fn if self.weight_loader.is_native_fp8_model else vllm_config.model_config.hf_config.quantization_config.get(
                    "tpu_settings", {}).get("mlp_dtype", jnp.float4_e2m1fn)

                router = DeepSeekV3Router(
                    hidden_size=hidden_size,
                    num_experts=num_local_experts,
                    num_experts_per_tok=num_experts_per_token,
                    n_groups=n_group,
                    topk_groups=4,
                    norm_topk_prob=True,
                    rngs=self.rng,
                    routed_scaling_factor=routed_scaling_factor,
                    dtype=dtype,
                    moe_backend=self.moe_backend,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                    ed_sharding=(None, None),
                    e_sharding=(None, ))

                # routed experts
                custom_module = JaxMoE(
                    dtype=dtype,
                    num_local_experts=num_local_experts,
                    apply_expert_weight_before_computation=False,
                    expert_axis_name=self.expert_axis_name,
                    num_expert_parallelism=self.num_expert_parallelism,
                    hidden_size=hidden_size,
                    intermediate_size_moe=moe_intermediate_size,
                    num_experts_per_tok=num_experts_per_token,
                    mesh=self.mesh,
                    hidden_act=hidden_act,
                    rngs=self.rng,
                    quant_config=self.vllm_config.quant_config,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA,
                                       ShardingAxisName.MOE_TENSOR),
                    activation_ffw_ted=(ShardingAxisName.MLP_DATA, None,
                                        ShardingAxisName.MOE_TENSOR),
                    edf_sharding=(None, ShardingAxisName.MOE_TENSOR,
                                  ShardingAxisName.ATTN_DATA_EXPERT),
                    efd_sharding=(None, ShardingAxisName.ATTN_DATA_EXPERT,
                                  ShardingAxisName.MOE_TENSOR),
                    moe_backend=self.moe_backend,
                    qwix_quantized_weight_dtype=moe_dtype
                    if self.weight_loader.is_model_quantized else None,
                    router=router)

                # shared experts
                shared_experts = DeepseekV3MLP(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=num_shared_experts *
                    moe_intermediate_size,
                    rngs=self.rng,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA,
                                       ShardingAxisName.MOE_TENSOR),
                    df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                    fd_sharding=(ShardingAxisName.MLP_TENSOR, None))

                mlp_layer = DeepseekV3MoE(
                    experts=custom_module,
                    shared_experts=shared_experts,
                    routed_scaling_factor=routed_scaling_factor,
                )

            block = DeepseekV3DecoderLayer(
                layer_idx=i,
                input_layernorm=input_layernorm,
                post_attention_layernorm=post_attention_layernorm,
                self_attn=_create_deepseek_attention(),
                custom_module=mlp_layer)

            self.layers.append(block)

        self.final_norm = RMSNorm(
            dims=hidden_size,
            rngs=self.rng,
            epsilon=rms_norm_eps,
            with_scale=True,
            dtype=dtype,
        )

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=hidden_size,
                              dtype=dtype,
                              rngs=self.rng,
                              vd_sharding=(ShardingAxisName.MLP_TENSOR, None),
                              dv_sharding=(None, ShardingAxisName.MLP_TENSOR))

        if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
            self._print_model_architecture()

    def _print_model_architecture(self):
        num_display_layers = 5

        logger.debug("### Embedding ###")
        nnx.display(self.embedder)

        logger.debug(f"\n### First {num_display_layers} Layers ###")
        # Loop through the slice and display each layer
        for i, layer in enumerate(self.layers[:num_display_layers]):
            logger.debug(f"\n--- Layer {i} ---")
            nnx.display(layer)

        logger.debug("\n### LM Head ###")
        nnx.display(self.lm_head)

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng)
        self.weight_loader.load_weights(self)
        self.initialize_cache()
        # TODO (jacobplatin): remove this once we switch to using JaxAutoWeightsLoader
        process_modules_after_loading(self, self.mesh)

    def initialize_cache(self):
        # Initialize RoPE caches after weights are loaded and before JIT compilation.
        for layer in self.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn,
                                                       'rope'):
                if hasattr(layer.self_attn.rope, 'initialize_cache'):
                    layer.self_attn.rope.initialize_cache(self.mesh)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]:
        is_prefill = False
        x = self.embedder.encode(input_ids)
        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x = block(x, is_prefill, kv_cache,
                                    attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)

        return kv_caches, final_activation, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head.decode(hidden_states)


def process_modules_after_loading(module, mesh):
    """Recursively call process_weights_after_loading on modules with quant_method.

    TODO (jacobplatin): remove this once we switch to using JaxAutoWeightsLoader
    """
    # Process this module if it has a quant_method
    if hasattr(module, 'quant_method') and module.quant_method is not None:
        if hasattr(module.quant_method, 'process_weights_after_loading'):
            module.quant_method.process_weights_after_loading(module, mesh)

    for name, value in vars(module).items():
        if isinstance(value, nnx.Module):
            process_modules_after_loading(value, mesh)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, nnx.Module):
                    process_modules_after_loading(item, mesh)


def weights_dequant_cpu(x: torch.Tensor,
                        s: torch.Tensor,
                        output_dtype: jnp.dtype,
                        block_size: int = 128) -> torch.Tensor:
    assert x.dim() == 2 and s.dim() == 2, "Both x and s must be 2D tensors"
    M, N = x.shape

    x = x.to(torch.float32)
    s = s.to(torch.float32)
    y = torch.empty_like(x)

    M_main = (M // block_size) * block_size
    N_main = (N // block_size) * block_size

    if M_main > 0 and N_main > 0:
        x_main = x[:M_main, :N_main]
        s_main = s[:(M // block_size), :(N // block_size)]

        x_reshaped = x_main.view(M // block_size, block_size, N // block_size,
                                 block_size).permute(0, 2, 1, 3)
        s_reshaped = s_main.view(M // block_size, N // block_size, 1, 1)
        y_main = (x_reshaped * s_reshaped).permute(0, 2, 1,
                                                   3).reshape(M_main, N_main)

        y[:M_main, :N_main] = y_main

    if N_main < N:
        for i in range(0, M_main, block_size):
            block = x[i:i + block_size, N_main:N]
            scale = s[i // block_size, N // block_size]
            y[i:i + block_size, N_main:N] = block * scale

    if M_main < M:
        for j in range(0, N, block_size):
            block = x[M_main:M, j:j + block_size]
            scale = s[M // block_size, j // block_size]
            y[M_main:M, j:j + block_size] = block * scale

    return y.to(j2t_dtype(jnp.dtype(output_dtype)))
