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
from dataclasses import InitVar, dataclass
from itertools import islice
from typing import Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flax.typing import Sharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Float
from torchax.ops.mappings import j2t_dtype
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.kernels.mla.v1.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.kernels.ragged_paged_attention.v3.tuned_block_sizes import \
    get_tuned_block_sizes
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.base import create_param, sharded_initializer
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.layers import FlaxUtils, RMSNorm
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.moe.utils import (get_expert_parallelism,
                                                select_moe_backend)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.layers.jax.rope import DeepseekScalingRotaryEmbedding
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator

KVCache = Tuple[jax.Array, jax.Array]

logger = init_logger(__name__)
init_fn = nnx.initializers.uniform()


def _weight_init(random_init: bool):
    return sharded_initializer if random_init else nnx.initializers.uniform()


# A map from JAX dtype to the corresponding PyTorch integer dtype for raw memory viewing.
DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
}

modeling_flax_utils = FlaxUtils()

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


@dataclass(kw_only=True)
class DeepseekV3BaseAttention(JaxModule):
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

    quant_config: Optional[QuantizationConfig] = None

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

        weight_init = _weight_init(self.random_init)

        self.q_down_proj = JaxEinsum(
            einsum_str="TD,DA->TA",
            kernel_shape=(self.D, self.q_lora_rank),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.q_da_sharding),
        )

        self.q_up_proj = JaxEinsum(
            einsum_str="TA,AP->TP",
            kernel_shape=(self.q_lora_rank, self.N * self.qk_head_dim),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ap_sharding),
        )

        self.kv_down_proj = JaxEinsum(
            einsum_str="SD,DA -> SA",
            kernel_shape=(self.D, self.kv_lora_rank + self.qk_rope_head_dim),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init,
                                              self.kv_da_sharding),
        )

        self.o_proj = JaxEinsum(
            einsum_str="TR,RD->TD",
            kernel_shape=(self.N * self.v_head_dim, self.D),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.rd_sharding),
        )

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
            self, x: jax.Array, kv_cache: KVCache,
            attention_metadata: AttentionMetadata
    ) -> Tuple[KVCache, jax.Array]:
        """Performs the forward pass of the attention module.  Expects that the
        child class has implemented the `compute_q_projection`, `compute_kv_projection`,
        and `compute_attention` methods.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
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
                q_data, kv_data, kv_cache, md)

            outputs_TNH = self.process_output(outputs_TNH)

            if outputs_TNH.shape[-1] != self.v_head_dim:
                outputs_TNH = outputs_TNH[..., :self.v_head_dim]

            with jax.named_scope("o_proj"):
                outputs_TR = outputs_TNH.reshape(outputs_TNH.shape[0],
                                                 self.N * self.v_head_dim)
                o_TD = self.o_proj(outputs_TR)

            return new_kv_cache, o_TD


@dataclass(kw_only=True)
class DeepseekV3Attention(DeepseekV3BaseAttention):
    """Standard Multi-Head Attention (MHA) for DeepSeek models."""

    def setup_specific_layers(self, rngs: nnx.Rngs) -> None:
        weight_init = _weight_init(self.random_init)
        self.kv_up_proj = JaxEinsum(
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.N * (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ap_sharding),
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
        q_TA = self.q_down_proj(x_q_TD)
        q_TA = self.q_rms_norm(q_TA)
        q_TP = self.q_up_proj(q_TA)
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

        kv_SA = self.kv_down_proj(x_SD)

        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        k_rope_SNH = k_rope_SH[..., None, :]
        k_rope_SNH = self.rope.apply_rope(input_positions, k_rope_SNH)
        assert k_rope_SNH.shape[1] == 1

        k_rope_SNH = jnp.broadcast_to(
            k_rope_SNH, (k_rope_SNH.shape[0], self.N, self.qk_rope_head_dim))

        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_rms_norm(kv_SA)
        kv_SA = nnx.with_sharding_constraint(kv_SA, self.keyvalue_skh)

        kv_SL = self.kv_up_proj(kv_SA)
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
                          kv_cache: KVCache,
                          md: AttentionMetadata) -> Tuple[jax.Array, KVCache]:
        """
        Computes self-attention for MHA.

        Args:
            q_data: The query tensor of shape `(tokens_query, num_query_heads, head_dim)`.
            kv_data: Tuple of key-value tensors of shape `(tokens_kv, num_query_heads, d_model)`.
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
        weight_init = _weight_init(self.random_init)
        self.k_up_proj = JaxEinsum(
            einsum_str="TNH,ANH->TNA",
            kernel_shape=(self.kv_lora_rank, self.N, self.qk_nope_head_dim),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.anh_sharding),
        )

        self.v_up_proj = JaxEinsum(
            einsum_str="TNA,ANH->TNH",
            kernel_shape=(self.kv_lora_rank, self.N, self.v_head_dim),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.anh_sharding),
        )

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
        q_TA = self.q_down_proj(x_q_TD)
        q_TA = self.q_rms_norm(q_TA)
        q_TP = self.q_up_proj(q_TA)
        q_TNH = q_TP.reshape(q_TA.shape[0], self.N, self.qk_head_dim)

        q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = self.rope.apply_rope(input_positions, q_rope_TNH)

        q_TNA = self.k_up_proj(q_nope_TNH)

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
        kv_SA = self.kv_down_proj(x_SD)

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
                          kv_data: Tuple[jax.Array,
                                         jax.Array], kv_cache: KVCache,
                          md: AttentionMetadata) -> Tuple[KVCache, jax.Array]:
        """
        Computes the attention for MLA.

        Args:
            q_data: A tuple of query tensor of shape `(tokens_query, num_query_heads, q_lora_rank)` and
                rope tensor of shape `(tokens_query, num_query_heads, head_dim)`.
            kv_data: A tuple of key-value tensor of shape `(tokens_kv, q_lora_rank)` and
                rope tensor of shape `(tokens_kv, head_dim)`.
            kv_cache: The key-value cache.
            md: The attention metadata.

        Returns:
            A tuple of key-value cache and output tensor of shape `(tokens_query, num_query_heads, q_lora_rank)`.
        """

        q_TNA, q_rope_TNH = q_data
        k_SA, k_rope_SH = kv_data

        q_scale = k_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            # TODO: May need to apply quantization separately for k_c & k_pe
            k_SA, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                  k_SA,
                                  value=None,
                                  k_scale=k_scale)
            k_rope_SH, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                       k_rope_SH,
                                       value=None,
                                       k_scale=k_scale)

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
                num_queries_per_block=num_queries_per_block,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=k_scale)

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
        outputs_TNH = self.v_up_proj(outputs_TNA)
        return outputs_TNH


@dataclass(kw_only=True)
class DeepseekV3MLP(JaxModule):
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
    quant_config: Optional[QuantizationConfig] = None

    rngs: InitVar[nnx.Rngs]

    def __call__(self, x_TD):
        """Performs the forward pass of the FFW layer.

        Args:
            x_TD: The input tensor of shape either `(sequence, d_model)`

        Returns:
            The output tensor of shape `(batch, sequence, d_model)`.
        """
        x_TD = jnp.asarray(x_TD, self.dtype)
        x_TD = nnx.with_sharding_constraint(x_TD, self.activation_ffw_td)
        with jax.named_scope("wi_0"):
            gating_TF = self.gate_proj(x_TD)
            activated_gating_TF = modeling_flax_utils.ACT2FN[self.hidden_act](
                gating_TF)
        with jax.named_scope("wi_1"):
            up_proj_TF = self.up_proj(x_TD)
        fuse_TF = activated_gating_TF * up_proj_TF
        with jax.named_scope("wo"):
            output_TD = self.down_proj(fuse_TF)

        return output_TD

    def __post_init__(self, rngs: nnx.Rngs):
        D = self.hidden_size
        F = self.intermediate_size
        weight_init = _weight_init(self.random_init)

        self.gate_proj = JaxEinsum(
            einsum_str="TD,DF->TF",
            kernel_shape=(D, F),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.df_sharding),
        )
        self.up_proj = JaxEinsum(
            einsum_str="TD,DF->TF",
            kernel_shape=(D, F),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.df_sharding),
        )
        self.down_proj = JaxEinsum(
            einsum_str="TF,FD->TD",
            kernel_shape=(F, D),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.fd_sharding),
        )


@dataclass(kw_only=True)
class DeepseekV3MoE(JaxModule):
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


class DeepseekV3DecoderLayer(JaxModule):
    """
    Implementats the DecoderLayer for DeepseekV3.
    """

    def __init__(
            self,
            input_layernorm: JaxRmsNorm,
            post_attention_layernorm: JaxRmsNorm,
            self_attn: Union[DeepseekV3Attention, DeepseekV3MLA],

            # MLP can be either the Dense MLP (for first k layers) or DeepseekV2MoE
            # TODO: rename to mlp? custom_module seems needlessly confusing
            custom_module: nnx.Module | DeepseekV3MoE | DeepseekV3MLP,
            prefix: str = ""):
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = self_attn
        self.mlp = custom_module

    def __call__(
        self, kv_cache: List[jax.Array], x_TD: jax.Array,
        attention_metadata: AttentionMetadata
    ) -> Tuple[List[jax.Array], jax.Array]:

        # Run Self-Attention
        residual = x_TD
        hidden_states = self.input_layernorm(x_TD)
        new_cache, attn_output = self.self_attn(hidden_states, kv_cache,
                                                attention_metadata)
        hidden_states = residual + attn_output

        # Run MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        custom_module_output = self.mlp(hidden_states)

        # Residual
        hidden_states = residual + custom_module_output

        return new_cache, hidden_states


@dataclass
class DeepSeekV3Router(JaxModule):
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
    quant_config: Optional[QuantizationConfig] = None

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

        scores_TE = self.gate_proj(x_TD)
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
        weight_init = _weight_init(self.random_init)
        self.gate_proj = JaxEinsum(
            einsum_str="TD,DE->TE",
            kernel_shape=(D, E),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ed_sharding),
        )
        self.bias_E = create_param(rngs,
                                   shape=(E, ),
                                   dtype=self.router_bias_dtype,
                                   sharding=self.e_sharding,
                                   random_init=self.random_init)


@dataclass
class DeepSeekV3(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config,
                 prefix: str = ""):
        self.vllm_config = vllm_config

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

        # TODO (jacobplatin): we will resolve this issue in a forthcoming PR that will refactor weight loading
        if vllm_config.load_config.load_format == "dummy" and self.moe_backend in MoEBackend.fused_moe_backends(
        ):
            raise ValueError(
                f"Random / dummy weights are not supported for {MoEBackend.fused_moe_backends()} backends right now."
            )

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank
        hf_config = vllm_config.model_config.hf_config

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hf_config.hidden_size,
                dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def _create_deepseek_attention(
        ) -> Union[DeepseekV3MLA, DeepseekV3Attention]:
            if self.use_mla_kernel:
                query_tnh_spec = P(ShardingAxisName.MLP_TENSOR, None, None)
                keyvalue_skh_spec = P(ShardingAxisName.MLP_TENSOR, None)
                attn_o_tnh_spec = P(ShardingAxisName.MLP_TENSOR, None, None)
                anh_sharding = (None, ShardingAxisName.MLP_TENSOR, None)
            else:
                query_tnh_spec = P(None, ShardingAxisName.MLP_TENSOR, None)
                keyvalue_skh_spec = P(None, ShardingAxisName.MLP_TENSOR, None)
                attn_o_tnh_spec = P(None, ShardingAxisName.MLP_TENSOR, None)
            rd_sharding = (ShardingAxisName.MLP_TENSOR, None)
            ap_sharding = (None, ShardingAxisName.MLP_TENSOR)
            q_da_sharding = (None, ShardingAxisName.MLP_TENSOR)
            kv_da_sharding = (None, ShardingAxisName.MLP_TENSOR)

            if self.vllm_config.additional_config.get("replicate_attn_weights",
                                                      False):
                rd_sharding = ()
                ap_sharding = ()
                q_da_sharding = ()
                kv_da_sharding = ()
                if self.use_mla_kernel:
                    anh_sharding = ()

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
                rngs=rng,
                quant_config=quant_config,
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

        def get_decoder_layer(i):
            input_layernorm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                dtype=dtype,
                quant_config=quant_config,
                rngs=rng,
            )

            post_attention_layernorm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                dtype=dtype,
                quant_config=quant_config,
                rngs=rng,
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
                    rngs=rng,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                    df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                    fd_sharding=(ShardingAxisName.MLP_TENSOR, None),
                    quant_config=quant_config)
            else:
                # MoE Layer
                # moe_dtype = jnp.float8_e4m3fn if self.weight_loader.is_native_fp8_model else vllm_config.model_config.hf_config.quantization_config.get(
                #     "tpu_settings", {}).get("mlp_dtype", jnp.float4_e2m1fn)

                router = DeepSeekV3Router(
                    hidden_size=hidden_size,
                    num_experts=num_local_experts,
                    num_experts_per_tok=num_experts_per_token,
                    n_groups=n_group,
                    topk_groups=4,
                    norm_topk_prob=True,
                    rngs=rng,
                    routed_scaling_factor=routed_scaling_factor,
                    dtype=dtype,
                    moe_backend=self.moe_backend,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                    ed_sharding=(None, None),
                    e_sharding=(None, ),
                    quant_config=quant_config)

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
                    rngs=rng,
                    quant_config=quant_config,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA,
                                       ShardingAxisName.MOE_TENSOR),
                    activation_ffw_ted=(ShardingAxisName.MLP_DATA, None,
                                        ShardingAxisName.MOE_TENSOR),
                    edf_sharding=(None, ShardingAxisName.MOE_TENSOR,
                                  ShardingAxisName.ATTN_DATA_EXPERT),
                    efd_sharding=(None, ShardingAxisName.ATTN_DATA_EXPERT,
                                  ShardingAxisName.MOE_TENSOR),
                    moe_backend=self.moe_backend,
                    qwix_quantized_weight_dtype=None,
                    router=router)

                # shared experts
                shared_experts = DeepseekV3MLP(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=num_shared_experts *
                    moe_intermediate_size,
                    rngs=rng,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                    df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                    fd_sharding=(ShardingAxisName.MLP_TENSOR, None),
                    quant_config=quant_config)

                mlp_layer = DeepseekV3MoE(
                    experts=custom_module,
                    shared_experts=shared_experts,
                    routed_scaling_factor=routed_scaling_factor,
                )

            return DeepseekV3DecoderLayer(
                input_layernorm=input_layernorm,
                post_attention_layernorm=post_attention_layernorm,
                self_attn=_create_deepseek_attention(),
                custom_module=mlp_layer,
                prefix=f"{prefix}.{i}")

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers, get_decoder_layer)

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
                scale_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                 (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def _print_model_architecture(self):
        num_display_layers = 5

        logger.debug("### Embedding ###")
        nnx.display(self.embed_tokens)

        logger.debug(f"\n### First {num_display_layers} Layers ###")
        # Loop through the slice and display each layer
        for i, layer in enumerate(self.layers[:num_display_layers]):
            logger.debug(f"\n--- Layer {i} ---")
            nnx.display(layer)

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

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
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class DeepseekV3ForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        if getattr(vllm_config.model_config, "quantization", None) == "fp8":
            # `get_tpu_quantization_config` returns None for "fp8" because
            # the work in #1623 is not fully merged. So this block overrides
            # the logic to return Fp8Config when model_config indicates fp8.
            # TODO(#1623): Remove this block when `get_tpu_quantization_config`
            # is updated.
            from tpu_inference.layers.jax.quantization.fp8 import Fp8Config
            hg_quant_config = getattr(vllm_config.model_config.hf_config,
                                      "quantization_config", {})
            vllm_config.quant_config = Fp8Config(hg_quant_config)

        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = None
        mesh = None

        self.model = DeepSeekV3(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            quant_config=vllm_config.quant_config,
            prefix="model",
        )

        model_config = vllm_config.model_config
        if self.model.is_last_rank:
            vocab_size = model_config.get_vocab_size()
            hidden_size = model_config.hf_config.hidden_size
            self.lm_head = JaxEinsum(
                einsum_str="TD,DV->TV",
                kernel_shape=(hidden_size, vocab_size),
                dtype=model_config.dtype,
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix="lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()

        self.model.initialize_cache()

        if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
            self.model._print_model_architecture()

            logger.debug("\n### LM Head ###")
            nnx.display(self.lm_head)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )

        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)


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
