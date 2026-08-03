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

import os
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Float
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import \
    ShardingAxisNameBase as ShardingAxisName
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.mla import (MLAAttention,
                                                    MLABaseAttention)
from tpu_inference.layers.jax.base import _init_fn as init_fn
from tpu_inference.layers.jax.base import create_param, sharded_initializer
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.layers import FlaxUtils
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLmHead
from tpu_inference.layers.jax.mlp import GatedMLP
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
from tpu_inference.models.jax.utils.weight_utils import (JaxAutoWeightsLoader,
                                                         LoadableWithIterator)

KVCache = Tuple[jax.Array, jax.Array]

logger = init_logger(__name__)


def _weight_init(random_init: bool):
    return sharded_initializer if random_init else nnx.initializers.uniform()


modeling_flax_utils = FlaxUtils()

# TODO: read these configs from HF config.
num_local_experts: int = 256
vocab_size: int = 129280
hidden_size: int = 7168
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
expert_axis_name = ShardingAxisName.ATTN_DATA_EXPERT

# MLA lives in the shared layer module so other MLA-family models can
# reuse it; these aliases keep the DeepSeek-V3 names in place.
DeepseekV3BaseAttention = MLABaseAttention
DeepseekV3MLA = MLAAttention


@dataclass(kw_only=True)
class DeepseekV3Attention(DeepseekV3BaseAttention):
    """Standard Multi-Head Attention (MHA) for DeepSeek models."""

    def __post_init__(self, rngs: nnx.Rngs):
        super().__post_init__(rngs)

        weight_init = _weight_init(self.random_init)
        self.kv_b_proj = JaxEinsum(
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.N * (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ap_sharding),
            prefix=self.prefix + ".kv_b_proj",
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
        q_TA = self.q_a_proj(x_q_TD)
        q_TA = self.q_a_layernorm(q_TA)
        q_TP = self.q_b_proj(q_TA)
        q_TNH = q_TP.reshape(q_TA.shape[0], self.N, self.qk_head_dim)

        q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        q_rope_TNH = self.rope.apply_rope(input_positions, q_rope_TNH)
        q_TNH = jnp.concatenate([q_nope_TNH, q_rope_TNH], axis=-1)

        return lax.with_sharding_constraint(q_TNH, self.query_tnh)

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

        kv_SA = self.kv_a_proj_with_mqa(x_SD)

        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        k_rope_SNH = k_rope_SH[..., None, :]
        k_rope_SNH = self.rope.apply_rope(input_positions, k_rope_SNH)
        assert k_rope_SNH.shape[1] == 1

        k_rope_SNH = jnp.broadcast_to(
            k_rope_SNH, (k_rope_SNH.shape[0], self.N, self.qk_rope_head_dim))

        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_a_layernorm(kv_SA)
        kv_SA = lax.with_sharding_constraint(kv_SA, self.keyvalue_skh)

        kv_SL = self.kv_b_proj(kv_SA)
        kv_nope_SNH = kv_SL.reshape(kv_SA.shape[0], self.N,
                                    self.qk_nope_head_dim + self.v_head_dim)

        k_nope_SNH = kv_nope_SNH[..., :self.qk_nope_head_dim]
        v_SNH = kv_nope_SNH[..., self.qk_nope_head_dim:]

        k_SNH = jnp.concatenate([k_nope_SNH, k_rope_SNH], axis=-1)

        # Shard
        k_SNH = lax.with_sharding_constraint(k_SNH, self.keyvalue_skh)
        v_SNH = lax.with_sharding_constraint(v_SNH, self.keyvalue_skh)

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
            P(ShardingAxisName.BATCH, None,
              ShardingAxisName.ATTN_HEAD),  # kv_cache
            P(),  # md.seq_lens: Replicated
            P(),  # page_indices_flat: Replicated
            P(),  # query_start_loc: Replicated
            P(),  # distribution: Replicated
        )

        out_specs = (self.attn_o_nth, P(None, None,
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


# The gated FFN lives in the shared layer module so other models can reuse
# it; this keeps the DeepSeek-V3 name in place.
DeepseekV3MLP = GatedMLP


@dataclass(kw_only=True)
class SharedFusedMoe(JaxMoE):
    """
    Corresponds to vLLM's SharedFusedMoe.
    Handles the routed and shared experts + the relevant forward pass.

    Reference here: https://github.com/vllm-project/vllm/blob/168ee03e1cbba2b962adbc704b16762b266be184/vllm/model_executor/layers/fused_moe/shared_fused_moe.py#L14
    """
    shared_experts: Optional[DeepseekV3MLP] = None

    routed_scaling_factor: float = 1.0

    def __call__(self, x_TD: jax.Array):
        # Compute Routed Experts
        final_hidden_states, expert_indices = super().__call__(x_TD)

        final_hidden_states *= self.routed_scaling_factor

        # (Maybe) Compute Shared Experts
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x_TD)
            final_hidden_states += shared_output

        return final_hidden_states, expert_indices


class DeepseekV2Moe(JaxModule):
    """Jax implementation of Deepseek MoE layer
    
    vllm ref. https://github.com/vllm-project/vllm/blob/168ee03e1cbba2b962adbc704b16762b266be184/vllm/model_executor/models/deepseek_v2.py#L225
    """

    def __init__(self,
                 *,
                 mesh,
                 dtype,
                 num_expert_parallelism,
                 moe_backend,
                 quant_config,
                 scoring_func,
                 rng,
                 prefix: str = "",
                 enable_return_routed_experts: bool = False):

        self.gate = DeepSeekV3Router(
            hidden_size=hidden_size,
            num_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_token,
            n_groups=n_group,
            topk_groups=4,
            norm_topk_prob=True,
            rngs=rng,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            moe_backend=moe_backend,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            ed_sharding=P(None, None),
            e_sharding=P(None, ),
            scoring_func=scoring_func,
            quant_config=quant_config)

        # shared experts
        self.shared_experts = DeepseekV3MLP(
            dtype=dtype,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            intermediate_size=num_shared_experts * moe_intermediate_size,
            rngs=rng,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            df_sharding=P(None, ShardingAxisName.ATTN_HEAD),
            fd_sharding=P(ShardingAxisName.ATTN_HEAD, None),
            quant_config=quant_config)

        # routed experts
        if moe_backend == MoEBackend.GMM_TP:
            moe_activation_ffw_td = P(ShardingAxisName.MLP_DATA, None)
            moe_activation_ffw_ted = P(ShardingAxisName.MLP_DATA, None,
                                       ShardingAxisName.MOE_TENSOR)
            moe_edf_sharding = P(None, ShardingAxisName.ATTN_DATA_EXPERT,
                                 ShardingAxisName.MOE_TENSOR)
            moe_efd_sharding = P(None, ShardingAxisName.MOE_TENSOR,
                                 ShardingAxisName.ATTN_DATA_EXPERT)
        else:
            moe_activation_ffw_td = P(ShardingAxisName.MLP_DATA,
                                      ShardingAxisName.MOE_TENSOR)
            moe_activation_ffw_ted = P(ShardingAxisName.MLP_DATA, None,
                                       ShardingAxisName.MOE_TENSOR)
            moe_edf_sharding = P(ShardingAxisName.ATTN_DATA_EXPERT, None, None)
            moe_efd_sharding = P(ShardingAxisName.ATTN_DATA_EXPERT, None, None)

        self.experts = SharedFusedMoe(
            dtype=dtype,
            num_local_experts=num_local_experts,
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            hidden_size=hidden_size,
            intermediate_size_moe=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_token,
            mesh=mesh,
            hidden_act=hidden_act,
            rngs=rng,
            quant_config=quant_config,
            activation_ffw_td=moe_activation_ffw_td,
            activation_ffw_ted=moe_activation_ffw_ted,
            edf_sharding=moe_edf_sharding,
            efd_sharding=moe_efd_sharding,
            moe_backend=moe_backend,
            qwix_quantized_weight_dtype=None,
            # It's abnormal prefix here because we are using dataclass for SharedFusedMoe and JaxMoe.
            # The proper way is to change both to normal class, set prefix=prefix+".mlp" here,
            # then in __init__, pass prefix+".experts" to super().__init__.
            prefix=f"{prefix}.experts",
            router=self.gate,
            shared_experts=self.shared_experts,
            scoring_func=scoring_func,
            enable_return_routed_experts=enable_return_routed_experts,
            routed_scaling_factor=routed_scaling_factor)

    def __call__(self, x_TD: jax.Array):
        return self.experts(x_TD)


class DeepseekV3DecoderLayer(JaxModule):
    """
    Implementats the DecoderLayer for DeepseekV3.
    """

    def __init__(
            self,
            input_layernorm: JaxRmsNorm,
            post_attention_layernorm: JaxRmsNorm,
            self_attn: Union[DeepseekV3Attention, DeepseekV3MLA],

            # MLP can be either the Dense MLP (for first k layers) or SharedFusedMoe
            mlp: nnx.Module | SharedFusedMoe | DeepseekV3MLP,
            prefix: str = ""):
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = self_attn
        self.mlp = mlp

    def __call__(
        self, x_TD: jax.Array, *, kv_cache: List[jax.Array],
        attention_metadata: AttentionMetadata
    ) -> Tuple[List[jax.Array], jax.Array, Optional[jax.Array]]:

        # Run Self-Attention
        residual = x_TD
        hidden_states = self.input_layernorm(x_TD)
        new_cache, attn_output = self.self_attn(hidden_states, kv_cache,
                                                attention_metadata)
        hidden_states = residual + attn_output

        # Run MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        expert_indices = None
        if isinstance(mlp_output, tuple):
            mlp_output, expert_indices = mlp_output

        # Residual
        hidden_states = residual + mlp_output

        return new_cache, hidden_states, expert_indices


class DeepSeekV3Router(JaxEinsum):
    """Router module for Mixture-of-Experts (MoE) layers.

    This module determines which experts each token should be routed to based on the input.
    """

    def __init__(
            self,
            hidden_size: int,
            num_experts: int,
            num_experts_per_tok: int,
            n_groups: int,
            topk_groups: int,
            norm_topk_prob: bool,
            routed_scaling_factor,
            dtype: jnp.dtype,
            rngs: nnx.Rngs,
            # Sharding Attributes
            activation_ffw_td: P = P(),
            ed_sharding: P = P(),
            e_sharding: P = P(),
            random_init: bool = False,
            quant_config: Optional[QuantizationConfig] = None,
            router_bias_dtype: jnp.dtype = jnp.float32,
            scoring_func: str = "sigmoid",
            moe_backend: MoEBackend = MoEBackend.DENSE_MAT):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_groups = n_groups
        self.topk_groups = topk_groups
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype
        self.activation_ffw_td = activation_ffw_td
        self.ed_sharding = ed_sharding
        self.e_sharding = e_sharding
        self.random_init = random_init
        self.quant_config = quant_config
        self.router_bias_dtype = router_bias_dtype
        self.scoring_func = scoring_func
        self.moe_backend = moe_backend
        """Generates the router kernel (weights and bias) for routing."""
        D = self.hidden_size
        E = self.num_experts
        weight_init = _weight_init(self.random_init)
        JaxEinsum.__init__(
            self,
            einsum_str="TD,DE->TE",
            kernel_shape=(D, E),
            rngs=rngs,
            # DS model has gate weights unquantized, but not mentioned in the config.
            quant_config=None,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ed_sharding),
            # The gate matmul runs in fp32 (see __call__); HIGHEST keeps it
            # there on TPU instead of letting XLA drop to a bf16 pass.
            precision=jax.lax.Precision.HIGHEST,
        )
        self.e_score_correction_bias = create_param(
            rngs,
            shape=(E, ),
            dtype=self.router_bias_dtype,
            sharding=self.e_sharding,
            random_init=self.random_init)

    def get_topk_indices(self, scores_TE: Float) -> Float:
        """Get the topk indices of the scores.

        Args:
            scores_TE: The scores to get the topk indices of. Shape (sequence, num_experts).

        Returns:
            The topk indices of the scores. Shape (sequence, num_experts_per_tok).
        """

        scores_TE = scores_TE + self.e_score_correction_bias
        if self.n_groups > 1:
            experts_per_group = self.num_experts // self.n_groups
            group_scores_TGM = jnp.reshape(
                scores_TE, (-1, self.n_groups, experts_per_group))
            group_scores_TG2 = jax.lax.top_k(group_scores_TGM, k=2)[0]
            group_scores_TG = jnp.sum(group_scores_TG2, axis=-1)
            group_indices = jax.lax.top_k(group_scores_TG,
                                          k=self.topk_groups)[1]

            # Apply mask at the group level before flattening
            mask_TG1 = jax.nn.one_hot(
                group_indices,
                self.n_groups).sum(axis=1)[..., None].astype(jnp.bool_)

            # Apply mask to each group of experts
            group_scores_TGM = jnp.where(mask_TG1, group_scores_TGM, -jnp.inf)

            scores_TE = jnp.reshape(group_scores_TGM, (-1, self.num_experts))

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
        x_TD = lax.with_sharding_constraint(x_TD, self.activation_ffw_td)

        # Expert assignments are accumulated in high precision to preserve accuracy.
        # See: https://github.com/vllm-project/vllm/blob/e89a91d9275cd8ac086fe04476b41675a9ebbd5c/vllm/model_executor/layers/fused_moe/cpu_fused_moe.py#L59
        #
        # The reference implementations run the gate itself in fp32 --
        # `DeepseekV3TopkRouter` / `KimiMoEGate` both do
        # `F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))`
        # -- so casting only the *output* is not enough: on TPU the fp32 einsum
        # is otherwise lowered to bf16 multiplies, and experts whose scores are
        # within bf16 rounding of each other swap places in the top-k. Cast the
        # activations up front and rely on `precision=HIGHEST` (set in __init__)
        # for the multiply. The gate is [hidden, num_experts], so this costs
        # nothing next to the expert matmuls.
        #
        # NOTE: this is a deliberate precision/behavior change to the existing
        # DeepSeek-V3 path (it previously ran the gate matmul in bf16 and only
        # cast the logits to fp32 afterwards): near-tie top-k routing decisions
        # can differ, so DSv3 nightly baselines may shift slightly. A routing
        # parity test pinning fp32 gate behavior lands with the K3 model.
        logits_TE = super().__call__(x_TD.astype(jnp.float32))

        # TODO(gpolovets): add back support for DeepSeek routing.
        if self.moe_backend in MoEBackend.fused_moe_backends():
            return logits_TE

        # Apply scoring function (Sigmoid/Softmax) to get probabilities
        if self.scoring_func == "sigmoid":
            probs_TE = jax.nn.sigmoid(logits_TE)
        elif self.scoring_func == "softmax":
            probs_TE = jax.nn.softmax(logits_TE, axis=-1)
        else:
            probs_TE = logits_TE

        # Add Aux-Loss-Free bias to the activation outputs during topk selection.
        topk_indices_TX = self.get_topk_indices(probs_TE)

        # The actual weights do not include the bias terms.
        weights_TX = jnp.take_along_axis(probs_TE, topk_indices_TX, axis=-1)

        if self.norm_topk_prob:
            weights_TX /= jnp.sum(weights_TX, axis=-1)[..., None] + 1e-20

        return weights_TX.astype(self.dtype), topk_indices_TX


@dataclass
class DeepSeekV3(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config,
                 prefix: str = ""):
        self.vllm_config = vllm_config
        self.enable_return_routed_experts = self.vllm_config.model_config.enable_return_routed_experts

        self.use_mla_kernel: bool = self.vllm_config.model_config.use_mla

        logger.info(f"Is using MLA kernel in DeepSeek: {self.use_mla_kernel}")

        self.mesh = mesh

        self.num_expert_parallelism = get_expert_parallelism(
            expert_axis_name, self.mesh)
        total_tensor_parallelsim = self.vllm_config.sharding_config.tp_size * \
                                        self.vllm_config.sharding_config.attn_dp_size
        self.use_ep = self.num_expert_parallelism > 1 and total_tensor_parallelsim == 1
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
        dtype = vllm_config.model_config.dtype
        scoring_func = getattr(hf_config, "scoring_func", "sigmoid")

        if self.is_first_rank:
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hf_config.hidden_size,
                param_dtype=dtype,
                dtype=dtype,
                embedding_init=nnx.with_partitioning(
                    init_fn, (ShardingAxisName.MLP_TENSOR, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.rope_emb = DeepseekScalingRotaryEmbedding(
            rotary_dim=qk_rope_head_dim,
            rope_theta=rope_theta,
            original_max_position_embeddings=rope_scaling[
                "original_max_position_embeddings"],
            scaling_factor=rope_scaling["factor"],
            dtype=dtype,
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
            mscale_value=rope_scaling["mscale"],
            mscale_all_dim=rope_scaling["mscale_all_dim"],
        )

        def _create_deepseek_attention(
                i: int) -> Union[DeepseekV3MLA, DeepseekV3Attention]:
            if self.use_mla_kernel:
                # q_NTA is [N, T, A].
                query_nth_spec = P(None, ShardingAxisName.ATTN_DATA, None)
                query_tnh_spec = P(ShardingAxisName.ATTN_DATA, None, None)
                keyvalue_skh_spec = P(ShardingAxisName.ATTN_DATA, None)
                # Kernel output is [N, T, A]
                attn_o_nth_spec = P(None, ShardingAxisName.ATTN_DATA, None)
                anh_sharding = (None, ShardingAxisName.ATTN_HEAD, None)
            else:
                query_tnh_spec = P(None, ShardingAxisName.MLP_TENSOR)
                keyvalue_skh_spec = P(None, ShardingAxisName.MLP_TENSOR)
                attn_o_nth_spec = P(None, ShardingAxisName.MLP_TENSOR)
            rd_sharding = P(ShardingAxisName.ATTN_HEAD, None)
            ap_sharding = P(None, ShardingAxisName.ATTN_HEAD)
            q_da_sharding = P(None, ShardingAxisName.ATTN_HEAD)
            kv_da_sharding = P(None, ShardingAxisName.ATTN_HEAD)

            attn_cls = None
            if self.use_mla_kernel:
                attn_cls = DeepseekV3MLA
            else:
                attn_cls = DeepseekV3Attention
                assert num_attention_heads == num_key_value_heads, "Expected same number of of attention heads and key value heads for MHA."

            kwargs = dict(
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
                rope=self.rope_emb,
                rope_mscale_all_dim=rope_scaling["mscale_all_dim"],
                dtype=dtype,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                rngs=rng,
                quant_config=quant_config,
                activation_attention_td=P(ShardingAxisName.ATTN_DATA, None),
                activation_q_td=P(ShardingAxisName.ATTN_DATA, None),
                query_nth=query_nth_spec,
                query_tnh=query_tnh_spec,
                keyvalue_skh=keyvalue_skh_spec,
                activation_attention_out_td=P(ShardingAxisName.ATTN_DATA,
                                              None),
                attn_o_nth=attn_o_nth_spec,
                q_da_sharding=q_da_sharding,
                ap_sharding=ap_sharding,
                kv_da_sharding=kv_da_sharding,
                rd_sharding=rd_sharding,
                prefix=f"{prefix}.layers.{i}.self_attn",
            )
            if self.use_mla_kernel:
                kwargs.update(anh_sharding=anh_sharding)

            return attn_cls(**kwargs)

        def get_decoder_layer(layer_index: int):
            input_layernorm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                dtype=dtype,
                param_dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
            )

            post_attention_layernorm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                dtype=dtype,
                param_dtype=dtype,
                rngs=rng,
                quant_config=quant_config,
            )

            # Logic to determine if this layer is Dense or MoE
            # * The first k layers are always dense.
            # * Subsequent layers are MoE if interleave_moe_layer_step conditions are met
            if layer_index < first_k_dense_replace:
                is_moe_layer = False
            else:
                is_moe_layer = ((layer_index + 1) %
                                interleave_moe_layer_step == 0)

            if not is_moe_layer:
                # Dense Layer (used for first k layers or interleaved dense layers)
                mlp_layer = DeepseekV3MLP(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=ffw_intermediate_size,
                    rngs=rng,
                    activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
                    df_sharding=P(None, ShardingAxisName.ATTN_HEAD),
                    fd_sharding=P(ShardingAxisName.ATTN_HEAD, None),
                    quant_config=quant_config)
            else:
                # MoE Layer
                mlp_layer = DeepseekV2Moe(
                    mesh=self.mesh,
                    dtype=dtype,
                    num_expert_parallelism=self.num_expert_parallelism,
                    moe_backend=self.moe_backend,
                    quant_config=quant_config,
                    scoring_func=scoring_func,
                    rng=rng,
                    prefix=f"{prefix}.layers.{layer_index}.mlp",
                    enable_return_routed_experts=self.
                    enable_return_routed_experts)

            return DeepseekV3DecoderLayer(
                input_layernorm=input_layernorm,
                post_attention_layernorm=post_attention_layernorm,
                self_attn=_create_deepseek_attention(layer_index),
                mlp=mlp_layer,
                prefix=f"{prefix}.layers.{layer_index}")

        # hf_config.num_hidden_layers is 61, which ignores the last MTP layer.
        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers, get_decoder_layer)

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                 (None, )),
                rngs=rng,
                quant_config=quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def initialize_cache(self):
        # Initialize RoPE cache once after weights are loaded.
        self.rope_emb.initialize_cache()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array, Optional[jax.Array]]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed_tokens(input_ids)

        all_expert_ids = []
        for i, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            kv_cache, x, expert_ids = layer(
                x,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata,
            )
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            kv_caches[i] = kv_cache
        x = self.norm(x)

        stacked_expert_ids = jnp.stack(all_expert_ids,
                                       axis=0) if all_expert_ids else None
        return kv_caches, x, stacked_expert_ids


class DeepseekV3ForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

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
            self.lm_head = JaxLmHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                param_dtype=model_config.dtype,
                dtype=model_config.dtype,
                rngs=rng,
                kernel_init=nnx.with_partitioning(
                    init_fn, (None, ShardingAxisName.MLP_TENSOR)),
                prefix="lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()

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
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        kv_caches, x, expert_indices = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )

        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )

        return kv_caches, x, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable) -> set[str]:
        if not isinstance(weights, Iterable):
            # Use next parent class in MRO.
            return super().load_weights(weights)

        start_ignore_layer_num = len(self.model.layers)
        end_ignore_layer_num = 62  # last layer is MTP, we ignore it for now
        loader = JaxAutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"]
                           if not hasattr(self, 'lm_head') else []),
            skip_substrs=[
                f"layers.{i}"
                for i in range(start_ignore_layer_num, end_ignore_layer_num)
            ],
        )
        loaded = loader.load_weights(weights)

        self.model.initialize_cache()

        # Display model arch
        if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
            logger.debug("Model architecture and parameter dtypes:")
            num_layers_to_display = 5
            should_skip_layer_display = False
            for name, param in self.named_parameters():
                if f"layers.{num_layers_to_display}." in name:
                    should_skip_layer_display = True
                if should_skip_layer_display and "layers." in name:
                    continue
                v: jax.Array = param.value
                logger.debug(f"{name} : {v.dtype}{v.shape} on {v.device}")

        return loaded
