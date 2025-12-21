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
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torchax.ops.mappings import j2t_dtype
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.quantization import u8_unpack_e2m1
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.deepseek_v3_attention import MLA
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.moe.deepseek_v3_moe import (DeepSeekV3Router,
                                                          SparseMoE)
from tpu_inference.layers.jax.moe.moe import MoE
from tpu_inference.layers.jax.transformer_block import (
    SharedExpertsTransformerBlock, TransformerBlock)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    get_param, model_weights_generator, print_param_info)

logger = init_logger(__name__)

# A map from JAX dtype to the corresponding PyTorch integer dtype for raw memory viewing.
DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
}


@dataclass
class DeepSeekV3(nnx.Module):

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

        self.random_init = force_random_weights or self.vllm_config.additional_config.get(
            "random_weights", False)
        self.sparse_matmul = self.vllm_config.additional_config.get(
            "sparse_matmul", False)

        if isinstance(self.sparse_matmul, str):
            self.sparse_matmul = self.sparse_matmul.lower() == "true"
        else:
            self.sparse_matmul = bool(self.sparse_matmul)

        if self.sparse_matmul:
            logger.info("sparse matmul is enabled")
        else:
            logger.info("sparse matmul is disabled, using dense matmul")
        self.mesh = mesh

        self.weight_loader = DeepSeekV3WeightLoader(
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
            use_mla_kernel=self.use_mla_kernel)

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 dtype=dtype,
                                 rngs=self.rng,
                                 vd_sharding=(ShardingAxisName.MLP_TENSOR,
                                              None),
                                 random_init=self.random_init)

        self.layers = []

        def _create_mla() -> MLA:
            if self.use_mla_kernel:
                query_tnh_spec = P(ShardingAxisName.MLP_TENSOR, None, None)
                keyvalue_skh_spec = P(ShardingAxisName.MLP_TENSOR, None)
                attn_o_tnh_spec = P(ShardingAxisName.MLP_TENSOR, None, None)

            else:
                query_tnh_spec = P(None, ShardingAxisName.MLP_TENSOR, None)
                keyvalue_skh_spec = P(None, ShardingAxisName.MLP_TENSOR, None)
                attn_o_tnh_spec = P(None, ShardingAxisName.MLP_TENSOR, None)

            return MLA(
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                rms_norm_eps=rms_norm_eps,
                v_head_dim=v_head_dim,
                mesh=self.mesh,
                use_mla_kernel=self.use_mla_kernel,
                random_init=self.random_init,
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
                q_da_sharding=(None, ShardingAxisName.VOCAB),
                ap_sharding=(None, ShardingAxisName.MLP_TENSOR),
                anh_sharding=(None, ShardingAxisName.MLP_TENSOR, None),
                kv_da_sharding=(None, ShardingAxisName.VOCAB),
                rd_sharding=(ShardingAxisName.MLP_TENSOR, None))

        for i in range(first_k_dense_replace):
            block = TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    with_scale=True,
                    dtype=dtype,
                    rngs=self.rng,
                ),
                pre_mlp_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    with_scale=True,
                    dtype=dtype,
                    rngs=self.rng,
                ),
                attn=_create_mla(),
                custom_module=DenseFFW(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=ffw_intermediate_size,
                    rngs=self.rng,
                    df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                    fd_sharding=(ShardingAxisName.MLP_TENSOR, None),
                    random_init=self.random_init))

            self.layers.append(block)

        for i in range(first_k_dense_replace, num_layers):
            is_moe_layer = ((i + 1) % interleave_moe_layer_step == 0)
            router = DeepSeekV3Router(
                random_init=self.random_init,
                hidden_size=hidden_size,
                num_experts=num_local_experts,
                num_experts_per_tok=num_experts_per_token,
                n_groups=n_group,
                topk_groups=4,
                norm_topk_prob=True,
                rngs=self.rng,
                routed_scaling_factor=2.5,
                dtype=dtype,
                activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                ed_sharding=(ShardingAxisName.MLP_TENSOR, None),
                e_sharding=(ShardingAxisName.MLP_TENSOR, ))
            if self.sparse_matmul:
                # TODO: orginize the SparseMoE and DenseMoE better given they share most interfaces
                custom_module = SparseMoE(
                    dtype=dtype,
                    num_local_experts=num_local_experts,
                    apply_expert_weight_before_computation=False,
                    hidden_size=hidden_size,
                    intermediate_size_moe=moe_intermediate_size,
                    num_experts_per_tok=num_experts_per_token,
                    mesh=self.mesh,
                    hidden_act=hidden_act,
                    rngs=self.rng,
                    random_init=self.random_init,
                    activation_ffw_td=(ShardingAxisName.MLP_TENSOR, None),
                    activation_ffw_ted=(ShardingAxisName.MLP_DATA, None, None),
                    edf_sharding=(ShardingAxisName.MLP_TENSOR, None, None),
                    efd_sharding=(ShardingAxisName.MLP_TENSOR, None, None),
                    quantized_dtype=self.weight_loader.quant_dtype
                    if self.weight_loader.is_model_quantized else None,
                    router=router) if is_moe_layer else DenseFFW(
                        dtype=dtype,
                        hidden_act=hidden_act,
                        hidden_size=hidden_size,
                        intermediate_size=ffw_intermediate_size,
                        rngs=self.rng,
                        random_init=self.random_init,
                        df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                        fd_sharding=(ShardingAxisName.MLP_TENSOR, None))
            else:
                custom_module = MoE(
                    dtype=dtype,
                    num_local_experts=num_local_experts,
                    apply_expert_weight_before_computation=False,
                    hidden_size=hidden_size,
                    intermediate_size_moe=moe_intermediate_size,
                    hidden_act=hidden_act,
                    rngs=self.rng,
                    random_init=self.random_init,
                    activation_ffw_td=(ShardingAxisName.MLP_DATA, None),
                    activation_ffw_ted=(ShardingAxisName.MLP_DATA, None, None),
                    edf_sharding=(ShardingAxisName.MLP_TENSOR, None, None),
                    efd_sharding=(ShardingAxisName.MLP_TENSOR, None, None),
                    router=router) if is_moe_layer else DenseFFW(
                        dtype=dtype,
                        hidden_act=hidden_act,
                        hidden_size=hidden_size,
                        intermediate_size=ffw_intermediate_size,
                        rngs=self.rng,
                        random_init=self.random_init,
                        df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                        fd_sharding=(ShardingAxisName.MLP_TENSOR, None))

            shared_experts = DenseFFW(
                dtype=dtype,
                hidden_act=hidden_act,
                hidden_size=hidden_size,
                intermediate_size=num_shared_experts * moe_intermediate_size,
                rngs=self.rng,
                random_init=self.random_init,
                df_sharding=(None, ShardingAxisName.MLP_TENSOR),
                fd_sharding=(ShardingAxisName.MLP_TENSOR, None))

            pre_attention_norm = RMSNorm(
                dims=hidden_size,
                rngs=self.rng,
                random_init=self.random_init,
                epsilon=rms_norm_eps,
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=hidden_size,
                rngs=self.rng,
                random_init=self.random_init,
                epsilon=rms_norm_eps,
                with_scale=True,
                dtype=dtype,
            )

            block = SharedExpertsTransformerBlock(
                custom_module=custom_module,
                attn=_create_mla(),
                pre_attention_norm=pre_attention_norm,
                pre_mlp_norm=pre_mlp_norm,
                shared_experts=shared_experts)
            self.layers.append(block)

        self.final_norm = RMSNorm(
            dims=hidden_size,
            rngs=self.rng,
            random_init=self.random_init,
            epsilon=rms_norm_eps,
            with_scale=True,
            dtype=dtype,
        )

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=hidden_size,
                              dtype=dtype,
                              rngs=self.rng,
                              vd_sharding=(ShardingAxisName.MLP_TENSOR, None),
                              dv_sharding=(None, ShardingAxisName.MLP_TENSOR),
                              random_init=self.random_init)

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

    def initialize_cache(self):
        # Initialize RoPE caches after weights are loaded and before JIT compilation.
        for layer in self.layers:
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'rope'):
                if hasattr(layer.attn.rope, 'initialize_cache'):
                    layer.attn.rope.initialize_cache(self.mesh)

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


@dataclass
class DeepSeekV3WeightLoader:

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
                 use_mla_kernel=False):
        self.num_layers = num_layers
        self.names_and_weights_generator = model_weights_generator(
            model_name_or_path=vllm_config.model_config.model,
            framework="pt",
            download_dir=vllm_config.load_config.download_dir)
        self.is_verbose = vllm_config.additional_config.get(
            "is_verbose", None) is not None
        self.num_routed_experts = num_local_experts
        self.attn_heads = attn_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.model_dtype = model_dtype
        self.use_mla_kernel = use_mla_kernel

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
            "layers.*.pre_attention_norm.scale",
            "model.layers.*.post_attention_layernorm.weight":
            "layers.*.pre_mlp_norm.scale",
            # attention (MLA)
            "model.layers.*.self_attn.q_a_layernorm.weight":
            "layers.*.attn.q_rms_norm.scale",
            "model.layers.*.self_attn.kv_a_layernorm.weight":
            "layers.*.attn.kv_rms_norm.scale",
            "model.layers.*.self_attn.q_a_proj.weight":
            "layers.*.attn.kernel_q_down_proj_DA",
            "model.layers.*.self_attn.q_b_proj.weight":
            "layers.*.attn.kernel_q_up_proj_AP",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight":
            "layers.*.attn.kernel_kv_down_proj_DA",
            "model.layers.*.self_attn.kv_b_proj.weight":
            "layers.*.attn.kernel_kv_up_proj_AL",
            "model.layers.*.self_attn.o_proj.weight":
            "layers.*.attn.kernel_o_proj_RD",
            # Dense ffw
            "model.layers.*.mlp.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_DF",
            "model.layers.*.mlp.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_DF",
            "model.layers.*.mlp.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_FD",
            # MOE(routed experts)
            "model.layers.*.mlp.gate.weight":
            "layers.*.custom_module.router.kernel_DE",
            "model.layers.*.mlp.gate.e_score_correction_bias":
            "layers.*.custom_module.router.bias_E",
            "model.layers.*.mlp.experts.*.gate_proj.weight":
            "layers.*.custom_module.kernel_gating_EDF",
            "model.layers.*.mlp.experts.*.down_proj.weight":
            "layers.*.custom_module.kernel_down_proj_EFD",
            "model.layers.*.mlp.experts.*.up_proj.weight":
            "layers.*.custom_module.kernel_up_proj_EDF",
            # MOE(shared experts)
            "model.layers.*.mlp.shared_experts.down_proj.weight":
            "layers.*.shared_experts.kernel_down_proj_FD",
            "model.layers.*.mlp.shared_experts.gate_proj.weight":
            "layers.*.shared_experts.kernel_gating_DF",
            "model.layers.*.mlp.shared_experts.up_proj.weight":
            "layers.*.shared_experts.kernel_up_proj_DF",
        }
        if self.use_mla_kernel:
            self._loaded_to_standardized_keys.update({
                "model.layers.*.self_attn.k_b_proj.weight":
                "layers.*.attn.kernel_k_up_proj_ANH",
                "model.layers.*.self_attn.v_b_proj.weight":
                "layers.*.attn.kernel_v_up_proj_ANH",
            })
        # TODO (jacobplatin): we should not be hard-coding these
        self.scale_dtype, self.quant_dtype = jnp.bfloat16, jnp.float8_e4m3fn

        self.is_model_quantized = not vllm_config.additional_config.get(
            "skip_quantization", False)

        if self.is_model_quantized:
            # NOTE: this is only needed for pre-quantized models when doing random weight loading
            # because the scales that Qwix configures by default don't necessarily match the
            # scales in practice
            # TODO (jacobplatin): remove or clean this up
            self.scale_shape_map_for_random_weight_loading = {
                # MoE experts (3D)
                "custom_module.kernel_down_proj_EFD": (256, 8, 7168),
                "custom_module.kernel_gating_EDF": (256, 28, 2048),
                "custom_module.kernel_up_proj_EDF": (256, 28, 2048),
                # Shared experts (2D)
                "shared_experts.kernel_down_proj_FD": (8, 7168),
                "shared_experts.kernel_gating_DF": (28, 2048),
                "shared_experts.kernel_up_proj_DF": (28, 2048),
                # Dense FFW (2D)
                "custom_module.kernel_gating_DF": (28, 18432),
                "custom_module.kernel_up_proj_DF": (28, 18432),
                "custom_module.kernel_down_proj_FD": (72, 7168),
                # Attention (3D for MLA, 2D for the rest)
                "attn.kernel_q_down_proj_DA": (28, 1536),
                "attn.kernel_q_up_proj_AP": (6, 24576),
                "attn.kernel_kv_down_proj_DA": (28, 576),
                "attn.kernel_kv_up_proj_AL": (2, 32768),
                "attn.kernel_o_proj_RD": (64, 7168),
                "attn.kernel_k_up_proj_ANH": (2, 128, 128),  # MLA
                "attn.kernel_v_up_proj_ANH": (2, 128, 128),  # MLA
            }

            # TODO (jacobplatin): remove this check eventually!
            assert self.quant_dtype == jnp.float8_e4m3fn, f"Expected quant_dtype to be float8_e4m3fn for DeepSeek but got {self.quant_dtype}"

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
        quantized_weights = {}
        quantized_scales = {}
        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
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
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_down_proj_weights)
                        if scale is not None:
                            stacked_scales = self._process_moe_weights(
                                loaded_name, scale,
                                mlp_experts_down_proj_scales)
                    if "gate_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_gate_proj_weights)
                        if scale is not None:
                            stacked_scales = self._process_moe_weights(
                                loaded_name, scale,
                                mlp_experts_gate_proj_scales)
                    if "up_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_up_proj_weights)
                        if scale is not None:
                            stacked_scales = self._process_moe_weights(
                                loaded_name, scale, mlp_experts_up_proj_scales)
                    if stacked_weights is not None:
                        weight_bytes, weight_shards = self._load_individual_weight(
                            loaded_name,
                            stacked_weights,
                            model_params,
                            model_for_loading.mesh,
                            scale=stacked_scales)
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
                            v_scale = scale_reshaped[:,
                                                     self.qk_nope_head_dim:, :]
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
