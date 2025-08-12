import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_dtypes
import torch
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.attention.deepseek_v3_attention import MLA
from tpu_commons.models.jax.common.constants import KVCacheType
from tpu_commons.models.jax.common.layers import (DenseFFW, Embedder, LMhead,
                                                  RMSNorm)
from tpu_commons.models.jax.common.model import Model
from tpu_commons.models.jax.common.moe.deepseek_moe import DeepSeekV3Router
from tpu_commons.models.jax.common.moe.moe import MoE
from tpu_commons.models.jax.common.transformer_block import (
    SharedExpertsTransformerBlock, TransformerBlock)
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.utils.weight_utils import (
    get_param, hf_model_weights_iterator, print_param_info, reshape_params)

logger = init_logger(__name__)


@dataclass
class DeepSeekV3(Model):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: jax.Array,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)

        # Currently the runner will always set a mesh, so the custom default sharding (when
        #  no sharding is set in vllm config) doesn't take effect.
        # TODO(fhzhang): figure out whether we need to actually enable this.
        #    strategy_dict = {
        #        "tensor_parallelism": 4,
        #        "expert_parallelism": 2
        #    }  # todo: update this.

        num_layers: int = 61
        num_local_experts: int = 256

        vocab_size: int = 129280
        hidden_size: int = 7168
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
            num_local_experts=num_local_experts)

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 dtype=dtype,
                                 vd_sharding=NamedSharding(
                                     self.mesh,
                                     P(('data', 'expert', 'model'), None)),
                                 prelogit_td=NamedSharding(self.mesh, P()),
                                 mesh=self.mesh,
                                 random_init=self.random_init)
        self.embedder.generate_kernel(self.rng)

        self.layers = []

        def _create_mla() -> MLA:
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
                random_init=self.random_init,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=v_head_dim,  # MLA uses v_head_dim as head_dim
                dtype=dtype,
                activation_attention_td=NamedSharding(self.mesh,
                                                      P(None, 'model')),
                activation_q_td=NamedSharding(self.mesh, P(None, 'model')),
                query_tnh=NamedSharding(self.mesh, P(None, 'model', None)),
                keyvalue_skh=NamedSharding(self.mesh, P(None, 'model', None)),
                activation_attention_out_td=NamedSharding(
                    self.mesh, P(None, 'model')),
                keyvalue_cache_lskh=NamedSharding(self.mesh,
                                                  P(None, None, 'model',
                                                    None)),
                attn_o_tnh=NamedSharding(self.mesh, P(None, 'model', None)),
                q_da_sharding=NamedSharding(self.mesh, P(None, 'model')),
                anh_sharding=NamedSharding(self.mesh, P(None, 'model', None)),
                kv_da_sharding=NamedSharding(self.mesh, P(None, 'model')),
                nhd_sharding=NamedSharding(self.mesh, P('model', None, None)),
                query_ktnph=NamedSharding(self.mesh,
                                          P('model', None, None, None, None)),
                keyvalue_cache_nbkph=NamedSharding(
                    self.mesh, P(None, None, 'model', None, None)),
                attn_o_ktnph=NamedSharding(self.mesh,
                                           P('model', None, None, None, None)))

        for i in range(first_k_dense_replace):
            block = TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=hidden_size,
                    mesh=self.mesh,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    activation_ffw_td=NamedSharding(self.mesh, P()),
                    with_scale=True,
                    dtype=dtype,
                ),
                pre_mlp_norm=RMSNorm(
                    dims=hidden_size,
                    mesh=self.mesh,
                    random_init=self.random_init,
                    activation_ffw_td=NamedSharding(self.mesh, P()),
                    epsilon=rms_norm_eps,
                    with_scale=True,
                    dtype=dtype,
                ),
                attn=_create_mla(),
                custom_module=DenseFFW(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=ffw_intermediate_size,
                    mesh=self.mesh,
                    df_sharding=NamedSharding(self.mesh,
                                              P(None, ('model', 'expert'))),
                    fd_sharding=NamedSharding(self.mesh,
                                              P(('model', 'expert'), None)),
                    activation_ffw_td=NamedSharding(self.mesh, P()),
                    random_init=self.random_init))

            self.layers.append(block)

        for i in range(first_k_dense_replace, num_layers):
            is_moe_layer = ((i + 1) % interleave_moe_layer_step == 0)
            router = DeepSeekV3Router(
                mesh=self.mesh,
                random_init=self.random_init,
                hidden_size=hidden_size,
                num_experts=num_local_experts,
                num_experts_per_tok=num_experts_per_token,
                n_groups=n_group,
                topk_groups=4,
                norm_topk_prob=True,
                routed_scaling_factor=2.5,
                dtype=dtype,
                activation_ffw_td=NamedSharding(self.mesh, P('data', None)),
                ed_sharding=NamedSharding(self.mesh, P('expert', None)),
                e_sharding=NamedSharding(self.mesh, P('expert')))
            custom_module = MoE(
                dtype=dtype,
                num_local_experts=num_local_experts,
                apply_expert_weight_before_computation=False,
                hidden_size=hidden_size,
                intermediate_size_moe=moe_intermediate_size,
                hidden_act=hidden_act,
                mesh=self.mesh,
                random_init=self.random_init,
                activation_ffw_td=NamedSharding(self.mesh, P('data', None)),
                activation_ffw_ted=NamedSharding(self.mesh,
                                                 P('data', 'expert', None)),
                edf_sharding=NamedSharding(self.mesh, P(
                    'expert', None, 'model')),
                efd_sharding=NamedSharding(self.mesh, P(
                    'expert', 'model', None)),
                router=router) if is_moe_layer else DenseFFW(
                    dtype=dtype,
                    hidden_act=hidden_act,
                    hidden_size=hidden_size,
                    intermediate_size=ffw_intermediate_size,
                    mesh=self.mesh,
                    random_init=self.random_init,
                    df_sharding=NamedSharding(self.mesh,
                                              P(None, ('model', 'expert'))),
                    fd_sharding=NamedSharding(self.mesh,
                                              P(('model', 'expert'), None)),
                    activation_ffw_td=NamedSharding(self.mesh, P()))

            shared_experts = DenseFFW(
                dtype=dtype,
                hidden_act=hidden_act,
                hidden_size=hidden_size,
                intermediate_size=num_shared_experts * moe_intermediate_size,
                mesh=self.mesh,
                random_init=self.random_init,
                df_sharding=NamedSharding(self.mesh,
                                          P(None, ('model', 'expert'))),
                fd_sharding=NamedSharding(self.mesh,
                                          P(('model', 'expert'), None)),
                activation_ffw_td=NamedSharding(self.mesh, P()))

            pre_attention_norm = RMSNorm(
                dims=hidden_size,
                mesh=self.mesh,
                random_init=self.random_init,
                epsilon=rms_norm_eps,
                activation_ffw_td=NamedSharding(self.mesh, P()),
                with_scale=True,
                dtype=dtype,
            )

            pre_mlp_norm = RMSNorm(
                dims=hidden_size,
                mesh=self.mesh,
                random_init=self.random_init,
                activation_ffw_td=NamedSharding(self.mesh, P()),
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

        for i in range(len(self.layers)):
            self.layers[i].generate_kernel(self.rng)

        self.final_norm = RMSNorm(
            dims=hidden_size,
            mesh=self.mesh,
            random_init=self.random_init,
            activation_ffw_td=NamedSharding(self.mesh, P()),
            epsilon=rms_norm_eps,
            with_scale=True,
            dtype=dtype,
        )
        self.final_norm.generate_kernel(self.rng)

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
            prelogit_td=NamedSharding(self.mesh, P()),
            vd_sharding=NamedSharding(self.mesh,
                                      P(('data', 'expert', 'model'), None)),
            dv_sharding=NamedSharding(self.mesh,
                                      P(None, ('data', 'expert', 'model'))),
            mesh=self.mesh,
            random_init=self.random_init)
        self.lm_head.generate_kernel(self.rng)

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)
        self.weight_loader.load_weights(self)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False
        x = self.embedder.encode(input_ids)
        for (i, block) in enumerate(self.layers):
            kv_cache = kv_caches[i]
            new_kv_cache, x = block(x, is_prefill, kv_cache,
                                    attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)

        return kv_caches, final_activation

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head.decode(hidden_states)


@dataclass
class DeepSeekV3WeightLoader:

    def __init__(self, vllm_config: VllmConfig, num_layers, hidden_size,
                 q_lora_rank, kv_lora_rank, attn_heads, qk_nope_head_dim,
                 qk_rope_head_dim, v_head_dim, num_local_experts):

        self.num_layers = num_layers
        self.names_and_weights_generator = hf_model_weights_iterator(
            model_name_or_path=vllm_config.model_config.model,
            framework="pt",
            filter_regex="")
        self.num_routed_experts = num_local_experts

        self._transpose_map = {
            # dense mlp
            r"mlp\.down_proj": (1, 0),
            r"mlp\.gate_proj": (1, 0),
            r"mlp\.up_proj": (1, 0),
            # mla
            r"q_a_proj": (1, 0),
            r"q_b_proj": (2, 0, 1),
            r"kv_a_proj_with_mqa": (1, 0),
            r"kv_b_proj": (2, 0, 1),
            r"o_proj": (1, 2, 0),
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
        self._weight_shape_map = {
            "q_b_proj":
            (attn_heads, qk_nope_head_dim + qk_rope_head_dim, q_lora_rank),
            "kv_b_proj":
            (attn_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank),
            "o_proj": (hidden_size, attn_heads, v_head_dim)
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
            "layers.*.attn.kernel_q_up_proj_ANH",
            "model.layers.*.self_attn.kv_a_proj_with_mqa.weight":
            "layers.*.attn.kernel_kv_down_proj_DA",
            "model.layers.*.self_attn.kv_b_proj.weight":
            "layers.*.attn.kernel_kv_up_proj_ANH",
            "model.layers.*.self_attn.o_proj.weight":
            "layers.*.attn.kernel_o_proj_NHD",
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
        expert_num = re.search(r"experts\.(\d+)", loaded_name).group(1)
        if layer_num not in weights_dict:
            weights_dict[layer_num] = {}
        weights_dict[layer_num][expert_num] = loaded_weight
        # Stack all the weights from the expert in this layer
        if len(weights_dict[layer_num]) == self.num_routed_experts:
            weight_list = []
            for expert_index in range(self.num_routed_experts):
                weight_list.append(weights_dict[layer_num][str(expert_index)])
            stacked_weights = torch.stack(weight_list, axis=0)
            del weights_dict[layer_num]
            return stacked_weights
        return None

    def _load_individual_weight(self, name, weight, model_params, model_mesh):
        mapped_name = self.map_loaded_to_standardized_name(name)
        model_weight = get_param(model_params, mapped_name)
        logger.debug(
            f"{name}: {weight.shape}  -->  {mapped_name}: {model_weight.value.shape}"
        )

        # Convert weights from torch into numpy
        # TODO: set cast_type based on model weight's type.
        cast_type = ml_dtypes.bfloat16
        weight = weight.to(torch.float32).numpy().astype(cast_type)

        # Reshape and transpose weights if necessary.
        weight = reshape_params(name, weight, self._weight_shape_map)
        weight = self._transpose_params(name, weight)
        if model_weight.value.shape != weight.shape:
            raise ValueError(
                f"Loaded shape for {name}: {weight.shape} "
                f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
            )
        model_weight.value = shard_put(weight,
                                       model_weight.sharding.spec,
                                       mesh=model_mesh)
        model_weight.value.block_until_ready()
        del weight
        print_param_info(model_weight, name)
        return model_weight.value.nbytes / 1e9, model_weight.value.addressable_shards[
            0].data.nbytes / 1e9

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        logger.warning(
            f"loaded_to_standardized_keys: {self._loaded_to_standardized_keys}"
        )
        cumulative_global_memory = 0
        cumulative_local_memory = 0
        mlp_experts_gate_proj_weights = {}
        mlp_experts_up_proj_weights = {}
        mlp_experts_down_proj_weights = {}
        fp8_weights = {}
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
                if loaded_weight.dtype == torch.float8_e4m3fn:
                    fp8_weights[loaded_name] = loaded_weight
                    continue
                if loaded_name.endswith(".weight_scale_inv"):
                    # assuming weights are loaded before scales.
                    weight_name = loaded_name.replace(".weight_scale_inv",
                                                      ".weight")

                    loaded_weight = weights_dequant_cpu(
                        fp8_weights[weight_name], loaded_weight)
                    loaded_name = weight_name
                    del fp8_weights[weight_name]
                # concat mlp.experts weights
                if "mlp.experts" in loaded_name:
                    if "down_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_down_proj_weights)
                    if "gate_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_gate_proj_weights)
                    if "up_proj" in loaded_name:
                        stacked_weights = self._process_moe_weights(
                            loaded_name, loaded_weight,
                            mlp_experts_up_proj_weights)
                    if stacked_weights is not None:
                        weight_bytes, weight_shards = self._load_individual_weight(
                            loaded_name, stacked_weights, model_params,
                            model_for_loading.mesh)
                        cumulative_global_memory += weight_bytes
                        cumulative_local_memory += weight_shards
                        logger.info(
                            f"Cumulative global memory: {cumulative_global_memory} GB"
                        )
                        logger.info(
                            f"Cumulative local memory: {cumulative_local_memory} GB"
                        )
                else:
                    weight_bytes, weight_shards = self._load_individual_weight(
                        loaded_name, loaded_weight, model_params,
                        model_for_loading.mesh)
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
        del fp8_weights
        # TODO: validate that all of the model_params were accounted for as well.
        nnx.update(model_for_loading, model_params)


def weights_dequant_cpu(x: torch.Tensor,
                        s: torch.Tensor,
                        block_size: int = 128) -> torch.Tensor:
    assert x.dim() == 2 and s.dim() == 2, "Both x and s must be 2D tensors"
    M, N = x.shape

    x = x.to(torch.float32)
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            row_start = i
            row_end = min(i + block_size, M)
            col_start = j
            col_end = min(j + block_size, N)
            block = x[row_start:row_end, col_start:col_end]
            scale = s[i // block_size, j // block_size]
            y[row_start:row_end, col_start:col_end] = (block * scale).to(
                torch.get_default_dtype())

    return y
