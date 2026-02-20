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
import re
from itertools import islice
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.attention.llama4_attention import (
    Llama4Attention, Llama4VisionAttention)
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.misc import shard_put
from tpu_inference.layers.jax.moe.moe import JaxMoE, Router
from tpu_inference.layers.jax.pp_utils import (PPMissingLayer,
                                               get_start_end_layer)
from tpu_inference.layers.jax.rope import Llama4VisionRotaryEmbedding
from tpu_inference.layers.jax.transformer_block import \
    SharedExpertsTransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    BaseWeightLoader, _is_pp_missing_layer, convert_torch_to_jax_with_view,
    get_param, print_param_info, reshape_params, transpose_params)

logger = init_logger(__name__)


class Llama4WeightLoader(BaseWeightLoader):

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        super().__init__(vllm_config,
                         framework="pt",
                         filter_regex="language_model")
        self.is_verbose = getattr(vllm_config.additional_config, "is_verbose",
                                  False)
        self.interleave_moe_layer_step = getattr(
            vllm_config.model_config.hf_config.text_config,
            "interleave_moe_layer_step", 1)

        self.quantization_config = getattr(vllm_config.model_config.hf_config,
                                           "quantization_config", None)
        self.expert_weights_buffer = {}
        self.expert_prefix = "shared_expert."

        transpose_mappings_to_quantization = {
            "down_proj": (1, 0),
            "gate_proj": (1, 0),
            "up_proj": (1, 0),
        }

        self._transpose_map = {
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "router": (1, 0),
            f"{self.expert_prefix}down_proj": (1, 0),
            f"{self.expert_prefix}gate_proj": (1, 0),
            f"{self.expert_prefix}up_proj": (1, 0),
            "feed_forward.down_proj": (1, 0),
            "feed_forward.gate_proj": (1, 0),
            "feed_forward.up_proj": (1, 0),
            "o_proj": (1, 2, 0),
            "lm_head": (1, 0),
        }

        if self.quantization_config and self.expert_prefix:
            self._transpose_map.update(transpose_mappings_to_quantization)

        self._weight_shape_map = {
            "q_proj": (attn_heads, attn_head_dim, hidden_size),
            "k_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            "v_proj": (num_key_value_heads, attn_head_dim, hidden_size),
            # o_proj is inverted: https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/llama4/modeling_llama4.py#L298
            "o_proj": (hidden_size, attn_heads, attn_head_dim),
        }

        # Set the mappings from loaded parameter keys to standardized names.\
        # 1. EXPERT_MAPPINGS_FUSED: Used for non-quantized (e.g., BF16) checkpoints.
        #    - This format typically comes from standard checkpoints where 'gate' and 'up' projection weights might be combined (FUSED) into a single tensor.
        #    - Expert weights are usually stacked, with the expert dimension (E) being the first dimension.
        EXPERT_MAPPINGS_FUSED = {
            "language_model.model.layers.*.feed_forward.experts.down_proj":
            "layers.*.moe_ffw.kernel_down_proj_EFD",
            "language_model.model.layers.*.feed_forward.experts.gate_up_proj":
            "layers.*.moe_ffw.kernel_up_proj_EDF",
        }

        # 2. EXPERT_MAPPINGS_UNFUSED: Specifically designed for quantized checkpoints (e.g., FP8).
        #    - Quantized checkpoints store each expert's weights separately and explicitly separate the 'weight' (quantized value) from the 'weight_scale' (quantization scale).
        #    - The mapping captures both the `.weight` and `.weight_scale` components. This allows the loader to aggregate (stack) the individual expert weights and scales.
        EXPERT_MAPPINGS_UNFUSED = {
            "language_model.model.layers.*.feed_forward.experts.*.down_proj.weight":
            "layers.*.moe_ffw.kernel_down_proj_EFD",
            "language_model.model.layers.*.feed_forward.experts.*.down_proj.weight_scale":
            "layers.*.moe_ffw.kernel_down_proj_EFD",
            "language_model.model.layers.*.feed_forward.experts.*.gate_proj.weight":
            "layers.*.moe_ffw.kernel_gating_EDF",
            "language_model.model.layers.*.feed_forward.experts.*.gate_proj.weight_scale":
            "layers.*.moe_ffw.kernel_gating_EDF",
            "language_model.model.layers.*.feed_forward.experts.*.up_proj.weight":
            "layers.*.moe_ffw.kernel_up_proj_EDF",
            "language_model.model.layers.*.feed_forward.experts.*.up_proj.weight_scale":
            "layers.*.moe_ffw.kernel_up_proj_EDF",
        }

        self._loaded_to_standardized_keys = {
            "language_model.model.embed_tokens.weight":
            "embedder.input_embedding_table_VD",
            "language_model.lm_head.weight":
            "lm_head.input_embedding_table_DV",
            "language_model.model.norm.weight":
            "final_norm.scale",
            "language_model.model.layers.*.input_layernorm.weight":
            "layers.*.pre_attention_norm.scale",
            "language_model.model.layers.*.post_attention_layernorm.weight":
            "layers.*.pre_mlp_norm.scale",
            "language_model.model.layers.*.self_attn.q_proj.weight":
            "layers.*.attn.kernel_q_proj_DNH",
            "language_model.model.layers.*.self_attn.k_proj.weight":
            "layers.*.attn.kernel_k_proj_DKH",
            "language_model.model.layers.*.self_attn.v_proj.weight":
            "layers.*.attn.kernel_v_proj_DKH",
            "language_model.model.layers.*.self_attn.o_proj.weight":
            "layers.*.attn.kernel_o_proj_NHD",
            "language_model.model.layers.*.feed_forward.router.weight":
            "layers.*.moe_ffw.router.kernel_DE",
            # shared experts
            "language_model.model.layers.*.feed_forward.shared_expert.down_proj.weight":
            "layers.*.shared_experts.kernel_down_proj_FD",
            "language_model.model.layers.*.feed_forward.shared_expert.gate_proj.weight":
            "layers.*.shared_experts.kernel_gating_DF",
            "language_model.model.layers.*.feed_forward.shared_expert.up_proj.weight":
            "layers.*.shared_experts.kernel_up_proj_DF",
            # dense layers
            "language_model.model.layers.*.feed_forward.down_proj.weight":
            "layers.*.dense_ffw.kernel_down_proj_FD",
            "language_model.model.layers.*.feed_forward.up_proj.weight":
            "layers.*.dense_ffw.kernel_up_proj_DF",
            "language_model.model.layers.*.feed_forward.gate_proj.weight":
            "layers.*.dense_ffw.kernel_gating_DF",
        }

        if self.quantization_config is None:
            self._loaded_to_standardized_keys.update(EXPERT_MAPPINGS_FUSED)
        else:
            self._loaded_to_standardized_keys.update(EXPERT_MAPPINGS_UNFUSED)
        self.pp_missing_layers = []

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            layer_num = self._get_layer_num(loaded_key)
            layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)

            expert_match = re.search(r"experts\.(\d+)", layer_key)
            if expert_match:
                # Key for lookup eg: layers.*.feed_forward.experts.*.down_proj.weight
                layer_key = re.sub(r"experts\.\d+", "experts.*", layer_key)

            mapped_key = self._loaded_to_standardized_keys.get(
                layer_key, loaded_key)
            mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                mapped_key)
        else:
            mapped_key = self._loaded_to_standardized_keys.get(
                loaded_key, loaded_key)
        return mapped_key

    def _map_llama4_gate_up_proj(self, model_for_loading: nnx.Module,
                                 model_params: nnx.State, loaded_name: str,
                                 loaded_weight: jax.Array):
        """HF's gate_up_proj is a fused tensor of gate and up projections. It needs to be split."""

        cast_type = jnp.dtype(jnp.bfloat16)
        # loaded_weight is a jax.Array when framework="flax", otherwise it's bfloat16
        if not isinstance(loaded_weight, jax.Array):
            loaded_weight = convert_torch_to_jax_with_view(
                loaded_weight, cast_type)

        split_weights = jnp.split(loaded_weight, 2, axis=-1)
        layer_num = self._get_layer_num(loaded_name)

        for split_type in ["gate", "up"]:
            split_loaded_name = loaded_name.replace("gate_up_proj",
                                                    f"{split_type}_proj")
            if split_type == "gate":
                mapped_name = "layers.*.moe_ffw.kernel_gating_EDF"
                loaded_weight = split_weights[0]
            else:
                mapped_name = "layers.*.moe_ffw.kernel_up_proj_EDF"
                loaded_weight = split_weights[1]

            mapped_name = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                 mapped_name)
            if _is_pp_missing_layer(mapped_name, self.pp_missing_layers):
                logger.debug(
                    f"Skip loading {mapped_name=} as it doesn't belong to this PP stage."
                )
                continue
            mapped_model_weight = get_param(model_params, mapped_name)

            if mapped_model_weight.value.shape != loaded_weight.shape:
                raise ValueError(
                    f"Loaded shape for {split_loaded_name}: {loaded_weight.shape} "
                    f"does not match model shape for {mapped_name}: {mapped_model_weight.value.shape}!"
                )

            mapped_model_weight.value = shard_put(loaded_weight,
                                                  mapped_model_weight.sharding,
                                                  mesh=model_for_loading.mesh)
            logger.debug(
                f"{split_loaded_name}: {loaded_weight.shape}  -->  {mapped_name}: {mapped_model_weight.value.shape}"
            )
            if self.is_verbose:
                print_param_info(mapped_model_weight, mapped_name)

    def _get_layer_num(self, loaded_key: str) -> Optional[int]:
        """
        Extracts the layer number from a HuggingFace weight key string.
        Returns the layer number (int) or None if no layer number is found.
        """
        match = re.search(r"layers\.(\d+)", loaded_key)
        if match:
            return int(match.group(1))
        return None

    def _get_expert_num(self, loaded_key: str) -> Optional[int]:
        """
        Extracts the expect number from a HuggingFace weight key string.
        Returns the expect number (int) or None if no expect number is found.
        """
        match = re.search(r"experts\.(\d+)\.", loaded_key)
        if match:
            return int(match.group(1))
        return None

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)

        for path, module in nnx.iter_graph(model_for_loading):
            if isinstance(module, PPMissingLayer):
                # this layer name is layers.{i} or final_norm, embedder, lm_head, etc
                layer_name = ".".join([str(s) for s in path])
                self.pp_missing_layers.append(layer_name)

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.get_weights_iterator():
                is_moe_layer = False
                layer_num = self._get_layer_num(loaded_name)
                expert_num = self._get_expert_num(loaded_name)
                # Quantized (FP8) checkpoints unstack the expert weights, while unquantized (BF16) checkpoints keep them stacked.
                is_unfused_expert = self.quantization_config is not None and expert_num is not None
                is_scale = loaded_name.endswith(".weight_scale")
                skip_layer = False
                if is_unfused_expert:
                    mapped_name = self.map_loaded_to_standardized_name(
                        loaded_name)
                    if _is_pp_missing_layer(mapped_name,
                                            self.pp_missing_layers):
                        skip_layer = True
                        logger.debug(
                            f"Skip loading {mapped_name} as it doesn't belong to this PP stage."
                        )
                        continue
                    model_weight = get_param(model_params, mapped_name)

                    if is_scale:
                        cast_type = model_weight.array.scale.value.dtype
                    else:
                        cast_type = model_weight.array.qvalue.value.dtype

                    loaded_weight = convert_torch_to_jax_with_view(
                        loaded_weight, cast_type)
                    loaded_weight = transpose_params(loaded_name,
                                                     loaded_weight,
                                                     self._transpose_map)

                    buffer_key = f"{mapped_name}_{'scale' if is_scale else 'qvalue'}"
                    if buffer_key not in self.expert_weights_buffer:
                        self.expert_weights_buffer[buffer_key] = {}
                    self.expert_weights_buffer[buffer_key][
                        expert_num] = loaded_weight
                    continue
                if skip_layer:
                    continue

                if layer_num is not None:
                    is_moe_layer = (layer_num + 1) % \
                            self.interleave_moe_layer_step == 0
                    self.expert_prefix = "shared_expert." if is_moe_layer else ""

                if "gate_up_proj" in loaded_name:
                    self._map_llama4_gate_up_proj(model_for_loading,
                                                  model_params, loaded_name,
                                                  loaded_weight)
                    continue

                mapped_name = self.map_loaded_to_standardized_name(loaded_name)
                if _is_pp_missing_layer(mapped_name, self.pp_missing_layers):
                    logger.debug(
                        f"Skip loading {mapped_name} as it doesn't belong to this PP stage."
                    )
                    continue
                model_weight = get_param(model_params, mapped_name)

                cast_type = model_weight.value.dtype
                if not isinstance(loaded_weight, jax.Array):
                    logger.debug(
                        f"Converting PyTorch tensor {loaded_name} to JAX {cast_type}"
                    )
                    loaded_weight = convert_torch_to_jax_with_view(
                        loaded_weight, cast_type)

                if not loaded_name.endswith(".bias"):
                    loaded_weight = reshape_params(loaded_name, loaded_weight,
                                                   self._weight_shape_map)
                    loaded_weight = transpose_params(loaded_name,
                                                     loaded_weight,
                                                     self._transpose_map)
                if model_weight.value.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                        f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                    )
                logger.debug(
                    f"Transformed parameter {loaded_name} to {mapped_name}: {loaded_weight.shape} --> {model_weight.value.shape}"
                )
                model_weight.value = shard_put(loaded_weight,
                                               model_weight.sharding,
                                               mesh=model_for_loading.mesh)
                if self.is_verbose:
                    print_param_info(model_weight, loaded_name)
            with jax.default_device(jax.devices("cpu")[0]):
                for buffer_key, expert_map in self.expert_weights_buffer.items(
                ):
                    sorted_exp_nums = sorted(expert_map.keys())
                    aggregated_weight = jnp.stack(
                        [expert_map[k] for k in sorted_exp_nums], axis=0)
                    is_scale = buffer_key.endswith("_scale")
                    base_mapped_name = buffer_key.replace("_scale",
                                                          "").replace(
                                                              "_qvalue", "")
                    if _is_pp_missing_layer(base_mapped_name,
                                            self.pp_missing_layers):
                        logger.debug(
                            f"Skip loading {base_mapped_name} as it doesn't belong to this PP stage."
                        )
                        continue
                    model_weight = get_param(model_params, base_mapped_name)

                    assert hasattr(
                        model_weight, 'array'
                    ), f"Expected MoE weight '{base_mapped_name}' to be a quantized array (qarray)"

                    if is_scale:
                        loaded_name = f"{base_mapped_name}.array.scale.value"
                        if model_weight.array.scale.value.shape != aggregated_weight.shape:
                            raise ValueError(
                                f"[AGGREGATED] Loaded shape for {buffer_key}: {aggregated_weight.shape}"
                                f"does not match model shape for {loaded_name}: {model_weight.array.scale.value.shape}!"
                            )

                        model_weight.array.scale.value = shard_put(
                            aggregated_weight,
                            model_weight.array.scale.sharding,
                            mesh=model_for_loading.mesh)

                    elif aggregated_weight.itemsize < 2:  # check model weight elem nbits < 16
                        loaded_name = f"{base_mapped_name}.array.qvalue.value"
                        if model_weight.array.qvalue.value.shape != aggregated_weight.shape:
                            raise ValueError(
                                f"[AGGREGATED] Loaded shape for {buffer_key}: {aggregated_weight.shape}"
                                f"does not match model shape for {loaded_name}: {model_weight.array.qvalue.value.shape}!"
                            )

                        model_weight.array.qvalue.value = shard_put(
                            aggregated_weight,
                            model_weight.array.qvalue.sharding,
                            mesh=model_for_loading.mesh)

                    logger.debug(
                        f"Aggregated and loaded {loaded_name}: {aggregated_weight.shape}"
                    )

                    if self.is_verbose:
                        print_param_info(model_weight, loaded_name)

        nnx.update(model_for_loading, model_params)


class Llama4ForCausalLM(nnx.Module):
    WeightLoader = Llama4WeightLoader

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        text_config = model_config.hf_config.text_config

        self.rng = nnx.Rngs(rng)
        self.mesh = mesh
        self.is_verbose = getattr(self.vllm_config.additional_config,
                                  "is_verbose", False)

        # Currently the runner will always set a mesh, so the custom default sharding (when
        #  no sharding is set in vllm config) doesn't take effect.
        # TODO(fhzhang): figure out whether we need to actually enable this.
        #    strategy_dict = {"tensor_parallelism": 4, "expert_parallelism": 2}

        self.vocab_size = model_config.get_vocab_size()
        self.hidden_size = model_config.get_hidden_size()

        dtype: jnp.dtype = jnp.bfloat16

        self.num_layers: int = getattr(text_config, "num_hidden_layers", 48)

        self.intermediate_size_moe: int = getattr(text_config,
                                                  "intermediate_size", 8192)
        self.intermediate_size_mlp = getattr(text_config,
                                             "intermediate_size_mlp", 16384)

        # num_local_experts: uses 16 experts for Llama-4-Scout-17B-16E-Instruct and uses 128 experts Llama-4-Maverick-17B-128E-Instruct.
        # The default value is set to 16 for compatibility with Llama-4-Scout.
        self.num_local_experts: int = getattr(text_config, "num_local_experts",
                                              16)
        self.hidden_act: str = getattr(text_config, "hidden_act", "silu")
        self.no_rope_layer_interval = 4

        # interleave_moe_layer_step has a layer step of 2 to interleave MoE and dense layers for Llama-4-Maverick-17B-128E-Instruct.
        # The default value is set to 1 for compatibility with Llama-4-Scout.
        self.interleave_moe_layer_step = getattr(text_config,
                                                 "interleave_moe_layer_step",
                                                 1)

        self.num_attention_heads = getattr(text_config, "num_attention_heads",
                                           40)
        self.num_key_value_heads = getattr(text_config, "num_key_value_heads",
                                           8)
        self.head_dim = getattr(text_config, "head_dim", 128)

        self.num_shared_experts = getattr(text_config, "num_experts_per_tok",
                                          1)
        self.rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-5)

        self.rope_scaling = getattr(text_config, "rope_scaling", None)
        if self.rope_scaling:
            self.rope_scaling["scale_factor"] = self.rope_scaling.pop("factor")

        self.use_qk_norm = getattr(text_config, "use_qk_norm", True)

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (model_config.hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embedder = Embedder(vocab_size=self.vocab_size,
                                     hidden_size=self.hidden_size,
                                     dtype=dtype,
                                     vd_sharding=(('data', 'expert', 'model'),
                                                  None),
                                     rngs=self.rng,
                                     random_init=force_random_weights)
        else:
            self.embedder = PPMissingLayer()

        layers = []
        self.start_layer, self.end_layer = get_start_end_layer(
            self.num_layers,
            get_pp_group().rank_in_group,
            get_pp_group().world_size)
        for i in range(self.start_layer):
            layers.append(PPMissingLayer())

        for i in range(self.start_layer, self.end_layer):
            # For Llama4-Scout, all layers are MoE layers.
            # This can be adjusted for other variants.
            is_moe_layer = (i + 1) % \
                            self.interleave_moe_layer_step == 0

            # Llama-4-Scout config: It has "no_rope_layers": []
            use_attention_rope = (i + 1) % self.no_rope_layer_interval != 0

            router = Router(dtype=dtype,
                            hidden_size=self.hidden_size,
                            num_experts=self.num_local_experts,
                            num_experts_per_tok=1,
                            router_act="sigmoid",
                            rngs=self.rng,
                            activation_ffw_td=('data', None),
                            ed_sharding=(None, None),
                            random_init=force_random_weights,
                            mesh=self.mesh)

            moe_ffw = JaxMoE(
                dtype=dtype,
                mesh=self.mesh,
                num_local_experts=self.num_local_experts,
                apply_expert_weight_before_computation=True,
                hidden_size=self.hidden_size,
                intermediate_size_moe=self.intermediate_size_moe,
                expert_axis_name=None,
                num_expert_parallelism=1,
                hidden_act=self.hidden_act,
                router=router,
                rngs=self.rng,
                activation_ffw_td=('data', None),
                activation_ffw_ted=P('data', 'expert', None),
                edf_sharding=('model', None, None),
                efd_sharding=('model', None, None),
                quant_config=vllm_config.quant_config,
                random_init=force_random_weights) if is_moe_layer else None

            dense_ffw = DenseFFW(dtype=dtype,
                                 hidden_act=self.hidden_act,
                                 hidden_size=self.hidden_size,
                                 intermediate_size=self.intermediate_size_mlp,
                                 random_init=force_random_weights,
                                 rngs=self.rng,
                                 df_sharding=(None, 'model'),
                                 fd_sharding=('model', None),
                                 activation_ffw_td=('data', None),
                                 mesh=self.mesh) if not is_moe_layer else None

            attn = Llama4Attention(
                hidden_size=self.hidden_size,
                dtype=dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rope_theta=500000.0,
                # https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
                rope_scaling=self.rope_scaling,
                rngs=self.rng,
                rope_input_ordering="interleaved",
                temperature_tuning=True,
                temperature_tuning_scale=0.1,
                temperature_tuning_floor_scale=8192,
                use_qk_norm=self.use_qk_norm,
                attention_chunk_size=None if use_attention_rope else 8192,
                mesh=self.mesh,
                random_init=force_random_weights,
                activation_attention_td=NamedSharding(self.mesh,
                                                      P('data', 'model')),
                activation_q_td=NamedSharding(self.mesh, P('data', 'model')),
                query_tnh=P('data', 'model', None),
                keyvalue_skh=P('data', 'model', None),
                activation_attention_out_td=NamedSharding(
                    self.mesh, P('data', 'model')),
                attn_o_tnh=P('data', 'model', None),
                dnh_sharding=(None, 'model', None),
                dkh_sharding=(None, 'model', None),
                nhd_sharding=('model', None, None),
            )

            shared_experts = DenseFFW(
                dtype=dtype,
                hidden_act=self.hidden_act,
                hidden_size=self.hidden_size,
                intermediate_size=self.num_shared_experts *
                self.intermediate_size_moe,
                rngs=self.rng,
                random_init=force_random_weights,
                df_sharding=(None, 'model'),
                fd_sharding=('model', None),
                activation_ffw_td=('data', None),
                mesh=self.mesh) if is_moe_layer else None

            pre_attention_norm = RMSNorm(
                dims=self.hidden_size,
                random_init=force_random_weights,
                epsilon=self.rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=dtype,
                activation_ffw_td=NamedSharding(self.mesh, P('data', None)),
            )

            pre_mlp_norm = RMSNorm(
                dims=self.hidden_size,
                epsilon=self.rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=dtype,
                random_init=force_random_weights,
                activation_ffw_td=NamedSharding(self.mesh, P('data', None)),
            )

            block = SharedExpertsTransformerBlock(
                moe_ffw=moe_ffw if is_moe_layer else None,
                dense_ffw=dense_ffw if not is_moe_layer else None,
                shared_experts=shared_experts if is_moe_layer else None,
                attn=attn,
                pre_attention_norm=pre_attention_norm,
                pre_mlp_norm=pre_mlp_norm,
                use_attention_rope=use_attention_rope)
            layers.append(block)

        for i in range(self.end_layer, self.num_layers):
            layers.append(PPMissingLayer())

        self.layers = nnx.List(layers)
        if self.is_last_rank:
            self.final_norm = RMSNorm(
                dims=self.hidden_size,
                epsilon=self.rms_norm_eps,
                rngs=self.rng,
                with_scale=True,
                dtype=dtype,
                random_init=force_random_weights,
                activation_ffw_td=NamedSharding(self.mesh, P()),
            )
            self.lm_head = LMhead(vocab_size=self.vocab_size,
                                  hidden_size=self.hidden_size,
                                  dtype=dtype,
                                  rngs=self.rng,
                                  vd_sharding=(('data', 'expert', 'model'),
                                               None),
                                  dv_sharding=(None, ('data', 'expert',
                                                      'model')),
                                  random_init=force_random_weights)
        else:
            self.final_norm = PPMissingLayer()
            self.lm_head = PPMissingLayer()

        if self.is_verbose:
            self._print_model_architecture()

    def _print_model_architecture(self):
        num_display_layers = max(self.interleave_moe_layer_step,
                                 self.no_rope_layer_interval)

        logger.info("### Embedding ###")
        nnx.display(self.embedder)

        logger.info(f"\n### First {num_display_layers} Layers ###")
        # Loop through the slice and display each layer
        for i, layer in enumerate(self.layers[:num_display_layers]):
            logger.info(f"\n--- Layer {i} ---")
            nnx.display(layer)

        logger.info("\n### LM Head ###")
        nnx.display(self.lm_head)

    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng)

        weight_loader = self.WeightLoader(
            vllm_config=self.vllm_config,
            hidden_size=self.hidden_size,
            attn_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attn_head_dim=self.head_dim)
        weight_loader.load_weights(self)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions: Optional[jax.Array] = None,
        _layer_name_to_kv_cache: Optional[Tuple[Tuple[str, int]]] = None,
        _lora_metadata: Any = None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]:
        is_prefill = False
        if self.is_first_rank:
            x_TD = self.embedder.encode(input_ids)
        else:
            assert intermediate_tensors is not None
            x_TD = intermediate_tensors["hidden_states"]

        for (i, block) in enumerate(
                islice(self.layers, self.start_layer, self.end_layer)):
            kv_cache = kv_caches[i]
            new_kv_cache, x_TD = block(x_TD, is_prefill, kv_cache,
                                       attention_metadata)
            jax.block_until_ready(x_TD)
            kv_caches[i] = new_kv_cache

        if not self.is_last_rank:
            return kv_caches, JaxIntermediateTensors({"hidden_states":
                                                      x_TD}), []

        final_activation_TD = self.final_norm(x_TD)
        return kv_caches, final_activation_TD, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        logits_TV = jnp.dot(hidden_states,
                            self.lm_head.input_embedding_table_DV.value)
        return logits_TV


# --- Vision Classes ---


class JAXUnfoldConvolution(nnx.Module):
    """
    A module that performs an "unfold" convolution operation,
    similar to a patch embedding layer in Vision Transformers.

    It reshapes input images into non-overlapping patches and then
    applies a linear projection to embed these patches into a higher-dimensional space.

    Attributes:
        kernel_size: The size of the image patches (e.g., 16 for 16x16 patches).
        num_channels: The number of input channels in the image (e.g., 3 for RGB).
        linear: A linear layer to project the flattened patches into the hidden size.
    """

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.kernel_size = cfg.patch_size
        self.num_channels = cfg.num_channels
        patch_flat_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size

        self.linear = nnx.Linear(
            patch_flat_dim,
            cfg.hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(),
                                              (None, "model")),
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        # Check if NCHW, flip to NHWC
        if inputs.shape[1] == self.num_channels and inputs.shape[
                -1] != self.num_channels:
            inputs = jnp.transpose(inputs, (0, 2, 3, 1))

        batch_size, height, width, channels = inputs.shape
        patch_size = self.kernel_size

        # 1. Reshape to separate tiles
        patches = inputs.reshape(batch_size, height // patch_size, patch_size,
                                 width // patch_size, patch_size, channels)

        # 2. Transpose to group grid cells
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))

        # 3. Flatten patches
        patches = patches.reshape(batch_size, -1,
                                  patch_size * patch_size * channels)

        # 4. Project
        hidden_states = self.linear(patches)

        return hidden_states


class JAXLlama4VisionMLP(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.fc1 = nnx.Linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.glorot_uniform(), (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros,
                                            ("model", )),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.glorot_uniform(), ("model", None)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None, )),
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class JAXLlama4VisionEncoderLayer(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.hidden_size = cfg.hidden_size
        num_attention_heads = getattr(cfg, "num_attention_heads", 16)
        num_key_value_heads = getattr(
            cfg,
            "num_key_value_heads",  #cfg doesn't have this value
            num_attention_heads)
        rope_theta = getattr(cfg, "rope_theta", 10000.0)

        self.self_attn = Llama4VisionAttention(
            hidden_size=cfg.hidden_size,
            dtype=dtype,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
            head_dim=cfg.hidden_size // num_attention_heads,
            rope_theta=rope_theta,
            rope_scaling=None,
            rngs=rngs,
            rope_input_ordering="interleaved",
            temperature_tuning=False,
            use_qk_norm=False,
            mesh=mesh,
            is_causal=False,  # Forces bidirectional mask for ViT Encoder
            temperature_tuning_floor_scale=0,
            temperature_tuning_scale=0.0,
            activation_attention_td=None,
            activation_attention_out_td=None,
            activation_q_td=P('data', 'model'),
        )

        self.mlp = JAXLlama4VisionMLP(cfg,
                                      rngs=rngs,
                                      dtype=dtype,
                                      random_init=random_init)

        self.input_layernorm = nnx.LayerNorm(
            cfg.hidden_size,
            epsilon=cfg.norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, P()),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, P()))
        self.post_attention_layernorm = nnx.LayerNorm(
            cfg.hidden_size,
            epsilon=cfg.norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, P()),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, P()))

    def __call__(self, hidden_state: jax.Array, freqs_ci_stacked: jax.Array,
                 **kwargs) -> jax.Array:

        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)

        original_shape = hidden_state.shape
        B, S, D = original_shape

        hidden_state_2D = hidden_state.reshape(-1, original_shape[-1])

        vision_metadata = AttentionMetadata(input_positions=jnp.array([]))
        new_kv_cache, attention_output_2D = self.self_attn(
            x=hidden_state_2D,
            is_prefill=True,
            kv_cache=None,  # Vision Encoder does not use KV cache
            attention_metadata=vision_metadata,
            freqs_cis=freqs_ci_stacked,
            use_attention_rope=True,
            **kwargs)

        attention_output = attention_output_2D.reshape(original_shape)

        hidden_state = residual + attention_output

        residual = hidden_state

        # MLP
        hidden_state = self.post_attention_layernorm(hidden_state)

        hidden_state_2D = hidden_state.reshape(B * S, D)

        hidden_state_2D = self.mlp(hidden_state_2D)

        hidden_state = hidden_state_2D.reshape(B, S, D)

        hidden_state = residual + hidden_state

        return hidden_state


class JAXLlama4VisionEncoder(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        num_layers = cfg.num_hidden_layers if 'num_hidden_layers' in cfg else 34
        self.layers = nnx.List([
            JAXLlama4VisionEncoderLayer(cfg,
                                        rngs=rngs,
                                        dtype=dtype,
                                        random_init=random_init,
                                        mesh=mesh) for _ in range(num_layers)
        ])

    def __call__(self, hidden_states: jax.Array, freqs_ci_stacked: jax.Array,
                 **kwargs) -> jax.Array:
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, freqs_ci_stacked,
                                          **kwargs)
        return hidden_states


def jax_pixel_shuffle(input_tensor: jax.Array,
                      shuffle_ratio: float) -> jax.Array:
    """
    Rearranges elements in a tensor of shape [B, L, C] according to a shuffle ratio.

    This operation reshapes the sequence of patches into a 2D grid and then redistributes
    elements between the spatial dimensions and the channel dimension. It is commonly
    used in vision models for spatial downsampling (ratio < 1) or upsampling (ratio > 1)
    while preserving information in the channel dimension.

    Args:
        input_tensor: Input tensor of shape [batch_size, num_patches, channels].
            Assumes num_patches is a perfect square.
        shuffle_ratio: The ratio used to scale the spatial dimensions (Height and Width).

    Returns:
        The shuffled tensor of shape [batch_size, new_num_patches, new_channels].
    """
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    # Reshape to [batch_size, patch_size, patch_size, channels]
    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    # Reshape 1: [batch_size, height, width * shuffle_ratio, channels / shuffle_ratio]
    reshaped_tensor = input_tensor.reshape(batch_size, height,
                                           int(width * shuffle_ratio),
                                           int(channels / shuffle_ratio))
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

    # Reshape 2: [batch_size, height * shuffle_ratio, width * shuffle_ratio, channels / (shuffle_ratio^2)]
    reshaped_tensor = reshaped_tensor.reshape(
        batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)))
    reshaped_tensor = reshaped_tensor.transpose(0, 2, 1, 3)

    # Reshape back to [batch_size, num_new_patches, channels_out]
    output_tensor = reshaped_tensor.reshape(batch_size, -1,
                                            reshaped_tensor.shape[-1])
    return output_tensor


class JAXLlama4VisionMLP2(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16):
        cfg = config
        self.fc1 = nnx.Linear(
            cfg.intermediate_size,
            cfg.projector_input_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.glorot_uniform(), (None, "model")),
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            cfg.projector_output_dim,
            cfg.projector_output_dim,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.glorot_uniform(), ("model", None)),
            rngs=rngs,
        )

        self.dropout_rate = cfg.projector_dropout

        if self.dropout_rate > 0:
            self.dropout = nnx.Dropout(self.dropout_rate, rngs=rngs)
        else:
            self.dropout = nnx.Dropout(0.0)

    def __call__(self,
                 hidden_states: jax.Array,
                 deterministic: bool = False) -> jax.Array:
        # First linear layer with GELU activation
        hidden_states = self.fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)

        # Apply dropout
        hidden_states = self.dropout(hidden_states,
                                     deterministic=deterministic)

        # Second linear layer with GELU activation
        hidden_states = self.fc2(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        return hidden_states


class JAXLlama4VisionPixelShuffleMLP(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.pixel_shuffle_ratio = cfg.pixel_shuffle_ratio
        self.pixel_shuffle_mlp = JAXLlama4VisionMLP2(cfg,
                                                     rngs=rngs,
                                                     dtype=dtype)

    def __call__(self,
                 encoded_patches: jax.Array,
                 deterministic: bool = False) -> jax.Array:
        # Apply pixel shuffle operation
        encoded_patches = jax_pixel_shuffle(encoded_patches,
                                            self.pixel_shuffle_ratio)

        # Apply MLP transformation
        result = self.pixel_shuffle_mlp(encoded_patches,
                                        deterministic=deterministic)
        return result


class JAXLlama4VisionModel(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 vision_rope: Llama4VisionRotaryEmbedding,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.scale = cfg.hidden_size**-0.5
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.norm_eps = cfg.norm_eps
        self.num_channels = cfg.num_channels

        # Number of patches = (grid_size**2) + 1 for the [CLS] token
        self.num_patches = (self.image_size // self.patch_size)**2 + 1

        self.patch_embedding = JAXUnfoldConvolution(cfg,
                                                    rngs=rngs,
                                                    dtype=dtype,
                                                    random_init=random_init)

        key_cls = rngs.params()
        key_pos = rngs.params()

        self.class_embedding = nnx.Param(
            self.scale *
            jax.random.normal(key_cls, (self.hidden_size, ), dtype=dtype),
            sharding=P())
        self.positional_embedding_vlm = nnx.Param(
            self.scale * jax.random.normal(
                key_pos, (self.num_patches, self.hidden_size), dtype=dtype),
            sharding=P(None, "model"))

        self.layernorm_pre = nnx.LayerNorm(
            self.hidden_size,
            epsilon=self.norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, P()),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, P()))
        self.layernorm_post = nnx.LayerNorm(
            self.hidden_size,
            epsilon=self.norm_eps,
            dtype=dtype,
            rngs=rngs,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, P()),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, P()))

        self.model = JAXLlama4VisionEncoder(cfg,
                                            rngs=rngs,
                                            mesh=mesh,
                                            dtype=dtype,
                                            random_init=random_init)

        self.vision_rope = vision_rope

        self.vision_adapter = JAXLlama4VisionPixelShuffleMLP(
            cfg, rngs=rngs, dtype=dtype, random_init=random_init)

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        input_shape = pixel_values.shape
        # This model expects NCHW format. We handle NHWC and 5D (video) inputs.
        if len(input_shape) == 5:
            # (Batch, Time, Channel, Height, Width) -> (B*T, C, H, W)
            b, t, c, h, w = input_shape
            pixel_values = pixel_values.reshape(b * t, c, h, w)
        elif len(input_shape) == 4 and input_shape[-1] == self.num_channels:
            # (Batch, Height, Width, Channel) -> (B, C, H, W)
            b, h, w, c = input_shape
            t = 1
            pixel_values = jnp.transpose(pixel_values, (0, 3, 1, 2))
        else:
            b, c, h, w = input_shape
            t = 1

        # 1. Unfold convolution
        hidden_states = self.patch_embedding(pixel_values)

        # 2. Add class embedding
        class_embedding_expanded = self.class_embedding.value[
            None, None, :].repeat(hidden_states.shape[0], axis=0)
        hidden_states = jnp.concatenate(
            [hidden_states, class_embedding_expanded], axis=1)

        # 3. Add positional embedding
        hidden_states += self.positional_embedding_vlm.value

        # 4. Transformation layers
        hidden_states = self.layernorm_pre(hidden_states)
        freqs_ci_stacked = self.vision_rope()
        hidden_states = self.model(hidden_states, freqs_ci_stacked)

        hidden_states = self.layernorm_post(hidden_states)

        encoder_out_no_cls = hidden_states[:, :-1, :]

        # 5. Remove CLS token
        hidden_states = encoder_out_no_cls

        # 6. Vision Adapter (Pixel Shuffle MLP)
        hidden_states = self.vision_adapter(hidden_states)

        # 7. Reshape back to [B, T * N_patches, H_out]
        _, _, patch_dim = hidden_states.shape
        return hidden_states.reshape(b, -1, patch_dim)


class JAXLlama4MultiModalProjector(nnx.Module):

    def __init__(self,
                 config: dict,
                 rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.bfloat16,
                 random_init: bool = False):
        cfg = config
        self.linear = nnx.Linear(
            cfg["vision_config"].vision_output_dim,
            cfg["text_config"].hidden_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(),
                                              (None, "model")),
            rngs=rngs,
        )

    def __call__(self, image_features: jax.Array) -> jax.Array:
        # image_features: [batch, num_patches, vision_output_dim=4096]
        hidden_states = self.linear(image_features)
        return hidden_states  # Output shape: [batch, num_patches, text_hidden_size=5120]


# --- END: Vision Classes ---