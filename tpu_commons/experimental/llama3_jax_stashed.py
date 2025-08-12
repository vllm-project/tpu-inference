# TODO: Update documentation

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import (
    Attention, AttentionMetadata)
from tpu_commons.models.jax.common.constants import KVCacheType
from tpu_commons.models.jax.common.layers import (DenseFFW, Embedder, LMhead,
                                                  RMSNorm)
from tpu_commons.models.jax.common.model import Model
from tpu_commons.models.jax.common.transformer_block import TransformerBlock
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.utils.weight_utils import (get_model_weights_files,
                                                       get_param,
                                                       model_weights_generator,
                                                       reshape_params,
                                                       transpose_params)

logger = init_logger(__name__)


class LlamaForCausalLM(Model):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: jax.Array,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng)
        self.mesh = mesh

        model_name = self.vllm_config.model_config.model.lower()
        if "70b" in model_name:
            logger.info("Initializing Llama3 70B model variant.")
            self.hidden_size = 8192
            num_layers = 80
            self.num_attention_heads = 64
            self.num_key_value_heads = 8
            intermediate_size = 28672
        elif "8b" in model_name:
            logger.info("Initializing Llama3 8B model variant.")
            self.hidden_size = 4096
            num_layers = 32
            self.num_attention_heads = 32
            self.num_key_value_heads = 8
            intermediate_size = 14336
        else:
            raise ValueError(
                f"Could not determine Llama3 variant (8B or 70B) from model name: '{model_name}'. "
                "Please ensure '8b' or '70b' is in the model path.")

        dtype = jnp.bfloat16
        self.head_dim = 128
        rope_theta = 500000.0
        vocab_size = 128256
        rms_norm_eps = 1e-5

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=self.hidden_size,
                                 dtype=dtype,
                                 mesh=self.mesh,
                                 random_init=force_random_weights,
                                 vd_sharding=NamedSharding(
                                     self.mesh, P("model", None)),
                                 prelogit_td=NamedSharding(self.mesh, P()))
        self.embedder.generate_kernel(self.rng)

        self.layers = [
            TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=self.hidden_size,
                    mesh=self.mesh,
                    random_init=force_random_weights,
                    epsilon=rms_norm_eps,
                    activation_ffw_td=NamedSharding(self.mesh, P()),
                    with_scale=True,
                    dtype=dtype,
                ),
                pre_mlp_norm=RMSNorm(
                    dims=self.hidden_size,
                    mesh=self.mesh,
                    random_init=force_random_weights,
                    activation_ffw_td=NamedSharding(self.mesh, P()),
                    epsilon=rms_norm_eps,
                    with_scale=True,
                    dtype=dtype,
                ),
                attn=Attention(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    rope_theta=rope_theta,
                    rope_scaling={},
                    dtype=dtype,
                    mesh=self.mesh,
                    random_init=force_random_weights,
                    dnh_sharding=NamedSharding(self.mesh,
                                               P(None, "model", None)),
                    dkh_sharding=NamedSharding(self.mesh,
                                               P(None, "model", None)),
                    nhd_sharding=NamedSharding(self.mesh,
                                               P("model", None, None)),
                    activation_q_td=NamedSharding(self.mesh, P()),
                    query_tnh=NamedSharding(self.mesh, P(None, "model", None)),
                    keyvalue_skh=NamedSharding(self.mesh,
                                               P(None, "model", None)),
                    keyvalue_cache_lskh=NamedSharding(
                        self.mesh, P(None, None, "model", None)),
                    attn_o_tnh=NamedSharding(self.mesh, P(None, "model",
                                                          None)),
                ),
                custom_module=DenseFFW(
                    dtype=dtype,
                    hidden_act="silu",
                    hidden_size=self.hidden_size,
                    intermediate_size=intermediate_size,
                    mesh=self.mesh,
                    df_sharding=NamedSharding(self.mesh, P(None, "model")),
                    fd_sharding=NamedSharding(self.mesh, P("model", None)),
                    activation_ffw_td=NamedSharding(self.mesh, P()),
                    random_init=force_random_weights),
            ) for _ in range(num_layers)
        ]
        for i in range(len(self.layers)):
            self.layers[i].generate_kernel(self.rng)

        self.final_norm = RMSNorm(
            dims=self.hidden_size,
            mesh=self.mesh,
            random_init=force_random_weights,
            activation_ffw_td=NamedSharding(self.mesh, P()),
            epsilon=rms_norm_eps,
            with_scale=True,
            dtype=dtype,
        )
        self.final_norm.generate_kernel(self.rng)

        self.lm_head = LMhead(vocab_size=vocab_size,
                              hidden_size=self.hidden_size,
                              dtype=dtype,
                              mesh=self.mesh,
                              prelogit_td=NamedSharding(self.mesh, P()),
                              vd_sharding=None,
                              dv_sharding=NamedSharding(
                                  self.mesh, P(None, 'model')),
                              random_init=force_random_weights)
        self.lm_head.generate_kernel(self.rng)

    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        self.rng = nnx.Rngs(rng)
        weight_loader = Llama3WeightLoader(
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
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array]:
        is_prefill = False
        with jax.named_scope("llama_embed_input"):  #Embedding
            x_TD = self.embedder.encode(input_ids)

        with jax.named_scope("llama_model_transformer_blocks"):
            for (i, layer) in enumerate(self.layers):
                kv_cache = kv_caches[i]

                # The first layer is unscoped to avoid JAX tracing issues.
                # JAX's profiler may incorrectly apply the scope name from the first
                # layer's kernel compilation to all subsequent layers. Skipping the
                # first layer ensures distinct scope names for the remaining layers.
                if i == 0:
                    new_kv_cache, x_TD = layer(x_TD, is_prefill, kv_cache,
                                               attention_metadata)
                else:
                    with jax.named_scope(f'layer_{i}'):
                        new_kv_cache, x_TD = layer(x_TD, is_prefill, kv_cache,
                                                   attention_metadata)

                kv_caches[i] = new_kv_cache

        with jax.named_scope(
                "llama_final_norm"):  #Norm after last transformer block
            final_activation_TD = self.final_norm(x_TD)

        return kv_caches, final_activation_TD

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        with jax.named_scope("llama_lm_head_projection"
                             ):  #LM head projection to produce logits
            logits_TV = jnp.dot(hidden_states,
                                self.lm_head.input_embedding_table_DV.value)

        return logits_TV


class Llama3WeightLoader:

    def __init__(self, vllm_config: VllmConfig, hidden_size, attn_heads,
                 num_key_value_heads, attn_head_dim):
        self._transpose_map = {
            "lm_head": (1, 0),
            "gate_proj": (1, 0),
            "up_proj": (1, 0),
            "down_proj": (1, 0),
            "q_proj": (2, 0, 1),
            "k_proj": (2, 0, 1),
            "v_proj": (2, 0, 1),
            "o_proj": (1, 2, 0),
        }
        self._weight_shape_map = {
            "q_proj": (attn_heads, -1, hidden_size),
            "k_proj": (num_key_value_heads, -1, hidden_size),
            "v_proj": (num_key_value_heads, -1, hidden_size),
            "o_proj": (hidden_size, attn_heads, -1),
        }
        self._bias_shape_map = {
            "q_proj.bias": (attn_heads, attn_head_dim),
            "k_proj.bias": (num_key_value_heads, attn_head_dim),
            "v_proj.bias": (num_key_value_heads, attn_head_dim)
        }

        # Set the mappings from loaded parameter keys to standardized names.
        self._loaded_to_standardized_keys = {
            "model.embed_tokens": "embedder.input_embedding_table_VD",
            "model.layers.*.input_layernorm":
            "layers.*.pre_attention_norm.scale",
            "model.layers.*.mlp.down_proj":
            "layers.*.custom_module.kernel_down_proj_FD",
            "model.layers.*.mlp.gate_proj":
            "layers.*.custom_module.kernel_gating_DF",
            "model.layers.*.mlp.up_proj":
            "layers.*.custom_module.kernel_up_proj_DF",
            "model.layers.*.post_attention_layernorm":
            "layers.*.pre_mlp_norm.scale",
            "model.layers.*.self_attn.k_proj":
            "layers.*.attn.kernel_k_proj_DKH",
            "model.layers.*.self_attn.o_proj":
            "layers.*.attn.kernel_o_proj_NHD",
            "model.layers.*.self_attn.q_proj":
            "layers.*.attn.kernel_q_proj_DNH",
            "model.layers.*.self_attn.v_proj":
            "layers.*.attn.kernel_v_proj_DKH",
            "model.norm": "final_norm.scale",
            "lm_head": "lm_head.input_embedding_table_DV"
        }
        self.vllm_config = vllm_config

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        # Find the corresponding model key using the HF key
        if "layer" in loaded_key:
            layer_num_match = re.search(r"layers\.(\d+)", loaded_key)
            if layer_num_match:
                layer_num = layer_num_match.group(1)
                layer_key = re.sub(r"layers\.\d+", "layers.*", loaded_key)
                mapped_key = self._loaded_to_standardized_keys.get(
                    layer_key, layer_key)
                mapped_key = re.sub(r"layers\.\*", f"layers.{layer_num}",
                                    mapped_key)
                return mapped_key

        return self._loaded_to_standardized_keys.get(loaded_key, loaded_key)

    def load_weights_single_thread(self, model_params, weights_file, mesh):
        for loaded_name, loaded_weight in model_weights_generator(
                weights_file, framework="flax"):
            old_param_name = loaded_name
            if loaded_name.endswith(".weight"):
                loaded_name = loaded_name.removesuffix(".weight")
            mapped_name = self.map_loaded_to_standardized_name(loaded_name)
            model_weight = get_param(model_params, mapped_name)

            if model_weight is None:
                logger.warning(
                    f"Could not find a matching model parameter for loaded weight: '{old_param_name}' (mapped to: '{mapped_name}')"
                )
                continue

            logger.debug(
                f"{old_param_name}: {loaded_weight.shape}  -->  {mapped_name}: {model_weight.value.shape}"
            )
            if loaded_name.endswith(".bias"):
                loaded_weight = reshape_params(loaded_name, loaded_weight,
                                               self._bias_shape_map)
            else:
                loaded_weight = reshape_params(loaded_name, loaded_weight,
                                               self._weight_shape_map)
                loaded_weight = transpose_params(loaded_name, loaded_weight,
                                                 self._transpose_map)
            if model_weight.value.shape != loaded_weight.shape:
                raise ValueError(
                    f"Loaded shape for {loaded_name}: {loaded_weight.shape} "
                    f"does not match model shape for {mapped_name}: {model_weight.value.shape}!"
                )
            model_weight.value = shard_put(loaded_weight,
                                           model_weight.sharding.spec,
                                           mesh=mesh)

    def load_weights(self, model_for_loading: nnx.Module):
        model_params = nnx.state(model_for_loading)
        model_path = self.vllm_config.model_config.model
        weights_files = get_model_weights_files(model_path)
        max_workers = min(64, len(weights_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.load_weights_single_thread, model_params,
                                weights_file, model_for_loading.mesh)
                for weights_file in weights_files
            ]
            for future in futures:
                future.result()

        # TODO: validate that all of the model_params were accounted for as well.
        nnx.update(model_for_loading, model_params)
