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
from vllm.config import VllmConfig

from tpu_inference.layers.jax.attention.gpt_oss_attention import (
    AttentionMetadata, GptOssAttention)
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.moe.gpt_oss_moe import GptOssMoE, GptOssRouter
from tpu_inference.layers.jax.transformer_block import TransformerBlock
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
class GptOss(nnx.Module):
    """
    JAX implementation of the GPT-OSS model architecture.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: jax.Array,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        assert mesh is not None

        self.vllm_config = vllm_config
        self.hf_config = vllm_config.model_config.hf_config
        self.rng = nnx.Rngs(rng)

        num_layers: int = self.hf_config.num_hidden_layers
        num_experts: int = self.hf_config.num_local_experts
        vocab_size: int = self.hf_config.vocab_size
        num_attention_heads: int = self.hf_config.num_attention_heads
        num_key_value_heads: int = self.hf_config.num_key_value_heads
        head_dim: int = self.hf_config.head_dim
        hidden_size: int = self.hf_config.hidden_size
        ffw_intermediate_size: int = self.hf_config.intermediate_size
        num_experts_per_token: int = self.hf_config.num_experts_per_tok
        rms_norm_eps: float = self.hf_config.rms_norm_eps
        swiglu_limit: float = self.hf_config.swiglu_limit

        rope_theta: float = self.hf_config.rope_theta
        rope_scaling_factor: float = self.hf_config.rope_scaling["factor"]
        rope_ntk_alpha: float = self.hf_config.rope_scaling["beta_slow"]
        rope_ntk_beta: float = self.hf_config.rope_scaling["beta_fast"]
        initial_context_length: int = self.hf_config.rope_scaling[
            "original_max_position_embeddings"]

        dtype: jnp.dtype = jnp.bfloat16

        self.sliding_window = self.hf_config.sliding_window

        self.random_init = force_random_weights or self.vllm_config.additional_config.get(
            "random_weights", False)
        self.mesh = mesh

        self.embedder = Embedder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=self.rng,
            vd_sharding=(('data', 'model'), None),
            random_init=self.random_init,
        )

        self.layers = []
        for i in range(num_layers):
            attn = GptOssAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=dtype,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                rope_theta=rope_theta,
                initial_context_length=initial_context_length,
                rope_scaling_factor=rope_scaling_factor,
                rope_ntk_alpha=rope_ntk_alpha,
                rope_ntk_beta=rope_ntk_beta,
                rngs=self.rng,
                random_init=self.random_init,
                query_tnh=P(None, 'model', None),
                keyvalue_skh=P(None, 'model', None),
                attn_o_tnh=P(None, 'model', None),
                dnh_sharding=(None, 'model', None),
                dkh_sharding=(None, 'model', None),
                nhd_sharding=('model', None, None),
                mesh=self.mesh,
            )

            # MoE MLP block
            router = GptOssRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_token,
                rngs=self.rng,
                dtype=dtype,
                router_act='softmax',
                random_init=self.random_init,
                activation_ffw_td=('data', None),
                ed_sharding=('model', None),
                e_sharding=('model', ),
            )

            moe_mlp = GptOssMoE(
                dtype=dtype,
                num_local_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size_moe=ffw_intermediate_size,
                rngs=self.rng,
                random_init=self.random_init,
                router=router,
                swiglu_limit=swiglu_limit,
                # Sharding configuration
                activation_ffw_td=('data', None),
                edf_sharding=('model', None, None),
                efd_sharding=('model', None, None),
                ed_sharding=('model', None),
            )

            block = TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=dtype,
                    rngs=self.rng,
                ),
                pre_mlp_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=dtype,
                    rngs=self.rng,
                ),
                attn=attn,
                custom_module=moe_mlp,
            )
            self.layers.append(block)
        # Note: ALL RMSNorm does not upcast input to float32, while the pytorch does
        self.final_norm = RMSNorm(
            dims=hidden_size,
            rngs=self.rng,
            random_init=self.random_init,
            epsilon=rms_norm_eps,
            dtype=dtype,
        )

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=self.rng,
            vd_sharding=(('data', 'model'), None),
            dv_sharding=(None, ('data', 'model')),
            random_init=self.random_init,
        )

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        """Loads and transforms all weights from a checkpoint"""
        self.rng = nnx.Rngs(rng)

        # Format: 'hf_key': ('jax_model_path', transform_function, target_shape)
        transforms = {
            "transpose_reshape": lambda w, shape: w.T.reshape(shape),
            "reshape": lambda b, shape: b.reshape(shape),
            "transpose": lambda w, _: w.T,
        }

        mappings = {
            # Embeddings, Norms, and LM Head
            "model.embed_tokens.weight": ("embedder.input_embedding_table_VD",
                                          None, None),
            "lm_head.weight": ("lm_head.input_embedding_table_DV",
                               transforms["transpose"], None),
            "model.norm.weight": ("final_norm.scale", None, None),
            "model.layers.*.input_layernorm.weight":
            ("layers.*.pre_attention_norm.scale", None, None),
            "model.layers.*.post_attention_layernorm.weight":
            ("layers.*.pre_mlp_norm.scale", None, None),

            # Attention Weights
            "model.layers.*.self_attn.q_proj.weight":
            ("layers.*.attn.kernel_q_DNH", transforms["transpose_reshape"],
             (self.hf_config.hidden_size, self.hf_config.num_attention_heads,
              self.hf_config.head_dim)),
            "model.layers.*.self_attn.k_proj.weight":
            ("layers.*.attn.kernel_k_DKH", transforms["transpose_reshape"],
             (self.hf_config.hidden_size, self.hf_config.num_key_value_heads,
              self.hf_config.head_dim)),
            "model.layers.*.self_attn.v_proj.weight":
            ("layers.*.attn.kernel_v_DKH", transforms["transpose_reshape"],
             (self.hf_config.hidden_size, self.hf_config.num_key_value_heads,
              self.hf_config.head_dim)),
            "model.layers.*.self_attn.o_proj.weight":
            ("layers.*.attn.kernel_o_proj_NHD",
             transforms["transpose_reshape"],
             (self.hf_config.num_attention_heads, self.hf_config.head_dim,
              self.hf_config.hidden_size)),

            # Attention Biases
            "model.layers.*.self_attn.q_proj.bias":
            ("layers.*.attn.bias_q_NH", transforms["reshape"],
             (self.hf_config.num_attention_heads, self.hf_config.head_dim)),
            "model.layers.*.self_attn.k_proj.bias":
            ("layers.*.attn.bias_k_KH", transforms["reshape"],
             (self.hf_config.num_key_value_heads, self.hf_config.head_dim)),
            "model.layers.*.self_attn.v_proj.bias":
            ("layers.*.attn.bias_v_KH", transforms["reshape"],
             (self.hf_config.num_key_value_heads, self.hf_config.head_dim)),
            "model.layers.*.self_attn.o_proj.bias": ("layers.*.attn.bias_o_D",
                                                     None, None),

            # Sinks
            "model.layers.*.self_attn.sinks": ("layers.*.attn.sinks_N", None,
                                               None),

            # MoE Weights
            "model.layers.*.mlp.router.weight":
            ("layers.*.custom_module.router.kernel_DE",
             transforms["transpose"], None),
            "model.layers.*.mlp.router.bias":
            ("layers.*.custom_module.router.bias_E", None, None),
            "model.layers.*.mlp.experts.gate_up_proj":
            ("layers.*.custom_module.mlp1_weight_EDF2", None, None),
            "model.layers.*.mlp.experts.gate_up_proj_bias":
            ("layers.*.custom_module.mlp1_bias_EF2", None, None),
            "model.layers.*.mlp.experts.down_proj":
            ("layers.*.custom_module.mlp2_weight_EFD", None, None),
            "model.layers.*.mlp.experts.down_proj_bias":
            ("layers.*.custom_module.mlp2_bias_ED", None, None),
        }

        model_params = nnx.state(self)
        is_verbose = self.vllm_config.additional_config.get(
            "is_verbose", False)

        names_and_weights_generator = model_weights_generator(
            model_name_or_path=self.vllm_config.model_config.model,
            framework="pt",
            download_dir=self.vllm_config.load_config.download_dir)

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in names_and_weights_generator:
                hf_pattern = re.sub(r"layers\.(\d+)", "layers.*", loaded_name)
                if hf_pattern not in mappings:
                    logger.warning(
                        f"No mapping found for checkpoint tensor: {loaded_name}. Skipping."
                    )
                    continue

                jax_path_template, transform_fn, target_shape = mappings[
                    hf_pattern]

                layer_num_match = re.search(r"layers\.(\d+)", loaded_name)
                jax_path = jax_path_template
                if layer_num_match:
                    jax_path = jax_path_template.replace(
                        "*", layer_num_match.group(1))

                model_weight = get_param(model_params, jax_path)
                cast_type = model_weight.value.dtype

                if jax_path_template == "layers.*.attn.sinks_N":
                    # Checkpoint is bf16, but we have to upcast sinks to f32, as required by RPA_v3 kernel
                    weight_np = jnp.array(
                        loaded_weight.to(torch.float32).numpy())
                else:
                    torch_view_type = DTYPE_VIEW_MAP.get(jnp.dtype(cast_type))
                    if torch_view_type:
                        # Avoid unnecessary upcasting and mem copy by viewing the tensor's
                        # raw data as integers before converting to a JAX array.
                        weight_np = jnp.array(
                            loaded_weight.view(torch_view_type).numpy()).view(
                                cast_type)
                    else:
                        raise ValueError(
                            f"Unsupported dtype for tensor conversion: {cast_type}"
                        )

                if transform_fn:
                    transformed_weight = transform_fn(weight_np, target_shape)
                else:
                    transformed_weight = weight_np

                if model_weight.value.shape != transformed_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for '{jax_path}': Model expects {model_weight.value.shape}, but got {transformed_weight.shape} after transformation."
                    )

                def get_slice(index):
                    return transformed_weight[index]

                sharded_array = jax.make_array_from_callback(
                    transformed_weight.shape,
                    NamedSharding(self.mesh, P(*model_weight.sharding)),
                    get_slice)
                model_weight.value = sharded_array

                if is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(self, model_params)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]:
        is_prefill = False
        x = self.embedder.encode(input_ids)

        for i, block in enumerate(self.layers):
            kv_cache = kv_caches[i]
            current_sliding_window = self.sliding_window if i % 2 == 0 else None
            attention_metadata.sliding_window = current_sliding_window

            new_kv_cache, x = block(x, is_prefill, kv_cache,
                                    attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)
        return kv_caches, final_activation, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head.decode(hidden_states)
