# gpt_oss.py

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
from tpu_inference.layers.jax.attention.gpt_oss_attention import (GptOssAttention,
                                                           AttentionMetadata)
from tpu_inference.layers.jax.constants import KVCacheType
from tpu_inference.layers.jax.layers import DenseFFW, Embedder, LMhead, RMSNorm
from tpu_inference.layers.jax.moe.gpt_oss_moe import GptOssMoE, GptOssRouter
from tpu_inference.layers.jax.transformer_block import TransformerBlock
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (get_param,
                                                          model_weights_generator,
                                                          print_param_info)

logger = init_logger(__name__)


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
        self.rng = nnx.Rngs(rng)

        # Model hyperparameters from GPT-OSS config
        # TODO: verify the the default is 36(?)
        # TODO: update the parameters to hf_config, instead of being hardcoded 
        # after validating accuracy
        num_layers: int = 36
        num_experts: int = 128
        vocab_size: int = 201088
        num_attention_heads: int = 64
        num_key_value_heads: int = 8
        head_dim: int = 64
        hidden_size: int = 2880
        ffw_intermediate_size: int = 2880
        num_experts_per_token: int = 4
        sliding_window: int = 128
        swiglu_limit: float = 7.0
        rms_norm_eps: float = 1e-05
        rope_theta: float = 150000.0
        rope_scaling_factor: float = 32.0
        rope_ntk_alpha: float = 1.0
        rope_ntk_beta: float = 32.0
        initial_context_length: int = 4096
        dtype: jnp.dtype = jnp.bfloat16

        self.sliding_window = sliding_window
        self.random_init = force_random_weights or self.vllm_config.additional_config.get(
            "random_weights", False)
        self.mesh = mesh

        self.weight_loader = GptOssWeightLoader(
            vllm_config=vllm_config,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=ffw_intermediate_size,
            num_experts=num_experts,
        )

        self.embedder = Embedder(vocab_size=vocab_size,
                                 hidden_size=hidden_size,
                                 dtype=dtype,
                                 rngs=self.rng,
                                 vd_sharding=(('data', 'expert', 'model'),
                                              None),
                                 random_init=self.random_init)

        self.layers = []
        for i in range(num_layers):
            attn = GptOssAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                dtype=dtype,
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
                mesh=self.mesh
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
                e_sharding=('model',),
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
                ed_sharding=('model', None)
            )

            block = TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=jnp.float32,
                    rngs=self.rng,
                ),
                pre_mlp_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=jnp.float32,
                    rngs=self.rng,
                ),
                attn=attn,
                custom_module=moe_mlp
            )
            self.layers.append(block)

        self.final_norm = RMSNorm(
            dims=hidden_size,
            rngs=self.rng,
            random_init=self.random_init,
            epsilon=rms_norm_eps,
            dtype=jnp.float32,
        )

        self.unembedding = LMhead(vocab_size=vocab_size,
                              hidden_size=hidden_size,
                              dtype=dtype,
                              rngs=self.rng,
                              vd_sharding=(('data', 'expert', 'model'), None),
                              dv_sharding=(None, ('data', 'expert', 'model')),
                              random_init=self.random_init)

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
    ) -> Tuple[List[KVCacheType], jax.Array, List[jax.Array]]:
        is_prefill = False
        x = self.embedder.encode(input_ids)

        for i, block in enumerate(self.layers):
            kv_cache = kv_caches[i]
            # TODO: Only apply sliding window to every other layer
            current_sliding_window = self.sliding_window if i % 2 == 0 else None
            attention_metadata.sliding_window = current_sliding_window
            
            new_kv_cache, x = block(x, is_prefill, kv_cache, attention_metadata)
            kv_caches[i] = new_kv_cache

        final_activation = self.final_norm(x)
        return kv_caches, final_activation, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.unembedding.decode(hidden_states)


@dataclass
class GptOssWeightLoader:
    """
    Handles loading weights from a PyTorch checkpoint into the JAX GptOss model.
    """

    def __init__(self, vllm_config: VllmConfig, num_layers, hidden_size,
                 num_attention_heads, num_key_value_heads, head_dim,
                 intermediate_size, num_experts):

        self.vllm_config = vllm_config
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.names_and_weights_generator = model_weights_generator(
            model_name_or_path=vllm_config.model_config.model,
            framework="pt",
            download_dir=vllm_config.load_config.download_dir)
        self.is_verbose = vllm_config.additional_config.get(
            "is_verbose", False)

        self._transpose_map = {
            r"attn\.out\.weight": (1, 0),
            r"mlp\.gate\.weight": (1, 0),
            r"mlp\.mlp2_weight": (0, 2, 1),
            r"unembedding\.weight": (1, 0),
        }

