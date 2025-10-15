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
                e_sharding=('model'),
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

        # Note: RMSNorm does not upcast input to float32, while the pytorch does
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
            # Only apply sliding window to every other layer
            current_sliding_window = self.sliding_window if i % 2 == 0 else 0
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

        # JAX layers expect separate Q, K, V kernels, and Gating/Up projections for FFW.
        # These mappings point to the standardized names in our JAX model.
        self._loaded_to_standardized_keys = {
            # Embeddings
            "embedding.weight": "embedder.input_embedding_table_VD",
            "unembedding.weight": "lm_head.input_embedding_table_DV",
            # Final Norm
            "norm.weight": "final_norm.scale",
            # Per-layer norms
            "block.*.attn.norm.scale": "layers.*.pre_attention_norm.scale",
            "block.*.mlp.norm.scale": "layers.*.pre_mlp_norm.scale",
            # Attention
            "block.*.attn.out.weight": "layers.*.attn.kernel_o_proj_DH",
            # MoE Router
            "block.*.mlp.gate.weight": "layers.*.custom_module.router.kernel_DE",
            "block.*.mlp.gate.bias": "layers.*.custom_module.router.bias_E",
            # MoE Experts (Down projection)
            "block.*.mlp.mlp2_weight": "layers.*.custom_module.kernel_down_proj_EFD",
            "block.*.mlp.mlp2_bias": "layers.*.custom_module.bias_down_proj_ED",
        }

    def map_loaded_to_standardized_name(self, loaded_key: str) -> str:
        """Maps a HuggingFace checkpoint key to our standardized JAX model key."""
        if "block" in loaded_key:
            layer_num = re.search(r"block\.(\d+)", loaded_key).group(1)
            layer_key = re.sub(r"block\.\d+", "block.*", loaded_key)
            mapped_key = self._loaded_to_standardized_keys.get(layer_key, loaded_key)
            return re.sub(r"layers\.\*", f"layers.{layer_num}", mapped_key)
        else:
            return self._loaded_to_standardized_keys.get(loaded_key, loaded_key)

    def _transpose_params(self, param_key: str, param_tensor: jax.Array):
        for key, value in self._transpose_map.items():
            if re.search(key, param_key):
                return jnp.transpose(param_tensor, value)
        return param_tensor

    def _process_moe_weights(self, loaded_name, loaded_weight, weights_dict):
        """Accumulates expert weights for a layer until all are collected."""
        layer_num_match = re.search(r"block\.(\d+)", loaded_name)
        if not layer_num_match:
            return None
        layer_num = layer_num_match.group(1)

        # Expert weights are not individually numbered in the checkpoint. They are a single tensor.
        # This function will split them and prepare for stacking.
        if layer_num not in weights_dict:
            weights_dict[layer_num] = []

        weights_dict[layer_num].append(loaded_weight)
        
        # In this model, expert weights are already stacked in the checkpoint.
        # So we can return immediately.
        stacked_weights = loaded_weight 
        del weights_dict[layer_num]
        return stacked_weights

    def _load_individual_weight(self, name: str, weight: torch.Tensor,
                                model_params: nnx.State, model_mesh: Mesh):
        """Loads a single weight tensor into the JAX model."""
        mapped_name = self.map_loaded_to_standardized_name(name)
        model_weight = get_param(model_params, mapped_name)
        sharding = model_weight.sharding

        cast_type = model_weight.value.dtype
        weight_np = jnp.array(weight.numpy()).astype(cast_type)
        weight_np = self._transpose_params(name, weight_np)

        if model_weight.value.shape != weight_np.shape:
            raise ValueError(
                f"Loaded shape for '{name}': {weight_np.shape} "
                f"does not match model shape for '{mapped_name}': {model_weight.value.shape}!"
            )

        def get_slice(index):
            return weight_np[index]

        sharded_array = jax.make_array_from_callback(
            weight_np.shape, NamedSharding(model_mesh, P(*sharding)), get_slice)

        model_weight.value = sharded_array

        if self.is_verbose:
            print_param_info(model_weight, name)
        
        del weight

    def load_weights(self, model_for_loading: nnx.Module):
        """Main weight loading loop."""
        model_params = nnx.state(model_for_loading)
        
        mlp1_weights = {}
        mlp1_biases = {}

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in self.names_and_weights_generator:
                if "block" in loaded_name:
                    layer_num = int(re.search(r"block\.(\d+)", loaded_name).group(1))
                    if layer_num >= self.num_layers:
                        continue

                # --- Special Handling for Split Weights ---

                # 1. Handle combined QKV weight
                if "attn.qkv.weight" in loaded_name:
                    q_dim = self.num_attention_heads * self.head_dim
                    k_dim = self.num_key_value_heads * self.head_dim
                    
                    q_w, k_w, v_w = torch.split(
                        loaded_weight,
                        [q_dim, k_dim, k_dim],
                        dim=0
                    )
                    
                    # Transpose from (out, in) -> (in, out)
                    q_w = torch.transpose(q_w, 0, 1)
                    k_w = torch.transpose(k_w, 0, 1)
                    v_w = torch.transpose(v_w, 0, 1)

                    layer_prefix = self.map_loaded_to_standardized_name(loaded_name.split(".attn.")[0])
                    self._load_individual_weight(f"{layer_prefix}.attn.kernel_q_HHD", q_w, model_params, model_for_loading.mesh)
                    self._load_individual_weight(f"{layer_prefix}.attn.kernel_k_HHD", k_w, model_params, model_for_loading.mesh)
                    self._load_individual_weight(f"{layer_prefix}.attn.kernel_v_HHD", v_w, model_params, model_for_loading.mesh)
                    continue

                # 2. Handle MoE MLP's first layer (gating and up-projection)
                if "mlp.mlp1_weight" in loaded_name:
                    layer_num = re.search(r"block\.(\d+)", loaded_name).group(1)
                    mlp1_weights[layer_num] = loaded_weight
                    continue
                if "mlp.mlp1_bias" in loaded_name:
                    layer_num = re.search(r"block\.(\d+)", loaded_name).group(1)
                    mlp1_biases[layer_num] = loaded_weight
                    
                    # Once bias is loaded, we have both weight and bias, so process them
                    weight_tensor = mlp1_weights.pop(layer_num)
                    bias_tensor = mlp1_biases.pop(layer_num)

                    # Split into gating and up-projection parts
                    gate_w, up_w = torch.chunk(weight_tensor, 2, dim=1)
                    gate_b, up_b = torch.chunk(bias_tensor, 2, dim=1)

                    # Transpose from (E, 2I, H) to (E, H, I)
                    gate_w = gate_w.permute(0, 2, 1).contiguous()
                    up_w = up_w.permute(0, 2, 1).contiguous()

                    layer_prefix = self.map_loaded_to_standardized_name(loaded_name.split(".mlp.")[0])
                    
                    # Load into the DenseFFW experts inside the MoE layer
                    self._load_individual_weight(f"{layer_prefix}.custom_module.kernel_gating_EDF", gate_w, model_params, model_for_loading.mesh)
                    self._load_individual_weight(f"{layer_prefix}.custom_module.kernel_up_proj_EDF", up_w, model_params, model_for_loading.mesh)
                    self._load_individual_weight(f"{layer_prefix}.custom_module.bias_gating_ED", gate_b, model_params, model_for_loading.mesh)
                    self._load_individual_weight(f"{layer_prefix}.custom_module.bias_up_proj_ED", up_b, model_params, model_for_loading.mesh)
                    continue

                # --- Standard Weight Loading ---
                self._load_individual_weight(loaded_name, loaded_weight, model_params, model_for_loading.mesh)
        
        nnx.update(model_for_loading, model_params)