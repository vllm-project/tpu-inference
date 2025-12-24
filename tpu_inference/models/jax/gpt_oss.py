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

from tpu_inference.layers.common.quant_methods import MXFP4
from tpu_inference.layers.common.quantization import (
    dequantize_tensor_from_mxfp4_packed, e8m0_to_fp32, u8_unpack_e2m1)
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
            vd_sharding=P(('data', 'model'), None),
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
                query_tnh=P("data", 'model', None),
                keyvalue_skh=P("data", 'model', None),
                attn_o_tnh=P("data", 'model', None),
                dnh_sharding=P(None, 'model', None),
                dkh_sharding=P(None, 'model', None),
                nhd_sharding=P('model', None, None),
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
                activation_ffw_td=P('data', None),
                ed_sharding=P('model', None),
                e_sharding=P('model'),
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
                activation_ffw_td=P('data', None),
                edf_sharding=P('model', None, None),
                efd_sharding=P('model', None, None),
                ed_sharding=P('model', None),
            )

            block = TransformerBlock(
                pre_attention_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=dtype,
                    rngs=self.rng,
                    activation_ffw_td=P('data', None),
                ),
                pre_mlp_norm=RMSNorm(
                    dims=hidden_size,
                    random_init=self.random_init,
                    epsilon=rms_norm_eps,
                    dtype=dtype,
                    rngs=self.rng,
                    activation_ffw_td=P('data', None),
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
            activation_ffw_td=P('data', None),
        )

        self.lm_head = LMhead(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            dtype=dtype,
            rngs=self.rng,
            vd_sharding=P(('data', 'model'), None),
            dv_sharding=P(None, ('data', 'model')),
            random_init=self.random_init,
        )

    # For compatibility with flax.
    def apply(self, variables, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def load_weights(self, rng: PRNGKey, cache_dir: Optional[str] = None):
        """Loads and transforms all weights from a checkpoint"""
        self.rng = nnx.Rngs(rng)

        # Determine quantization method from HF config (config.json)
        quant_method = (self.hf_config.quantization_config["quant_method"]
                        if hasattr(self.hf_config, "quantization_config") else
                        None)

        # Format: 'hf_key': ('jax_model_path', transform_function, target_shape)
        transforms = {
            "transpose_reshape": lambda w, shape: w.T.reshape(shape),
            "reshape": lambda b, shape: b.reshape(shape),
            "transpose": lambda w, _: w.T,
            "swap_last2": lambda w, _: w.swapaxes(-1, -2),
        }

        # MXFP4 checkpoints swap last two dims for MoE to place packed dim at most minor
        swap_mlp_transform = transforms[
            "swap_last2"] if quant_method == MXFP4 else None

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
            ("layers.*.custom_module.mlp1_weight_EDF2", swap_mlp_transform,
             None),
            "model.layers.*.mlp.experts.gate_up_proj_bias":
            ("layers.*.custom_module.mlp1_bias_EF2", None, None),
            "model.layers.*.mlp.experts.down_proj":
            ("layers.*.custom_module.mlp2_weight_EFD", swap_mlp_transform,
             None),
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

        # Build a pool of weights with MXFP4 experts combined if neededs
        pool: dict[str, torch.Tensor | tuple] = (self._build_mxfp4_pool(
            names_and_weights_generator,
            mappings) if quant_method == MXFP4 else {
                loaded_name: loaded_weight
                for loaded_name, loaded_weight in names_and_weights_generator
            })

        with jax.default_device(jax.devices("cpu")[0]):
            for loaded_name, loaded_weight in pool.items():
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

                prepared_weight = loaded_weight
                if isinstance(loaded_weight, tuple):
                    # Loaded weight is an MXFP4 tuple
                    blocks_u8, scales_u8 = loaded_weight
                    # Quantized param (QArray): set qvalue/scale directly and skip regular path
                    if hasattr(model_weight, "array"):  # QArray check
                        codes_fp32_t = u8_unpack_e2m1(blocks_u8).astype(
                            jnp.float32)
                        scales_fp32_t = e8m0_to_fp32(scales_u8)
                        self._load_mxfp4(
                            model_weight=model_weight,
                            codes_fp32_t=codes_fp32_t,
                            scales_fp32_t=scales_fp32_t,
                            transform_fn=transform_fn,
                        )
                        if is_verbose:
                            print_param_info(model_weight, loaded_name)
                        continue
                    # Not a QArray: dequantize MXFP4 to BF16 full weights
                    prepared_weight = dequantize_tensor_from_mxfp4_packed(
                        blocks_u8, scales_u8)

                # Single regular-tensor load call (BF16 or dequantized MXFP4)
                cast_type = model_weight.value.dtype
                self._load_regular_param(
                    model_weight=model_weight,
                    loaded_weight=prepared_weight,
                    cast_type=cast_type,
                    transform_fn=transform_fn,
                    target_shape=target_shape,
                    jax_path_template=jax_path_template,
                )

                if is_verbose:
                    print_param_info(model_weight, loaded_name)

        nnx.update(self, model_params)

    def _build_mxfp4_pool(self, names_and_weights_generator, mappings):
        """Collect MXFP4 weights into a pool keeping tuples (blocks_u8, scales_u8).

        Combines *_blocks and *_scales pairs and stores uint8 tensors together.
        Non-expert tensors are kept as-is. Raises if any expert bundle is incomplete.
        """
        pool: dict[str, torch.Tensor | tuple] = {}
        pending_experts: dict[str, dict[str, torch.Tensor]] = {}
        for loaded_name, loaded_weight in names_and_weights_generator:
            if loaded_name.endswith("_blocks") or loaded_name.endswith(
                    "_scales"):
                base = loaded_name[:-7]
                entry = pending_experts.setdefault(base, {})
                if loaded_name.endswith("_blocks"):
                    entry["blocks"] = loaded_weight
                else:
                    entry["scales"] = loaded_weight

                # If we have both parts, place raw pair into the main pool
                if "blocks" in entry and "scales" in entry:
                    hf_pattern = re.sub(r"layers\.(\d+)", "layers.*", base)
                    if hf_pattern not in mappings:
                        raise ValueError(
                            f"No mapping found for expert tensor: {base}")
                    pool[base] = (entry["blocks"], entry["scales"])
                    # Remove from pending to free memory
                    pending_experts.pop(base, None)
            else:
                pool[loaded_name] = loaded_weight

        # Enforce completeness of expert bundles
        if pending_experts:
            details = []
            for base, entry in pending_experts.items():
                missing = [k for k in ("blocks", "scales") if k not in entry]
                details.append(
                    f"{base} (missing: {', '.join(missing) if missing else 'unknown'})"
                )
            raise RuntimeError(
                "Incomplete MXFP4 expert bundle(s) encountered: " +
                ", ".join(details))
        return pool

    def _load_mxfp4(self,
                    model_weight,
                    codes_fp32_t,
                    scales_fp32_t,
                    transform_fn=None):
        """Assign decoded MXFP4 codes/scales into a QArray (qvalue/scale)."""

        qv = model_weight.array.qvalue
        sv = model_weight.array.scale
        q_dtype = qv.value.dtype
        s_dtype = sv.value.dtype

        exp_q_shape = tuple(qv.value.shape)
        exp_s_shape = tuple(sv.value.shape)

        # Apply optional transform (e.g., swap last two dims) before conversion
        if transform_fn is not None:
            codes_fp32_t = transform_fn(codes_fp32_t, None)
            scales_fp32_t = transform_fn(scales_fp32_t, None)

        # Convert from torch.Tensor to numpy before creating JAX arrays
        codes_fp32_t = codes_fp32_t.detach().cpu().numpy()
        scales_fp32_t = scales_fp32_t.detach().cpu().numpy()

        codes_jnp = jnp.asarray(codes_fp32_t).astype(q_dtype)
        scales_jnp = jnp.asarray(scales_fp32_t).astype(s_dtype)

        def get_q_slice(index):
            return codes_jnp[index]

        def get_s_slice(index):
            return scales_jnp[index]

        q_sharded = jax.make_array_from_callback(
            exp_q_shape, NamedSharding(self.mesh, P(*qv.sharding)),
            get_q_slice)
        s_sharded = jax.make_array_from_callback(
            exp_s_shape, NamedSharding(self.mesh, P(*sv.sharding)),
            get_s_slice)

        model_weight.array.qvalue.value = q_sharded
        model_weight.array.scale.value = s_sharded

    def _load_regular_param(self, model_weight, loaded_weight: torch.Tensor,
                            cast_type, transform_fn, target_shape,
                            jax_path_template: str):
        """Assign a regular tensor (non-MXFP4) into the model param with transform applied."""
        if jax_path_template == "layers.*.attn.sinks_N":
            # Checkpoint is bf16, but we have to upcast sinks to f32, as required by RPA_v3 kernel
            weight_np = jnp.array(loaded_weight.to(torch.float32).numpy())
        else:
            torch_view_type = DTYPE_VIEW_MAP.get(jnp.dtype(cast_type))
            if torch_view_type:
                weight_np = jnp.array(
                    loaded_weight.view(torch_view_type).numpy()).view(
                        cast_type)
            else:
                raise ValueError(
                    f"Unsupported dtype for tensor conversion: {cast_type}")

        transformed_weight = transform_fn(
            weight_np, target_shape) if transform_fn else weight_np

        if model_weight.value.shape != transformed_weight.shape:
            raise ValueError(
                f"Shape mismatch: model expects {model_weight.value.shape}, but got {transformed_weight.shape} after transform."
            )

        def get_slice(index):
            return transformed_weight[index]

        sharded_array = jax.make_array_from_callback(
            transformed_weight.shape,
            NamedSharding(self.mesh, P(*model_weight.sharding)), get_slice)
        model_weight.value = sharded_array

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
