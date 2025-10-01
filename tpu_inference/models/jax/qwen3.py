from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from transformers import Qwen3Config, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention_interface import attention
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen2 import Qwen2DecoderLayer
from tpu_inference.models.jax.qwen2 import Qwen2MLP as Qwen3MLP
from tpu_inference.models.jax.qwen2 import Qwen2Model
from tpu_inference.models.jax.utils.weight_utils import (get_default_maps,
                                                       load_hf_weights)
from tpu_inference.models.vllm.jax_linear_common import sharded_quantized_matmul

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def get_param_path(target_param: nnx.Param,
                   root_module: nnx.Module) -> Optional[str]:
    """
    Finds the key path of a specific nnx.Param instance within an nnx.Module.

    Args:
        target_param: The nnx.Param instance to search for.
        root_module: The root nnx.Module to search within.

    Returns:
        The dotted key path string if the param is found, otherwise None.
    """
    # Get the State PyTree containing all nnx.Param objects in the module
    params_state = nnx.state(root_module, nnx.Param)

    # Flatten the state tree, getting paths to each leaf
    flat_params_with_paths, _ = jax.tree_util.tree_flatten_with_path(
        params_state)

    # Iterate through all paths and param leaves
    for path, leaf_param in flat_params_with_paths:
        # Check for object identity using 'is'
        if leaf_param is target_param:
            return jax.tree_util.keystr(path, simple=True,
                                        separator='.').removesuffix(".value")

    # Parameter instance not found in the module's Param state
    return "Not found"


class QuantizedLinear(nnx.Linear):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            rngs: nnx.Rngs,  # Required keyword argument for nnx.Linear
            model: nnx.Module,
            **kwargs  # To catch other optional arguments for nnx.Linear
    ):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         rngs=rngs,
                         **kwargs)
        self.model = model

    # Overrided from nnx.linear.__call__, started by copy and paste.
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Applies a quantized matmul to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
        kernel = self.kernel.value
        path = get_param_path(kernel, self.model)
        if path.endswith(".kernel"):
            path = path.removesuffix(".kernel")
        # Find the weight scale maching to the given kernel by path.
        weight_scale = None
        for scale_key, scale_value in self.model.quant_scales.items():
            if path in scale_key:
                weight_scale = scale_value
                break
        if weight_scale is None:
            raise ValueError(f"weight scale was not set for {path}")

        # Convert back to fp8 as it was casted up to fp32 from fp8 while loading weights.
        kernel = kernel.astype(jnp.float8_e4m3fn)
        # Trasponse the weight shape as the quantized matmul will take it as (n_output_feature, n_input_feature).
        kernel = jnp.transpose(kernel)
        # Make the weight scale shpae as (x), because it comes as (x, 1).
        if len(weight_scale.shape) == 2:
            weight_scale = jnp.reshape(weight_scale, weight_scale.shape[0])
        # Set the weight sharding to None as it was.
        weight_sharding = P(None, None)
        # The line below is to print weight array sharding, but it prints None as mentioned above.
        #jax.debug.inspect_array_sharding(kernel, callback=lambda sharding: print(f"Kernel sharding: {sharding}"))
        bias = self.bias.value if self.bias is not None else None

        y = sharded_quantized_matmul(
            inputs,
            kernel,
            weight_scale,
            self.model.mesh,
            weight_sharding,
        )
        assert self.use_bias == (bias is not None)
        if bias is not None:
            y += jnp.reshape(bias, (1, ) * (y.ndim - 1) + (-1, ))
        return y


class Qwen3QuantizedMLP(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 model: nnx.Module):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = config.hidden_act

        self.gate_proj = QuantizedLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            model=model,
        )
        self.up_proj = QuantizedLinear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            model=model,
        )
        self.down_proj = QuantizedLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            model=model,
        )
        self.act_fn = modeling_flax_utils.ACT2FN[act]

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, kv_cache_dtype: str):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # q: (T, N, H)
        q = self.q_proj(x)
        q = self.q_norm(q)
        q = apply_rope(q, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = self.k_norm(k)
        k = apply_rope(k, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # v: (T, K, H)
        v = self.v_proj(x)
        # o: (T, N, H)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # TODO(kyuyeunk/jacobplatin): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = utils.quantize_kv(k, v, self.kv_cache_quantized_dtype,
                                     k_scale, v_scale)
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Qwen3DecoderLayer(Qwen2DecoderLayer):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, model: nnx.Module, kv_cache_dtype: str):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = Qwen3Attention(config=config,
                                        dtype=dtype,
                                        rng=rng,
                                        mesh=mesh,
                                        kv_cache_dtype=kv_cache_dtype)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = Qwen3QuantizedMLP(
            config=config,
            dtype=dtype,
            rng=rng,
            model=model,
        )


class Qwen3Model(Qwen2Model):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh,
                 model: nnx.Module) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.layers = [
            Qwen3DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                model=model,
                # TODO (jacobplatin): we should refactor this to pass a dtype (or config) directly
                kv_cache_dtype=vllm_config.cache_config.cache_dtype)
            for _ in range(hf_config.num_hidden_layers)
        ]
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        if model_config.hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )


class Qwen3ForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.quant_scales = {}
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
            model=self,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
        )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.model.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.model.lm_head.value)
        return logits

    def load_weights(self, rng_key: jax.Array):
        # NOTE: Since we are using nnx.eval_shape to init the model,
        # we have to pass dynamic arrays here for __call__'s usage.
        self.rng = nnx.Rngs(rng_key)

        # Key: path to a HF layer weight
        # Value: path to a nnx layer weight
        mappings = {
            "model.embed_tokens": "model.embed.embedding",
            "model.layers.*.input_layernorm":
            "model.layers.*.input_layernorm.scale",
            "model.layers.*.mlp.down_proj":
            "model.layers.*.mlp.down_proj.kernel",
            "model.layers.*.mlp.gate_proj":
            "model.layers.*.mlp.gate_proj.kernel",
            "model.layers.*.mlp.up_proj": "model.layers.*.mlp.up_proj.kernel",
            "model.layers.*.post_attention_layernorm":
            "model.layers.*.post_attention_layernorm.scale",
            "model.layers.*.self_attn.k_norm":
            "model.layers.*.self_attn.k_norm.scale",
            "model.layers.*.self_attn.k_proj":
            "model.layers.*.self_attn.k_proj.kernel",
            "model.layers.*.self_attn.o_proj":
            "model.layers.*.self_attn.o_proj.kernel",
            "model.layers.*.self_attn.q_norm":
            "model.layers.*.self_attn.q_norm.scale",
            "model.layers.*.self_attn.q_proj":
            "model.layers.*.self_attn.q_proj.kernel",
            "model.layers.*.self_attn.v_proj":
            "model.layers.*.self_attn.v_proj.kernel",
            "model.norm": "model.norm.scale",
        }

        # Add lm_head mapping only if it's not tied to embeddings
        if not self.vllm_config.model_config.hf_config.tie_word_embeddings:
            mappings.update({
                "lm_head": "model.lm_head",
            })

        metadata_map = get_default_maps(self.vllm_config, self.mesh, mappings)
        load_hf_weights(vllm_config=self.vllm_config,
                        model=self,
                        metadata_map=metadata_map,
                        mesh=self.mesh)
