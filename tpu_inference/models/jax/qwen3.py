from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.typing import Shape
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from transformers import Qwen3Config, modeling_flax_utils
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention_interface import attention
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.linear_common import sharded_quantized_matmul
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.qwen2 import Qwen2DecoderLayer, Qwen2Model
from tpu_inference.models.jax.utils.weight_utils import (get_default_maps,
                                                         load_hf_weights)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def parse_einsum_subscripts(einsum_str: str) -> dict[str, list[str]]:
    """Parses the einsum string to identify and categorize labels."""
    einsum_str = einsum_str.replace(' ', '')
    parts = einsum_str.split('->')
    if len(parts) != 2:
        raise ValueError(f"Invalid einsum string format: '{einsum_str}'")
    input_strs, output_str = parts

    input_parts = input_strs.split(',')
    if len(input_parts) != 2:
        raise ValueError(
            f"Einsum string must have exactly two operands. Got: '{einsum_str}'"
        )
    lhs_subscript, rhs_subscript = input_parts[0], input_parts[1]

    lhs_labels = list(lhs_subscript)
    rhs_labels = list(rhs_subscript)
    out_labels = list(output_str)

    in1_set = set(lhs_labels)
    in2_set = set(rhs_labels)
    out_set = set(out_labels)

    if not out_set.issubset(in1_set | in2_set):
        raise ValueError(
            f"Output labels {out_set - (in1_set | in2_set)} not found in inputs"
        )
    if len(out_labels) != len(out_set):
        raise ValueError(f"Output labels must be unique. Got: '{output_str}'")

    # Categorize dimensions
    batch_dims = sorted(list(in1_set & in2_set & out_set))
    contract_dims = sorted(list((in1_set & in2_set) - out_set))
    # Labels appearing only in LHS (form the M dimension)
    lhs_non_contract_dims = sorted(list(in1_set - in2_set))
    # Labels appearing only in RHS (form the N dimension)
    rhs_non_contract_dims = sorted(list(in2_set - in1_set))

    # Output labels must be the union of batch and non-contracting dimensions
    expected_out_set = set(batch_dims + lhs_non_contract_dims +
                           rhs_non_contract_dims)
    if out_set != expected_out_set:
        raise ValueError(
            f"Output labels set {out_set} do not match the expected set {expected_out_set}"
        )

    return {
        "lhs_labels": lhs_labels,
        "rhs_labels": rhs_labels,
        "out_labels": out_labels,
        "batch_dims": batch_dims,
        "contract_dims": contract_dims,
        "lhs_non_contract_dims": lhs_non_contract_dims,
        "rhs_non_contract_dims": rhs_non_contract_dims,
    }


def prepare_rhs_transform(einsum_str: str, rhs_shape: Shape) -> dict[str, any]:
    """
    Parses einsum string and RHS shape to get information for RHS transformation.
    Call this at PREPROCESSING time.
    """
    subscripts = parse_einsum_subscripts(einsum_str)
    rhs_labels = subscripts["rhs_labels"]
    batch_dims = subscripts["batch_dims"]
    contract_dims = subscripts["contract_dims"]
    rhs_non_contract_dims = subscripts["rhs_non_contract_dims"]

    if len(rhs_labels) != len(rhs_shape):
        raise ValueError(
            f"RHS subscript '{''.join(rhs_labels)}' length {len(rhs_labels)} mismatch with shape {rhs_shape}"
        )

    rhs_label_to_axis = {label: i for i, label in enumerate(rhs_labels)}

    def get_shape_for_dims(shape: Shape, label_to_axis: dict[str, int],
                           dims: List[str]) -> Tuple[int, ...]:
        return tuple(shape[label_to_axis[d]] for d in dims)

    batch_shape_tuple = get_shape_for_dims(rhs_shape, rhs_label_to_axis,
                                           batch_dims)
    k_shape_tuple = get_shape_for_dims(rhs_shape, rhs_label_to_axis,
                                       contract_dims)
    n_shape_tuple = get_shape_for_dims(rhs_shape, rhs_label_to_axis,
                                       rhs_non_contract_dims)

    b_prod = np.prod(batch_shape_tuple) if batch_shape_tuple else 1
    k_prod = np.prod(k_shape_tuple) if k_shape_tuple else 1
    n_prod = np.prod(n_shape_tuple) if n_shape_tuple else 1

    return {
        "subscripts": subscripts,
        "b_prod": b_prod,
        "k_prod": k_prod,
        "n_prod": n_prod,
        "batch_shape_tuple": batch_shape_tuple,
        "k_shape_tuple": k_shape_tuple,
        "n_shape_tuple": n_shape_tuple,
    }


def prepare_lhs_and_output_transform(
        lhs_shape: Shape, rhs_parsed_info: dict[str, any]) -> dict[str, any]:
    """
    Uses LHS shape and precomputed RHS info to get all information
    needed for LHS transformation and output reconstruction.
    Call this at RUNTIME.
    """
    subscripts = rhs_parsed_info["subscripts"]
    lhs_labels = subscripts["lhs_labels"]
    out_labels = subscripts["out_labels"]
    batch_dims = subscripts["batch_dims"]
    contract_dims = subscripts["contract_dims"]
    lhs_non_contract_dims = subscripts["lhs_non_contract_dims"]
    rhs_non_contract_dims = subscripts["rhs_non_contract_dims"]

    if len(lhs_labels) != len(lhs_shape):
        raise ValueError(
            f"LHS subscript '{''.join(lhs_labels)}' length {len(lhs_labels)} mismatch with shape {lhs_shape}"
        )

    lhs_label_to_axis = {label: i for i, label in enumerate(lhs_labels)}

    def get_shape_for_dims(shape: Shape, label_to_axis: dict[str, int],
                           dims: List[str]) -> Tuple[int, ...]:
        return tuple(shape[label_to_axis[d]] for d in dims)

    lhs_batch_shape = get_shape_for_dims(lhs_shape, lhs_label_to_axis,
                                         batch_dims)
    lhs_k_shape = get_shape_for_dims(lhs_shape, lhs_label_to_axis,
                                     contract_dims)
    m_shape_tuple = get_shape_for_dims(lhs_shape, lhs_label_to_axis,
                                       lhs_non_contract_dims)
    m_prod = np.prod(m_shape_tuple) if m_shape_tuple else 1

    # Validate shapes between LHS and RHS for shared dimensions
    if lhs_batch_shape != rhs_parsed_info["batch_shape_tuple"]:
        raise ValueError(
            f"Batch dimension shapes mismatch: {lhs_batch_shape} (LHS) vs {rhs_parsed_info['batch_shape_tuple']} (RHS) for dims {batch_dims}"
        )
    if lhs_k_shape != rhs_parsed_info["k_shape_tuple"]:
        raise ValueError(
            f"Contracting dimension shapes mismatch: {lhs_k_shape} (LHS) vs {rhs_parsed_info['k_shape_tuple']} (RHS) for dims {contract_dims}"
        )

    # Check if reshape is needed for LHS
    num_batch_dims = len(batch_dims)
    num_contract_dims = len(contract_dims)
    num_lhs_non_contract_dims = len(lhs_non_contract_dims)

    if num_batch_dims == 0 and num_lhs_non_contract_dims <= 1 and num_contract_dims <= 1:
        needs_lhs_reshape = False
    else:
        needs_lhs_reshape = True

    # Info for output transformation
    n_shape_tuple = rhs_parsed_info["n_shape_tuple"]
    batch_shape_tuple = rhs_parsed_info["batch_shape_tuple"]

    out_label_to_size = dict()
    for i, d in enumerate(batch_dims):
        out_label_to_size[d] = batch_shape_tuple[i]
    for i, d in enumerate(lhs_non_contract_dims):
        out_label_to_size[d] = m_shape_tuple[i]
    for i, d in enumerate(rhs_non_contract_dims):
        out_label_to_size[d] = n_shape_tuple[i]

    final_output_shape_tuple = tuple(out_label_to_size[d] for d in out_labels)
    # Combine all necessary info for runtime transformations
    full_parsed_info = rhs_parsed_info.copy()
    full_parsed_info.update({
        "m_prod": m_prod,
        "final_output_shape_tuple": final_output_shape_tuple,
        "needs_lhs_reshape": needs_lhs_reshape,
    })
    return full_parsed_info


def transform_lhs_for_matmul(lhs: jax.Array,
                             parsed_info: dict[str, any]) -> jax.Array:
    """
    Transforms the LHS einsum operand using info from prepare_lhs_and_output_transform.
    LHS -> (B, M, K) or (M, K)
    """
    if parsed_info["needs_lhs_reshape"]:
        target_shape = (parsed_info["m_prod"], parsed_info["k_prod"])
        lhs = jnp.reshape(lhs, target_shape)

    return lhs


def transform_matmul_output_to_einsum(
        matmul_output: jax.Array, parsed_info: dict[str, any]) -> jax.Array:
    """
    Reshapes and transposes the output of jnp.matmul back to the
    expected einsum output shape using info from prepare_lhs_and_output_transform.
    matmul_output is expected to be (B, M, N) or (M, N).
    """
    final_output = jnp.reshape(matmul_output,
                               parsed_info["final_output_shape_tuple"])
    return final_output


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
        # Sharding should be transposed as well as the weight shapes were transposed for quantized matmul.
        self.weight_sharding = P(self.kernel.sharding[0],
                                 self.kernel.sharding[1])

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
        bias = self.bias.value if self.bias is not None else None

        y = sharded_quantized_matmul(
            inputs,
            kernel,
            weight_scale,
            self.model.mesh,
            self.weight_sharding,
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


class JaxEinsumLayer(nnx.Einsum):

    def __init__(self,
                 einsum_str: str,
                 kernel_shape: Shape,
                 bias_shape: Optional[Shape] = None,
                 *,
                 model: nnx.Module,
                 **kwargs):
        super().__init__(einsum_str, kernel_shape, bias_shape, **kwargs)
        self.model = model
        self.rhs_parsed_info = prepare_rhs_transform(self.einsum_str,
                                                     self.kernel_shape)
        self.weight_sharding = P(self.kernel.sharding[0],
                                 self.kernel.sharding[1])

    def __call__(self,
                 inputs: jax.Array,
                 einsum_str: Optional[str] = None) -> jax.Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.
          einsum_str: a string to denote the einsum equation. The equation must
            have exactly two operands, the lhs being the input passed in, and
            the rhs being the learnable kernel. Exactly one of ``einsum_str``
            in the constructor argument and call argument must be not None,
            while the other must be None.

        Returns:
          The transformed input.
        """
        einsum_str = nnx.module.first_from(
            einsum_str,
            self.einsum_str,
            error_msg="""No `einsum_str` argument was provided to Einsum
            as either a __call__ argument, or class attribute.""",
        )
        einsum_str = einsum_str.replace(' ', '')
        self._einsum_str_check(einsum_str)

        kernel = self.kernel.value
        bias = self.bias.value if self.bias is not None else None
        #inputs, kernel, bias = self.promote_dtype(
        #(
        #inputs,
        #self.kernel[...],
        #self.bias[...] if self.bias is not None else self.bias,
        #),
        #dtype=self.dtype,
        #)
        # We use einsum_op_kwargs for BC compatibility with
        # user custom self.einsum_op method which may not have
        # preferred_element_type argument to avoid breaking
        # existing code
        einsum_op_kwargs = {}
        #if self.preferred_element_type is not None:
        #einsum_op_kwargs["preferred_element_type"] = self.preferred_element_type
        #
        y = self._quantized_einsum(einsum_str,
                                   inputs,
                                   kernel,
                                   precision=self.precision,
                                   **einsum_op_kwargs)

        if bias is not None:
            broadcasted_bias_shape = self._infer_broadcasted_bias_shape(
                einsum_str, inputs, kernel)
            y += jnp.reshape(bias, broadcasted_bias_shape)
        return y

    def _quantized_einsum(self, einsum_str, inputs, kernel, precision):
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

        parsed_info = prepare_lhs_and_output_transform(inputs.shape,
                                                       self.rhs_parsed_info)
        inputs = transform_lhs_for_matmul(inputs, parsed_info)
        y = sharded_quantized_matmul(
            inputs,
            kernel,
            weight_scale,
            self.model.mesh,
            self.weight_sharding,
        )
        y = transform_matmul_output_to_einsum(y, parsed_info)
        return y


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, kv_cache_dtype: str, model: nnx.Module):
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

        self.q_proj = JaxEinsumLayer(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            model=model,
            param_dtype=dtype,
            # P(None, "model", None) -> P("model", None, None) due to weigh transpose.
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.q_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.k_proj = JaxEinsumLayer(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            model=model,
            param_dtype=dtype,
            # P(None, "model", None) -> P("model", None, None) due to weigh transpose.
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.k_norm = nnx.RMSNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.v_proj = JaxEinsumLayer(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            model=model,
            param_dtype=dtype,
            # P(None, "model", None) -> P("model", None, None) due to weigh transpose.
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.o_proj = JaxEinsumLayer(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            model=model,
            param_dtype=dtype,
            # P("model", None, None) -> P(None, "model", None) due to weigh transpose.
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def preprocess_weights(self):
        self.q_proj.transform_weights()
        self.k_proj.transform_weights()
        self.v_proj.transform_weights()
        self.o_proj.transform_weights()

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
                 mesh: Mesh, kv_cache_dtype: str, model: nnx.Module):
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
                                        kv_cache_dtype=kv_cache_dtype,
                                        model=model)
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
