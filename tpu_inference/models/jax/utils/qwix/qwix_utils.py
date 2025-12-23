# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import functools
import os
from typing import TYPE_CHECKING, Callable, List

import jax
import jax.numpy as jnp
import qwix
import qwix.pallas as qpl
import yaml
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwix._src.core.qarray import QArray
from qwix._src.providers import ptq

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.logger import init_logger
from tpu_inference.runner.kv_cache import (DEFAULT_KV_CACHE_DTYPE,
                                           create_kv_caches)
from tpu_inference.utils import device_array

logger = init_logger(__name__)

QUANTIZATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE = 2000
DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS = 512
DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS = 256
DEFAULT_MAX_NUM_BLOCKS_PER_REQ = 16

DEFAULT_DEEPSEEK_FP4_MLP_MOE_FP8_ATTN_CONFIG = {
    "qwix": {
        "use_abstract_model":
        True,
        "scale_dtype":
        "bfloat16",
        "rules": [
            # Exclude router from quantization
            {
                "module_path": ".*.custom_module.router.*",
                "weight_qtype": None,
            },
            # Avoid the combine expert ops
            {
                "module_path": ".*combine_experts.*",
                "weight_qtype": None,
            },
            # Attention layers: keep FP8 for weights and activations
            {
                "module_path": ".*.attn.*",
                "weight_qtype": "float8_e4m3fn",
                "act_qtype": "float8_e4m3fn",
            },
            # MoE experts: use FP4 for expert weights
            {
                "module_path": ".*.custom_module.*",
                "weight_qtype": "float4_e2m1fn",
                "act_qtype": "float8_e4m3fn",
                "tile_size": 256,
            },
            # Shared experts: also FP4
            {
                "module_path": ".*.shared_experts.*",
                "weight_qtype": "float4_e2m1fn",
                "act_qtype": "float8_e4m3fn",
                "tile_size": 256,
            },
            {
                "module_path": ".*",
                "weight_qtype": "float8_e4m3fn",
                "act_qtype": "float8_e4m3fn",
            },
        ],
    }
}

DEFAULT_LLAMA4_FP8_CONFIG = {
    "qwix": {
        "use_abstract_model":
        True,
        "scale_dtype":
        "bfloat16",
        "rules": [
            {
                "module_path": "layers.*.moe_ffw",
                "op_names": "einsum",
                "weight_qtype": "float8_e4m3fn",
                "act_qtype": "float8_e4m3fn",
            },
        ],
    }
}

# Default Qwix config for GPT-OSS MXFP4 checkpoints.
# Notes:
# - We quantize only the MoE expert weights by default (router stays in BF16).
# - We use Qwix's abstract-model path so weights can be set directly into QArray
#   fields during weight loading (similar to DeepSeek's flow).
# - Activation quantization is not set but Qwix would pickup MoE sum if activated
DEFAULT_GPT_OSS_FP4_CONFIG = {
    "qwix": {
        "use_abstract_model":
        True,
        "scale_dtype":
        "bfloat16",
        "rules": [
            {
                "module_path": ".*custom_module",
                "weight_qtype": "float4_e2m1fn",
                "act_qtype": None,
                "tile_size": 32,
            },
        ],
    }
}


def parse_qwix_config_to_rules(
        qwix_config: List[dict]) -> List[qwix.QuantizationRule]:
    """
    Parse a list of dictionaries containing Qwix quantization rules into a list of QuantizationRule objects.

    Args:
        qwix_config: a dictionary containing the Qwix quantization rules

    Returns:
        a list of QuantizationRule objects
    """
    rules = []
    for rule in qwix_config:
        rules.append(qwix.QuantizationRule(**rule))

    return rules


def qwix_quantize_nnx_model(model: nnx.Module, qwix_config: List[dict],
                            rng: jax.Array, mesh: Mesh, num_hidden_layers: int,
                            kv_cache_block_size: int,
                            kv_cache_num_kv_heads: int,
                            kv_cache_head_size: int,
                            kv_cache_dtype: str) -> nnx.Module:
    """
    Quantizes a Flax NNX model using Qwix.

    Args:
        model: the model to quantize
        qwix_config: a list of dictionaries, where each dictionary corresponds to a Qwix quantization rule
            For example:
            [
                {
                    "module_path": ".*attn.*",
                    "weight_qtype": "int8",
                },
                {
                    "module_path": ".*mlp.*",
                    "weight_qtype": "int8",
                    "act_qtype": "int8",
                    "tile_size": None,
                },
            ]
        rng: the random number generator to use
        mesh: the mesh to use
        num_hidden_layers: the number of hidden layers in the model
        kv_cache_page_size: the page size of the kv cache
        kv_cache_num_kv_heads: the number of kv heads
        head_size: the head size of the kv cache
        kv_cache_dtype: the dtype of the kv cache

    Returns:
        model: the quantized model
    """
    qwix_rules = parse_qwix_config_to_rules(qwix_config)
    logger.info(f"Qwix rules: {qwix_rules}")
    logger.info(f"Memory usage before applying quantization of params: "
                f"hbm={utils.hbm_usage_gb(jax.local_devices())}Gb")

    if kv_cache_dtype != "auto":
        kv_cache_jnp_dtype = utils.to_jax_dtype(kv_cache_dtype)
    else:
        kv_cache_jnp_dtype = DEFAULT_KV_CACHE_DTYPE

    kv_caches = create_kv_caches(
        num_blocks=DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE,
        block_size=kv_cache_block_size,
        num_kv_heads=kv_cache_num_kv_heads,
        head_size=kv_cache_head_size,
        mesh=mesh,
        layer_names=[f"layer.{i}" for i in range(num_hidden_layers)],
        cache_dtype=kv_cache_jnp_dtype,
        use_mla=model.vllm_config.model_config.use_mla,
    )

    dp_size = model.vllm_config.sharding_config.total_dp_size

    # NOTE: the inputs don't need to match the actual ones, as long as the consumed weights are the same
    input_ids = jax.random.randint(rng,
                                   (DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS, ),
                                   0,
                                   100,
                                   dtype=jnp.int32)
    positions = jax.random.randint(rng,
                                   (DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS, ),
                                   0,
                                   100,
                                   dtype=jnp.int32)
    block_tables = jax.random.randint(rng,
                                      (DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS *
                                       DEFAULT_MAX_NUM_BLOCKS_PER_REQ, ),
                                      0,
                                      100,
                                      dtype=jnp.int32)
    query_start_loc = jax.random.randint(
        rng, (DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS + dp_size, ),
        0,
        100,
        dtype=jnp.int32)
    seq_lens = jax.random.randint(rng,
                                  (DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS, ),
                                  0,
                                  100,
                                  dtype=jnp.int32)
    num_seqs = jax.random.randint(rng, (1, ), 0, 100, dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, num_seqs[0]] * dp_size,
                                     dtype=jnp.int32)

    (input_ids, positions, block_tables,
     query_start_loc, seq_lens, request_distribution) = device_array(
         mesh, (input_ids, positions, block_tables, query_start_loc, seq_lens,
                request_distribution))

    model_input = {
        "kv_caches":
        kv_caches,
        "input_ids":
        input_ids,
        "attention_metadata":
        AttentionMetadata(input_positions=positions,
                          block_tables=block_tables,
                          seq_lens=seq_lens,
                          query_start_loc=query_start_loc,
                          request_distribution=request_distribution),
    }
    model = qwix.quantize_model(model, qwix.PtqProvider(qwix_rules),
                                **model_input)
    return model


def quantization_config_file_path_to_dict(
        quantization_config_file_path: str) -> dict:
    """
    Converts a quantization config YAML file path to a dictionary.

    The expected format of the quantization config YAML file is as follows:
    ```yaml
        qwix:
            # optional, defaults to False if not specified
            use_abstract_model: True
            rules:
                # NOTE: each entry corresponds to a qwix.QuantizationRule
                - module_path: '.*attn.*'
                weight_qtype: 'int8'
                - module_path: '.*'
                weight_qtype: 'int8'
                act_qtype: 'int8'
    ```

    Args:
        quantization_config_file_path: the path to the quantization config YAML file

    Returns:
        a dictionary containing the quantization config
    """
    all_entries = os.listdir(QUANTIZATION_CONFIG_PATH)
    for filename in all_entries:
        if filename == quantization_config_file_path:
            path = os.path.join(QUANTIZATION_CONFIG_PATH, filename)
            with open(path, "r") as f:
                return yaml.safe_load(f)
    raise ValueError(
        f"Could not find quantization config file with name '{quantization_config_file_path}' in 'tpu_inference/models/jax/utils/quantization/configs."
    )


def apply_qwix_quantization(
        vllm_config: "VllmConfig", model_or_model_fn: Callable | nnx.Module,
        rng: jax.Array, mesh: Mesh,
        apply_to_abstract_model: bool) -> nnx.Module | Callable:
    """
    Will apply quantization if a valid quantization config with Qwix rules is provided.  See README
    for more details on Qwix.

    Note that we currently support different methods for applying Qwix quantization.  The typical
    approach is to apply quantization on the concrete model, which already has the weights
    loaded in.  However, for models like DeepSeek, which are already quantized, we need to
    first create the abstract model, then apply Qwix quantization to the abstract model, and
    finally load the weights in.  To use the latter approach, you will need to modify the
    model weight loading code appropriately (see deepseek_v3.py for an example) and
    pass and `use_abstract_model=True` in the quantization config.

    Args:
        vllm_config: the base VLLM config
        model_or_model_fn: if `apply_to_abstract_model` is True, this will be a Callable that returns the abstract model
            (e.g. _create_abstract_model).  Otherwise, this will be the concrete model (nnx.Module).
        rng: JAX RNG
        mesh: model Mesh
        apply_to_abstract_model: if True, we will apply Qwix quantization to the abstract model, which
            assumes that, during weight loading, the caller will thus override the QArray weights
            (see deepseek_v3.py for an example).  Otherwise, we will will apply Qwix quantization to the
            concrete model, which already has the weights loaded in.

    Returns:
        Either the concrete model (nnx.Module) or the abstract model (Callable) (if `apply_to_abstract_model` is True)
    """
    qwix_config = None
    if quantization_config := vllm_config.additional_config.get(
            "quantization"):
        qwix_config = quantization_config.get("qwix").get("rules")
    if not qwix_config:
        return model_or_model_fn

    logging_abstract_model_str = "abstract" if apply_to_abstract_model else "concrete"
    logger.info(
        f"Applying Qwix quantization on {logging_abstract_model_str} model")

    block_size = vllm_config.cache_config.block_size
    model_config = vllm_config.model_config

    # Pad num_kv_heads to multiple of TP size
    num_kv_heads = utils.get_padded_num_heads(
        model_config.get_total_num_kv_heads(), mesh.shape["model"])

    # Pad head_dim to multiple of 128
    head_size = model_config.get_head_size()
    head_size = utils.get_padded_head_dim(head_size)

    kv_cache_dtype = vllm_config.cache_config.cache_dtype

    if not apply_to_abstract_model:
        assert isinstance(model_or_model_fn, nnx.Module)
        qwix_quantize_nnx_model_with_config = functools.partial(
            qwix_quantize_nnx_model, qwix_config=qwix_config)
        # NOTE: it's REALLY important `qwix_quantize_nnx_model_with_config` is jitted
        # or else you'll run into hanging
        model_or_model_fn = nnx.jit(
            qwix_quantize_nnx_model_with_config,
            donate_argnums=(0, ),
            static_argnames=(
                "mesh",
                "num_hidden_layers",
                "kv_cache_block_size",
                "kv_cache_num_kv_heads",
                "kv_cache_head_size",
                "kv_cache_dtype",
            ))(model=model_or_model_fn,
               rng=rng,
               mesh=mesh,
               num_hidden_layers=vllm_config.model_config.hf_config.
               num_hidden_layers,
               kv_cache_block_size=block_size,
               kv_cache_num_kv_heads=num_kv_heads,
               kv_cache_head_size=head_size,
               kv_cache_dtype=kv_cache_dtype)

        return model_or_model_fn

    hf_config = vllm_config.model_config.hf_config
    if hasattr(hf_config, "text_config") and hasattr(hf_config.text_config,
                                                     "num_hidden_layers"):
        num_hidden_layers = hf_config.text_config.num_hidden_layers
        logger.info(
            f"Using num_hidden_layers from hf_config.text_config: {num_hidden_layers}"
        )
    elif hasattr(hf_config, "num_hidden_layers"):
        num_hidden_layers = hf_config.num_hidden_layers
        logger.info(
            f"Using num_hidden_layers directly from hf_config: {num_hidden_layers}"
        )
    else:
        raise AttributeError(
            "Could not find 'num_hidden_layers' in hf_config or hf_config.text_config."
        )

    qwix_quantize_fn_for_eval = functools.partial(
        qwix_quantize_nnx_model,
        qwix_config=qwix_config,
        mesh=mesh,
        num_hidden_layers=num_hidden_layers,
        kv_cache_block_size=block_size,
        kv_cache_num_kv_heads=num_kv_heads,
        kv_cache_head_size=head_size,
        kv_cache_dtype=kv_cache_dtype)

    def create_and_quantize_model_factory() -> Callable:
        """
        Helper function to create and quantize the abstract model.
        """
        model = model_or_model_fn()
        # Handle the DeepSeek case, where this needs to be called in the abstract model
        if hasattr(model, 'initialize_cache'):
            model.initialize_cache()
        return qwix_quantize_fn_for_eval(model=model, rng=rng)

    return create_and_quantize_model_factory


def apply_qwix_on_abstract_model(vllm_config: "VllmConfig") -> bool:
    """
    Determines whether to apply Qwix quantization on the abstract model (e.g. for DeepSeek)
    or the concrete model.  See `apply_qwix_quantization` for more details on the differences
    between these two approaches.
    Args:
        vllm_config: the vllm config
    Returns:
        whether to apply Qwix quantization on the abstract model
    """
    quantization_config = vllm_config.additional_config.get("quantization", {})
    return quantization_config.get("qwix", {}).get("use_abstract_model", False)


def get_default_qwix_quantization_config(
        hf_config: dict, skip_quantization: bool) -> dict | None:
    """
    Some models are pre-quantized and in those cases, we want to return a default set of
    Qwix quantization rules (instead of forcing the user to pass in a quantization config each time).

    Note that if a user passes in a quantization config (via `additional_config`), then
    we'll use that instead of this function.

    Args:
        model_type: the name of the model
        quant_method: the quantization method
        skip_quantization: whether to skip quantization.  In this case, we'll return None

    Returns:
        a dictionary containing the default Qwix quantization rules
    """
    if skip_quantization:
        return None
    model_type = hf_config.model_type.lower() if hasattr(
        hf_config, "model_type") else None
    quant_method = hf_config.quantization_config["quant_method"] if hasattr(
        hf_config, "quantization_config") else None
    # TODO (jacobplatin): remove this so that we can support various quantization types + make
    # more flexible
    # NOTE (jacobplatin): we'll default to mixed FP8 (attention) + FP4 (MoE experts)
    # for DeepSeek
    if model_type == "deepseek_v3" and quant_method == "fp8":
        config = copy.deepcopy(DEFAULT_DEEPSEEK_FP4_MLP_MOE_FP8_ATTN_CONFIG)

        # Dynamically fetch block size from HF config if available
        # Config fmt: 'weight_block_size': [1, 512] -> we want the 2nd dim for tile_size
        # NOTE: if the checkpoint is not 1D subchannel, we will throw an error
        hf_quant_config = hf_config.quantization_config
        assert "weight_block_size" in hf_quant_config, "Expected weight_block_size in quantization_config"
        block_size = hf_quant_config["weight_block_size"]
        if isinstance(block_size, (list, tuple)) and len(block_size) == 2:
            assert block_size[
                0] == 1, f"Expected first dimension to be 1 (unchanneled), but got {block_size[0]}! If you are trying to run quantized DeepSeek, we currently only support 1D-subchannel quantization and those models can be found here: https://huggingface.co/collections/jrplatin/deepseek-r1-1d-subchannel"
            tile_size = block_size[1]
            assert tile_size > 1, f"Expected tile_size > 1 for DeepSeek, but got {tile_size}"
            logger.info(
                f"Detected DeepSeek tile_size from config: {tile_size}")

            # Update tile_size in the rules, since we might not always use a 1D subchannel size of
            # 256
            for rule in config["qwix"]["rules"]:
                if "tile_size" in rule:
                    rule["tile_size"] = tile_size
        else:
            raise ValueError(
                f"Invalid weight_block_size config: {block_size}, expected a list/tuple of length 2"
            )

        return config
    elif model_type == "llama4" and quant_method == "compressed-tensors":
        return DEFAULT_LLAMA4_FP8_CONFIG
    # MXFP4 (GPT-OSS): provide a default configuration to quantize MoE experts via Qwix
    elif model_type == "gpt_oss" and quant_method == "mxfp4":
        return DEFAULT_GPT_OSS_FP4_CONFIG


def update_vllm_config_for_qwix_quantization(vllm_config: "VllmConfig"):
    """
    Updates the vLLM config to unpack the Qwix quantization config if it exists.
    By default, we'll check if the checkpoint is quantized and update the
    Qwix quantization config to use the default quantization config if it exists,
    but we'll override this if the user passes in a quantization config via `additional_config`.
    """
    # Automatically detect whether checkpoint is quantized and update the
    # Qwix quantization config accordingly
    # NOTE: if a Qwix config is provided (via the`additional_config`), we'll
    # use that instead
    hf_config = vllm_config.model_config.hf_config
    default_quantization_config = get_default_qwix_quantization_config(
        hf_config, vllm_config.additional_config.get("skip_quantization",
                                                     False))

    maybe_existing_quantization_config = vllm_config.additional_config.get(
        "quantization")
    if maybe_existing_quantization_config:
        logger.warning("Overwriting default Qwix quantization config with "
                       "user provided quantization config.")
    elif default_quantization_config is not None:
        vllm_config.additional_config[
            "quantization"] = default_quantization_config

    # Validate additional config
    if additional_config := vllm_config.additional_config:
        # Try loading/parsing the quantization config so that we can fail fast
        if quantization_config := additional_config.get("quantization"):
            try:
                # NOTE: Qwix quantization supports two paths:
                #  1. quantization config file (which we need to parse to a dictionary)
                #  2. quantization config JSON
                if isinstance(quantization_config, str):
                    quantization_config = quantization_config_file_path_to_dict(
                        quantization_config)
                    # NOTE: unpack the quantization config now so we don't need to keep doing this every time
                    vllm_config.additional_config[
                        "quantization"] = quantization_config
                parse_qwix_config_to_rules(
                    quantization_config["qwix"]["rules"])
            except Exception as e:
                raise ValueError(
                    f"Invalid quantization config; please see README for details on quantization config: {e}"
                )


def get_random_sharded_array(key: PRNGKey, mesh: Mesh, param: nnx.Param,
                             param_shape: tuple, dtype: jnp.dtype,
                             param_name: str) -> jax.Array:
    """
    Returns a random sharded array for the given parameter for the given shape.

    Args:
        key: The random key.
        mesh: The mesh to use for sharding.
        param: The parameter.
        param_shape: The shape of the parameter.
        dtype: The dtype of the parameter.
        param_name: The name of the parameter.

    Returns:
        A random sharded array for the given parameter for the given shape.
    """
    is_int = jnp.issubdtype(dtype, jnp.integer)
    if is_int:
        # These need to be JAX arrays or else you'll run into an overflow error
        minval = jnp.array(jnp.iinfo(dtype).min, dtype=dtype)
        maxval = jnp.array(jnp.iinfo(dtype).max, dtype=dtype)
        weight = jax.random.randint(key, param_shape, minval, maxval, dtype)
    else:
        # NOTE: _uniform() in random.py does not accept float4_e2m1fn
        # Error: "TypeError: uniform only accepts 8-, 16-, 32-, or 64-bit dtypesgot float4_e2m1fn."
        # Workaround: call function with dtype jnp.float8_e4m3fn and cast back to float4_e2m1fn
        if dtype != "float4_e2m1fn":
            weight = jax.random.normal(key, param_shape, dtype)
        else:
            weight = jax.random.normal(key, param_shape,
                                       jnp.float8_e4m3fn).astype(dtype)

    def get_slice(index):
        return weight[index]

    try:
        sharded_array = jax.make_array_from_callback(
            param_shape, NamedSharding(mesh, P(*param.sharding)), get_slice)
    except (ValueError, TypeError):
        logger.warning(
            f"Could not create sharded scale for {param_name} with shape {param_shape} and sharding {param.sharding}, skipping sharding..."
        )
        sharded_array = jax.make_array_from_callback(param_shape,
                                                     NamedSharding(mesh, P()),
                                                     get_slice)

    return sharded_array


def load_random_weights_into_qwix_abstract_model(rng: PRNGKey,
                                                 model: nnx.Module, mesh: Mesh,
                                                 quantization_config: dict):
    """
    Loads random weights for an abstract, Qwix-quantized model.

    Args:
        rng: The random key.
        state: The state of the model.
        mesh: The mesh.
        model: The model.
        quantization_config: The quantization config for the model.
    """
    logger.info("Initializing Qwix-quantized model with random weights...")
    # TODO (jacobplatin): clean up this logic
    scale_dtype = model.weight_loader.scale_dtype
    scale_shape_map = model.weight_loader.scale_shape_map_for_random_weight_loading if hasattr(
        model.weight_loader,
        'scale_shape_map_for_random_weight_loading') else {}
    quantization_block_sizes = quantization_config["weight_block_size"]
    assert len(
        quantization_block_sizes
    ) == 2, f"Expected only 2 quantization block sizes but got {quantization_block_sizes}"

    # Iterate through all variables and initialize them

    for path, param in nnx.iter_graph(model):
        if not isinstance(param, nnx.Variable):
            continue
        if path[0] == 'rng' and path[-1] == "key":
            param.value = rng
            continue
        is_qwix_scale = (path[-1] == 'scale' and path[-2] == "array")
        param_dtype = scale_dtype if is_qwix_scale else param.value.dtype
        param_shape = param.value.shape
        if is_qwix_scale:
            key = f"{path[2]}.{path[3]}"

            if key in scale_shape_map:
                param_shape = scale_shape_map[key]
            else:
                raise ValueError(
                    f"Scale shape for {key} not found in scale_shape_map.")
        param.value = get_random_sharded_array(
            rng, mesh, param, param_shape, param_dtype,
            ".".join([str(x) for x in path]))

    # Handles the DeepSeek case, where this needs to be called to make the cache weights
    # concrete
    if hasattr(model, 'initialize_cache'):
        model.initialize_cache()
    logger.info("Done initializing Qwix-quantized model with random weights")


def manually_quantize_qwix_weight(weight: jax.Array, qtype: jnp.dtype,
                                  channelwise_axes: List[int],
                                  tiled_axes: dict,
                                  calibration_method: str) -> QArray:
    """
    Manually quantizes a weight tensor using Qwix.  Only needed for the SparseMatmul DeepSeek case right now, since
    otherwise, Qwix will handle this automatically (through our application of `qwix.quantize_model`).
    """
    # TODO (jacobplatin): clean this up; this is needed because of issues with Qwix quantizing the `shard_map` in SpraseMatmul
    how_to_quantize = ptq.qarray.HowToQuantize(
        qtype=qtype,
        channelwise_axes=channelwise_axes,
        tiled_axes=tiled_axes,
        calibration_method=calibration_method)

    return ptq.create_quantized_param(weight, how_to_quantize)


def manually_quantize_qwix_activation(inputs: jax.Array, rule_name: str,
                                      qtype: jnp.dtype,
                                      channelwise_axes: List[int],
                                      tiled_axes: dict,
                                      calibration_method: str) -> QArray:
    """
    Manually quantizes an activation tensor using Qwix.  Needed for the SparseMatmul
    DeepSeek MoE case currently.

    Args:
        inputs: The activation tensor to quantize.
        rule_name: The name of the quantization rule to use.
        qtype: The quantization type.
        channelwise_axes: The channelwise axes to quantize.
        tiled_axes: The tiled axes to quantize.
        calibration_method: The calibration method to use.

    Returns:
        The quantized activation tensor.
    """
    rule = qpl.get_current_rule(rule_name)
    lhs_how = ptq.qarray.HowToQuantize(qtype=qtype,
                                       channelwise_axes=channelwise_axes,
                                       tiled_axes=tiled_axes,
                                       calibration_method=calibration_method)
    # This is needed because we aren't passing `act_name` right now
    assert not rule.act_static_scale, "Static scale not supported right now"

    # channelwise_axes should be set to (a subset of) non-contraction axes. e.g.
    # for ragged_dot [m, k] x [g, k, n], they are [0] and [0, 2]
    # TODO (jacobplatin): add support for `act_name`
    return ptq.quantize_act(inputs, lhs_how, rule, "")


def get_quant_dtype_from_qwix_config(
        vllm_config: "VllmConfig") -> tuple[jnp.dtype, jnp.dtype]:
    """
    Gets the quantization dtype from the Qwix config.

    Args:
        vllm_config: The VllmConfig object.

    Returns:
        A tuple of the scale dtype and quant dtype.
    """
    qwix_config = vllm_config.additional_config.get("quantization",
                                                    {}).get("qwix", {})
    scale_dtype = getattr(jnp, qwix_config.get("scale_dtype", "bfloat16"))
    quant_dtype = None
    # TODO (jacobplatin): this needs to be much more robust
    for rule in qwix_config.get("rules", []):
        if rule.get("module_path") == ".*":
            quant_dtype_str = rule.get("weight_qtype", "")
            assert quant_dtype_str, "Quantization dtype not found in Qwix config! We currently expect your Qwix config to have a rule with module_path '.*' and a weight_qtype."
            quant_dtype = getattr(jnp, quant_dtype_str)
    return scale_dtype, quant_dtype
