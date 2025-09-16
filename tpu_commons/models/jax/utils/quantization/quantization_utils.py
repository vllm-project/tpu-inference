# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
from typing import TYPE_CHECKING, Callable, List

import jax
import jax.numpy as jnp
import qwix
import yaml
from flax import nnx
from jax.sharding import Mesh

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from tpu_commons import utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.runner.kv_cache import create_kv_caches
from tpu_commons.utils import device_array

logger = init_logger(__name__)

QUANTIZATION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs")
DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE = 2000
DEFAULT_NUM_TOKENS_FOR_MODEL_INPUTS = 512
DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS = 256
DEFAULT_MAX_NUM_BLOCKS_PER_REQ = 16


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

    kv_cache_jnp_dtype = utils.TPU_STR_DTYPE_TO_JAX_DTYPE.get(
        kv_cache_dtype.lower().strip())

    kv_caches = create_kv_caches(
        num_blocks=DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE,
        block_size=kv_cache_block_size,
        num_kv_heads=kv_cache_num_kv_heads,
        head_size=kv_cache_head_size,
        mesh=mesh,
        layer_names=[f"layer.{i}" for i in range(num_hidden_layers)],
        cache_dtype=kv_cache_jnp_dtype)

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
        rng, (DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS + 1, ),
        0,
        100,
        dtype=jnp.int32)
    seq_lens = jax.random.randint(rng,
                                  (DEFAULT_MAX_NUM_SEQS_FOR_MODEL_INPUTS, ),
                                  0,
                                  100,
                                  dtype=jnp.int32)
    num_seqs = jax.random.randint(rng, (1, ), 0, 100, dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, num_seqs[0]], dtype=jnp.int32)

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
        f"Could not find quantization config file with name '{quantization_config_file_path}' in 'tpu_commons/models/jax/utils/quantization/configs."
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

    qwix_quantize_fn_for_eval = functools.partial(
        qwix_quantize_nnx_model,
        qwix_config=qwix_config,
        mesh=mesh,
        num_hidden_layers=vllm_config.model_config.hf_config.num_hidden_layers,
        kv_cache_block_size=block_size,
        kv_cache_num_kv_heads=num_kv_heads,
        kv_cache_head_size=head_size)

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
