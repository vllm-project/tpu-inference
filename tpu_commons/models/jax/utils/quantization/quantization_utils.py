# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Union

import jax
import jax.numpy as jnp
import qwix
import yaml
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_commons import utils_jax as utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.runner.utils import create_kv_caches

logger = init_logger(__name__)

MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)

DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE = 2000

DEFAULT_QWIX_RULES = lambda quant_dtype: [  # noqa: E731
    qwix.QuantizationRule(
        module_path='.*attn.*',
        weight_qtype=quant_dtype,  # quantizes weights in int8.
    ),
    qwix.QuantizationRule(
        # module_path='.*',
        module_path='.*mlp.*',
        weight_qtype=quant_dtype,
        act_qtype=quant_dtype,
        tile_size=None,
    ),
]


def parse_quantization_yaml_file_to_rules(quantization_config_file_path: str):
    """
    Parse a yaml file containing Qwix quantization rules into a list of QuantizationRule objects.

    Args:
        quantization_config_file_path: the path to the yaml file

    Returns:
        a list of QuantizationRule objects
    """
    with open(quantization_config_file_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    rules = []
    for rule in yaml_dict['rules']:
        rules.append(qwix.QuantizationRule(**rule))

    return rules


def qwix_quantize_nnx_model(model: nnx.Module,
                            quant_dtype: str | jnp.dtype,
                            rng: jax.Array,
                            mesh: Mesh,
                            num_hidden_layers: int,
                            kv_cache_block_size: int,
                            kv_cache_num_combined_kv_heads: int,
                            kv_cache_head_size: int,
                            kv_cache_quant_dtype: Optional[Union[
                                str, jnp.dtype]] = None,
                            rules_file_path: str = None) -> nnx.Module:
    """
    Quantizes a Flax NNX model using Qwix.

    Args:
        model: the model to quantize
        quant_dtype: the dtype to quantize the model to
        rng: the random number generator to use
        mesh: the mesh to use
        num_hidden_layers: the number of hidden layers in the model
        kv_cache_page_size: the page size of the kv cache
        num_combined_kv_heads: the number of combined kv heads
        head_size: the head size of the kv cache
        kv_cache_quant_dtype: the dtype to quantize the kv cache to (optional)
        rules_file_path: the path to the YAML file containing the quantization rules.
            See the README for more information on how to create/use this file.
            (optional)

    Returns:
        model: the quantized model
    """
    # TODO: add support for rules!
    if quant_dtype is not None and rules_file_path is not None:
        raise ValueError(
            "Cannot specify both quantization rules and quantization dtype in your quantization config"
        )
    qwix_rules = parse_quantization_yaml_file_to_rules(
        rules_file_path
    ) if rules_file_path is not None else DEFAULT_QWIX_RULES(quant_dtype)
    logger.info(f"Qwix rules: {qwix_rules}")
    logger.info(f"Memory usage before applying quantization of params: "
                f"hbm={utils.hbm_usage_gb(jax.local_devices())}Gb")

    kv_caches = create_kv_caches(
        num_blocks=DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE,
        block_size=kv_cache_block_size,
        num_kv_heads=kv_cache_num_combined_kv_heads,
        head_size=kv_cache_head_size,
        mesh=mesh,
        layer_names=[f"layer.{i}" for i in range(num_hidden_layers)],
        devices=jax.local_devices(),
        kv_cache_quant_dtype=kv_cache_quant_dtype,
    )

    def _device_array(*args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

    input_ids = jax.random.randint(rng, (512, ), 0, 100, dtype=jnp.int32)
    positions = jax.random.randint(rng, (512, ), 0, 100, dtype=jnp.int32)
    slot_mapping_metadata = jax.random.randint(rng, (3, 512),
                                               0,
                                               100,
                                               dtype=jnp.int32)
    num_slices = jax.random.randint(rng, (1, ), 0, 100, dtype=jnp.int32)
    block_tables = jax.random.randint(rng, (256, 16), 0, 100, dtype=jnp.int32)
    query_start_loc = jax.random.randint(rng, (257, ), 0, 100, dtype=jnp.int32)
    seq_lens = jax.random.randint(rng, (256, ), 0, 100, dtype=jnp.int32)
    num_seqs = jax.random.randint(rng, (1, ), 0, 100, dtype=jnp.int32)

    (input_ids, positions, slot_mapping_metadata, num_slices, block_tables,
     query_start_loc, seq_lens, num_seqs) = _device_array(
         (input_ids, positions, slot_mapping_metadata, num_slices,
          block_tables, query_start_loc, seq_lens, num_seqs))

    model_input = {
        "kv_caches":
        kv_caches,
        "input_ids":
        input_ids,
        "attention_metadata":
        AttentionMetadata(
            input_positions=positions,
            slot_mapping=slot_mapping_metadata,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            num_slices=num_slices,
        ),
    }
    model = qwix.quantize_model(model, qwix.PtqProvider(qwix_rules),
                                **model_input)
    return model


def quantize(x: jax.Array, quant_dtype: jnp.dtype):
    """Quantizes uses a per-tensor approach.
      TODO (jacobplatin): support a per-token approach

    Args:
        x: the value to quantize
        quant_dtype: the dtype to quantize to

    Returns:
         x (jax.Array): the quantized value
         scale (jax.Array): the scale factor (of shape (1,))
          NOTE: this should really be a float, but static types don't play
          nicely with JAX tracing
    """
    # Would be nicer to do this as a dictionary, but indexing with
    # a jnp.dtype didn't work for some reason
    if quant_dtype == jnp.int8:
        dtype_max = MAX_INT8
    elif quant_dtype == jnp.int4:
        dtype_max = MAX_INT4
    elif quant_dtype == jnp.float8_e4m3fn:
        dtype_max = E4M3_MAX
    else:
        raise ValueError(f"Unsupported quant dtype: {quant_dtype}")

    scale = jnp.max(jnp.abs(x)) / dtype_max

    # Ensure scales are not zero to avoid division by zero errors.
    scale = jnp.maximum(scale, 1e-6)

    x = (x / scale).astype(quant_dtype)

    # Upcast to float32 to avoid a SMEM Mosaic error with bfloat16
    # NOTE: the scales are really floats but static types don't play
    # nicely with JAX tracing
    scale = scale.reshape(-1).astype(jnp.float32)

    return x, scale
