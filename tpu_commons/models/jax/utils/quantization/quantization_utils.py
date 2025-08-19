# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import List

import jax
import jax.numpy as jnp
import qwix
import yaml
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from qwix import pallas as qpl

from tpu_commons import utils
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.runner.utils import create_kv_caches

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
                            kv_cache_head_size: int) -> nnx.Module:
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
        rules_file_path: the path to the YAML file containing the quantization rules.
            See the README for more information on how to create/use this file.
            (optional)

    Returns:
        model: the quantized model
    """
    qwix_rules = parse_qwix_config_to_rules(qwix_config)
    logger.info(f"Qwix rules: {qwix_rules}")
    logger.info(f"Memory usage before applying quantization of params: "
                f"hbm={utils.hbm_usage_gb(jax.local_devices())}Gb")

    kv_caches = create_kv_caches(
        num_blocks=DEFAULT_NUM_BLOCKS_FOR_JIT_KV_CACHE,
        block_size=kv_cache_block_size,
        num_kv_heads=kv_cache_num_kv_heads,
        head_size=kv_cache_head_size,
        mesh=mesh,
        layer_names=[f"layer.{i}" for i in range(num_hidden_layers)],
        devices=jax.local_devices(),
    )

    def _device_array(*args, sharding=None, **kwargs) -> jax.Array:
        if sharding is None:
            sharding = NamedSharding(mesh, PartitionSpec(None))
        return jax.device_put(*args, device=sharding, **kwargs)

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

    (input_ids, positions, block_tables, query_start_loc, seq_lens,
     request_distribution) = _device_array(
         (input_ids, positions, block_tables, query_start_loc, seq_lens,
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


def load_and_unpack_gptq_int4(weight_dict):
    """
  TODO
  """
    g_idx = weight_dict['g_idx']
    qweight = weight_dict['qweight']
    qzeros = weight_dict['qzeros']
    scales = weight_dict['scales']

    # g_idx is always even.
    tile_count = scales.shape[0]
    assert jnp.all(
        g_idx.reshape(tile_count, -1) == jnp.arange(tile_count).reshape(
            tile_count, 1))

    # .transpose()
    qvalue = qweight.view(jnp.int4) - 8
    # .transpose()
    zp = qzeros.view(jnp.int4) - 7
    if jnp.all(zp == 0):
        zp = None
    scale = scales.astype(jnp.float32)  # .transpose()
    return qpl.QArray(qvalue, scale, zp, jnp.int4)


def load_and_unpack_gptq_int8(weight_dict):
    g_idx = weight_dict['g_idx']
    qweight = weight_dict['qweight']
    qzeros = weight_dict['qzeros']
    scales = weight_dict['scales']

    # g_idx is always even.
    tile_count = scales.shape[0]
    assert jnp.all(
        g_idx.reshape(tile_count, -1) == jnp.arange(tile_count).reshape(
            tile_count, 1))

    qvalue = qweight.transpose().view(jnp.int8) - 128
    zp = qzeros.view(jnp.int8).transpose() - 127
    if jnp.all(zp == 0):
        zp = None
    scale = scales.astype(jnp.float32).transpose()
    return qpl.QArray(qvalue, scale, zp, jnp.int8)
