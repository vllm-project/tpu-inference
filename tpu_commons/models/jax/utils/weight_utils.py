"""Utilities for downloading model weights from HuggingFace."""

import functools
import glob
import os
import re
from typing import Any, Dict, Generator

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from safetensors import safe_open

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.utils import file_utils

logger = init_logger(__name__)

HF_WEIGHTS_FORMAT = "*.safetensors"
FULL_DOWNLOAD_DISK_RATIO = 0.9


def hf_model_weights_iterator(
    model_name: str,
    framework: str,
) -> Generator[tuple, Any, None]:
    weights_files = []
    weights_location = "local"
    if os.path.isdir(model_name):
        weights_files = glob.glob(os.path.join(model_name, HF_WEIGHTS_FORMAT))
    elif file_utils.is_gcs_path(model_name):
        local_free_disk_size = file_utils.get_free_disk_size()
        model_size = file_utils.get_gcs_model_weights_size(
            model_name, HF_WEIGHTS_FORMAT)
        if model_size < local_free_disk_size * FULL_DOWNLOAD_DISK_RATIO:
            logger.info(f"Downloading weights from GCS {model_name}")
            weights_files = file_utils.download_model_weights_from_gcs(
                model_name, HF_WEIGHTS_FORMAT)
        else:
            weights_files = file_utils.list_gcs_dir(model_name,
                                                    HF_WEIGHTS_FORMAT)
            weights_location = "gcs"
    elif file_utils.is_hf_repo(model_name):
        local_free_disk_size = file_utils.get_free_disk_size()
        model_size = file_utils.get_hf_model_weights_size(
            model_name, HF_WEIGHTS_FORMAT)
        if model_size < local_free_disk_size * FULL_DOWNLOAD_DISK_RATIO:
            logger.info(f"Downloading weights from HF {model_name}")
            weights_files = file_utils.download_model_weights_from_hf(
                model_name, HF_WEIGHTS_FORMAT)
        else:
            weights_files = file_utils.list_hf_repo(model_name,
                                                    HF_WEIGHTS_FORMAT)
            weights_location = "hf"
    else:
        raise ValueError(
            f"{model_name} must be a local path, or a gcs path, or a HF model id."
        )

    if len(weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any {HF_WEIGHTS_FORMAT} files in {model_name}.")

    if weights_location != "local":
        logger.warning(
            "Weights files are not downloaded to local disk at once due to insufficient disk space. "
            "They will be downloaded on the fly during loading.")

    # Sort to ensure the order of files is consistent.
    weights_files.sort()

    for st_file in weights_files:
        logger.info(f"Loading weights from {st_file}")
        if weights_location == "gcs":
            st_file = file_utils.download_model_weights_from_gcs(
                model_name, os.path.basename(st_file))[0]
        elif weights_location == "hf":
            st_file = file_utils.download_model_weights_from_hf(
                model_name, os.path.basename(st_file))[0]
        # NOTE: We enforce loading tensors on CPU here.
        # Because otherwise the tensor will be loaded on TPU:0 by default,
        # although the tensor would eventually be sharded across multiple TPUs,
        # it would lead to OOM on TPU:0 for large models.
        with jax.default_device(jax.devices("cpu")[0]):
            with safe_open(st_file, framework=framework) as f:
                for name in f.keys():
                    weight_tensor = f.get_tensor(name)
                    yield name, weight_tensor
        if weights_location != "local":
            file_utils.delete_file(st_file)


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def get_num_q_heads_by_tp(num_q_heads: int, num_kv_heads: int,
                          tp_size: int) -> int:
    num_kv_heads_by_tp = get_num_kv_heads_by_tp(num_kv_heads, tp_size)
    kv_repeats = num_kv_heads_by_tp // num_kv_heads
    q_repeats = 1
    if num_q_heads % tp_size != 0:
        if (num_q_heads * kv_repeats) % tp_size != 0:
            raise ValueError(
                f"Cannot make q_heads divisible by TP={tp_size} properly. Consider other padding strategies."
            )
        q_repeats = kv_repeats

    return q_repeats * num_q_heads


def get_param(params: nnx.State, path: str) -> nnx.State:
    keys = path.split(".")
    current_level = params
    for key in keys:
        if key.isdigit():
            current_level = current_level[int(key)]
        else:
            if key in current_level:
                current_level = current_level[key]
            else:
                raise ValueError(f"{path} is not a valid param path")
    return current_level


def load_hf_weights(vllm_config, model: nnx.Module, mappings: Dict[str, str],
                    mesh: Mesh):
    transpose_keys = {
        "lm_head": (1, 0),
        "gate_proj": (1, 0),
        "up_proj": (1, 0),
        "down_proj": (1, 0),
        "q_proj": (0, 2, 1),
        "k_proj": (0, 2, 1),
        "v_proj": (0, 2, 1),
        "o_proj": (1, 2, 0),
    }
    shard = functools.partial(shard_put, mesh=mesh)

    model_path = vllm_config.model_config.model
    hf_config = vllm_config.model_config.hf_config
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    hidden_size = hf_config.hidden_size

    reshape_keys = {
        "q_proj": (num_heads, -1, hidden_size),
        "k_proj": (num_kv_heads, -1, hidden_size),
        "v_proj": (num_kv_heads, -1, hidden_size),
        "o_proj": (hidden_size, num_heads, -1),
    }

    params = nnx.state(model)
    for hf_key, hf_weight in hf_model_weights_iterator(model_path,
                                                       framework="flax"):
        hf_key = hf_key.removesuffix(".weight")

        # Find the corresponding model key using the HF key
        if "layer" in hf_key:
            layer_num = re.search(r"layers\.(\d+)", hf_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", hf_key)
            model_key = mappings[layer_key]
            model_key = re.sub(r"layers\.\*", f"layers.{layer_num}", model_key)
        else:
            model_key = mappings[hf_key]
        model_weight = get_param(params, model_key)

        logger.debug(
            f"{hf_key}: {hf_weight.shape}  -->  {model_key}: {model_weight.value.shape}"
        )

        # Reshape HF weight if needed
        for key in reshape_keys:
            if key in hf_key:
                hf_weight = jnp.reshape(hf_weight, reshape_keys[key])
                break
        # Transpose HF weight if needed
        for key in transpose_keys:
            if key in hf_key:
                hf_weight = jnp.transpose(hf_weight, transpose_keys[key])
                break
        assert model_weight.value.shape == hf_weight.shape

        # Update the model weight
        model_weight.value = shard(hf_weight, model_weight.sharding)

    nnx.update(model, params)
