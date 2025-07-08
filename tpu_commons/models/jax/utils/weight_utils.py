"""Utilities for downloading model weights from HuggingFace."""

import abc
import functools
import glob
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from safetensors import safe_open
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.model import ModelConfig
from tpu_commons.models.jax.common.sharding import ShardingConfig
from tpu_commons.models.jax.layers.misc import shard_put
from tpu_commons.models.jax.utils import file_utils

logger = init_logger(__name__)

HF_WEIGHTS_FORMAT = "*.safetensors"
FULL_DOWNLOAD_DISK_RATIO = 0.9


class ParameterType(str, Enum):
    weight = "weight"
    bias = "bias"


@dataclass
class TransformationConfig:
    transpose: Mapping[str, Any] = field(default_factory=dict)
    reshape: Mapping[str, Any] = field(default_factory=dict)


class WeightLoader(abc.ABC):
    # Create a doc string for this class.
    """Abstract base class for loading model weights.

    This class provides a common interface for loading model weights from various
    sources (currently supports HuggingFace) and applying necessary transformations
    (e.g., transposing, reshaping) before sharding and loading them into the
    model.
    Each implemeentation of the WeightLoader must define the procedure for loading the
    weights in its own load_weights() method.

    Args:
        vllm_config (VllmConfig): The VLLM configuration object.
        model_config (ModelConfig): The model configuration object.
        framework str: Type of backend to use (defaults to "flax")
        cache_dir str: An optional cache dir to load the model from (currently unused).
        sharding_cfg (ShardingConfig): The sharding configuration object.
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 model_config: ModelConfig,
                 framework: str = "flax",
                 cache_dir: Optional[str] = None,
                 sharding_cfg: Optional[ShardingConfig] = None):
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.sharding_cfg = sharding_cfg
        self.framework = framework
        self.cache_dir = cache_dir
        self.transformation_cfg = TransformationConfig()
        self.setup()

    def setup(self):
        self.names_and_weights_generator = hf_model_weights_iterator(
            model_name_or_path=self.vllm_config.model_config.model,
            framework=self.framework)

    def set_transpose_param_map(self,
                                transpose_param_dict: Mapping[str,
                                                              Tuple[int]]):
        self.transformation_cfg.transpose = transpose_param_dict

    def set_reshape_param_map(self, param_reshape_dict: Mapping[str,
                                                                Tuple[int]],
                              param_type: str):
        self.transformation_cfg.reshape[param_type] = param_reshape_dict

    def set_loaded_to_standardized_keys(
            self, loaded_to_standardized_keys: Mapping[str, str]):
        self.loaded_to_standardized_keys = loaded_to_standardized_keys

    def transpose_params(self, param_key: str, param_tensor: jax.Array):
        for key in self.transformation_cfg.transpose:
            if key in param_key:
                return jnp.transpose(param_tensor,
                                     self.transformation_cfg.transpose[key])
        return param_tensor  # Base case / no-op

    def reshape_params(self, param_key: str, param_tensor: jax.Array,
                       param_type: str):
        for key in self.transformation_cfg.reshape[param_type]:
            if key in param_key:
                reshape_shape = self.transformation_cfg.reshape[param_type][
                    key]
                return jnp.reshape(param_tensor, reshape_shape)
        return param_tensor  # Base case / no-op

    abc.abstractmethod

    def load_weights(self, model_for_loading: nnx.Module):
        raise NotImplementedError


def hf_model_weights_iterator(
    model_name_or_path: str,
    framework: str,
) -> Generator[tuple, Any, None]:
    weights_files = []
    weights_location = "local"
    if os.path.isdir(model_name_or_path):
        weights_files = glob.glob(
            os.path.join(model_name_or_path, HF_WEIGHTS_FORMAT))
    elif file_utils.is_gcs_path(model_name_or_path):
        local_free_disk_size = file_utils.get_free_disk_size()
        model_size = file_utils.get_gcs_model_weights_size(
            model_name_or_path, HF_WEIGHTS_FORMAT)
        if model_size < local_free_disk_size * FULL_DOWNLOAD_DISK_RATIO:
            logger.info(f"Downloading weights from GCS {model_name_or_path}")
            weights_files = file_utils.download_model_weights_from_gcs(
                model_name_or_path, HF_WEIGHTS_FORMAT)
        else:
            weights_files = file_utils.list_gcs_dir(model_name_or_path,
                                                    HF_WEIGHTS_FORMAT)
            weights_location = "gcs"
    elif file_utils.is_hf_repo(model_name_or_path):
        local_free_disk_size = file_utils.get_free_disk_size()
        model_size = file_utils.get_hf_model_weights_size(
            model_name_or_path, HF_WEIGHTS_FORMAT)
        if model_size < local_free_disk_size * FULL_DOWNLOAD_DISK_RATIO:
            logger.info(f"Downloading weights from HF {model_name_or_path}")
            weights_files = file_utils.download_model_weights_from_hf(
                model_name_or_path, HF_WEIGHTS_FORMAT)
        else:
            weights_files = file_utils.list_hf_repo(model_name_or_path,
                                                    HF_WEIGHTS_FORMAT)
            weights_location = "hf"
    else:
        raise ValueError(
            f"{model_name_or_path} must be a local path, or a gcs path, or a HF model id."
        )

    if len(weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any {HF_WEIGHTS_FORMAT} files in {model_name_or_path}."
        )

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
                model_name_or_path, os.path.basename(st_file))[0]
        elif weights_location == "hf":
            st_file = file_utils.download_model_weights_from_hf(
                model_name_or_path, os.path.basename(st_file))[0]
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
    shard = functools.partial(shard_put, mesh=mesh)

    model_config = vllm_config.model_config
    model_path = model_config.model
    hf_config = model_config.hf_config

    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    hidden_size = model_config.get_hidden_size()

    # NOTE(wenlong): we may need to pad head_dim to a multiple of 128 as required of kernels
    # Details can be seen at: tpu_commons/kernels/ragged_kv_cache_update.py::_kv_cache_update()

    head_dim_original = model_config.get_head_size()
    head_dim = head_dim_original
    head_dim_pad = 0
    if head_dim % 128 != 0:
        head_dim = 128
        head_dim_pad = head_dim - head_dim_original
        logger.info(f"Change head_dim from {head_dim_original} to {head_dim}")

    reshape_keys = {
        "q_proj": (num_heads, head_dim_original, hidden_size),
        "k_proj": (num_kv_heads, head_dim_original, hidden_size),
        "v_proj": (num_kv_heads, head_dim_original, hidden_size),
        "o_proj": (hidden_size, num_heads, head_dim_original),
    }
    bias_reshape_keys = {
        "q_proj.bias": (num_heads, head_dim_original),
        "k_proj.bias": (num_kv_heads, head_dim_original),
        "v_proj.bias": (num_kv_heads, head_dim_original)
    }
    transpose_keys = {
        "lm_head": (1, 0),
        "gate_proj": (1, 0),
        "up_proj": (1, 0),
        "down_proj": (1, 0),
        "q_proj": (2, 0, 1),
        "k_proj": (2, 0, 1),
        "v_proj": (2, 0, 1),
        "o_proj": (1, 2, 0),
    }

    params = nnx.state(model)
    for hf_key, hf_weight in hf_model_weights_iterator(model_path,
                                                       framework="flax"):

        if hf_key.endswith(".weight"):
            hf_key = hf_key.removesuffix(".weight")

        # Find the corresponding model key using the HF key
        if "layer" in hf_key:
            layer_num = re.search(r"layers\.(\d+)", hf_key).group(1)
            layer_key = re.sub(r"layers\.\d+", "layers.*", hf_key)
            model_key, model_sharding = mappings[layer_key]
            model_key = re.sub(r"layers\.\*", f"layers.{layer_num}", model_key)
        else:
            model_key, model_sharding = mappings[hf_key]
        model_weight = get_param(params, model_key)

        logger.debug(
            f"{hf_key}: {hf_weight.shape}  -->  {model_key}: {model_weight.value.shape} {model_sharding}"
        )

        if hf_key.endswith(".bias"):
            for key in bias_reshape_keys:
                if key in hf_key:
                    hf_weight = jnp.reshape(hf_weight, bias_reshape_keys[key])
                    if head_dim_pad:
                        hf_weight = jnp.pad(hf_weight,
                                            ((0, 0), (0, head_dim_pad)))
                    break
        else:
            for key in reshape_keys:
                if key in hf_key:
                    hf_weight = jnp.reshape(hf_weight, reshape_keys[key])
                    if head_dim_pad:
                        hf_weight = jnp.pad(hf_weight,
                                            ((0, 0), (0, head_dim_pad),
                                             (0, 0)))
                    break
            for key in transpose_keys:
                if key in hf_key:
                    hf_weight = jnp.transpose(hf_weight, transpose_keys[key])
                    break
            if head_dim_pad == 0:
                assert model_weight.value.shape == hf_weight.shape

        # Update the model weight
        model_weight.value = shard(hf_weight, model_sharding)

    nnx.update(model, params)
