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

from flax import nnx
from flax.nnx import statelib
import jax
import jaxtyping
import re
from typing import Any

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
    head_dim = model_config.get_head_size()

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
    reshape_keys = {
        "q_proj": (num_heads, -1, hidden_size),
        "k_proj": (num_kv_heads, -1, hidden_size),
        "v_proj": (num_kv_heads, -1, hidden_size),
        "o_proj": (hidden_size, num_heads, -1),
    }

    bias_reshape_keys = {
        "q_proj.bias": (num_heads, head_dim),
        "k_proj.bias": (num_kv_heads, head_dim),
        "v_proj.bias": (num_kv_heads, head_dim)
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
                    break
        else:
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
        model_weight.value = shard(hf_weight, model_sharding)

    nnx.update(model, params)

def build_flat_dict(flat_state, mappings):
    new_flat_dict = {}
    for keys, v in flat_state:
        path = '.'.join(str(key) for key in keys)
        mapped = False
        for src, (tgt, sharding) in mappings.items():
            regex = "^" + re.escape(tgt).replace("\\.\\*", r"\.(\d+)") + "$"
            matched = re.match(regex, path)
            if matched:
              # Extract wildcards if any
              wildcards = matched.groups()
              src_parts = []
              wc_index = 0
              for part in src.split("."):
                  if part == "*":
                      src_parts.append(wildcards[wc_index])
                      wc_index += 1
                  else:
                      src_parts.append(part)
              actual_src = ".".join(src_parts)
              new_flat_dict[actual_src] = v, sharding
              mapped = True
              break
        if not mapped:
          print(f"!!! No mapping for flat state: {keys}")
    return new_flat_dict


def transfer_state_with_mappings(src_state, tgt_state, mappings, shard=None):
    src_flat = src_state.flat_state()
    tgt_flat = tgt_state.flat_state()

    new_src_dict = build_flat_dict(tgt_flat, mappings)
    transpose_keys = {
        "q_proj": (1, 0, 2),
        "k_proj": (1, 0, 2),
        "v_proj": (1, 0, 2),
    }
    print(f"YY {new_src_dict=}")
    for src_keys, v in src_flat:
        flattened_src_keys = '.'.join(str(k) for k in src_keys)
        new_v = jnp.copy(v.value)
        print(f"Processing source key: {flattened_src_keys} and value: {new_v.shape}")
        if flattened_src_keys not in new_src_dict:
            print(f"!!! No mapping for source key: {flattened_src_keys}")
            continue
        sharding = new_src_dict[flattened_src_keys][1]

        # E.g. layers.*.attn.k_proj.w, layers.*.attn.k_proj.w_lora_a
        # E.g. layers.*.mlp.down_proj.kernel, layers.*.mlp.down_proj.kernel_lora_a
        if src_keys[-2] in transpose_keys and 'lora' not in src_keys[-1]:
            v_maybe_t = jnp.transpose(new_v, transpose_keys[src_keys[-2]])
        else:
            v_maybe_t = new_v
        assert new_src_dict[flattened_src_keys][0].value.shape == v_maybe_t.shape, \
            f"Shape mismatch for {flattened_src_keys}: {new_src_dict[flattened_src_keys][0].value.shape} vs {v_maybe_t.shape}"
        new_src_dict[flattened_src_keys][0].value = shard(v_maybe_t, sharding) if shard else v_maybe_t

    tgt_state = tgt_state.from_flat_path(tgt_flat)
    return tgt_state

class LoadType(Enum):
    ALL = 1
    LORA_ONLY = 2
    BASE_ONLY = 3

def load_nnx_weights(source_state, target_model: nnx.Module,
                     mappings: Dict[str, str], mesh: Mesh, load_type: LoadType = LoadType.ALL):
    """
    Load weights from a source nnx model into a target nnx model.

    Args:
        source_model: The nnx model to extract weights from
        target_model: The nnx model to load weights into
        mappings: Dictionary mapping source model keys to target model keys
                 Format: {"source_key": "target_key"}
        mesh: JAX mesh for sharding
    """
    import functools
    from tpu_commons.models.jax.layers.misc import shard_put

    shard = functools.partial(shard_put, mesh=mesh)

    _, target_lora, target_base, _, _ = nnx.split(target_model, nnx.LoRAParam, nnx.Param, nnx.RngKey, nnx.RngCount)
    source_lora, source_base, _, _ = nnx.split_state(source_state, nnx.LoRAParam, nnx.Param, nnx.RngKey, nnx.RngCount)

    if load_type == LoadType.ALL or load_type == LoadType.LORA_ONLY:
      updated_lora = transfer_state_with_mappings(source_lora, target_lora, mappings, shard)
      nnx.update(target_model, updated_lora)

    if load_type == LoadType.ALL or load_type == LoadType.BASE_ONLY:
      updated_base = transfer_state_with_mappings(source_base, target_base, mappings, shard)
      nnx.update(target_model, updated_base)

def apply_sharding(model_state, shardings, rng, mesh):
    flat_state = model_state.flat_state()
    new_state = {}

    for src_keys, v in flat_state:
        flattened_src_keys = '.'.join('*' if isinstance(k, int) else k for k in src_keys)
        if flattened_src_keys not in shardings:
          print(f"!!! No sharding found for {flattened_src_keys}")
          dim_tuple = ()
        else:
          dim_tuple = shardings[flattened_src_keys]
        pspec = jax.sharding.PartitionSpec(*dim_tuple)
        sharding = jax.sharding.NamedSharding(mesh, pspec)
        if v.type == nnx.Param:
          new_state[src_keys] = jax.device_put(jnp.zeros(v.value.shape, dtype=v.value.dtype), sharding)
        elif v.type == nnx.RngKey:
          new_state[src_keys] = rng
        elif v.type == nnx.RngCount:
          new_state[src_keys] = jax.device_put(jnp.array(0, dtype=v.value.dtype), sharding)
        else:
          raise ValueError(f"Unsupported type {v.type} for sharding")

    return model_state.from_flat_path(new_state)

