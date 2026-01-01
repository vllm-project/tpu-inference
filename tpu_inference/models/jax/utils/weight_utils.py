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
"""Utilities for downloading model weights from HuggingFace."""

import functools
import glob
import math
import os
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import torch
import torchax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from vllm.config import VllmConfig

from tpu_inference import envs, utils
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils import file_utils

logger = init_logger(__name__)

HF_WEIGHTS_FORMAT = "*.safetensors"

DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
}


@dataclass
class MetadataMap:
    name_map: dict[str, str] = field(default_factory=dict)
    transpose_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reshape_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    bias_reshape_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pad_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    bias_pad_map: dict[str, tuple[int, ...]] = field(default_factory=dict)


############ START Used by llama4, deepseek only for now START ############


def print_param_info(param: nnx.Param, name: str):
    logger.warning(f"Global shape for {name}: {param.value.shape}")
    logger.warning(f"Sharding for {name}: {param.sharding}")

    logger.warning(
        f"Shape of {name} on a single device: {param.value.addressable_shards[0].data.shape}"
    )


def transpose_params(param_key: str, param_tensor: jax.Array, transpose_map):
    for key, value in transpose_map.items():
        if key in param_key:
            return jnp.transpose(param_tensor, value)
    return param_tensor  # Base case / no-op


def reshape_params(param_key: str, param_tensor: jax.Array, shape_map):
    for key, new_shape in shape_map.items():
        if key in param_key:
            try:
                #TODO:(gpolovets) Add validation on whether reshape preserves data layout.
                return jnp.reshape(param_tensor, new_shape)
            except TypeError:
                raise TypeError(
                    f"Cannot reshape for key={key}, new_shape={new_shape}, param_shape={param_tensor.shape}"
                )
    return param_tensor  # Base case / no-op


def model_file_generator(
        model_name_or_path: str,
        download_dir: Optional[str]) -> Generator[str, None, None]:
    weights_files = get_model_weights_files(model_name_or_path, download_dir)
    for st_file in weights_files:
        yield st_file


def model_weights_generator(
    model_name_or_path: str,
    framework: str,
    filter_regex: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> Generator[tuple, None, None]:
    for st_file in model_file_generator(model_name_or_path, download_dir):
        for name, weight_tensor in model_weights_single_file_generator(
                st_file, framework, filter_regex):
            yield name, weight_tensor


def convert_torch_to_jax_with_view(loaded_weight: torch.Tensor,
                                   cast_type: jnp.dtype) -> jax.Array:
    """
    Converts a PyTorch tensor to a JAX array by reinterpreting its
    bit representation using a dtype view map.
    """
    torch_view_type = DTYPE_VIEW_MAP.get(jnp.dtype(cast_type))
    loaded_weight = jnp.array(
        loaded_weight.view(torch_view_type).numpy()).view(cast_type)
    return loaded_weight


############ END Used by llama4, deepseek only for now END ############


def get_model_weights_files(
        model_name_or_path: str,
        download_dir: Optional[str]) -> tuple[list[str], str]:
    """
    Helper to get weight files and their location.
    """

    if os.path.isdir(model_name_or_path):
        logger.info(f"Found weights from local: {model_name_or_path}")
        weights_files = glob.glob(
            os.path.join(model_name_or_path, HF_WEIGHTS_FORMAT))
    elif file_utils.is_hf_repo(model_name_or_path):
        logger.info(f"Downloading weights from HF {model_name_or_path}")
        weights_files = file_utils.download_model_weights_from_hf(
            model_name_or_path, download_dir, HF_WEIGHTS_FORMAT)
    else:
        raise ValueError(
            f"{model_name_or_path} must be a local directory, or a Huggingface model id."
        )

    if not weights_files:
        raise RuntimeError(
            f"Cannot find any {HF_WEIGHTS_FORMAT} files in {model_name_or_path}."
        )

    weights_files.sort()
    return weights_files


def model_weights_single_file_generator(
    weights_file: str,
    framework: str,
    filter_regex: Optional[str] = None,
) -> Generator[tuple, None, None]:
    logger.info(f"Loading weights from {weights_file}")
    # NOTE: We enforce loading tensors on CPU here.
    # Because otherwise the tensor will be loaded on TPU:0 by default,
    # although the tensor would eventually be sharded across multiple TPUs,
    # it would lead to OOM on TPU:0 for large models.
    with jax.default_device(jax.devices("cpu")[0]):
        with safe_open(weights_file, framework=framework) as f:
            for name in f.keys():
                if filter_regex is not None and not re.match(
                        filter_regex, name):
                    continue
                weight_tensor = f.get_tensor(name)
                yield name, weight_tensor


def get_param(params: nnx.State, path: str) -> nnx.State:
    keys = path.split(".")
    plevel = params
    for key in keys:
        if key.isdigit():
            plevel = plevel[int(key)]
        else:
            if key in plevel:
                plevel = plevel[key]
            else:
                raise ValueError(f"{path} is not a valid param path")
    return plevel


def get_param_and_sharding(params: nnx.State, shardings: Any,
                           path: str) -> tuple[nnx.State, nnx.State]:
    keys = path.split(".")
    plevel = params
    slevel = shardings
    for key in keys:
        if key.isdigit():
            plevel = plevel[int(key)]
            slevel = slevel[int(key)]
        else:
            if key in plevel:
                plevel = plevel[key]
                slevel = slevel[key]
            else:
                raise ValueError(f"{path} is not a valid param path")
    return plevel, slevel.value


def shard_put(x: jax.Array, shardings, mesh: jax.sharding.Mesh) -> jax.Array:
    # Single device sharding requires this special handling
    # to avoid the recursive jit error.
    if math.prod(mesh.axis_sizes) == 1:
        return jax.device_put(x, mesh.devices.flatten()[0])

    if isinstance(shardings, tuple):
        return jax.device_put(x, NamedSharding(mesh, P(*shardings)))
    else:
        return jax.device_put(x, shardings)


def get_default_maps(model_config, mesh: Mesh,
                     name_map: dict[str, str]) -> MetadataMap:
    """Load weights from one model weights file to the model, run on single thread."""
    sharding_size = mesh.shape["model"]

    hf_config = model_config.hf_config

    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    hidden_size = model_config.get_hidden_size()

    # Pad head_dim for kernel performance.
    head_dim_original = model_config.get_head_size()

    reshape_keys: dict[str, tuple[int, ...]] = {
        "q_proj": (num_heads, head_dim_original, hidden_size),
        "k_proj": (num_kv_heads, head_dim_original, hidden_size),
        "v_proj": (num_kv_heads, head_dim_original, hidden_size),
        "o_proj": (hidden_size, num_heads, head_dim_original),
    }
    bias_reshape_keys: dict[str, tuple[int, ...]] = {
        "q_proj.bias": (num_heads, head_dim_original),
        "k_proj.bias": (num_kv_heads, head_dim_original),
        "v_proj.bias": (num_kv_heads, head_dim_original)
    }
    transpose_keys: dict[str, tuple[int, ...]] = {
        "lm_head": (1, 0),
        "fc": (1, 0),
        "gate_proj": (1, 0),
        "up_proj": (1, 0),
        "down_proj": (1, 0),
        "q_proj": (2, 0, 1),
        "k_proj": (2, 0, 1),
        "v_proj": (2, 0, 1),
        "o_proj": (1, 2, 0),
    }

    # # get vision config
    if model_config.is_multimodal_model:
        # TODO: Wenlong: Do not consider padding for now
        transpose_keys.update({
            "attn.proj": (1, 0),
            "attn.qkv": (1, 0),
            "visual.merger.mlp": (1, 0),
            "visual.patch_embed.proj": (2, 3, 4, 1, 0),
        })

    # key: (padding_dim, padding_size)
    pad_keys: dict[str, tuple[int, ...]] = {
        "q_proj": (1, sharding_size // num_heads),
        "k_proj": (1, sharding_size // num_kv_heads),
        "v_proj": (1, sharding_size // num_kv_heads),
        "o_proj": (0, sharding_size // num_heads),
    }
    bias_pad_keys: dict[str, tuple[int, ...]] = {
        "q_proj.bias": (0, sharding_size // num_heads),
        "k_proj.bias": (0, sharding_size // num_kv_heads),
        "v_proj.bias": (0, sharding_size // num_kv_heads),
    }

    return MetadataMap(name_map=name_map,
                       reshape_map=reshape_keys,
                       bias_reshape_map=bias_reshape_keys,
                       transpose_map=transpose_keys,
                       pad_map=pad_keys,
                       bias_pad_map=bias_pad_keys)


def _load_and_shard_weight(vllm_config,
                           params: nnx.State,
                           shardings: Any,
                           metadata_map: MetadataMap,
                           mesh: Mesh,
                           hf_key: str,
                           hf_weight: jax.Array,
                           keep_original_dtype_keys_regex: list[str]
                           | None = None,
                           pp_missing_layers: list[str] | None = None):
    name_map = metadata_map.name_map
    reshape_keys = metadata_map.reshape_map
    bias_reshape_keys = metadata_map.bias_reshape_map
    transpose_keys = metadata_map.transpose_map
    pad_keys = metadata_map.pad_map
    bias_pad_keys = metadata_map.bias_pad_map

    shard = functools.partial(shard_put, mesh=mesh)

    model_config = vllm_config.model_config

    # Pad head_dim for kernel performance.
    head_dim_original = model_config.get_head_size()
    head_dim = utils.get_padded_head_dim(head_dim_original)
    head_dim_pad = head_dim - head_dim_original

    # Check if the key should retain its original dtype
    keep_original_dtype = False
    if keep_original_dtype_keys_regex:
        for pattern in keep_original_dtype_keys_regex:
            if re.match(pattern, hf_key):
                keep_original_dtype = True
                break

    # Converting to config's dtype
    if not keep_original_dtype and hf_weight.dtype != model_config.dtype:
        logger.warning(
            f"Converting dtype for {hf_key} from {hf_weight.dtype} to {model_config.dtype}"
        )
        hf_weight = hf_weight.astype(model_config.dtype)

    if hf_key.endswith(".weight"):
        hf_key = hf_key.removesuffix(".weight")

    # Find the corresponding model key using the HF key
    if "layers" in hf_key:
        layer_num = re.search(r"layers\.(\d+)", hf_key).group(1)
        layer_key = re.sub(r"layers\.\d+", "layers.*", hf_key)
        model_key = name_map[layer_key]
        model_key = re.sub(r"layers\.\*", f"layers.{layer_num}", model_key)
    elif "blocks" in hf_key:
        layer_num = re.search(r"blocks\.(\d+)", hf_key).group(1)
        layer_key = re.sub(r"blocks\.\d+", "blocks.*", hf_key)
        model_key = name_map[layer_key]
        model_key = re.sub(r"blocks\.\*", f"blocks.{layer_num}", model_key)
    else:
        if hf_key not in name_map and hf_key == "lm_head":
            logger.warning(f"Skip loading {hf_key} due to tie_word_embeddings")
            return
        if hf_key not in name_map and "t2d" in hf_key:
            logger.warning(
                f"Skip loading {hf_key} as it's not used in eagle-3 for now")
            return
        model_key = name_map.get(hf_key, hf_key)

    if pp_missing_layers and _is_pp_missing_layer(hf_key, pp_missing_layers):
        logger.warning(
            f"Skip loading {hf_key} as it doesn't belong to this PP stage.")
        return
    model_weight, model_sharding = get_param_and_sharding(
        params, shardings, model_key)

    logger.debug(
        "before transform | "
        f"{hf_key}: {hf_weight.shape} --> {model_key}: {model_weight.value.shape} {model_sharding}"
    )

    if hf_key.endswith(".bias"):
        for key in bias_reshape_keys:
            if key in hf_key:
                hf_weight = jnp.reshape(hf_weight, bias_reshape_keys[key])
                if head_dim_pad > 0:
                    hf_weight = jnp.pad(hf_weight, ((0, 0), (0, head_dim_pad)))
                break
    else:
        for key in reshape_keys:
            if key in hf_key:
                hf_weight = jnp.reshape(hf_weight, reshape_keys[key])
                if head_dim_pad > 0:
                    if "o_proj" in key:
                        hf_weight = jnp.pad(hf_weight, ((0, 0), (0, 0),
                                                        (0, head_dim_pad)))
                    else:
                        hf_weight = jnp.pad(hf_weight,
                                            ((0, 0), (0, head_dim_pad),
                                             (0, 0)))
                break
        for key in transpose_keys:
            if key in hf_key:
                hf_weight = jnp.transpose(hf_weight, transpose_keys[key])
                break

    # Pad num-kv-heads
    if hf_key.endswith(".bias"):
        for key, value in bias_pad_keys.items():
            dim = value[0]
            dim_size = value[1]
            if key in hf_key and dim_size != 0:
                hf_weight = jnp.repeat(hf_weight, dim_size, axis=dim)
                break
    else:
        for key, value in pad_keys.items():
            dim = value[0]
            dim_size = value[1]
            if key in hf_key and dim_size != 0:
                hf_weight = jnp.repeat(hf_weight, dim_size, axis=dim)
                break

    logger.debug(
        "after transform | "
        f"{hf_key}: {hf_weight.shape} --> {model_key}: {model_weight.value.shape} {model_sharding}"
    )

    if head_dim_pad == 0:
        assert model_weight.value.shape == hf_weight.shape, f"{hf_key}: {model_weight.value.shape} != {hf_weight.shape}"

    # Update the model weight
    spec = model_weight.sharding.spec if isinstance(
        model_weight.sharding, NamedSharding) else model_weight.sharding
    model_weight.value = shard(hf_weight, spec)


def _is_pp_missing_layer(hf_key: str, pp_missing_layers: list[str]) -> bool:
    has_digit = any(char.isdigit() for char in hf_key)
    # add the suffix after digits to avoid it matches "layers.10" with "layers.1"
    suffix = "." if has_digit else ""
    return any(f'{pp_missing_layer}{suffix}' in hf_key
               for pp_missing_layer in pp_missing_layers)


def _load_hf_weights_on_thread(
    vllm_config: VllmConfig,
    params: nnx.State,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    weights_file: str,
    filter_regex: Optional[str] = None,
    keep_original_dtype_keys_regex: Optional[list[str]] = None,
    pp_missing_layers: list[str] | None = None,
):
    """Loads weights from a single weights file."""
    try:
        shardings = nnx.get_named_sharding(params, mesh)
    except TypeError:
        shardings = params

    for hf_key, hf_weight in model_weights_single_file_generator(
            weights_file, framework="flax", filter_regex=filter_regex):
        _load_and_shard_weight(
            vllm_config,
            params,
            shardings,
            metadata_map,
            mesh,
            hf_key,
            hf_weight,
            keep_original_dtype_keys_regex,
            pp_missing_layers,
        )


def load_hf_weights(
    vllm_config: VllmConfig,
    model: nnx.Module,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    filter_regex: Optional[str] = None,
    is_draft_model: bool = False,
    keep_original_dtype_keys_regex: Optional[list[str]] = None,
    pp_missing_layers: list[str] | None = None,
):
    """Load weights into a JAX model from either an iterator or files."""
    params = nnx.state(model)
    try:
        shardings = nnx.get_named_sharding(params, mesh)
    except TypeError:
        shardings = params
    weights_iterator = None
    if hasattr(vllm_config.model_config, "model_weights_iterator"):
        weights_iterator = vllm_config.model_config.model_weights_iterator
    env = torchax.default_env()
    # The weights_iterator is used in RunAI model streamer integration.
    if weights_iterator is not None:
        for hf_key, hf_weight in weights_iterator:
            if filter_regex and not re.match(filter_regex, hf_key):
                continue

            # Since the weights_iterator yields Pytorch tensors (torch.Tensor),
            # we need to convert them to JAX arrays (jax.Array).
            hf_weight_jax = env.t2j_copy(hf_weight)

            _load_and_shard_weight(
                vllm_config,
                params,
                shardings,
                metadata_map,
                mesh,
                hf_key,
                hf_weight_jax,
                keep_original_dtype_keys_regex,
                pp_missing_layers=pp_missing_layers,
            )
    else:
        # File-based path (multi-threaded)
        if is_draft_model:
            model_path = vllm_config.speculative_config.draft_model_config.model
        else:
            model_path = vllm_config.model_config.model
        weights_files = get_model_weights_files(
            model_path, vllm_config.load_config.download_dir)
        max_workers = min(64, len(weights_files))
        # NOTE(xiang): Disable multi-threading mode if running on multi-host.
        # Because multi-threading would cause different JAX processes to load
        # different weights at the same time.
        if envs.TPU_MULTIHOST_BACKEND == "ray":
            max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _load_hf_weights_on_thread,
                    vllm_config,
                    params,
                    metadata_map,
                    mesh,
                    weights_file,
                    filter_regex=filter_regex,
                    keep_original_dtype_keys_regex=
                    keep_original_dtype_keys_regex,
                    pp_missing_layers=pp_missing_layers,
                ) for weights_file in weights_files
            ]
            for future in futures:
                future.result()

    check_all_loaded(params)
    nnx.update(model, params)


def check_all_loaded(params: nnx.State):

    def _check(x: Any):
        if isinstance(x, nnx.Param) and isinstance(x.value,
                                                   jax.ShapeDtypeStruct):
            raise ValueError(f"The param does not load weights: {x}")

    jax.tree.map(_check, params)


def build_flat_dict(flat_state, mappings):
    """Build a new flat dictionary from the flat state using the provided mappings."""
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
            logger.info(f"!!! No mapping for flat state: {keys}")
    return new_flat_dict


def transfer_state_with_mappings(src_state,
                                 tgt_state,
                                 mappings,
                                 transpose_keys=None,
                                 shard=None):
    """Transfer state from src_state to tgt_state using the provided mappings."""
    src_flat = src_state.flat_state()
    tgt_flat = tgt_state.flat_state()

    new_src_dict = build_flat_dict(tgt_flat, mappings)
    logger.info(f"{mappings=}")
    logger.info(f"{transpose_keys=}")
    for src_keys, v in src_flat:
        flattened_src_keys = '.'.join(str(k) for k in src_keys)
        new_v = jnp.copy(v.value)
        logger.info(
            f"Processing source key: {flattened_src_keys} and value: {new_v.shape} {new_v.dtype}"
        )
        if flattened_src_keys not in new_src_dict:
            logger.info(f"!!! No mapping for source key: {flattened_src_keys}")
            continue
        sharding = new_src_dict[flattened_src_keys][1]

        # E.g. layers.*.attn.k_proj.w, layers.*.attn.k_proj.w_lora_a
        # E.g. layers.*.mlp.down_proj.kernel, layers.*.mlp.down_proj.kernel_lora_a
        if transpose_keys is not None \
          and ((src_keys[-1] in transpose_keys) and ('lora' not in src_keys[-1])):
            v_maybe_t = jnp.transpose(new_v, transpose_keys[src_keys[-1]])
        else:
            v_maybe_t = new_v

        to_update_value = new_src_dict[flattened_src_keys][0].value
        assert to_update_value.shape == v_maybe_t.shape, \
            f"Shape mismatch for {flattened_src_keys}: {to_update_value.shape} vs {v_maybe_t.shape}"

        if to_update_value.dtype != v_maybe_t.dtype:
            logger.info(
                f"Type mismatch between external model and vLLM model. Converting {v_maybe_t.dtype=} to {to_update_value.dtype=}"
            )
            v_maybe_t = v_maybe_t.astype(to_update_value.dtype)

        new_src_dict[flattened_src_keys][0].value = shard(
            v_maybe_t, sharding) if shard else v_maybe_t

    tgt_state = tgt_state.from_flat_path(tgt_flat)
    return tgt_state
