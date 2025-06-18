# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
from ray._private.accelerators import TPUAcceleratorManager

GBYTES = 1024 * 1024 * 1024

_megacore = False


def enable_megacore() -> None:
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    return _megacore


def get_local_available_devices() -> int:
    return TPUAcceleratorManager.get_current_node_num_accelerators()


def set_visible_device_ids(tpu_ids: List[int]) -> None:
    validate = TPUAcceleratorManager.validate_resource_request_quantity(
        len(tpu_ids))
    if not validate[0]:
        raise ValueError(validate[1])
    tpu_ids = [str(tpu_id) for tpu_id in tpu_ids]
    TPUAcceleratorManager.set_current_process_visible_accelerator_ids(tpu_ids)


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def hbm_usage_bytes(devices: Any) -> List[Tuple[int, int]]:
    usage = []
    for device in devices:
        hbm_used = device.memory_stats()["bytes_in_use"]
        hbm_limit = device.memory_stats()["bytes_limit"]
        usage.append((hbm_used, hbm_limit))
    return usage


def hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
    usage = hbm_usage_bytes(devices)
    usage = [(round(used / GBYTES, 2), round(limit / GBYTES, 2))
             for used, limit in usage]
    return usage


def array_info(name: str, x: jax.Array) -> str:
    rep = f"{name} | shape={x.shape} | dtype={x.dtype} | sharding={x.sharding}"
    return rep


def calculate_prefill_tflops_per_device(
    num_model_parameters: int,
    prefill_length: int,
    vllm_model_config:
    Any,  # <--- MODIFIED: Now takes vLLM's ModelConfig (e.g., vllm_config.model_config)
    log: bool = True
) -> Tuple[float, float, float]:
    """
  Calculates the TFLOPs per device for a prefill step based on model parameters and sequence length.
  Based on the formula from MaxText (arxiv.org/pdf/2204.02311.pdf Appendix B).

  Args:
    num_model_parameters: Total count of learnable parameters in the model.
    prefill_length: The length of the input prompt/sequence being prefilled.
    vllm_model_config: The vLLM ModelConfig object. Expected to have:
                       .hf_config.num_attention_heads
                       .hf_config.num_hidden_layers
                       .hf_config.hidden_size
                       .dtype (for model parameters dtype)
    log: If True, prints the TFLOPs breakdown.

  Returns:
    A tuple containing: (total_tflops, learnable_weight_tflops, causal_attention_tflops)
  """
    # Extract relevant config parameters from the Hugging Face model config NESTED inside vllm_model_config
    num_query_heads = vllm_model_config.hf_config.num_attention_heads  # <--- MODIFIED
    num_decoder_layers = vllm_model_config.hf_config.num_hidden_layers  # <--- MODIFIED
    head_dim = vllm_model_config.hf_config.hidden_size // vllm_model_config.hf_config.num_attention_heads  # <--- MODIFIED

    learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count(
    ) / 1e12
    noncasual_attention_flops = (4 * num_query_heads * num_decoder_layers *
                                 head_dim * prefill_length**2 /
                                 jax.device_count() / 1e12)
    causal_attention_tflops = noncasual_attention_flops / 2

    total_tflops = learnable_weight_tflops + causal_attention_tflops

    if log:
        print(
            "Per prefill step per device: \n",
            f"\tTotal TFLOPs: {total_tflops:.2f} \n",
            f"\t\tLearnable weight TFLOPs: {learnable_weight_tflops:.2f} ",
            f"({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n",
            f"\t\tCausal attention TFLOPs: {causal_attention_tflops:.2f} ",
            f"({100 * causal_attention_tflops/total_tflops:.2f})% of Total",
        )
    return total_tflops, learnable_weight_tflops, causal_attention_tflops


def count_model_parameters(params: Any) -> int:
    """
    Counts the total number of parameters (individual scalar values) in a JAX PyTree.
    This includes all arrays within the PyTree.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(lambda x: x.size, params))


def get_kv_cache_size_bytes(
    kv_cache_config: Any, vllm_model_config: Any
) -> int:  # <--- MODIFIED: Now takes vLLM's ModelConfig
    """
    Calculates the total size of the KV cache in bytes.

    Args:
        kv_cache_config: The KV cache configuration object (e.g., vllm_config.cache_config).
                         Expected attributes: .num_blocks, .block_size.
        vllm_model_config: The vLLM ModelConfig object. Expected attributes:
                           .get_total_num_kv_heads(), .get_head_size(),
                           and .dtype (for the model's data type, which KV cache typically matches).

    Returns:
        int: Total memory size of the KV cache in bytes.
    """
    num_blocks = kv_cache_config.num_gpu_blocks  # Total number of allocated KV cache blocks
    block_size = kv_cache_config.block_size  # Number of tokens stored per block

    # Access from vllm_model_config
    num_kv_heads = vllm_model_config.get_total_num_kv_heads()  # <--- MODIFIED
    head_size = vllm_model_config.get_head_size()  # <--- MODIFIED

    # Access dtype from vllm_model_config
    dtype_itemsize = jnp.dtype(
        vllm_model_config.dtype).itemsize  # <--- MODIFIED

    total_kv_cache_bytes = num_blocks * block_size * num_kv_heads * head_size * 2 * dtype_itemsize
    return total_kv_cache_bytes


def get_model_size_bytes(
    num_model_parameters: int, vllm_model_config: Any
) -> int:  # <--- MODIFIED: Now takes vLLM's ModelConfig
    """
    Calculates the total size of model parameters in bytes.

    Args:
        num_model_parameters: Total count of individual learnable parameters in the model.
        vllm_model_config: The vLLM ModelConfig object. Expected attribute: .dtype (for the model parameters' data type).

    Returns:
        int: Total memory size of model parameters in bytes.
    """
    # Access dtype from vllm_model_config
    dtype_itemsize = jnp.dtype(
        vllm_model_config.dtype).itemsize  # <--- MODIFIED

    return num_model_parameters * dtype_itemsize
