# SPDX-License-Identifier: Apache-2.0
from typing import Any, List, Tuple

import jax
from ray._private.accelerators import TPUAcceleratorManager

import jax.numpy as jnp # Though not directly used in the formula, good to have for JAX utils
import jax.tree_util    # Needed for counting parameters (see helper function below)

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
    num_model_parameters: int,  # Total count of learnable parameters in the model
    prefill_length: int,        # The length of the input prompt/sequence being prefilled
    model_hf_config: Any,       # The Hugging Face model configuration object (e.g., LlamaConfig)
                                # containing attributes like num_attention_heads, num_hidden_layers, hidden_size.
    log: bool = True
) -> Tuple[float, float, float]:
  """
  Calculates the TFLOPs per device for a prefill step based on model parameters and sequence length.
  Based on the formula from MaxText (arxiv.org/pdf/2204.02311.pdf Appendix B).

  Args:
    num_model_parameters: Total count of learnable parameters in the model.
    prefill_length: The length of the input prompt/sequence being prefilled.
    model_hf_config: The Hugging Face model configuration object. Expected to have:
                     .num_attention_heads (for num_query_heads)
                     .num_hidden_layers (for num_decoder_layers)
                     .hidden_size (to derive head_dim)
    log: If True, prints the TFLOPs breakdown.

  Returns:
    A tuple containing: (total_tflops, learnable_weight_tflops, causal_attention_tflops)
  """
  # Extract relevant config parameters from the Hugging Face model config
  num_query_heads = model_hf_config.num_attention_heads
  num_decoder_layers = model_hf_config.num_hidden_layers
  
  # head_dim is typically hidden_size / num_attention_heads
  # Use integer division as head_dim is usually an integer.
  head_dim = model_hf_config.hidden_size // model_hf_config.num_attention_heads 

  # Convert to TFLOPs (1 TFLOP = 10^12 FLOPs)
  
  # Learnable weight TFLOPs (2 * Parameters * Sequence_Length / Devices)
  # This accounts for FLOPs in dense layers, typically 2 * Input_dim * Output_dim for matmul.
  # Summing over all parameters gives approximate total weights FLOPs.
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length / jax.device_count() / 1e12

  # Non-causal attention FLOPs (simplified approximation for self-attention)
  # The formula 4 * H * L * D_h * S^2 is a common approximation for attention layers.
  # H: num_heads, L: num_layers, D_h: head_dim, S: sequence_length
  noncasual_attention_flops = (
      4
      * num_query_heads
      * num_decoder_layers
      * head_dim
      * prefill_length**2
      / jax.device_count()
      / 1e12
  )

  # Causal attention (used in autoregressive models like LLMs) is roughly half of non-causal
  # due to the masking preventing attention to future tokens.
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
    return jax.tree_util.tree_reduce(lambda x, y: x + y, jax.tree_util.tree_map(lambda x: x.size, params))