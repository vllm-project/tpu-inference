# SPDX-License-Identifier: Apache-2.0
from bisect import bisect_left
from typing import Any, List, Tuple, Union

import jax
import jax.tree_util
import numpy as np
from ray._private.accelerators import TPUAcceleratorManager

from tpu_commons.core.jetstream_commons.engine import PATHWAYS_ENABLED

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
        if PATHWAYS_ENABLED:
            # The Pathways backend doesn't support memory_stats().
            # TODO(fhzhang): find the proper way to support this.
            usage.append((32384, 33550237184))
        else:
            hbm_used = device.memory_stats()["bytes_in_use"]
            hbm_limit = device.memory_stats()["bytes_limit"]
            print(hbm_used, hbm_limit)
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
        vllm_model_config: Any,
        log: bool = True) -> Tuple[float, float, float]:
    """
  Calculates the TFLOPs per device for a prefill step based on model parameters and sequence length.
  Based on the formula from MaxText (https://github.com/AI-Hypercomputer/maxtext/blob/028bc3ca0a352a8836e979121f3cb4c6bc60b3ed/MaxText/maxtext_utils.py#L322).

  Args:
    num_model_parameters: Total count of learnable parameters in the model.
    prefill_length: The length of the input prompt/sequence being prefilled.
    vllm_model_config: The vLLM ModelConfig object.
    log: If True, prints the TFLOPs breakdown.

  Returns:
    A tuple containing: (total_tflops, learnable_weight_tflops, causal_attention_tflops)
  """
    num_query_heads = vllm_model_config.hf_config.num_attention_heads
    num_decoder_layers = vllm_model_config.hf_config.num_hidden_layers
    head_dim = vllm_model_config.hf_config.hidden_size // vllm_model_config.hf_config.num_attention_heads

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


def take_nearest_length(lengths: list[int], length: int) -> int:
    """Gets the nearest length to the right in a set of lengths.

  Args:
    lengths: A list of integers.
    length: An integer.

  Returns:
    The nearest length to the right in the list.
  """
    pos = bisect_left(lengths, length)
    if pos == len(lengths):
        return lengths[-1]
    return lengths[pos]


def pad_tokens(
    tokens: Union[List[int], np.ndarray],
    pad_id: int,
    prefill_lengths: List[int],
    return_as_list: bool = True,
) -> Tuple[Union[jax.Array, np.ndarray], int]:
    """Pads tokens to the nearest prefill length that is equal to or greater
     than the token length.

  Args:
    tokens: Tokens.
    pad_id: Pad ID.
    prefill_lengths: Buckets to pad the sequence to for static compilation.
    return_as_list: Whether to return the padded tokens as a list.

  Returns:
    tokens: Tokenized into integers.
    true_length: Actual length of the non-padded sequence.
  """
    assert pad_id == 0, "Further logic required if pad_id not 0."
    if isinstance(tokens, list):
        tokens = np.array(tokens)
    true_length = tokens.shape[-1]
    padded_length = take_nearest_length(prefill_lengths, true_length)
    padding = padded_length - true_length
    if padding < 0:
        print("Provided sequence longer than available.")
        # Take the last N tokens if we have too many.
        padded_tokens = tokens[-padded_length:]
    else:
        padded_tokens = np.pad(tokens, (0, padding),
                               constant_values=(pad_id, ))
    if return_as_list:
        padded_tokens = padded_tokens.tolist()
    return padded_tokens, true_length
