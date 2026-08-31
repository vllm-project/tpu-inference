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

from abc import ABC, abstractmethod
import ctypes
import ctypes.util
import gc
from typing import Optional

import jax
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers import linear as vllm_linear

from tpu_inference import envs

logger = init_logger(__name__)


def _free_torch_storage(tensor: Optional[torch.Tensor]) -> None:
    """Safely frees the underlying CPU memory storage of a PyTorch tensor.

    Tries `untyped_storage().resize_(0)` first, with fallback to `set_(torch.storage.UntypedStorage())`
    for 0-dim scalars or float8 dtypes that cannot be resized in-place.
    """
    if tensor is None:
        return
    try:
        tensor.untyped_storage().resize_(0)
    except Exception:
        try:
            tensor.set_(torch.storage.UntypedStorage())
        except Exception:
            pass


def _release_host_memory() -> None:
    """Frees CPU host memory and trims malloc arena if incremental loading is enabled."""
    if not (getattr(envs, "VLLM_INCREMENTAL_FP8_LOADING", False) or getattr(envs, "VLLM_INCREMENTAL_MXFP4_LOADING", False)):
        return
    gc.collect()
    jax.effects_barrier()
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            ctypes.CDLL(libc_name).malloc_trim(0)
    except Exception as e:
        logger.debug(f"malloc_trim failed: {e}")


def _log_memory_stats(layer_name: str = "") -> None:
    try:
        import psutil, resource
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        print(
            f"[RAM Trace] Layer {layer_name} sharded & freed | "
            f"Process RSS: {rss_gb:.2f} GB | Peak RSS: {max_rss_gb:.2f} GB",
            flush=True,
        )
    except Exception as e:
        print(f"[RAM Trace Error] {e}", flush=True)


class VllmQuantizationMethod(ABC):

    def maybe_process_linear_weights(
        self,
        layer: torch.nn.Module,
        param_name: str,
        args,
        kwargs,
        num_proj: int,
        log_prefix: str = "",
    ):
        """Shared tracking logic for incremental sharding of linear layers."""
        if isinstance(layer, vllm_linear.QKVParallelLinear):
            if len(args) == 1:
                shard_id = args[0]
                layer._loaded_weights.add((param_name, shard_id))
            else:
                layer._loaded_weights.add((param_name, "q"))
                layer._loaded_weights.add((param_name, "k"))
                layer._loaded_weights.add((param_name, "v"))
        elif isinstance(layer, vllm_linear.MergedColumnParallelLinear):
            if len(args) == 1:
                shard_id = args[0]
                layer._loaded_weights.add((param_name, shard_id))
            else:
                for i in range(len(layer.output_sizes)):
                    layer._loaded_weights.add((param_name, i))
        else:
            layer._loaded_weights.add(param_name)

        expected_count = num_proj * len(
            dict(layer.named_parameters(recurse=False)))
        if len(layer._loaded_weights) == expected_count:
            prefix_str = f"[{log_prefix}] " if log_prefix else ""
            logger.debug(
                f"{prefix_str}Start sharding weights for layer {type(layer)}")
            self.process_weights_after_loading(layer)
            logger.debug(
                f"{prefix_str}Complete sharding weights for layer {type(layer)}"
            )

    @abstractmethod
    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError
