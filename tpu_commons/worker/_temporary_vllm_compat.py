# SPDX-License-Identifier: Apache-2.0

#
# WARNING: This is a temporary compatibility module.
#
#
# THE PROBLEM:
# The ideal dependency injection pattern dictates that the "producer" of data
# (in this case, the vLLM engine) should be responsible for adapting its data
# into the abstract format that the "consumer" (the TPU worker) expects.
#
# However, this would require a simultaneous code change in both the `vllm` and
# `tpu_commons` repositories. Such cross-repository changes are difficult to
# coordinate, slow to land, and can easily cause breakages if the releases
# are not perfectly synchronized.
#
#
# THE TEMPORARY SOLUTION:
# To enable independent development and deployment, we are temporarily violating
# this pattern. We are making the consumer (`tpu_commons`) responsible for
# detecting and adapting the producer's raw data.
#
# This function checks if it has received a raw `vllm.SchedulerOutput` and,
# if so, wraps it in the appropriate adapter. This allows `vllm` to continue
# sending its raw data type without modification, decoupling the release cycles.
#
#
# THE FUTURE (HOW TO REMOVE THIS):
# This entire file should be deleted once the `vllm` repository has been updated.
# The required change in `vllm` is small and looks like this:
#
# --- SKELETON CODE FOR FUTURE vLLM CHANGE ---
# In the vLLM engine, where `execute_model` is called:
#
# from tpu_commons.adapters.vllm_adapters import VllmSchedulerOutputAdapter
# from vllm.v1.core.sched.output import SchedulerOutput
#
# # ... inside some method ...
#
# # OLD CODE:
# # concrete_work = SchedulerOutput(...)
# # self.tpu_backend.execute_model(concrete_work)
#
# # NEW CODE:
# concrete_work = SchedulerOutput(...)
# adapted_work = VllmSchedulerOutputAdapter(concrete_work)  # This line is added
# self.tpu_backend.execute_model(adapted_work)          # Pass the adapter
#
# --- END SKELETON CODE ---
#

import logging
from typing import Union

from vllm.lora.request import LoRARequest as VllmLoRARequest
# Import the concrete vLLM type for the check
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig as VllmKVCacheConfig

from tpu_commons.adapters.vllm_adapters import (VllmKVCacheConfigAdapter,
                                                VllmLoRARequestAdapter,
                                                VllmSchedulerOutputAdapter)
from tpu_commons.di.abstracts import (AbstractKVCacheConfig,
                                      AbstractLoRARequest,
                                      AbstractSchedulerOutput)

logger = logging.getLogger(__name__)


def adapt_scheduler_output_if_needed(
    scheduler_output: Union[AbstractSchedulerOutput, VllmSchedulerOutput]
) -> AbstractSchedulerOutput:
    """
    Checks if the input is a raw VllmSchedulerOutput and wraps it.
    If it's already an AbstractSchedulerOutput, it's passed through.
    """
    if isinstance(scheduler_output, VllmSchedulerOutput):
        # logger.warning(
        #     "Received raw VllmSchedulerOutput. Performing temporary, on-the-fly "
        #     "adaptation. This is a compatibility feature and should be removed "
        #     "once the vLLM engine is updated to provide an adapted object.")
        return VllmSchedulerOutputAdapter(scheduler_output)

    if isinstance(scheduler_output, AbstractSchedulerOutput):
        return scheduler_output

    raise TypeError(
        f"Unsupported type for scheduler_output: {type(scheduler_output)}")


def adapt_kv_cache_config_if_needed(
    kv_cache_config: Union[AbstractKVCacheConfig, VllmKVCacheConfig]
) -> AbstractKVCacheConfig:
    """
    Checks if the input is a raw VllmKVCacheConfig and wraps it.
    If it's already an AbstractKVCacheConfig, it's passed through.
    """
    if isinstance(kv_cache_config, VllmKVCacheConfig):
        # logger.warning(
        #     "Received raw VllmKVCacheConfig. Performing temporary, on-the-fly "
        #     "adaptation. This is a compatibility feature and should be removed "
        #     "once the vLLM engine is updated to provide an adapted object.")
        return VllmKVCacheConfigAdapter(kv_cache_config)

    if isinstance(kv_cache_config, AbstractKVCacheConfig):
        return kv_cache_config

    raise TypeError(
        f"Unsupported type for kv_cache_config: {type(kv_cache_config)}")


def adapt_lora_request_if_needed(
    lora_request: Union[AbstractLoRARequest, VllmLoRARequest]
) -> AbstractLoRARequest:
    """
    Checks if the input is a raw VllmLoRARequest and wraps it.
    If it's already an AbstractLoRARequest, it's passed through.
    """
    if isinstance(lora_request, VllmLoRARequest):
        # logger.warning(
        #     "Received raw VllmLoRARequest. Performing temporary, on-the-fly "
        #     "adaptation. This is a compatibility feature and should be removed "
        #     "once the vLLM engine is updated to provide an adapted object.")
        return VllmLoRARequestAdapter(lora_request)

    if isinstance(lora_request, AbstractLoRARequest):
        return lora_request

    raise TypeError(f"Unsupported type for lora_request: {type(lora_request)}")
