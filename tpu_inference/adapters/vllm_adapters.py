# SPDX-License-Identifier: Apache-2.0

from vllm.lora.request import LoRARequest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput

from tpu_inference.di.abstracts import (AbstractKVCacheConfig,
                                        AbstractKVCacheSpec,
                                        AbstractLoRARequest,
                                        AbstractModelRunnerOutput,
                                        AbstractSchedulerOutput)


class VllmModelRunnerOutputAdapter(AbstractModelRunnerOutput):

    def __init__(self, vllm_output: ModelRunnerOutput):
        self.vllm_output = vllm_output


class VllmSchedulerOutputAdapter(AbstractSchedulerOutput):

    def __init__(self, vllm_scheduler_output: SchedulerOutput):
        self.vllm_scheduler_output = vllm_scheduler_output


class VllmLoRARequestAdapter(AbstractLoRARequest):

    def __init__(self, vllm_lora_request: LoRARequest):
        self.vllm_lora_request = vllm_lora_request


class VllmKVCacheConfigAdapter(AbstractKVCacheConfig):

    def __init__(self, vllm_kv_cache_config: KVCacheConfig):
        self.vllm_kv_cache_config = vllm_kv_cache_config


class VllmKVCacheSpecAdapter(AbstractKVCacheSpec):

    def __init__(self, vllm_kv_cache_spec: KVCacheSpec):
        self.vllm_kv_cache_spec = vllm_kv_cache_spec
