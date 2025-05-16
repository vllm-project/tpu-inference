# Here we try to bring as much code as possible from Hex-LLM, instead of `tpu_torch_xla_runner.py` -> jax conversion.
from typing import Any, Optional

import jax.numpy as jnp
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import ModelRunnerOutput


class TPUModelRunner():

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: Any,
    ):
        self.vllm_config = vllm_config

    def load_model(self):
        pass

    def get_kv_cache_spec(self):
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=model_config.get_num_attention_heads(
                    parallel_config),
                head_size=model_config.get_head_size(),
                dtype=jnp.bfloat16,
                use_mla=False,
            )

        return kv_cache_spec

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def capture_model(self) -> None:
        pass

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        req_id_to_index = {}
        req_ids = []
        prompt_logprobs_dict = {}

        all_reqs = scheduler_output.scheduled_new_reqs + scheduler_output.scheduled_cached_reqs
        for i, seq in enumerate(all_reqs):
            req_id_to_index[seq.req_id] = i
            req_ids.append(seq.req_id)
            prompt_logprobs_dict[seq.req_id] = None

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            prompt_logprobs_dict=prompt_logprobs_dict,
            logprobs=None,
            spec_token_ids=None,
            sampled_token_ids=[[0] for _ in range(len(req_ids))],
        )
