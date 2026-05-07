# Copyright 2026 Google LLC
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

import os
from typing import Any, Literal
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.config import (AttentionConfig, CacheConfig, DeviceConfig,
                         KVTransferConfig, ModelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorWorkerMetadata
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tpu_inference.offload.tpu_offload_connector import (
    KVOffloadConnectorStats, TPUOffloadConnector, TPUOffloadConnectorMetadata)

EOS_TOKEN_ID = 50256


def create_model_runner_output(
    reqs: list[Request],
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
    use_eos: bool = False,
    token_id: int = 0,
    kv_connector_worker_meta: KVConnectorWorkerMetadata | None = None,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""

    # Make request data.
    req_ids = [req.request_id for req in reqs]
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    # Make sampled tokens.
    sampled_token = EOS_TOKEN_ID if use_eos else token_id
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    kv_connector_output = (
        None if (finished_sending is None and finished_recving is None
                 and invalid_block_ids is None
                 and kv_connector_worker_meta is None) else KVConnectorOutput(
                     finished_sending=finished_sending,
                     finished_recving=finished_recving,
                     invalid_block_ids=invalid_block_ids or set(),
                     kv_connector_worker_meta=kv_connector_worker_meta,
                 ))

    # Make output data structure.
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )


def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
    kv_connector_extra_config: dict[str, Any] | None = None,
    dtype: str = "float16",
    cache_dtype: str = "auto",
    hf_overrides: dict[str, Any] | None = None,
    attention_backend: str | None = None,
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail",
    kv_connector: str = "NixlConnector",
    kv_role: str = "kv_both",
    disable_hybrid_kv_cache_manager: bool | None = None,
) -> VllmConfig:
    """Initialize VllmConfig For Testing."""
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype=dtype,
        seed=42,
        hf_overrides=hf_overrides or {},
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=model_config.is_encoder_decoder,
        disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype=cache_dtype,
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector=kv_connector,
        kv_role=kv_role,
        enable_permute_local_kv=enable_permute_local_kv,
        kv_connector_extra_config=kv_connector_extra_config or {},
        kv_load_failure_policy=kv_load_failure_policy,
    )
    attention_config = AttentionConfig(backend=attention_backend)
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
        attention_config=attention_config,
    )


class TPURequestRunner:

    def __init__(self, block_size: int, num_blocks: int,
                 async_scheduling: bool):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.async_scheduling = async_scheduling
        self.req_id = -1

        # We need to mock envs to allow the scheduler to be created
        os.environ["TPU_OFFLOAD_NUM_CPU_CHUNKS"] = "100"
        os.environ["TPU_OFFLOAD_NUM_STAGING_BLOCKS"] = "100"
        os.environ["TPU_OFFLOAD_DECODE_SAVE"] = "1"

        vllm_config = create_vllm_config(
            block_size=block_size,
            max_num_batched_tokens=1000,
            disable_hybrid_kv_cache_manager=False,
            kv_connector="TPUOffloadConnector",
        )
        vllm_config.scheduler_config.async_scheduling = async_scheduling
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector=
            "tpu_inference.offload.tpu_offload_connector.TPUOffloadConnector",
            kv_role="kv_both",
        )
        kv_cache_groups = [
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )
        vllm_config.cache_config.num_gpu_blocks = num_blocks

        def mock_create_connector(config,
                                  role,
                                  kv_cache_config=None,
                                  **kwargs):
            return TPUOffloadConnector(config, role, kv_cache_config)

        with patch.object(KVConnectorFactory,
                          'create_connector',
                          side_effect=mock_create_connector):
            scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
            self.scheduler = scheduler_cls(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                log_stats=True,
                structured_output_manager=StructuredOutputManager(vllm_config),
                block_size=block_size,
            )

        self.scheduler_connector = self.scheduler.connector
        self.connector_scheduler = self.scheduler.connector.connector_scheduler

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(block_size, sha256)
        self.completed_stores = []
        self.completed_loads = []

    def new_request(self, token_ids: list[int]):
        self.req_id += 1
        sampling_params = SamplingParams(max_tokens=1000)
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
        req = Request(
            request_id=str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=self._block_hasher,
        )
        self.scheduler.add_request(req)

    def run(self,
            decoded_tokens: list[int],
            complete_transfers: bool = True,
            expected_stored_blocks: int = 0,
            expected_loaded_blocks: int = 0):
        tokens_iter = iter(decoded_tokens)
        token_id = next(tokens_iter, None)
        prev_scheduler_output = None
        prev_model_runner_output = None

        stores_this_run = []
        loads_this_run = []

        while True:
            if not self.scheduler.requests and not self.connector_scheduler._reqs_being_saved and not self.connector_scheduler._reqs_being_loaded:
                break

            scheduler_output = self.scheduler.schedule()
            kv_connector_metadata = scheduler_output.kv_connector_metadata

            finished_sending = set()
            finished_recving = set()
            stats = KVOffloadConnectorStats()

            if kv_connector_metadata is not None and isinstance(
                    kv_connector_metadata, TPUOffloadConnectorMetadata):
                for meta in kv_connector_metadata.requests_meta:
                    if meta.save_spec and not meta.save_spec.skip_save:
                        if complete_transfers:
                            stats.record_save(meta.req_id,
                                              meta.save_spec.dst_chunks)
                            stores_this_run.extend(meta.save_spec.src_blocks)
                    if meta.load_spec and meta.load_spec.can_load:
                        if complete_transfers:
                            stats.record_load(meta.req_id,
                                              meta.load_spec.src_chunks)
                            loads_this_run.extend(meta.load_spec.dst_blocks)
                    if meta.save_spec and meta.save_spec.is_final_save:
                        finished_sending.add(meta.req_id)
            connector_output = KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                kv_connector_stats=stats,
            )
            self.scheduler.connector.update_connector_output(connector_output)

            model_runner_output = create_model_runner_output(
                reqs=self.scheduler.running,
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                token_id=token_id or 0,
            )

            prev_token_id = token_id
            if self.scheduler.running:
                token_id = next(tokens_iter, None)

            if self.async_scheduling:
                if prev_model_runner_output is not None:
                    self.scheduler.update_from_output(
                        prev_scheduler_output, prev_model_runner_output)
                prev_scheduler_output = scheduler_output
                prev_model_runner_output = model_runner_output
            else:
                self.scheduler.update_from_output(scheduler_output,
                                                  model_runner_output)

            if (prev_token_id == EOS_TOKEN_ID and prev_token_id != token_id
                    and (self.scheduler.requests
                         or self.connector_scheduler._reqs_being_saved
                         or self.connector_scheduler._reqs_being_loaded)):
                continue

            if token_id is None:
                if self.async_scheduling:
                    self.scheduler.update_from_output(
                        prev_scheduler_output, prev_model_runner_output)
                break

        self.completed_stores.extend(stores_this_run)
        self.completed_loads.extend(loads_this_run)

        assert len(
            stores_this_run
        ) == expected_stored_blocks, f"Expected {expected_stored_blocks} stores, got {len(stores_this_run)}"
        assert len(
            loads_this_run
        ) == expected_loaded_blocks, f"Expected {expected_loaded_blocks} loads, got {len(loads_this_run)}"
