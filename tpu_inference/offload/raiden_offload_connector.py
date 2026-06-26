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
"""RaidenOffloadConnector manages KV cache data transfer using Raiden libraries."""

import copy
import os
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple, List, Dict

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import \
    KVConnectorStats
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.forward_context import ForwardContext
    from tpu_inference.runner.tpu_runner import TPUModelRunner

from tpu_inference import envs
from tpu_inference.logger import init_logger
from tpu_inference.offload.metrics import TPUKVCacheMetrics
from tpu_inference.offload.utils import CpuChunkId, ReqId

logger = init_logger(__name__)

try:
    from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager as RaidenKVCacheManager
    from tpu_raiden.api.jax.kv_cache_store import KVCacheStore, RaidenId
    _RAIDEN_IMPORT_ERROR = None
except ImportError:
    try:
        from tpu_raiden import KVCacheManager as RaidenKVCacheManager, KVCacheStore, RaidenId
        _RAIDEN_IMPORT_ERROR = None
    except Exception as _exc:  # pylint: disable=broad-except
        RaidenKVCacheManager = None
        KVCacheStore = None
        RaidenId = None
        _RAIDEN_IMPORT_ERROR = _exc


def to_raiden_hash(block_hash: BlockHash) -> bytes:
    """Converts vLLM BlockHash to a bytes representation for C++ KVCacheStore."""
    if isinstance(block_hash, bytes):
        return block_hash
    elif isinstance(block_hash, int):
        return (block_hash % (2**64)).to_bytes(8, byteorder='big')
    elif isinstance(block_hash, str):
        return block_hash.encode('utf-8')
    else:
        return (hash(block_hash) % (2**64)).to_bytes(8, byteorder='big')


@dataclass
class RaidenLocator:
    """Pure-Python, Ray-serializable container holding Raiden data locator info."""
    job_name: str
    job_replica_id: str
    data_name: str
    data_replica_idx: int

    def to_raiden_id(self) -> "RaidenId":
        if RaidenId is None:
            raise ImportError("RaidenId is not available.")
        return RaidenId(self.job_name, self.job_replica_id, self.data_name, self.data_replica_idx)


@dataclass
class RaidenSaveSpec:
    num_skip_leading_tokens: int
    num_total_tokens: int
    src_blocks: list[int]
    dst_chunks: list[int] = field(default_factory=list) # Filled on worker via auto-allocate
    dst_locators: list[RaidenLocator] = field(default_factory=list)
    block_hashes: list[bytes] = field(default_factory=list)
    is_final_save: bool = False
    skip_save: bool = False


@dataclass
class RaidenLoadSpec:
    num_matched_tokens: int
    src_chunks: list[int]
    dst_blocks: list[int]
    src_locators: list[RaidenLocator] = field(default_factory=list)
    can_load: bool = False
    num_skip_leading_tokens: int = 0


@dataclass
class RaidenReqMeta:
    req_id: str
    token_ids: list[int]
    local_block_ids: list[int]
    save_spec: Optional[RaidenSaveSpec] = None
    load_spec: Optional[RaidenLoadSpec] = None


@dataclass
class RaidenConnectorMetadata(KVConnectorMetadata):
    requests_meta: list[RaidenReqMeta] = field(default_factory=list)
    pending_unlocks: list[int] = field(default_factory=list) # Physical chunks to unlock in C++


@dataclass
class KVRaidenConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, dict[str, list[int]]] = {
            "finished_save_chunks": dict(),
            "finished_load_chunks": dict(),
        }

    def record_save(self, req: ReqId, saved_chunk_ids: list[int]):
        if req not in self.data["finished_save_chunks"]:
            self.data["finished_save_chunks"][req] = []
        self.data["finished_save_chunks"][req].extend(copy.deepcopy(saved_chunk_ids))

    def record_load(self, req: ReqId, loaded_chunk_ids: list[int]):
        if req not in self.data["finished_load_chunks"]:
            self.data["finished_load_chunks"][req] = []
        self.data["finished_load_chunks"][req].extend(copy.deepcopy(loaded_chunk_ids))

    def clone_and_reset(self) -> "KVRaidenConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    @property
    def num_finished_blocks(self) -> int:
        return len(self.data["finished_save_chunks"]) + len(self.data["finished_load_chunks"])

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        res = KVRaidenConnectorStats()
        res.data["finished_save_chunks"] = copy.deepcopy(self.data["finished_save_chunks"])
        res.data["finished_load_chunks"] = copy.deepcopy(self.data["finished_load_chunks"])
        if isinstance(other, KVRaidenConnectorStats) and other.data:
            for req, chunks in other.data.get("finished_save_chunks", {}).items():
                if req not in res.data["finished_save_chunks"]:
                    res.data["finished_save_chunks"][req] = []
                res.data["finished_save_chunks"][req].extend(copy.deepcopy(chunks))
            for req, chunks in other.data.get("finished_load_chunks", {}).items():
                if req not in res.data["finished_load_chunks"]:
                    res.data["finished_load_chunks"][req] = []
                res.data["finished_load_chunks"][req].extend(copy.deepcopy(chunks))
        return res

    def reduce(self) -> dict[str, int | float]:
        return {
            "saves": sum(len(c) for c in self.data["finished_save_chunks"].values()),
            "loads": sum(len(c) for c in self.data["finished_load_chunks"].values()),
        }

    def is_empty(self) -> bool:
        return not self.data["finished_save_chunks"] and not self.data["finished_load_chunks"]


@dataclass
class RaidenRequestTracker:
    req_id: str
    prompt_len: int
    block_ids: list[int]
    token_ids: list[int]
    save_watermark: int = 0
    is_decode_phase: bool = False

    def update(self, new_block_ids: list[int], new_token_ids: list[int]):
        if new_block_ids is None:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        self.block_ids.extend(new_block_ids)
        self.token_ids.extend(new_token_ids)
        if len(new_token_ids) == 1:
            self.is_decode_phase = True

    def reset_after_preempt(self):
        self.block_ids = []
        self.token_ids = []
        self.is_decode_phase = False


class RaidenOffloadConnector(KVConnectorBase_V1):
    """Manages KV cache offloading via Raiden libraries."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        logger.info("RaidenOffloadConnector: Entering __init__")
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = RaidenOffloadConnectorScheduler(vllm_config)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = RaidenOffloadConnectorWorker(vllm_config, self)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        return "NHD"

    @classmethod
    def build_kv_connector_stats(
        cls,
        data: dict[str, dict[str, int]] | None = None
    ) -> KVConnectorStats | None:
        return KVRaidenConnectorStats(data=data) if data is not None else KVRaidenConnectorStats()

    def get_num_new_matched_tokens(
            self, request: "Request",
            num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> RaidenConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def register_runner(self, runner: "TPUModelRunner") -> None:
        assert self.connector_worker is not None
        self.connector_worker.register_runner(runner)

    def start_load_kv(self, fwd_ctx: "ForwardContext") -> None:
        assert self.connector_worker is not None
        self.connector_worker.start_load_kv(fwd_ctx)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, **kwargs) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        self.connector_worker.wait_for_save()

    def get_finished(self,
                     finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()


class RaidenOffloadConnectorScheduler:
    """Coordinates logical state of Raiden KV cache offloading utilizing authoritative Raiden KVCacheStore."""

    def __init__(self, vllm_config: VllmConfig):
        if KVCacheStore is None:
            raise ImportError(
                "KVCacheStore is not importable from tpu_raiden. Please check your "
                f"PYTHONPATH and Raiden installation. Original error: {_RAIDEN_IMPORT_ERROR}"
            )
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.num_cpu_chunks = int(os.getenv("TPU_OFFLOAD_NUM_CPU_CHUNKS", 65536))
        
        self.job_name = os.getenv("RAIDEN_JOB_NAME", "tpu_inference")
        self.job_replica_id = os.getenv("CLOUD_TPU_TASK_ID", "0")
        self.data_name = os.getenv("RAIDEN_DATA_NAME", "kv_cache")
        logger.info(f"Raiden Scheduler Locator baseline: job={self.job_name}, replica={self.job_replica_id}, data={self.data_name}")

        # Authoritative logical C++ metadata store
        self.kv_store = KVCacheStore(capacity=self.num_cpu_chunks)
        
        # Track pending physical unlocks to send to workers
        self._pending_unlocks_to_send: list[int] = []
        
        self._request_trackers: dict[ReqId, RaidenRequestTracker] = {}
        self._unfinished_requests: dict[ReqId, "Request"] = {}
        self.load_specs: dict[ReqId, RaidenLoadSpec] = {}
        self._pre_load_specs: dict[ReqId, RaidenLoadSpec] = {}
        self._external_cache_hits: dict[ReqId, int] = {}
        
        # Map ReqId -> list(Hash) for active in-flight save tracking
        self._pending_save_hashes: dict[ReqId, list[bytes]] = defaultdict(list)
        
        self._reqs_being_saved = defaultdict[ReqId, set[CpuChunkId]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[CpuChunkId]](set)
        self.metrics_collector = TPUKVCacheMetrics.get_or_create()
        self._recorded_metrics_reqs: set[ReqId] = set()

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        if self.kv_store is None:
            return 0, False
        
        self.metrics_collector.record_lookup_request()
        
        raiden_hashes = [to_raiden_hash(h) for h in request.block_hashes]
        matched = self.kv_store.lookup(raiden_hashes)
        num_hits = len(matched)
        num_misses = len(raiden_hashes) - num_hits
        logger.debug(f"Raiden KVCacheStore lookup result for {request.request_id}: Block Hits={num_hits}, Block Misses={num_misses}")
        if num_computed_tokens == 0:
            print(f"[RaidenOffload] Lookup snapshot for {request.request_id}: Block Hits={num_hits}, Block Misses={num_misses}", flush=True)
        
        if num_hits > 0:
            matched_hashes = [m[0] for m in matched]
            # Pin in C++ store
            self.kv_store.pin(matched_hashes)
        
        num_matched_blocks = num_hits
        num_matched_tokens = num_matched_blocks * self.block_size
        num_computed_blocks = num_computed_tokens // self.block_size
        num_blocks_to_load = max(num_matched_blocks - num_computed_blocks, 0)

        if num_blocks_to_load > 0:
            src_chunk_ids = []
            src_locators = []
            load_start_idx = num_computed_blocks
            
            for i in range(load_start_idx, load_start_idx + num_blocks_to_load):
                _, raiden_ids = matched[i]
                r_id = raiden_ids[0]
                src_chunk_ids.append(r_id.data_replica_idx)
                src_locators.append(RaidenLocator(
                    r_id.job_name, r_id.job_replica_id, r_id.data_name, r_id.data_replica_idx
                ))
            
            dummy_dst_blocks = [-1] * num_blocks_to_load
            self._pre_load_specs[request.request_id] = RaidenLoadSpec(
                num_matched_tokens=num_matched_tokens,
                src_chunks=src_chunk_ids,
                dst_blocks=dummy_dst_blocks,
                src_locators=src_locators,
                num_skip_leading_tokens=num_computed_tokens,
            )

        prev_hit_tokens = self._external_cache_hits.get(request.request_id, 0)
        self._external_cache_hits[request.request_id] = num_matched_tokens
        
        if request.request_id not in self._recorded_metrics_reqs:
            self._recorded_metrics_reqs.add(request.request_id)
            total_hits = num_matched_tokens
            total_misses = max(0, request.num_tokens - total_hits)
            self.metrics_collector.record_cache_hit(total_hits)
            self.metrics_collector.record_cache_miss(total_misses)

            stats = self.metrics_collector.get_cumulative_stats()
            total_tokens = stats.lookup_hits + stats.lookup_miss
            hit_rate = (stats.lookup_hits / total_tokens * 100.0) if total_tokens > 0 else 0.0
            logger.debug(f"Cumulative Host Cache Hit Rate: {hit_rate:.2f}% (Hits: {stats.lookup_hits}, Miss: {stats.lookup_miss}, Inserts: {stats.insertions}, Evictions: {stats.evictions}, Queries: {stats.lookup_requests})")
        
        num_matched_for_scheduler = num_matched_tokens
        if num_matched_tokens > 0 and num_matched_tokens == request.num_tokens:
            num_matched_for_scheduler = num_matched_tokens - 1

        num_to_load = max(0, num_matched_for_scheduler - num_computed_tokens)
        return num_to_load, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        self._unfinished_requests[request.request_id] = request
        if num_external_tokens == 0:
            return
        load_spec = self._pre_load_specs.pop(request.request_id, None)
        if load_spec:
            skip_leading_blocks = load_spec.num_skip_leading_tokens // self.block_size
            num_blocks_to_load = len(load_spec.src_chunks)
            num_matched_blocks = num_blocks_to_load + skip_leading_blocks

            all_blocks = blocks.get_block_ids()[0]
            dst_blocks = all_blocks[skip_leading_blocks:num_matched_blocks]

            load_spec.dst_blocks = dst_blocks
            load_spec.can_load = True
            self.load_specs[request.request_id] = load_spec
            self._reqs_being_loaded[request.request_id] |= set(load_spec.src_chunks)

    def _prepare_save_spec(self, tracker: RaidenRequestTracker, is_finished: bool) -> Optional[RaidenSaveSpec]:
        req_id = tracker.req_id
        _request = self._unfinished_requests[req_id]

        num_tracked_tokens = len(tracker.token_ids)
        num_full_blocks = num_tracked_tokens // self.block_size
        adjusted_num_total_blocks = num_full_blocks
        adjusted_num_total_tokens = num_full_blocks * self.block_size

        block_hashes = _request.block_hashes
        raiden_hashes = [to_raiden_hash(h) for h in block_hashes]
        
        # C++ KVCacheStore manages LRU order authoritatively on access.

        has_new_tokens = adjusted_num_total_tokens > tracker.save_watermark
        should_save = False
        if has_new_tokens:
            if not tracker.is_decode_phase:
                should_save = True
            elif envs.TPU_OFFLOAD_DECODE_SAVE:
                if is_finished:
                    should_save = True
                else:
                    next_block_boundary = (tracker.save_watermark // self.block_size + 1) * self.block_size
                    if adjusted_num_total_tokens == next_block_boundary:
                        should_save = True

        save_spec = None
        if should_save:
            num_skip_leading_blocks = tracker.save_watermark // self.block_size
            num_skip_leading_tokens = num_skip_leading_blocks * self.block_size
            num_blocks_to_save = adjusted_num_total_blocks - num_skip_leading_blocks

            if num_blocks_to_save > 0:
                hashes_to_save = raiden_hashes[num_skip_leading_blocks:adjusted_num_total_blocks]
                self._pending_save_hashes[req_id].extend(hashes_to_save)

                src_block_ids = tracker.block_ids[num_skip_leading_blocks:adjusted_num_total_blocks]
                
                save_spec = RaidenSaveSpec(
                    num_skip_leading_tokens=num_skip_leading_tokens,
                    num_total_tokens=adjusted_num_total_tokens,
                    is_final_save=is_finished,
                    src_blocks=src_block_ids,
                    dst_chunks=[],  # Physical chunks are automatically allocated on the worker!
                    dst_locators=[],
                    block_hashes=hashes_to_save,
                )
                tracker.save_watermark = adjusted_num_total_tokens

        if is_finished and save_spec is None:
            save_spec = RaidenSaveSpec(
                num_skip_leading_tokens=tracker.save_watermark,
                num_total_tokens=tracker.save_watermark,
                src_blocks=[], dst_chunks=[], dst_locators=[],
                is_final_save=True, skip_save=True,
            )
        return save_spec

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> RaidenConnectorMetadata:
        metadata = RaidenConnectorMetadata(
            pending_unlocks=copy.deepcopy(self._pending_unlocks_to_send)
        )
        self._pending_unlocks_to_send.clear()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self.load_specs.pop(finished_req_id, None)
            self._recorded_metrics_reqs.discard(finished_req_id)

        for request in scheduler_output.scheduled_new_reqs:
            req_id = request.req_id
            _request = self._unfinished_requests.get(req_id, None)
            if not _request:
                continue
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_external_hits = self._external_cache_hits.pop(req_id, 0)
            
            num_total = min(request.num_computed_tokens + num_new_tokens, _request.num_tokens)
            tokens_for_tracker = request.prompt_token_ids[:num_total]
            initial_save = max(request.num_computed_tokens, num_external_hits)
            
            tracker = RaidenRequestTracker(
                req_id=req_id, prompt_len=len(request.prompt_token_ids),
                block_ids=copy.deepcopy(request.block_ids[0]),
                token_ids=tokens_for_tracker, save_watermark=initial_save,
            )
            self._request_trackers[req_id] = tracker
            
            load_spec = self.load_specs.pop(req_id, None)
            save_spec = self._prepare_save_spec(tracker, is_finished=False)
            if save_spec or (load_spec and load_spec.can_load):
                metadata.requests_meta.append(RaidenReqMeta(req_id, tracker.token_ids, tracker.block_ids, save_spec, load_spec))

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            _request = self._unfinished_requests.get(req_id, None)
            if _request is None:
                continue
            tracker = self._request_trackers[req_id]
            if req_id in cached_reqs.resumed_req_ids:
                tracker.reset_after_preempt()

            num_new = scheduler_output.num_scheduled_tokens[req_id]
            cur_total = min(_request.num_computed_tokens + num_new, _request.num_tokens)
            num_tracked = len(tracker.token_ids)
            new_tokens = _request.all_token_ids[num_tracked:cur_total] if cur_total > num_tracked else []
            new_blocks = cached_reqs.new_block_ids[i] or []
            
            tracker.update(new_blocks, new_tokens)
            
            load_spec = self.load_specs.pop(req_id, None)
            save_spec = self._prepare_save_spec(tracker, is_finished=False)
            if save_spec or (load_spec and load_spec.can_load):
                metadata.requests_meta.append(RaidenReqMeta(req_id, tracker.token_ids, tracker.block_ids, save_spec, load_spec))

        self._pre_load_specs.clear()
        self._external_cache_hits.clear()
        return metadata

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        if self.kv_store is not None:
            raiden_hashes = [to_raiden_hash(h) for h in request.block_hashes]
            # Release logical pins in C++ KVCacheStore
            self.kv_store.release(raiden_hashes)
        return False, None

    def update_connector_output(self, connector_output: KVConnectorOutput):
        if connector_output.kv_connector_stats and connector_output.kv_connector_stats.data is not None:
            stats = connector_output.kv_connector_stats
            
            for req_id, saved_chunks in stats.data["finished_save_chunks"].items():
                pending_hashes = self._pending_save_hashes.get(req_id, [])
                num_completed = len(saved_chunks)
                
                # Retrieve the corresponding block hashes in order
                completed_hashes = [pending_hashes.pop(0) for _ in range(num_completed)] if len(pending_hashes) >= num_completed else []
                
                for h, cid in zip(completed_hashes, saved_chunks):
                    if RaidenId is not None and self.kv_store is not None:
                        locator = RaidenLocator(self.job_name, self.job_replica_id, self.data_name, cid)
                        # 1. Insert into C++ metadata lookup store
                        # 1. Insert into C++ logical store and process evictions
                        all_inserted, evicted = self.kv_store.insert([h], [[locator.to_raiden_id()]], on_host=True)
                        if all_inserted:
                            self.metrics_collector.record_insertion(1)
                            
                            # Handle C++ LRU evictions immediately
                            if evicted:
                                for evict_hash, evict_slices in evicted:
                                    for slice_id in evict_slices:
                                        evict_cid = slice_id.data_replica_idx
                                        self._pending_unlocks_to_send.append(evict_cid)
                                        self.metrics_collector.record_eviction(1)
                                        logger.debug(f"[RaidenOffload] C++ Eviction: Staging physical unlock for chunk {evict_cid} (hash {evict_hash})")

                if not pending_hashes:
                    self._pending_save_hashes.pop(req_id, None)

                for chunk_id in saved_chunks:
                    self._reqs_being_saved[req_id].discard(chunk_id)
                if not self._reqs_being_saved[req_id]:
                    self._reqs_being_saved.pop(req_id, None)

            for req_id, loaded_chunks in stats.data["finished_load_chunks"].items():
                for chunk_id in loaded_chunks:
                    self._reqs_being_loaded[req_id].discard(chunk_id)
                if not self._reqs_being_loaded[req_id]:
                    self._reqs_being_loaded.pop(req_id, None)


class RaidenOffloadConnectorWorker:
    """Executes physical KV cache transfers via Raiden KVCacheManager."""

    def __init__(self, vllm_config: VllmConfig, connector: RaidenOffloadConnector):
        self.vllm_config = vllm_config
        self.connector = connector
        self.block_size = vllm_config.cache_config.block_size
        self.num_cpu_chunks = envs.TPU_OFFLOAD_NUM_CPU_CHUNKS
        
        self.job_replica_id = os.getenv("CLOUD_TPU_TASK_ID", "0")
        self.raiden_manager: Optional[RaidenKVCacheManager] = None
        self.offload_stats = KVRaidenConnectorStats()
        self._pending_saves: list[Any] = []

    def register_runner(self, runner: "TPUModelRunner"):
        if RaidenKVCacheManager is None:
            raise ImportError(
                "RaidenKVCacheManager is not importable from tpu_raiden. Please check your "
                f"PYTHONPATH and Raiden installation. Original error: {_RAIDEN_IMPORT_ERROR}"
            )
        
        kv_caches = runner.kv_caches
        if not kv_caches:
            raise ValueError("No KV caches registered in Runner.")
        
        logger.info(f"Initializing Raiden KVCacheManager on worker. host_blocks_to_allocate={self.num_cpu_chunks}, block_size={self.block_size}")
        self.raiden_manager = RaidenKVCacheManager(
            kv_caches=kv_caches,
            local_control_port=0,
            host_blocks_to_allocate=self.num_cpu_chunks,
            unsafe_skip_buffer_lock=True
        )
        logger.info("Raiden Worker: C++ KVCacheManager Initialized.")

    def start_load_kv(self, fwd_ctx: "ForwardContext"):
        if not self.raiden_manager:
            return
        meta: RaidenConnectorMetadata = self.connector._connector_metadata  # type: ignore
        if not meta:
            return
        
        # 1. Execute physical block unlocking before load
        if meta.pending_unlocks:
            logger.debug(f"Raiden Worker: Unlocking {len(meta.pending_unlocks)} physical Host RAM blocks in C++: {meta.pending_unlocks}")
            self.raiden_manager.unlock_blocks(meta.pending_unlocks)
        
        if not meta.requests_meta:
            return
        
        for req_meta in meta.requests_meta:
            load_spec = req_meta.load_spec
            if load_spec and load_spec.can_load:
                logger.debug(
                    f"Raiden Worker: Loading {len(load_spec.src_chunks)} chunks for {req_meta.req_id} "
                    f"from locators: {load_spec.src_locators}"
                )
                
                remote_src_chunks = []
                local_src_chunks = []
                local_dst_blocks = []
                
                for idx, loc in enumerate(load_spec.src_locators):
                    if loc.job_replica_id != self.job_replica_id:
                        remote_src_chunks.append((loc, load_spec.dst_blocks[idx]))
                    else:
                        local_src_chunks.append(loc.data_replica_idx)
                        local_dst_blocks.append(load_spec.dst_blocks[idx])
                
                if remote_src_chunks:
                    logger.info(f"Raiden Worker: Found {len(remote_src_chunks)} remote chunks to fetch over network! (E.g., {remote_src_chunks[0]})")
                
                if local_src_chunks:
                    self.raiden_manager.h2d(local_src_chunks, local_dst_blocks)
                
                self.offload_stats.record_load(req_meta.req_id, load_spec.src_chunks)

    def wait_for_save(self):
        if not self.raiden_manager:
            return
        meta: RaidenConnectorMetadata = self.connector._connector_metadata  # type: ignore
        if not meta or not meta.requests_meta:
            return
        
        for req_meta in meta.requests_meta:
            save_spec = req_meta.save_spec
            if save_spec and not save_spec.skip_save and save_spec.src_blocks:
                req_id = req_meta.req_id
                src_blocks = save_spec.src_blocks
                logger.debug(
                    f"Raiden Worker: Calling d2h_auto_allocate for {len(src_blocks)} blocks for {req_id}."
                )
                
                # Delegate physical allocation to C++ using d2h_auto_allocate!
                # This automatically allocates locked physical chunk IDs and issues the async copy.
                try:
                    allocated_chunks, future = self.raiden_manager.d2h_auto_allocate(
                        src_blocks
                    )
                except Exception as e:
                    logger.error(
                        f"Raiden Worker: d2h_auto_allocate failed for {req_id}: {e}"
                    )
                    continue
                
                # Reconstruct dst_locators now that physical chunk IDs are allocated
                job_name = self.connector.connector_scheduler.job_name if self.connector.connector_scheduler else "tpu_inference"
                dst_locators = [RaidenLocator(
                    job_name, self.job_replica_id, "kv_cache", cid
                ) for cid in allocated_chunks]
                
                self._pending_saves.append((future, req_id, allocated_chunks, save_spec.block_hashes, dst_locators))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        completed = []
        finished_saves = set()
        for item in self._pending_saves:
            future, req_id, chunks, hashes, locators = item
            try:
                future.Await()  # Synchronous wait for the JAX FFI future
                
                self.offload_stats.record_save(req_id, chunks)
                finished_saves.add(req_id)
                logger.debug(f"Raiden Worker: Save complete for {req_id}.")
            except Exception as e:
                logger.error(f"Failed to save Raiden chunks for {req_id}: {e}")
                # CRITICAL: Unlock the physically allocated blocks so they can be reused!
                self.raiden_manager.unlock_blocks(chunks)
            completed.append(item)
        
        for item in completed:
            self._pending_saves.remove(item)

        return set(), set()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        return self.offload_stats.clone_and_reset()
