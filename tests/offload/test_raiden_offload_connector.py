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

import sys
from unittest.mock import MagicMock

# Mock tpu_raiden module before importing connector to avoid strict ImportError
mock_raiden = MagicMock()
mock_raiden.KVCacheManager = MagicMock()
mock_raiden.KVCacheStore = MagicMock()

# Configure Mock RaidenId to have standard locator properties
class MockRaidenId:
    def __init__(self, job_name="", job_replica_id="", data_name="", data_replica_idx=0):
        self.job_name = job_name
        self.job_replica_id = job_replica_id
        self.data_name = data_name
        self.data_replica_idx = data_replica_idx

mock_raiden.RaidenId = MockRaidenId
sys.modules['tpu_raiden'] = mock_raiden

import os
import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from tpu_inference.offload.raiden_offload_connector import (
    RaidenOffloadConnector, RaidenOffloadConnectorScheduler,
    RaidenOffloadConnectorWorker, RaidenConnectorMetadata,
    RaidenLoadSpec, RaidenSaveSpec, RaidenLocator
)

_DEFAULT_BLOCK_SIZE = 16


class MockVllmConfig:
    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
        self.model_config = MagicMock()
        self.model_config.model = "test-raiden-model"
        self.cache_config = MagicMock()
        self.cache_config.block_size = block_size
        self.kv_transfer_config = MagicMock()


def create_request(request_id: str, num_tokens: int, block_size: int) -> Request:
    req = MagicMock(spec=Request)
    req.request_id = request_id
    req.req_id = request_id
    req.prompt_token_ids = list(range(num_tokens))
    req.all_token_ids = list(range(num_tokens))
    req.num_computed_tokens = 0
    req.block_size = block_size
    num_blocks = num_tokens // block_size
    req.block_ids = [list(range(num_blocks))]
    req.block_hashes = [f"hash_{i}".encode() for i in range(num_blocks)]
    req.num_tokens = num_tokens
    return req


def test_instantiation():
    vllm_config = MockVllmConfig()
    
    sched_conn = RaidenOffloadConnector(vllm_config, KVConnectorRole.SCHEDULER)
    assert sched_conn.connector_scheduler is not None
    assert isinstance(sched_conn.connector_scheduler, RaidenOffloadConnectorScheduler)
    
    worker_conn = RaidenOffloadConnector(vllm_config, KVConnectorRole.WORKER)
    assert worker_conn.connector_worker is not None
    assert isinstance(worker_conn.connector_worker, RaidenOffloadConnectorWorker)


def test_scheduler_no_hit():
    vllm_config = MockVllmConfig()
    
    mock_store_inst = MagicMock()
    mock_store_inst.lookup.return_value = []
    mock_raiden.KVCacheStore.return_value = mock_store_inst

    scheduler = RaidenOffloadConnectorScheduler(vllm_config)
    req = create_request("req_1", num_tokens=32, block_size=16)
    
    num_matched, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert num_matched == 0


def test_scheduler_build_meta():
    vllm_config = MockVllmConfig()
    
    mock_store_inst = MagicMock()
    mock_store_inst.lookup.return_value = []
    mock_raiden.KVCacheStore.return_value = mock_store_inst

    scheduler = RaidenOffloadConnectorScheduler(vllm_config)
    req = create_request("req_1", num_tokens=32, block_size=16)
    
    scheduler.update_state_after_alloc(req, MagicMock(), 0)
    
    sched_output = MagicMock(spec=SchedulerOutput)
    sched_output.finished_req_ids = set()
    sched_output.scheduled_new_reqs = [req]
    sched_output.scheduled_cached_reqs = MagicMock(req_ids=[], new_block_ids=[], resumed_req_ids=set())
    sched_output.num_scheduled_tokens = {"req_1": 32}

    meta: RaidenConnectorMetadata = scheduler.build_connector_meta(sched_output)
    
    assert len(meta.requests_meta) == 1
    req_meta = meta.requests_meta[0]
    assert req_meta.req_id == "req_1"
    
    assert req_meta.save_spec is not None
    assert isinstance(req_meta.save_spec, RaidenSaveSpec)
    assert req_meta.save_spec.num_total_tokens == 32
    assert len(req_meta.save_spec.src_blocks) == 2
    
    # Verify that destination locators/chunks are empty (allocated dynamically by the worker)
    assert len(req_meta.save_spec.dst_locators) == 0
    assert len(req_meta.save_spec.dst_chunks) == 0


def test_worker_register_and_load():
    vllm_config = MockVllmConfig()
    conn = RaidenOffloadConnector(vllm_config, KVConnectorRole.WORKER)
    worker = conn.connector_worker
    assert worker is not None
    
    runner = MagicMock()
    runner.kv_caches = [MagicMock()]
    
    worker.register_runner(runner)
    assert worker.raiden_manager is not None
    
    # Provide explicit full locator info in LoadSpec
    loc = RaidenLocator(job_name="tpu_inference", job_replica_id="0", data_name="kv_cache", data_replica_idx=10)
    load_spec = RaidenLoadSpec(num_matched_tokens=16, src_chunks=[10], dst_blocks=[1], src_locators=[loc], can_load=True)
    req_meta = MagicMock(req_id="req_load", load_spec=load_spec, save_spec=None)
    conn._connector_metadata = RaidenConnectorMetadata(requests_meta=[req_meta])
    
    worker.start_load_kv(MagicMock())
    
    stats = worker.get_kv_connector_stats()
    assert stats is not None
    assert "req_load" in stats.data["finished_load_chunks"]
    assert stats.data["finished_load_chunks"]["req_load"] == [10]


if __name__ == "__main__":
    pytest.main([__file__])
