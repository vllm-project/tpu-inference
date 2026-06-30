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
from unittest.mock import MagicMock, patch

# Install a mock `tpu_raiden` in sys.modules BEFORE importing pytest / vllm /
# tpu_inference / the connector. This lets the tests run even when the real
# tpu_raiden is NOT installed, and is intentionally done first because:
#   - the connector's module-level `from tpu_raiden import ...` resolves to these
#     mocks (otherwise it caches None and raises ImportError at use time), and
#   - tpu_inference's engine-first `import tpu_raiden.frameworks.jax._tpu_raiden_jax`
#     resolves through this mock (submodule import -> ModuleNotFoundError, which
#     tpu_inference swallows) instead of loading the real engine .so -- which would
#     crash on a standalone import. So the mock also shadows a real install.
mock_raiden = MagicMock()
mock_raiden.KVCacheManager = MagicMock()
mock_raiden.KVCacheStore = MagicMock()


# Configure Mock RaidenId to have standard locator properties
class MockRaidenId:

    def __init__(self,
                 job_name="",
                 job_replica_id="",
                 data_name="",
                 data_replica_idx=0):
        self.job_name = job_name
        self.job_replica_id = job_replica_id
        self.data_name = data_name
        self.data_replica_idx = data_replica_idx


mock_raiden.RaidenId = MockRaidenId
sys.modules['tpu_raiden'] = mock_raiden

import pytest  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorRole  # noqa: E402
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402

from tpu_inference.offload.raiden_offload_connector import (  # noqa: E402
    KVRaidenConnectorStats, RaidenConnectorMetadata, RaidenLoadSpec,
    RaidenLocator, RaidenOffloadConnector, RaidenOffloadConnectorScheduler,
    RaidenOffloadConnectorWorker, RaidenSaveSpec)

_DEFAULT_BLOCK_SIZE = 16


class MockVllmConfig:

    def __init__(self, block_size=_DEFAULT_BLOCK_SIZE):
        self.model_config = MagicMock()
        self.model_config.model = "test-raiden-model"
        self.cache_config = MagicMock()
        self.cache_config.block_size = block_size
        self.kv_transfer_config = MagicMock()


def create_request(request_id: str, num_tokens: int,
                   block_size: int) -> Request:
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


def make_matched(num_blocks: int, start_chunk: int = 100):
    """Build a kv_store.lookup() result: list[(hash_bytes, [RaidenId])]."""
    return [(f"hash_{i}".encode(), [
        MockRaidenId(job_name="tpu_inference",
                     job_replica_id="0",
                     data_name="kv_cache",
                     data_replica_idx=start_chunk + i)
    ]) for i in range(num_blocks)]


def make_scheduler(lookup_result=None, insert_result=None):
    """Create a scheduler whose KVCacheStore mock returns the given results."""
    store = MagicMock()
    store.lookup.return_value = [] if lookup_result is None else lookup_result
    if insert_result is not None:
        store.insert.return_value = insert_result
    mock_raiden.KVCacheStore.return_value = store
    scheduler = RaidenOffloadConnectorScheduler(MockVllmConfig())
    return scheduler, store


def test_instantiation():
    vllm_config = MockVllmConfig()

    sched_conn = RaidenOffloadConnector(vllm_config, KVConnectorRole.SCHEDULER)
    assert sched_conn.connector_scheduler is not None
    assert isinstance(sched_conn.connector_scheduler,
                      RaidenOffloadConnectorScheduler)

    worker_conn = RaidenOffloadConnector(vllm_config, KVConnectorRole.WORKER)
    assert worker_conn.connector_worker is not None
    assert isinstance(worker_conn.connector_worker,
                      RaidenOffloadConnectorWorker)


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
    sched_output.scheduled_cached_reqs = MagicMock(req_ids=[],
                                                   new_block_ids=[],
                                                   resumed_req_ids=set())
    sched_output.num_scheduled_tokens = {"req_1": 32}

    meta: RaidenConnectorMetadata = scheduler.build_connector_meta(
        sched_output)

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
    loc = RaidenLocator(job_name="tpu_inference",
                        job_replica_id="0",
                        data_name="kv_cache",
                        data_replica_idx=10)
    load_spec = RaidenLoadSpec(num_matched_tokens=16,
                               src_chunks=[10],
                               dst_blocks=[1],
                               src_locators=[loc],
                               can_load=True)
    req_meta = MagicMock(req_id="req_load",
                         load_spec=load_spec,
                         save_spec=None)
    conn._connector_metadata = RaidenConnectorMetadata(
        requests_meta=[req_meta])

    worker.start_load_kv(MagicMock())

    stats = worker.get_kv_connector_stats()
    assert stats is not None
    assert "req_load" in stats.data["finished_load_chunks"]
    assert stats.data["finished_load_chunks"]["req_load"] == [10]


# --------------------------------------------------------------------------- #
# Connector-level helpers / pure-Python classes
# --------------------------------------------------------------------------- #
def test_get_required_kvcache_layout():
    assert RaidenOffloadConnector.get_required_kvcache_layout(
        MockVllmConfig()) == "NHD"


def test_build_kv_connector_stats():
    empty = RaidenOffloadConnector.build_kv_connector_stats()
    assert isinstance(empty, KVRaidenConnectorStats)
    assert empty.is_empty()

    primed = RaidenOffloadConnector.build_kv_connector_stats(data={
        "finished_save_chunks": {
            "r": [1]
        },
        "finished_load_chunks": {}
    })
    assert isinstance(primed, KVRaidenConnectorStats)
    assert not primed.is_empty()


def test_raiden_locator_to_raiden_id():
    loc = RaidenLocator(job_name="job",
                        job_replica_id="rep",
                        data_name="data",
                        data_replica_idx=7)
    rid = loc.to_raiden_id()
    assert rid.job_name == "job"
    assert rid.job_replica_id == "rep"
    assert rid.data_name == "data"
    assert rid.data_replica_idx == 7


def test_connector_stats_record_aggregate_reduce():
    stats = KVRaidenConnectorStats()
    assert stats.is_empty()

    stats.record_save("r1", [1, 2])
    stats.record_load("r1", [3])
    assert not stats.is_empty()
    assert stats.reduce() == {"saves": 2, "loads": 1}
    # num_finished_blocks counts request keys in both maps (1 + 1).
    assert stats.num_finished_blocks == 2

    other = KVRaidenConnectorStats()
    other.record_save("r2", [9])
    merged = stats.aggregate(other)
    assert merged.data["finished_save_chunks"]["r1"] == [1, 2]
    assert merged.data["finished_save_chunks"]["r2"] == [9]

    # clone_and_reset returns the old snapshot and empties the live object.
    snapshot = stats.clone_and_reset()
    assert snapshot.data["finished_save_chunks"]["r1"] == [1, 2]
    assert stats.is_empty()


# --------------------------------------------------------------------------- #
# Scheduler: cache-hit / load path
# --------------------------------------------------------------------------- #
def test_scheduler_cache_hit():
    scheduler, store = make_scheduler(lookup_result=make_matched(2))
    req = create_request("req_hit", num_tokens=48, block_size=16)  # 3 blocks

    num_to_load, load_async = scheduler.get_num_new_matched_tokens(req, 0)

    # 2 blocks hit * 16 = 32 tokens; not a full-prefix hit (48), so load all 32.
    assert num_to_load == 32
    assert load_async is False
    store.pin.assert_called_once_with([b"hash_0", b"hash_1"])

    pre = scheduler._pre_load_specs["req_hit"]
    assert pre.src_chunks == [100, 101]
    assert pre.num_matched_tokens == 32
    assert pre.num_skip_leading_tokens == 0


def test_scheduler_full_prefix_hit_reserves_one_token():
    scheduler, _ = make_scheduler(lookup_result=make_matched(2))
    req = create_request("req_full", num_tokens=32, block_size=16)  # 2 blocks

    # All blocks hit => matched == num_tokens; must leave >=1 token to compute.
    num_to_load, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert num_to_load == 31


def test_scheduler_partial_computed_loads_remainder():
    scheduler, _ = make_scheduler(lookup_result=make_matched(3))
    req = create_request("req_part", num_tokens=64, block_size=16)  # 4 blocks

    # 3 blocks matched, 1 block (16 tok) already computed locally => load 2 blocks.
    num_to_load, _ = scheduler.get_num_new_matched_tokens(req, 16)
    assert num_to_load == 32  # (3*16) - 16
    pre = scheduler._pre_load_specs["req_part"]
    # load starts at the first uncomputed matched block -> chunks 101,102.
    assert pre.src_chunks == [101, 102]


def test_update_state_after_alloc_promotes_load():
    scheduler, _ = make_scheduler(lookup_result=make_matched(2))
    req = create_request("req_hit", num_tokens=48, block_size=16)
    scheduler.get_num_new_matched_tokens(req, 0)  # populates _pre_load_specs

    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12], )
    scheduler.update_state_after_alloc(req, blocks, num_external_tokens=32)

    assert "req_hit" not in scheduler._pre_load_specs  # consumed
    ls = scheduler.load_specs["req_hit"]
    assert ls.can_load is True
    assert ls.dst_blocks == [10, 11]  # skip_leading=0, 2 matched blocks
    assert scheduler._reqs_being_loaded["req_hit"] == {100, 101}


def test_update_state_after_alloc_no_external_is_noop():
    scheduler, _ = make_scheduler()
    req = create_request("req_x", num_tokens=32, block_size=16)
    scheduler.update_state_after_alloc(req, MagicMock(), num_external_tokens=0)
    assert "req_x" not in scheduler.load_specs
    assert scheduler._unfinished_requests["req_x"] is req


def test_request_finished_releases_pins():
    scheduler, store = make_scheduler()
    req = create_request("req_done", num_tokens=32, block_size=16)
    delay_free, params = scheduler.request_finished(req, [0, 1])
    assert delay_free is False
    assert params is None
    store.release.assert_called_once_with([b"hash_0", b"hash_1"])


# --------------------------------------------------------------------------- #
# Scheduler: reconcile worker output -> store insert + eviction
# --------------------------------------------------------------------------- #
def test_update_connector_output_inserts_and_stages_evictions():
    evicted = MockRaidenId(data_replica_idx=99)
    scheduler, store = make_scheduler(insert_result=(True, [(b"old_hash",
                                                             [evicted])]))
    scheduler._pending_save_hashes["req_s"] = [b"h0", b"h1"]

    stats = KVRaidenConnectorStats()
    stats.record_save("req_s", [50, 51])
    output = MagicMock()
    output.kv_connector_stats = stats

    scheduler.update_connector_output(output)

    # One insert per (hash, chunk) pair.
    assert store.insert.call_count == 2
    # Each insert evicted chunk 99 -> staged for physical unlock.
    assert scheduler._pending_unlocks_to_send.count(99) == 2
    # Pending hashes fully consumed.
    assert "req_s" not in scheduler._pending_save_hashes


# --------------------------------------------------------------------------- #
# Worker: save path (d2h_auto_allocate + get_finished)
# --------------------------------------------------------------------------- #
def make_worker():
    conn = RaidenOffloadConnector(MockVllmConfig(), KVConnectorRole.WORKER)
    worker = conn.connector_worker
    runner = MagicMock()
    runner.kv_caches = [MagicMock()]
    worker.register_runner(runner)
    # raiden_manager is the shared KVCacheManager mock instance; clear call
    # history (not return values) so per-test call assertions are isolated.
    worker.raiden_manager.reset_mock()
    return conn, worker


def test_worker_save_flow():
    conn, worker = make_worker()
    future = MagicMock()
    worker.raiden_manager.d2h_auto_allocate.return_value = ([5, 6], future)

    save_spec = RaidenSaveSpec(num_skip_leading_tokens=0,
                               num_total_tokens=32,
                               src_blocks=[0, 1],
                               block_hashes=[b"h0", b"h1"])
    req_meta = MagicMock(req_id="req_save",
                         save_spec=save_spec,
                         load_spec=None)
    conn._connector_metadata = RaidenConnectorMetadata(
        requests_meta=[req_meta])

    worker.wait_for_save()
    worker.raiden_manager.d2h_auto_allocate.assert_called_once_with([0, 1])
    assert len(worker._pending_saves) == 1

    worker.get_finished(set())
    future.Await.assert_called_once()
    assert worker._pending_saves == []

    stats = worker.get_kv_connector_stats()
    assert stats.data["finished_save_chunks"]["req_save"] == [5, 6]


def test_worker_skip_save_does_not_transfer():
    conn, worker = make_worker()
    save_spec = RaidenSaveSpec(num_skip_leading_tokens=0,
                               num_total_tokens=0,
                               src_blocks=[],
                               is_final_save=True,
                               skip_save=True)
    req_meta = MagicMock(req_id="req_skip",
                         save_spec=save_spec,
                         load_spec=None)
    conn._connector_metadata = RaidenConnectorMetadata(
        requests_meta=[req_meta])

    worker.wait_for_save()
    worker.raiden_manager.d2h_auto_allocate.assert_not_called()
    assert worker._pending_saves == []


def test_worker_get_finished_unlocks_on_save_failure():
    conn, worker = make_worker()
    failing = MagicMock()
    failing.Await.side_effect = RuntimeError("save boom")
    worker._pending_saves = [(failing, "req_fail", [7, 8], [b"h"], [])]

    # The failure path logs logger.error(...); patch the logger so the expected
    # error message doesn't pollute test output, and assert it was emitted.
    with patch("tpu_inference.offload.raiden_offload_connector.logger") \
            as mock_logger:
        worker.get_finished(set())
    mock_logger.error.assert_called_once()

    worker.raiden_manager.unlock_blocks.assert_called_once_with([7, 8])
    assert worker._pending_saves == []  # cleared even on failure
    assert "req_fail" not in worker.offload_stats.data["finished_save_chunks"]


if __name__ == "__main__":
    pytest.main([__file__])
