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
"""The connector's controller path, with the controller mocked.

Covers the orchestration only -- that the right units, spans, tags and block
ids reach the controller, and that a request is reported finished exactly when
every destination rank's transfer is. Whether those spans tile the
destination's byte space exactly is the job of
`tests/distributed/test_kv_pool_layout.py`, which drives a real
`RaidenController`.

The symmetric fall-through is asserted here too: with no `src_units` in the
handshake the connector must still take the index-matched `start_read` path,
so a producer and a consumer can be upgraded independently.
"""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("tpu_sync.rpc.raiden_controller")

from tpu_inference.distributed import kv_pool_layout as kvpl  # noqa: E402
from tpu_inference.distributed import tpu_raiden_connector as trc  # noqa: E402

CONTROLLER = "10.0.0.1:9700"
NUM_LAYERS = 4
NUM_BLOCKS = 64
HEAD_GROUPS = 8
PACKING = 2
HEAD_DIM = 128
DTYPE_BITS = 16
DTYPE_TAG = "bfloat16"


def _geometry(*, rank: int, parallelism: int, page_tokens: int):
    return kvpl.AttentionKVGeometry(
        num_layers=NUM_LAYERS,
        num_blocks=NUM_BLOCKS,
        page_tokens=page_tokens,
        head_groups=HEAD_GROUPS,
        head_groups_local=HEAD_GROUPS // parallelism,
        packing=PACKING,
        padded_head_dim=HEAD_DIM,
        dtype_bits=DTYPE_BITS,
        transfer_rank=rank,
        transfer_parallelism=parallelism,
        dtype_tag=DTYPE_TAG,
    )


class _RecordingFacade:
    """Stands in for `RaidenControllerClientFacade`."""

    def __init__(self, address, log):
        self.address = address
        self._log = log

    def coordinate_transfer(self, **kwargs):
        self._log.append(("coordinate", kwargs))
        return True


class _RecordingBlockClient:
    """Stands in for `ReshardRequestBlockClient`.

    Shares the caller's log with `_RecordingFacade` so the declaration and
    the coordination it precedes stay in one ordered sequence.
    """

    def __init__(self, address, log):
        self.address = address
        self._log = log

    def register_request_blocks(self,
                                req_id,
                                uuid,
                                unit,
                                block_ids,
                                pool_spans=()):
        self._log.append(
            ("register", req_id, unit, tuple(block_ids), tuple(pool_spans)))


class MockVllmConfig:

    def __init__(self,
                 *,
                 is_producer: bool,
                 block_size: int,
                 tp_size: int,
                 extra_config: dict | None = None):
        self.kv_transfer_config = MagicMock()
        self.kv_transfer_config.is_kv_producer = is_producer
        # Real dict semantics, not a MagicMock: the connector reads its
        # options through `get_from_extra_config(key, default)` and has to see
        # the default when a key is absent.
        self.kv_transfer_config.kv_connector_extra_config = dict(extra_config
                                                                 or {})
        self.kv_transfer_config.get_from_extra_config = (
            self.kv_transfer_config.kv_connector_extra_config.get)
        self.cache_config = MagicMock()
        self.cache_config.block_size = block_size
        self.model_config = MagicMock()
        self.model_config.max_model_len = 4096
        self.parallel_config = MagicMock()
        self.parallel_config.tensor_parallel_size = tp_size


def _scheduler(*,
               is_producer,
               block_size=128,
               tp_size=4,
               env=None,
               extra_config=None):
    with patch.dict(os.environ, env or {}, clear=False):
        return trc.TPUConnectorScheduler(
            MockVllmConfig(is_producer=is_producer,
                           block_size=block_size,
                           tp_size=tp_size,
                           extra_config=extra_config))


def _finished_request(req_id="req-0", num_computed_tokens=261):
    request = MagicMock()
    request.request_id = req_id
    request.status = trc.RequestStatus.FINISHED_LENGTH_CAPPED
    request.num_computed_tokens = num_computed_tokens
    return request


class TestSchedulerHandshake(unittest.TestCase):

    def test_symmetric_handshake_is_unchanged_without_a_controller(self):
        scheduler = _scheduler(is_producer=True,
                               env={"TPU_KV_CONTROLLER_ADDRESS": ""})
        self.assertFalse(scheduler.use_controller)
        _, params = scheduler.request_finished(_finished_request(),
                                               [10, 11, 12])
        self.assertEqual(
            set(params),
            {"uuid", "remote_block_ids", "remote_host", "remote_port"})

    def test_controller_handshake_lists_every_source_rank(self):
        scheduler = _scheduler(is_producer=True,
                               tp_size=4,
                               env={
                                   "TPU_KV_CONTROLLER_ADDRESS": CONTROLLER,
                                   "TPU_KV_DECODE_TP_SIZE": "2",
                               })
        geometry = _geometry(rank=0, parallelism=4, page_tokens=128)
        with patch.object(trc.dist_utils,
                          "get_local_kv_geometry",
                          return_value=kvpl.geometry_to_dict(geometry)):
            delay, params = scheduler.request_finished(_finished_request(),
                                                       [10, 11, 12])

        self.assertTrue(delay)
        # Every source rank is listed, contributing or not: the controller
        # requires transfer_rank contiguous from zero.
        self.assertEqual(
            [unit["data_replica_idx"] for unit in params["src_units"]],
            [0, 1, 2, 3])
        self.assertEqual(params["src_controller"], CONTROLLER)
        # 261 computed tokens at page 128: the trailing partial block is
        # dropped, so exactly two whole source pages are transferred.
        self.assertEqual(params["remote_block_ids"], [10, 11])
        self.assertEqual(params["num_tokens"], 2 * 128)
        # One req_id per *destination* rank: the controller's registration
        # store is keyed (req_id, unit), so reusing one would have the second
        # destination's spans overwrite the first's.
        self.assertEqual(params["dst_req_ids"], ["req-0#d0", "req-0#d1"])
        self.assertEqual(kvpl.geometry_from_dict(params["src_geometry"]),
                         geometry)

    def test_two_engines_of_a_role_register_distinct_units(self):
        # Two decode engines of the same parallelism both hold ranks 0..TP-1,
        # so rank alone cannot separate them: without the instance tag they
        # register byte-identical ids, the controller's registry keeps only
        # the last endpoints, and every push lands on one engine while the
        # other's loads never complete.
        first = kvpl.engine_instance_tag("10.0.1.1", 7200)
        second = kvpl.engine_instance_tag("10.0.2.2", 7200)
        self.assertNotEqual(first, second)
        for rank in range(4):
            a = kvpl.work_unit_id(trc._DECODE_ROLE, rank, first)
            b = kvpl.work_unit_id(trc._DECODE_ROLE, rank, second)
            self.assertNotEqual(a, b)
            # The controller keys push schedules by int(job_replica_id), so
            # the tag must never leak into that field.
            for unit in (a, b):
                self.assertEqual(int(unit.job_replica_id), rank)
                self.assertEqual(unit.data_replica_idx, rank)

    def test_handshake_units_match_what_the_worker_registers(self):
        # The scheduler names the producer's units in the handshake while its
        # workers register them from another process. Both derive the tag from
        # the same configuration, so they must agree without exchanging one.
        env = {
            "TPU_KV_CONTROLLER_ADDRESS": CONTROLLER,
            "TPU_KV_TRANSFER_PORT": "7100",
        }
        scheduler = _scheduler(is_producer=True, tp_size=4, env=env)
        geometry = _geometry(rank=0, parallelism=4, page_tokens=128)
        with patch.dict(os.environ, env, clear=False), \
             patch.object(trc.dist_utils, "get_host_ip",
                          return_value="10.0.3.3"), \
             patch.object(trc.dist_utils, "get_local_kv_geometry",
                          return_value=kvpl.geometry_to_dict(geometry)):
            scheduler.instance_tag = kvpl.engine_instance_tag(
                trc.dist_utils.get_host_ip(),
                trc.dist_utils.get_kv_transfer_port())
            _, params = scheduler.request_finished(_finished_request(),
                                                   [10, 11, 12])
            worker_tag = kvpl.engine_instance_tag(
                trc.dist_utils.get_host_ip(),
                int(trc.dist_utils.get_kv_transfer_port()))

        self.assertEqual(
            [kvpl.unit_from_dict(unit) for unit in params["src_units"]], [
                kvpl.work_unit_id(trc._PREFILL_ROLE, rank, worker_tag)
                for rank in range(4)
            ])

    def test_missing_worker_geometry_is_a_loud_error(self):
        scheduler = _scheduler(is_producer=True,
                               env={"TPU_KV_CONTROLLER_ADDRESS": CONTROLLER})
        with patch.object(trc.dist_utils,
                          "get_local_kv_geometry",
                          return_value=None):
            with self.assertRaisesRegex(RuntimeError, "never published"):
                scheduler.request_finished(_finished_request(), [10, 11, 12])

    def test_decode_rank_count_mismatch_is_rejected(self):
        scheduler = _scheduler(is_producer=False,
                               tp_size=2,
                               env={"TPU_KV_CONTROLLER_ADDRESS": CONTROLLER})
        params = {
            "src_units": [{}, {}, {}, {}],
            "dst_req_ids": ["a", "b", "c", "d"],
        }
        with self.assertRaisesRegex(ValueError, r'"decode_tp_size": 2'):
            scheduler._controller_load_fields(params)

    def test_decode_tp_size_is_read_from_the_connector_extra_config(self):
        """The peer's TP degree is a connector option, not a TPU-only env var.

        It rides in `--kv-transfer-config` alongside `kv_connector` and
        `kv_role`, so a recipe written for another backend's connector ports
        without moving settings into the environment.
        """
        scheduler = _scheduler(is_producer=True,
                               tp_size=4,
                               env={
                                   "TPU_KV_CONTROLLER_ADDRESS": CONTROLLER,
                                   "TPU_KV_DECODE_TP_SIZE": "",
                               },
                               extra_config={"decode_tp_size": 2})
        self.assertEqual(scheduler.num_decode_ranks, 2)

        geometry = _geometry(rank=0, parallelism=4, page_tokens=128)
        with patch.object(trc.dist_utils,
                          "get_local_kv_geometry",
                          return_value=kvpl.geometry_to_dict(geometry)):
            _, params = scheduler.request_finished(_finished_request(),
                                                   [10, 11, 12])
        # One destination request id per decode rank, from the declared width.
        self.assertEqual(len(params["dst_req_ids"]), 2)

    def test_extra_config_wins_over_the_environment_fallback(self):
        scheduler = _scheduler(is_producer=True,
                               tp_size=4,
                               env={
                                   "TPU_KV_CONTROLLER_ADDRESS": CONTROLLER,
                                   "TPU_KV_DECODE_TP_SIZE": "8",
                               },
                               extra_config={"decode_tp_size": 2})
        self.assertEqual(scheduler.num_decode_ranks, 2)

    def test_environment_fallback_still_applies_when_unset(self):
        scheduler = _scheduler(is_producer=True,
                               tp_size=4,
                               env={
                                   "TPU_KV_CONTROLLER_ADDRESS": CONTROLLER,
                                   "TPU_KV_DECODE_TP_SIZE": "8",
                               },
                               extra_config={})
        self.assertEqual(scheduler.num_decode_ranks, 8)

    def test_symmetric_is_assumed_when_neither_is_set(self):
        scheduler = _scheduler(is_producer=True,
                               tp_size=4,
                               env={
                                   "TPU_KV_CONTROLLER_ADDRESS": CONTROLLER,
                                   "TPU_KV_DECODE_TP_SIZE": "",
                               },
                               extra_config={})
        self.assertEqual(scheduler.num_decode_ranks, 4)

    def test_matched_tokens_follow_the_producer_not_our_page_size(self):
        """The two sides must agree on how much prefix actually arrives.

        The producer drops its own trailing partial block, so it sends
        `round_down(len(prompt), P.block_size)`. Re-deriving that from the
        decode block size answers differently for any prompt that is not a
        multiple of both, and the excess would be decoded against KV nobody
        ever wrote.
        """
        scheduler = _scheduler(is_producer=False,
                               block_size=64,
                               tp_size=1,
                               env={"TPU_KV_CONTROLLER_ADDRESS": CONTROLLER})
        request = MagicMock()
        request.prompt_token_ids = [0] * 200
        request.kv_transfer_params = {"num_tokens": 128}
        count, is_async = scheduler.get_num_new_matched_tokens(request, 0)
        # round_down(200, 64) would say 192; the producer only sent 128.
        self.assertEqual(count, 128)
        self.assertTrue(is_async)

    def test_matched_tokens_round_the_producer_count_to_our_pages(self):
        """A decode page coarser than the prefill page truncates further."""
        scheduler = _scheduler(is_producer=False,
                               block_size=128,
                               tp_size=1,
                               env={"TPU_KV_CONTROLLER_ADDRESS": CONTROLLER})
        request = MagicMock()
        request.prompt_token_ids = [0] * 200
        request.kv_transfer_params = {"num_tokens": 192}
        count, _ = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(count, 128)

    def test_matched_tokens_unchanged_on_the_symmetric_path(self):
        """No `num_tokens` in the handshake means the legacy derivation."""
        scheduler = _scheduler(is_producer=False,
                               block_size=128,
                               tp_size=1,
                               env={"TPU_KV_CONTROLLER_ADDRESS": ""})
        request = MagicMock()
        request.prompt_token_ids = [0] * 300
        request.kv_transfer_params = {"uuid": 1}
        count, _ = scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(count, 256)

    def test_load_without_controller_fields_falls_through(self):
        scheduler = _scheduler(is_producer=False,
                               env={"TPU_KV_CONTROLLER_ADDRESS": CONTROLLER})
        self.assertEqual(scheduler._controller_load_fields({"uuid": 1}), {})


class TestWorkerControllerPath(unittest.TestCase):

    def _worker(self,
                *,
                dst_parallelism=2,
                dst_page_tokens=64,
                is_producer=False):
        with patch.dict(os.environ, {"TPU_KV_CONTROLLER_ADDRESS": CONTROLLER},
                        clear=False):
            worker = trc.TPUConnectorWorker(
                MockVllmConfig(is_producer=is_producer,
                               block_size=dst_page_tokens,
                               tp_size=dst_parallelism))
        self.assertTrue(worker.use_controller)
        worker.shard_geometries = [
            _geometry(rank=rank,
                      parallelism=dst_parallelism,
                      page_tokens=dst_page_tokens)
            for rank in range(dst_parallelism)
        ]
        worker.shard_units = [
            kvpl.work_unit_id(trc._DECODE_ROLE, rank)
            for rank in range(dst_parallelism)
        ]
        worker.shard_managers = [MagicMock() for _ in range(dst_parallelism)]
        for manager in worker.shard_managers:
            manager.poll_stats.return_value = ([], [], [])
        worker._transfer_pool = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(worker._transfer_pool.shutdown)
        return worker

    def _load_meta(self,
                   *,
                   src_parallelism=4,
                   src_page_tokens=128,
                   num_tokens=256,
                   dst_req_ids=("req-0#d0", "req-0#d1")):
        src = _geometry(rank=0,
                        parallelism=src_parallelism,
                        page_tokens=src_page_tokens)
        return trc.LoadMeta(
            uuid=7,
            local_block_ids=[100, 101, 102, 103, 104],
            remote_block_ids=[10, 11],
            remote_host="10.0.0.1",
            remote_port=9100,
            src_units=[
                kvpl.unit_to_dict(kvpl.work_unit_id(trc._PREFILL_ROLE, rank))
                for rank in range(src_parallelism)
            ],
            src_geometry=kvpl.geometry_to_dict(src),
            src_controller=CONTROLLER,
            num_tokens=num_tokens,
            dst_req_ids=list(dst_req_ids),
        )

    def _run_load(self, worker, req_meta, req_id="req-0"):
        log = []
        with patch.object(trc, "RaidenControllerClientFacade",
                          lambda address: _RecordingFacade(address, log)), \
             patch.object(trc, "ReshardRequestBlockClient",
                          lambda address: _RecordingBlockClient(address, log)):
            worker._start_controller_load(req_id, req_meta)
            wait(worker._loads_in_flight[req_id])
        for future in worker._loads_in_flight[req_id]:
            self.assertIsNone(future.exception())
        return log

    def test_one_transfer_per_destination_rank(self):
        worker = self._worker()
        log = self._run_load(worker, self._load_meta())

        coordinations = [entry[1] for entry in log if entry[0] == "coordinate"]
        self.assertEqual(len(coordinations), 2)
        targeted = sorted(kwargs["dst_units"][0].data_replica_idx
                          for kwargs in coordinations)
        self.assertEqual(targeted, [0, 1])
        for kwargs in coordinations:
            # One destination unit per plan is a hard constraint of the
            # byte-span planner, which is why a TP2 decode fans out to two
            # calls rather than one call naming two shards.
            self.assertEqual(len(kwargs["dst_units"]), 1)
            self.assertEqual(kwargs["transfer_pool_tags"], [kvpl.KV_POOL_TAG])
            # Distinct per destination: at TP4 -> TP2 each source feeds both
            # of us, and a source rejects a second plan under a uuid it has
            # already registered.
            self.assertEqual(
                kwargs["uuid"],
                kvpl.dst_uuid(7, kwargs["dst_units"][0].data_replica_idx))
            self.assertEqual(kwargs["src_controller_address"], CONTROLLER)
            self.assertEqual(kwargs["num_tokens"], 256)
            # 256 tokens at 64 tokens per destination page: the four pages
            # those tokens actually fill, not every block vLLM allocated.
            self.assertEqual(kwargs["dst_device_block_ids"],
                             [100, 101, 102, 103])
            self.assertEqual(
                [unit.data_replica_idx for unit in kwargs["src_units"]],
                [0, 1, 2, 3])

    def test_each_destination_rank_gets_its_own_plan_uuid(self):
        """Every source here feeds both destinations, so the uuids must differ.

        A source keys its active plan by uuid and rejects a repeat with
        ALREADY_EXISTS, which arms only the first destination and leaves the
        second waiting on a push that never comes.
        """
        worker = self._worker()
        log = self._run_load(worker, self._load_meta())

        uuids = [
            kwargs["uuid"] for entry in log if entry[0] == "coordinate"
            for kwargs in [entry[1]]
        ]
        self.assertEqual(len(set(uuids)), 2)

    def test_non_contributing_source_ranks_register_no_spans(self):
        worker = self._worker()
        log = self._run_load(worker, self._load_meta())

        registrations = [entry for entry in log if entry[0] == "register"]
        # Every source rank registers for every destination rank: a listed
        # unit with no registration is "Missing producer block registration".
        self.assertEqual(len(registrations), 8)
        by_target = {}
        for _, req_id, unit, block_ids, spans in registrations:
            self.assertEqual(block_ids, (10, 11))
            by_target.setdefault(req_id, {})[unit.data_replica_idx] = spans
        # TP4 -> TP2 on the head axis: {0,1} feed d0 and {2,3} feed d1.
        self.assertEqual(
            {rank
             for rank, spans in by_target["req-0#d0"].items() if spans},
            {0, 1})
        self.assertEqual(
            {rank
             for rank, spans in by_target["req-0#d1"].items() if spans},
            {2, 3})

    def test_request_is_finished_only_when_every_rank_lands(self):
        worker = self._worker()
        self._run_load(worker, self._load_meta())

        # One rank's bytes have landed; the request is not loaded yet.
        worker.shard_managers[0].poll_stats.return_value = ([], ["req-0#d0"],
                                                            [])
        self.assertEqual(worker.get_finished(), (set(), set()))

        worker.shard_managers[0].poll_stats.return_value = ([], [], [])
        worker.shard_managers[1].poll_stats.return_value = ([], ["req-0#d1"],
                                                            [])
        self.assertEqual(worker.get_finished(), (set(), {"req-0"}))
        # Bookkeeping is released, not leaked.
        self.assertEqual(worker._loads_in_flight, {})
        self.assertEqual(worker._load_req_ids, {})
        self.assertEqual(worker._recvd_ids, set())

    def test_a_failed_reshard_is_never_reported_as_loaded(self):
        worker = self._worker()

        def _explode(address):
            raise RuntimeError("controller unreachable")

        with patch.object(trc, "RaidenControllerClientFacade", _explode):
            worker._start_controller_load("req-0", self._load_meta())
            wait(worker._loads_in_flight["req-0"])
        # Decoding against KV that was never written asserts inside vLLM and
        # takes the whole EngineCore down; the request must time out at the
        # API layer instead.
        self.assertEqual(worker.get_finished(), (set(), set()))
        self.assertEqual(worker._loads_in_flight, {})
        self.assertEqual(worker._load_req_ids, {})

    def test_producer_holds_blocks_until_every_push_completes(self):
        worker = self._worker(is_producer=True)
        metadata = trc.TPUConnectorMetadata(
            reqs_to_send={
                "req-0":
                trc.SendMeta(uuid=7,
                             local_block_ids=[10, 11],
                             expiration_time=1e18,
                             dst_req_ids=["req-0#d0", "req-0#d1"])
            })
        worker.process_send_load(metadata)
        # The controller fires the push over the listener socket; the producer
        # arms nothing.
        for manager in worker.shard_managers:
            manager.register_read.assert_not_called()

        worker.shard_managers[0].poll_stats.return_value = (["req-0#d0"], [],
                                                            [])
        self.assertEqual(worker.get_finished(), (set(), set()))
        worker.shard_managers[0].poll_stats.return_value = ([], [], [])
        worker.shard_managers[1].poll_stats.return_value = (["req-0#d1"], [],
                                                            [])
        self.assertEqual(worker.get_finished(), ({"req-0"}, set()))
        self.assertEqual(worker._sends_outstanding, {})

    def test_symmetric_load_still_takes_the_start_read_path(self):
        worker = self._worker()
        worker.kv_manager = MagicMock()
        worker.kv_manager.get_local_endpoints.return_value = [{"shards": [0]}]
        worker._parallelism = 1
        metadata = trc.TPUConnectorMetadata(
            reqs_to_load={
                "req-1":
                trc.LoadMeta(uuid=7,
                             local_block_ids=[100],
                             remote_block_ids=[10],
                             remote_host="10.0.0.1",
                             remote_port=9100)
            })
        worker.process_send_load(metadata)
        worker.kv_manager.start_read.assert_called_once()
        self.assertEqual(
            worker.kv_manager.start_read.call_args.kwargs["remote_endpoint"],
            "10.0.0.1:9100")
        self.assertEqual(worker._loads_in_flight, {})

    def test_post_load_notify_pass_does_not_touch_the_kv_manager(self):
        # Once the load lands, the scheduler re-issues the request with
        # remote_block_ids=None to release the producer -- and, because there
        # is nothing left to load, without any of the controller fields. Under
        # the controller there is no `kv_manager` at all (each device shard has
        # its own), so that pass must resolve no endpoint and do nothing.
        worker = self._worker()
        self.assertIsNone(worker.kv_manager)
        metadata = trc.TPUConnectorMetadata(
            reqs_to_load={
                "req-1":
                trc.LoadMeta(uuid=7,
                             local_block_ids=[100, 101],
                             remote_block_ids=None,
                             remote_host="10.0.0.1",
                             remote_port=9100)
            })
        worker.process_send_load(metadata)
        self.assertEqual(worker._loads_in_flight, {})
        self.assertEqual(worker._submitted, set())


if __name__ == "__main__":
    unittest.main()
