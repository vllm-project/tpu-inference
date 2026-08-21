# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import functools
import importlib.util
import pathlib
import sys
import types
from collections import defaultdict
from types import SimpleNamespace

import pytest


def _load_router(monkeypatch):
    repository_root = pathlib.Path(__file__).resolve().parents[2]
    config_path = (repository_root / "tpu_inference" / "runner" / "diffusion" /
                   "config.py")
    config_spec = importlib.util.spec_from_file_location(
        "tpu_inference.runner.diffusion.config", config_path)
    config_module = importlib.util.module_from_spec(config_spec)

    packages = {
        "tpu_inference":
        types.ModuleType("tpu_inference"),
        "tpu_inference.core":
        types.ModuleType("tpu_inference.core"),
        "tpu_inference.runner":
        types.ModuleType("tpu_inference.runner"),
        "tpu_inference.runner.diffusion":
        types.ModuleType("tpu_inference.runner.diffusion"),
        config_spec.name:
        config_module,
    }
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    config_spec.loader.exec_module(config_module)

    router_path = (repository_root / "tpu_inference" / "core" /
                   "diffusion_dp_router.py")
    router_spec = importlib.util.spec_from_file_location(
        "tpu_inference.core.diffusion_dp_router", router_path)
    router_module = importlib.util.module_from_spec(router_spec)
    monkeypatch.setitem(sys.modules, router_spec.name, router_module)
    router_spec.loader.exec_module(router_module)
    return router_module


def _config(round_robin_enabled: bool,
            max_num_seqs: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        additional_config={
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "block_size": 32,
                "mask_token_id": 151669,
                "sub_block_size": 8,
                "cohort_round_robin_dp": round_robin_enabled,
            },
        },
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def _request(request_id: str,
             data_parallel_rank: int | None = None,
             late_interaction_rank: int | None = None) -> SimpleNamespace:
    pooling_params = (None if late_interaction_rank is None else
                      SimpleNamespace(engine_index=late_interaction_rank))
    return SimpleNamespace(
        request_id=request_id,
        data_parallel_rank=data_parallel_rank,
        pooling_params=pooling_params,
    )


def _factory_template(
    vllm_config,
    executor_class,
    log_stats,
    client_addresses=None,
    client_count=1,
    client_index=0,
):
    return DPLBAsyncMPClient(  # noqa: F821
        vllm_config,
        executor_class,
        log_stats,
        client_addresses,
        client_count,
        client_index,
    )


def _install_fake_vllm(monkeypatch, *, router_has_extra_parameter=False):
    core_client = types.ModuleType("vllm.v1.engine.core_client")

    class DPLBAsyncMPClient:

        def __init__(
            self,
            vllm_config,
            executor_class,
            log_stats,
            client_addresses=None,
            client_count=1,
            client_index=0,
        ):
            del executor_class, log_stats, client_addresses
            self.vllm_config = vllm_config
            self.client_count = client_count
            self.client_index = client_index
            self.core_engines = [b"engine-0", b"engine-1"]
            self.eng_start_index = client_index
            self.lb_engines = [[0, 0], [0, 0]]
            self.reqs_in_flight = {}
            self.added_engines = []
            self.aborted_by_engine = []
            self.original_router_calls = 0

        def get_core_engine_for_request(self, request):
            self.original_router_calls += 1
            engine_index = request.data_parallel_rank
            if engine_index is None and request.pooling_params is not None:
                engine_index = request.pooling_params.engine_index
            if engine_index is None:
                engine_index = min(
                    range(len(self.core_engines)),
                    key=lambda index: (self.lb_engines[index][0] * 4 + self.
                                       lb_engines[index][1]),
                )
                self.lb_engines[engine_index][0] += self.client_count
            chosen_engine = self.core_engines[engine_index]
            self.reqs_in_flight[request.request_id] = chosen_engine
            return chosen_engine

        async def add_request_async(self, request):
            chosen_engine = self.get_core_engine_for_request(request)
            self.added_engines.append(chosen_engine)

        async def abort_requests_async(self, request_ids):
            by_engine = defaultdict(list)
            for request_id in request_ids:
                engine = self.reqs_in_flight.get(request_id)
                if engine is not None:
                    by_engine[engine].append(request_id)
            self.aborted_by_engine.extend(by_engine.items())

        @staticmethod
        async def process_engine_outputs(self, outputs):
            for request_id in outputs.finished_requests:
                self.reqs_in_flight.pop(request_id, None)

    if router_has_extra_parameter:

        def get_core_engine_for_request(self, request, unexpected=None):
            del unexpected
            return DPLBAsyncMPClient.get_core_engine_for_request(self, request)

        DPLBAsyncMPClient.get_core_engine_for_request = \
            get_core_engine_for_request

    core_client.DPLBAsyncMPClient = DPLBAsyncMPClient
    factory = types.FunctionType(
        _factory_template.__code__,
        core_client.__dict__,
        name="make_async_mp_client",
        argdefs=_factory_template.__defaults__,
    )

    @functools.wraps(factory)
    def instrumented_factory(*args, **kwargs):
        return factory(*args, **kwargs)

    class EngineCoreClient:
        make_async_mp_client = staticmethod(instrumented_factory)

    core_client.EngineCoreClient = EngineCoreClient

    engine = types.ModuleType("vllm.v1.engine")
    engine.core_client = core_client
    late_interaction = types.ModuleType("vllm.v1.pool.late_interaction")
    late_interaction.get_late_interaction_engine_index = (
        lambda pooling_params, _: None
        if pooling_params is None else pooling_params.engine_index)

    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": engine,
        "vllm.v1.engine.core_client": core_client,
        "vllm.v1.pool": types.ModuleType("vllm.v1.pool"),
        "vllm.v1.pool.late_interaction": late_interaction,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return core_client


async def _add_requests(client, requests):
    for request in requests:
        await client.add_request_async(request)


def test_round_robin_routes_complete_cohort_blocks(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.EngineCoreClient.make_async_mp_client(
        _config(True), object, False)

    asyncio.run(
        _add_requests(client, [_request(f"request-{i}") for i in range(64)]))

    assert client.added_engines == ([b"engine-0"] * 16 + [b"engine-1"] * 16 +
                                    [b"engine-0"] * 16 + [b"engine-1"] * 16)
    assert client.original_router_calls == 0


def test_next_cohort_follows_engine_that_clears_first(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(True), object, False)
    initial_requests = [_request(f"initial-{i}") for i in range(32)]
    asyncio.run(_add_requests(client, initial_requests))

    asyncio.run(
        type(client).process_engine_outputs(
            client,
            SimpleNamespace(finished_requests=[
                request.request_id for request in initial_requests[16:]
            ]),
        ))
    asyncio.run(
        _add_requests(client, [_request(f"refill-{i}") for i in range(16)]))

    assert client.added_engines[:32] == ([b"engine-0"] * 16 +
                                         [b"engine-1"] * 16)
    assert client.added_engines[32:] == [b"engine-1"] * 16


def test_equal_in_flight_counts_rotate_tie_break(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(True, max_num_seqs=2),
                                           object, False)

    for wave in range(4):
        requests = [_request(f"wave-{wave}-{i}") for i in range(2)]
        asyncio.run(_add_requests(client, requests))
        asyncio.run(
            type(client).process_engine_outputs(
                client,
                SimpleNamespace(finished_requests=[
                    request.request_id for request in requests
                ]),
            ))

    assert client.added_engines == ([b"engine-0"] * 2 + [b"engine-1"] * 2 +
                                    [b"engine-0"] * 2 + [b"engine-1"] * 2)


@pytest.mark.parametrize("request_count", [1, 4])
def test_partial_cohort_stays_on_one_engine(monkeypatch, request_count):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(True), object, False)

    asyncio.run(
        _add_requests(client,
                      [_request(f"request-{i}")
                       for i in range(request_count)]))

    assert client.added_engines == [b"engine-0"] * request_count


def test_explicit_late_and_existing_requests_do_not_advance_cohort_block(
        monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(True), object, False)

    asyncio.run(
        _add_requests(
            client,
            [
                _request("automatic-0"),
                _request("automatic-0"),
                _request("explicit", data_parallel_rank=1),
                _request("late", late_interaction_rank=0),
            ] + [_request(f"automatic-{i}")
                 for i in range(1, 16)] + [_request("next-cohort")],
        ))

    assert client.added_engines[:4] == [
        b"engine-0", b"engine-0", b"engine-1", b"engine-0"
    ]
    assert client.added_engines[4:19] == [b"engine-0"] * 15
    assert client.added_engines[19] == b"engine-1"
    assert client.original_router_calls == 2


def test_abort_uses_round_robin_request_mapping(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(True), object, False)
    requests = [_request(f"request-{i}") for i in range(17)]
    asyncio.run(_add_requests(client, requests))

    asyncio.run(client.abort_requests_async(["request-16", "request-0"]))

    assert client.reqs_in_flight["request-0"] == b"engine-0"
    assert client.reqs_in_flight["request-16"] == b"engine-1"
    assert dict(client.aborted_by_engine) == {
        b"engine-0": ["request-0"],
        b"engine-1": ["request-16"],
    }


def test_finished_request_id_can_be_reused_on_next_engine(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(True), object, False)
    request = _request("reused")
    asyncio.run(
        _add_requests(client, [request] +
                      [_request(f"request-{i}") for i in range(1, 16)]))

    asyncio.run(
        type(client).process_engine_outputs(
            client, SimpleNamespace(finished_requests=[request.request_id])))
    asyncio.run(client.add_request_async(request))

    assert client.added_engines[:16] == [b"engine-0"] * 16
    assert client.added_engines[16] == b"engine-1"
    assert client.reqs_in_flight["reused"] == b"engine-1"


def test_patch_is_idempotent(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)

    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    patched_client = core_client.DPLBAsyncMPClient
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()

    assert core_client.DPLBAsyncMPClient is patched_client


def test_patched_client_preserves_default_load_balancing(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()
    client = core_client.DPLBAsyncMPClient(_config(False), object, False)
    client.lb_engines = [[3, 1], [0, 0]]

    asyncio.run(client.add_request_async(_request("request-default")))

    assert client.added_engines == [b"engine-1"]
    assert client.original_router_calls == 1


def test_round_robin_rejects_multiple_api_clients(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()

    with pytest.raises(ValueError, match="requires client_count=1"):
        core_client.DPLBAsyncMPClient(_config(True),
                                      object,
                                      False,
                                      client_count=2)


def test_round_robin_rejects_invalid_cohort_size(monkeypatch):
    router = _load_router(monkeypatch)
    core_client = _install_fake_vllm(monkeypatch)
    router.patch_vllm_dp_client_for_block_diffusion_round_robin()

    with pytest.raises(ValueError, match="requires positive scheduler_config"):
        core_client.DPLBAsyncMPClient(_config(True, max_num_seqs=0), object,
                                      False)


def test_patch_fails_closed_on_unknown_vllm_router_signature(monkeypatch):
    router = _load_router(monkeypatch)
    _install_fake_vllm(monkeypatch, router_has_extra_parameter=True)

    with pytest.raises(RuntimeError,
                       match="Unsupported vLLM DPLBAsyncMPClient"):
        router.patch_vllm_dp_client_for_block_diffusion_round_robin()
