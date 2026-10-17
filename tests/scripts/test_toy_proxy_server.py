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

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

_REPO_ROOT = Path(__file__).parents[2]
_PROXY_PATHS = [
    _REPO_ROOT / ".buildkite" / "kubernetes" / "toy_proxy_server.py",
    _REPO_ROOT / "examples" / "disagg" / "toy_proxy_server.py",
]


def _load_proxy_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"toy_proxy_{path.parent.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_scripts_are_identical():
    assert _PROXY_PATHS[0].read_text() == _PROXY_PATHS[1].read_text()


@pytest.mark.parametrize("proxy_path",
                         _PROXY_PATHS,
                         ids=["buildkite", "examples"])
@pytest.mark.asyncio
async def test_lifespan_configures_unlimited_connections(proxy_path: Path):
    module = _load_proxy_module(proxy_path)
    module.global_args = types.SimpleNamespace(
        prefiller_instances=[("127.0.0.1", 8400)],
        decoder_instances=[("127.0.0.1", 9400)],
    )
    expected_limits = httpx.Limits(
        max_connections=None,
        max_keepalive_connections=None,
    )

    with patch.object(module.httpx, "AsyncClient",
                      wraps=httpx.AsyncClient) as mock_client_cls:
        async with module.lifespan(module.app):
            assert len(module.app.state.prefill_clients) == 1
            assert len(module.app.state.decode_clients) == 1
            assert mock_client_cls.call_count == 2
            for call in mock_client_cls.call_args_list:
                assert call.kwargs["timeout"] is None
                assert call.kwargs["limits"] == expected_limits

            for client_info in (module.app.state.prefill_clients +
                                module.app.state.decode_clients):
                client = client_info["client"]
                assert client.timeout == httpx.Timeout(None)
                pool = client._transport._pool
                assert pool._max_connections == sys.maxsize
                assert pool._max_keepalive_connections == sys.maxsize

    for client_info in (module.app.state.prefill_clients +
                        module.app.state.decode_clients):
        assert client_info["client"].is_closed


@pytest.mark.parametrize("proxy_path",
                         _PROXY_PATHS,
                         ids=["buildkite", "examples"])
@pytest.mark.asyncio
async def test_send_request_to_prefill_reads_and_closes_response(
        proxy_path: Path):
    module = _load_proxy_module(proxy_path)
    response = MagicMock(spec=httpx.Response)
    response.aread = AsyncMock(
        return_value=b'{"kv_transfer_params": {"uuid": "1"}}')
    response.aclose = AsyncMock()

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)

    orig_req = {
        "prompt": "hello",
        "stream": True,
        "max_tokens": 128,
        "max_completion_tokens": 128,
        "stream_options": {
            "include_usage": True
        },
    }
    ret = await module.send_request_to_prefill({"client": client},
                                               "/v1/completions", orig_req,
                                               "req-1")

    assert ret is response
    assert orig_req["stream"] is True
    assert orig_req["max_tokens"] == 128
    assert "stream_options" in orig_req

    sent_json = client.post.call_args.kwargs["json"]
    assert sent_json["stream"] is False
    assert sent_json["max_tokens"] == 1
    assert sent_json["max_completion_tokens"] == 1
    assert "stream_options" not in sent_json

    response.raise_for_status.assert_called_once()
    response.aread.assert_awaited_once()
    response.aclose.assert_awaited_once()


@pytest.mark.parametrize("proxy_path",
                         _PROXY_PATHS,
                         ids=["buildkite", "examples"])
@pytest.mark.asyncio
async def test_send_request_to_prefill_closes_response_on_error(
        proxy_path: Path):
    module = _load_proxy_module(proxy_path)
    request = httpx.Request("POST", "http://127.0.0.1:8400/v1/completions")
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=request, response=response)
    response.aread = AsyncMock()
    response.aclose = AsyncMock()

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=response)

    with pytest.raises(httpx.HTTPStatusError):
        await module.send_request_to_prefill({"client": client},
                                             "/v1/completions",
                                             {"prompt": "hi"}, "req-err")

    response.aread.assert_not_awaited()
    response.aclose.assert_awaited_once()


async def _read_http_request(reader: asyncio.StreamReader) -> dict:
    header_bytes = await reader.readuntil(b"\r\n\r\n")
    content_length = 0
    for line in header_bytes.decode().split("\r\n")[1:]:
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
            break
    body = await reader.readexactly(content_length
                                    ) if content_length else b"{}"
    return json.loads(body.decode())


@pytest.mark.parametrize("proxy_path",
                         _PROXY_PATHS,
                         ids=["buildkite", "examples"])
@pytest.mark.asyncio
async def test_high_concurrency_streaming_not_throttled_at_100(
        proxy_path: Path):
    module = _load_proxy_module(proxy_path)
    concurrency = 128
    active_decode = 0
    all_decode_connected = asyncio.Event()
    received_kv_uuids = []

    async def handle_prefill(reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter):
        try:
            req_body = await _read_http_request(reader)
            payload = json.dumps({
                "kv_transfer_params": {
                    "uuid": f"kv-{req_body['req_idx']}"
                }
            }).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" +
                f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()

    async def handle_decode(reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter):
        nonlocal active_decode
        try:
            req_body = await _read_http_request(reader)
            received_kv_uuids.append(
                req_body.get("kv_transfer_params", {}).get("uuid"))
            active_decode += 1
            if active_decode >= concurrency:
                all_decode_connected.set()
            await all_decode_connected.wait()
            payload = f"data: {req_body['req_idx']}\n\n".encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" +
                f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()

    prefill_server = await asyncio.start_server(handle_prefill,
                                                "127.0.0.1",
                                                0,
                                                backlog=256)
    decode_server = await asyncio.start_server(handle_decode,
                                               "127.0.0.1",
                                               0,
                                               backlog=256)
    prefill_port = prefill_server.sockets[0].getsockname()[1]
    decode_port = decode_server.sockets[0].getsockname()[1]

    module.global_args = types.SimpleNamespace(
        prefiller_instances=[("127.0.0.1", prefill_port)],
        decoder_instances=[("127.0.0.1", decode_port)],
    )

    try:
        async with module.lifespan(module.app):

            async def run_one(idx: int) -> bytes:
                scope_req = MagicMock()
                scope_req.app = module.app
                scope_req.json = AsyncMock(return_value={
                    "prompt": "test",
                    "req_idx": idx,
                    "stream": True,
                })
                streaming_resp = await module._handle_completions(
                    "/v1/completions", scope_req)
                chunks = [
                    chunk async for chunk in streaming_resp.body_iterator
                ]
                return b"".join(chunks)

            tasks = [
                asyncio.create_task(run_one(i)) for i in range(concurrency)
            ]
            results = await asyncio.wait_for(asyncio.gather(*tasks),
                                             timeout=5.0)

        assert active_decode == concurrency
        assert sorted(received_kv_uuids) == sorted(f"kv-{i}"
                                                   for i in range(concurrency))
        assert results == [
            f"data: {i}\n\n".encode() for i in range(concurrency)
        ]
    finally:
        all_decode_connected.set()
        prefill_server.close()
        decode_server.close()
        await prefill_server.wait_closed()
        await decode_server.wait_closed()
