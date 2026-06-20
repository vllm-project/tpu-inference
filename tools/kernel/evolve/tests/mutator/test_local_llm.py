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
"""Tests for the local-LLM (OpenAI-compatible) client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from tools.kernel.evolve.mutator.local_llm import LocalLlmClient


def _make_server(responder):
    received: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body)
            received["last_payload"] = payload
            received["last_headers"] = dict(self.headers)
            try:
                response = responder(payload)
            except Exception as err:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(err).encode())
                return
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            return

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, port, received


def test_local_llm_round_trip():

    def responder(payload):
        return {
            "id":
            "test",
            "object":
            "chat.completion",
            "model":
            payload["model"],
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello back"
                },
                "finish_reason": "stop",
            }],
        }

    srv, port, received = _make_server(responder)
    try:
        client = LocalLlmClient(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model="Qwen/Qwen3-0.6B")
        out = client.chat(system="sys", user="hi", max_tokens=64)
        assert out == "hello back"
        sent = received["last_payload"]
        assert sent["model"] == "Qwen/Qwen3-0.6B"
        assert sent["max_tokens"] == 64
        assert sent["messages"][0]["role"] == "system"
        assert sent["messages"][1]["role"] == "user"
    finally:
        srv.shutdown()


def test_local_llm_propagates_auth_header():

    def responder(payload):
        return {"choices": [{"message": {"content": "ok"}}]}

    srv, port, received = _make_server(responder)
    try:
        client = LocalLlmClient(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model="m",
            api_key="secret-token")
        client.chat(system="s", user="u")
        assert received["last_headers"].get(
            "Authorization") == "Bearer secret-token"
    finally:
        srv.shutdown()


def test_local_llm_retries_on_failure_then_succeeds():
    state = {"calls": 0}

    def responder(payload):
        state["calls"] += 1
        if state["calls"] < 2:
            raise RuntimeError("first call fails")
        return {"choices": [{"message": {"content": "ok"}}]}

    srv, port, received = _make_server(responder)
    try:
        client = LocalLlmClient(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model="m",
            max_retries=3,
            retry_backoff_s=0.01)
        out = client.chat(system="s", user="u")
        assert out == "ok"
        assert state["calls"] == 2
    finally:
        srv.shutdown()


def test_local_llm_raises_after_max_retries():

    def responder(payload):
        raise RuntimeError("always fails")

    srv, port, _ = _make_server(responder)
    try:
        client = LocalLlmClient(
            endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
            model="m",
            max_retries=2,
            retry_backoff_s=0.01)
        with pytest.raises(Exception):
            client.chat(system="s", user="u")
    finally:
        srv.shutdown()
