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
"""Tests for ``VertexAnthropicClient`` (Vertex AI Claude backend).

Network is mocked — we don't hit Vertex in unit tests. The construction
path and retry logic are exercised against a fake ``AnthropicVertex``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class _FakeText:
    type: str = "text"
    text: str = ""


@dataclass
class _FakeMessage:
    content: list[_FakeText]


class _FakeMessages:

    def __init__(self,
                 *,
                 responses: list[str],
                 exceptions: list[Exception] | None = None) -> None:
        self._responses = list(responses)
        self._exceptions = list(exceptions or [])
        self._idx = 0
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kw):
        self.last_kwargs = kw
        if self._exceptions and self._idx < len(self._exceptions):
            err = self._exceptions[self._idx]
            self._idx += 1
            if err is not None:
                raise err
        return _FakeMessage(content=[
            _FakeText(text=self._responses[self._idx % len(self._responses)])
        ])


class _FakeAnthropicVertex:

    def __init__(self,
                 *,
                 project_id: str,
                 region: str,
                 timeout: float = 60.0) -> None:
        self.project_id = project_id
        self.region = region
        self.timeout = timeout
        # Will be installed by the test.
        self.messages = _FakeMessages(responses=["ok"])


def _install_fake_anthropic(monkeypatch, *, fake_module: types.ModuleType):
    """Replace the real ``anthropic`` import for one test."""
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)


@pytest.fixture
def fake_anthropic(monkeypatch):
    mod = types.ModuleType("anthropic")
    mod.AnthropicVertex = _FakeAnthropicVertex

    class APIConnectionError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class InternalServerError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    mod.APIConnectionError = APIConnectionError
    mod.RateLimitError = RateLimitError
    mod.InternalServerError = InternalServerError
    mod.APIStatusError = APIStatusError
    mod.BadRequestError = BadRequestError
    _install_fake_anthropic(monkeypatch, fake_module=mod)
    # Also re-import the wrapper so it picks up the fake module.
    monkeypatch.delitem(sys.modules,
                        "tools.kernel.evolve.mutator.vertex_anthropic",
                        raising=False)
    return mod


def test_vertex_client_builds_with_env_project(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-proj")
    from tools.kernel.evolve.mutator.vertex_anthropic import \
        VertexAnthropicClient
    c = VertexAnthropicClient(model="claude-opus-4-8")
    assert c.whoami() == {
        "model": "claude-opus-4-8",
        "project_id": "test-proj",
        "region": "global",
    }
    assert c.model_id == "vertex:claude-opus-4-8@global"


def test_vertex_client_round_trip(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-proj")
    from tools.kernel.evolve.mutator.vertex_anthropic import \
        VertexAnthropicClient
    c = VertexAnthropicClient(model="claude-opus-4-8")
    # Replace the fake's response.
    c._client.messages = _FakeMessages(responses=["hello back"])
    out = c.chat(system="sys", user="ping", max_tokens=16)
    assert out == "hello back"
    sent = c._client.messages.last_kwargs
    assert sent["model"] == "claude-opus-4-8"
    assert sent["max_tokens"] == 16
    assert sent["system"] == "sys"
    assert sent["messages"][0]["role"] == "user"


def test_vertex_client_raises_when_no_project(monkeypatch, fake_anthropic):
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    from tools.kernel.evolve.mutator.vertex_anthropic import \
        VertexAnthropicClient
    with pytest.raises(RuntimeError, match="project_id not set"):
        VertexAnthropicClient(model="claude-opus-4-8")


def test_vertex_client_retries_on_rate_limit(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-proj")
    from tools.kernel.evolve.mutator.vertex_anthropic import \
        VertexAnthropicClient
    c = VertexAnthropicClient(model="claude-opus-4-8",
                              max_retries=3,
                              retry_backoff_sec=0.01)
    c._client.messages = _FakeMessages(
        responses=["recovered"],
        exceptions=[fake_anthropic.RateLimitError("slow down"), None])
    out = c.chat(system="s", user="u")
    assert out == "recovered"


def test_vertex_client_raises_after_max_retries(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "test-proj")
    from tools.kernel.evolve.mutator.vertex_anthropic import \
        VertexAnthropicClient
    c = VertexAnthropicClient(model="claude-opus-4-8",
                              max_retries=2,
                              retry_backoff_sec=0.01)
    c._client.messages = _FakeMessages(
        responses=["never"],
        exceptions=[
            fake_anthropic.APIConnectionError("net"),
            fake_anthropic.APIConnectionError("net")
        ])
    with pytest.raises(fake_anthropic.APIConnectionError):
        c.chat(system="s", user="u")
