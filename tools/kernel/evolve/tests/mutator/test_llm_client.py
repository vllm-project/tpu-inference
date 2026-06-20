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
"""Tests for LLM client backends."""

import pytest

from tools.kernel.evolve.mutator.llm_client import (CachingClient, LLMClient,
                                                    StubClient)


def test_stub_client_cycles_through_responses():
    client = StubClient(["r1", "r2", "r3"])
    assert client.chat(system="s", user="u") == "r1"
    assert client.chat(system="s", user="u") == "r2"
    assert client.chat(system="s", user="u") == "r3"
    # Wraps around.
    assert client.chat(system="s", user="u") == "r1"


def test_stub_client_implements_protocol():
    client = StubClient(["x"])
    assert isinstance(client, LLMClient)


def test_stub_client_requires_at_least_one_response():
    with pytest.raises(ValueError):
        StubClient([])


def test_caching_client_replays_on_same_input(tmp_path):
    inner = StubClient(["A", "B", "C"])
    cache = CachingClient(inner, cache_path=tmp_path / "cache.jsonl")
    r1 = cache.chat(system="s", user="u")
    r2 = cache.chat(system="s", user="u")
    assert r1 == r2 == "A"
    # Different prompt → next response.
    r3 = cache.chat(system="s", user="different")
    assert r3 == "B"


def test_caching_client_persists_across_instances(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    inner1 = StubClient(["A", "B"])
    cache1 = CachingClient(inner1, cache_path=cache_path)
    cache1.chat(system="s", user="u")
    # New inner with different responses; the cache should still return "A".
    inner2 = StubClient(["X", "Y"])
    cache2 = CachingClient(inner2, cache_path=cache_path)
    assert cache2.chat(system="s", user="u") == "A"
