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
"""LLM client backends for the mutator.

Three implementations ship:

* ``AnthropicClient`` — real Claude API. Requires ``ANTHROPIC_API_KEY``.
* ``StubClient`` — deterministic playback of pre-canned responses. Used in
  unit tests and the offline demo so the loop is exercisable without API
  credentials.
* ``CachingClient`` — wraps another client with a JSONL cache keyed by the
  ``(system, user)`` hash. Saves API spend during iteration.

All clients implement the same minimal Protocol.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal API the mutator depends on."""

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> str:
        ...

    @property
    def model_id(self) -> str:
        ...


class AnthropicClient:
    """Real Anthropic Messages API client.

    Defaults to Claude Opus 4.8 (the most capable available family member at
    time of writing). Override via ``model`` if you have a tighter latency
    budget — Sonnet 4.6 is the natural mid-tier choice; Haiku 4.5 is the
    cheap-fast option for the critic.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-8",
        api_key: str | None = None,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - tested via import path
            raise RuntimeError(
                "anthropic SDK not installed. `pip install anthropic`.") from e
        self._anthropic = anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; either export it or pass api_key=..."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

    @property
    def model_id(self) -> str:
        return self._model

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{
                        "role": "user",
                        "content": user
                    }],
                )
                # Concatenate text blocks; ignore tool-use blocks (we don't
                # ask the model to use tools).
                parts: list[str] = []
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                return "".join(parts)
            except (self._anthropic.APIConnectionError,
                    self._anthropic.RateLimitError,
                    self._anthropic.InternalServerError) as err:
                last_err = err
                wait = self.retry_backoff_sec * (2**attempt)
                logger.warning(
                    "Anthropic %s attempt %d/%d failed (%s); retrying in %.1fs",
                    self._model, attempt + 1, self.max_retries, err, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err


class StubClient:
    """Deterministic playback for tests and offline demos.

    Cycles through ``responses`` in order. Useful for unit-testing the
    orchestrator without paying for tokens and for shipping a
    reproducible end-to-end demo.
    """

    def __init__(
        self,
        responses: list[str],
        *,
        model_id: str = "stub",
    ) -> None:
        if not responses:
            raise ValueError("StubClient needs at least one response")
        self._responses = list(responses)
        self._idx = 0
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def chat(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        del system, user, max_tokens
        out = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return out


class CachingClient:
    """JSONL cache wrapping any LLMClient. Replays on hit, calls on miss."""

    def __init__(
        self,
        inner: LLMClient,
        cache_path: str | os.PathLike,
    ) -> None:
        self._inner = inner
        self._path = Path(cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, str] = {}
        if self._path.exists():
            with self._path.open("r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["response"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    @property
    def model_id(self) -> str:
        return f"cached:{self._inner.model_id}"

    def chat(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        key = self._key(system, user, max_tokens)
        if key in self._cache:
            return self._cache[key]
        resp = self._inner.chat(system=system,
                                user=user,
                                max_tokens=max_tokens)
        self._cache[key] = resp
        with self._path.open("a") as f:
            f.write(
                json.dumps({
                    "key": key,
                    "model": self._inner.model_id,
                    "response": resp,
                }) + "\n")
        return resp

    @staticmethod
    def _key(system: str, user: str, max_tokens: int) -> str:
        h = hashlib.sha256()
        h.update(system.encode("utf-8"))
        h.update(b"\n--\n")
        h.update(user.encode("utf-8"))
        h.update(b"\n--\n")
        h.update(str(max_tokens).encode("utf-8"))
        return h.hexdigest()
