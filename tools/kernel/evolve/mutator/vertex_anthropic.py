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
"""Claude on Vertex AI as an ``LLMClient``-shaped mutator.

Uses ``anthropic.AnthropicVertex`` so authentication piggybacks on the
``gcloud auth application-default`` chain — no separate API key needed.
Drop-in for ``AnthropicClient`` everywhere the orchestrator expects an
``LLMClient``.

Model IDs on Vertex follow the ``claude-<family>@<date>`` convention
(e.g. ``claude-opus-4-5@20250929``). The default below picks the strongest
generally-available model at the time of writing; override per call site
when you have specific quality/cost trade-offs.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# The ``global`` region is Anthropic-on-Vertex's auto-routing endpoint — the
# documented default. Specific region IDs (``us-east5`` etc.) also work but
# require per-region model enablement, which is more brittle than ``global``.
_DEFAULT_REGION = "global"
_DEFAULT_PROJECT_ENV_KEYS = ("ANTHROPIC_VERTEX_PROJECT_ID",
                             "GOOGLE_CLOUD_PROJECT")


class VertexAnthropicClient:
    """LLMClient-shaped wrapper around ``anthropic.AnthropicVertex``."""

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-8",
        project_id: str | None = None,
        region: str | None = None,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        timeout_sec: float = 120.0,
        accept_temperature: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "anthropic SDK not installed. `pip install anthropic`.") from e
        if not hasattr(anthropic, "AnthropicVertex"):
            raise RuntimeError(
                "Your anthropic SDK is too old; needs >=0.30 with "
                "AnthropicVertex.")
        proj = project_id
        if proj is None:
            for env in _DEFAULT_PROJECT_ENV_KEYS:
                proj = os.environ.get(env)
                if proj:
                    break
        if not proj:
            raise RuntimeError(
                "VertexAnthropicClient: project_id not set and none of "
                f"{_DEFAULT_PROJECT_ENV_KEYS} found in env.")
        reg = region or os.environ.get(
            "ANTHROPIC_VERTEX_REGION") or _DEFAULT_REGION
        self._anthropic = anthropic
        self._client = anthropic.AnthropicVertex(project_id=proj,
                                                 region=reg,
                                                 timeout=timeout_sec)
        self._model = model
        self._project_id = proj
        self._region = reg
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        # Opus 4.x on Vertex returns 400 "temperature is deprecated"; we
        # auto-detect this and skip the kwarg on subsequent calls.
        self._accept_temperature = accept_temperature

    @property
    def model_id(self) -> str:
        return f"vertex:{self._model}@{self._region}"

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kw = {
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{
                        "role": "user",
                        "content": user
                    }],
                }
                if temperature is not None and self._accept_temperature:
                    kw["temperature"] = temperature
                resp = self._client.messages.create(**kw)
                parts: list[str] = []
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                return "".join(parts)
            except self._anthropic.BadRequestError as err:
                # Opus 4.x rejects ``temperature``; flip the flag and retry
                # immediately without the kwarg.
                if (self._accept_temperature
                        and "temperature" in str(err).lower()
                        and "deprecated" in str(err).lower()):
                    self._accept_temperature = False
                    logger.warning(
                        "Vertex %s deprecated `temperature` — auto-disabling "
                        "for this client", self._model)
                    continue
                last_err = err
                wait = self.retry_backoff_sec * (2**attempt)
                logger.warning("Vertex %s 400 (%s); retrying in %.1fs",
                               self._model,
                               str(err)[:140], wait)
                time.sleep(wait)
            except (self._anthropic.APIConnectionError,
                    self._anthropic.RateLimitError,
                    self._anthropic.InternalServerError,
                    self._anthropic.APIStatusError) as err:
                last_err = err
                wait = self.retry_backoff_sec * (2**attempt)
                logger.warning(
                    "Vertex %s attempt %d/%d failed (%s); retrying in %.1fs",
                    self._model, attempt + 1, self.max_retries, err, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err

    # Convenience for diagnostics
    def whoami(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "project_id": self._project_id,
            "region": self._region,
        }
