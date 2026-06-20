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
"""Local LLM backend for the mutator.

Speaks the OpenAI Chat Completions wire format (``POST /v1/chat/completions``),
which both vLLM and OpenAI's API implement. Means we can plug in:

* A local vLLM server hosting Qwen3-32B (or 0.6B in the no-quality-required
  case) for self-evolving kernels on the same TPU.
* A SGLang server hosting Llama3-70B.
* Any OpenAI-compatible endpoint.

No SDK dependency — uses ``urllib.request`` directly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LocalLlmClient:
    """LLMClient-shaped client for any OpenAI-compatible chat endpoint."""
    endpoint: str  # e.g. 'http://localhost:8000/v1/chat/completions'
    model: str  # e.g. 'Qwen/Qwen3-0.6B'
    api_key: str | None = None  # optional bearer token
    temperature: float = 0.4
    timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0

    @property
    def model_id(self) -> str:
        return f"local:{self.model}"

    def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> str:
        body = json.dumps({
            "model":
            self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                },
            ],
            "temperature":
            self.temperature,
            "max_tokens":
            max_tokens,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        key = self.api_key or os.environ.get("LOCAL_LLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self.endpoint,
                                         data=body,
                                         method="POST",
                                         headers=headers)
            try:
                with urllib.request.urlopen(req,
                                            timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    return self._extract_text(payload)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, json.JSONDecodeError) as err:
                last_err = err
                wait = self.retry_backoff_s * (2**attempt)
                logger.warning(
                    "LocalLlmClient attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1, self.max_retries, err, wait)
                time.sleep(wait)
        assert last_err is not None
        raise last_err

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"unexpected payload: {payload}")
        msg = choices[0].get("message") or {}
        # vLLM and OpenAI both place text in ``message.content``.
        return msg.get("content", "") or ""
