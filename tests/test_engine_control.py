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
"""Unit tests for TPU engine lifecycle control and HBM preservation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from tpu_inference.core.control import EngineControlManager, get_control_router


class MockEngine:

    def __init__(self):
        self.pause_calls = []
        self.resume_calls = []
        self._is_paused = False
        self.model_executor = MagicMock()
        self.model_executor.synchronize_device = AsyncMock()

    async def pause_generation(self,
                               mode: str = "keep",
                               clear_cache: bool = False):
        self.pause_calls.append({"mode": mode, "clear_cache": clear_cache})
        self._is_paused = True

    async def resume_generation(self):
        self.resume_calls.append(True)
        self._is_paused = False

    async def is_paused(self) -> bool:
        return self._is_paused

    def get_num_waiting_requests(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_pause_preserves_tpu_hbm():
    mock_engine = MockEngine()
    manager = EngineControlManager(mock_engine)

    # Execute pause
    result = await manager.pause(drain_timeout_s=5.0)

    # Assert pause status and HBM preservation
    assert result["status"] == "PAUSED"
    assert result["hbm_retained"] is True
    assert manager.is_paused is True

    # Critical verification: clear_cache MUST be False and mode MUST be "keep"
    assert len(mock_engine.pause_calls) == 1
    assert mock_engine.pause_calls[0]["mode"] == "keep"
    assert mock_engine.pause_calls[0]["clear_cache"] is False

    # Verify device synchronization was called to flush inflight DMA work
    mock_engine.model_executor.synchronize_device.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_idempotency():
    mock_engine = MockEngine()
    manager = EngineControlManager(mock_engine)

    res1 = await manager.pause(drain_timeout_s=5.0)
    assert res1["status"] == "PAUSED"

    # Second call should be a no-op
    res2 = await manager.pause(drain_timeout_s=5.0)
    assert res2["status"] == "PAUSED"
    assert "already paused" in res2["message"].lower()
    assert len(mock_engine.pause_calls) == 1


@pytest.mark.asyncio
async def test_resume_generation():
    mock_engine = MockEngine()
    manager = EngineControlManager(mock_engine)

    await manager.pause(drain_timeout_s=5.0)
    assert manager.is_paused is True

    res = await manager.resume()
    assert res["status"] == "SERVING"
    assert manager.is_paused is False
    assert len(mock_engine.resume_calls) == 1


@pytest.mark.asyncio
async def test_status_reporting():
    mock_engine = MockEngine()
    manager = EngineControlManager(mock_engine)

    st1 = await manager.status()
    assert st1["status"] == "SERVING"
    assert st1["is_paused"] is False
    assert st1["hbm_retained"] is True

    await manager.pause(drain_timeout_s=5.0)
    st2 = await manager.status()
    assert st2["status"] == "PAUSED"
    assert st2["is_paused"] is True


def test_fastapi_control_endpoints():
    mock_engine = MockEngine()
    app = FastAPI()
    router = get_control_router(mock_engine)
    app.include_router(router)

    client = TestClient(app)

    # Initial liveness and readiness
    live_resp = client.get("/ctl/health/live")
    assert live_resp.status_code == 200

    ready_resp = client.get("/ctl/health/ready")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["status"] == "ready"

    # Trigger pause
    pause_resp = client.post("/ctl/pause")
    assert pause_resp.status_code == 200
    assert pause_resp.json()["status"] == "PAUSED"
    assert pause_resp.json()["hbm_retained"] is True

    # Readiness probe must fail with 503 while paused
    ready_while_paused = client.get("/ctl/health/ready")
    assert ready_while_paused.status_code == 503

    # Status must report PAUSED
    status_resp = client.get("/ctl/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "PAUSED"

    # Trigger resume
    resume_resp = client.post("/ctl/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["status"] == "SERVING"

    # Readiness probe recovers to 200
    ready_after_resume = client.get("/ctl/health/ready")
    assert ready_after_resume.status_code == 200
