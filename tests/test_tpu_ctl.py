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
"""Unit tests for tpu-ctl CLI tool."""

import os
from unittest.mock import patch

import pytest

from tpu_inference.tools import ctl


def test_get_worker_rank_detection(monkeypatch):
    # Default without env vars
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("JOB_COMPLETION_INDEX", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    assert ctl.get_worker_rank() == 0

    # From TPU_WORKER_ID
    monkeypatch.setenv("TPU_WORKER_ID", "2")
    assert ctl.get_worker_rank() == 2

    # From JOB_COMPLETION_INDEX
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "3")
    assert ctl.get_worker_rank() == 3

    # From hostname pattern
    monkeypatch.delenv("JOB_COMPLETION_INDEX", raising=False)
    monkeypatch.setenv("HOSTNAME", "tpu-job-1-xyz")
    assert ctl.get_worker_rank() == 1


def test_worker_silent_noop_on_pause(monkeypatch, capsys):
    monkeypatch.setenv("TPU_WORKER_ID", "1")
    code = ctl.main(["pause"])
    assert code == 0
    captured = capsys.readouterr()
    assert "[CTL-Worker] Silent local no-op on worker pod (Rank 1)." in captured.out


def test_worker_silent_noop_on_resume(monkeypatch, capsys):
    monkeypatch.setenv("TPU_WORKER_ID", "2")
    code = ctl.main(["resume"])
    assert code == 0
    captured = capsys.readouterr()
    assert "[CTL-Worker] Silent local no-op on worker pod (Rank 2)." in captured.out


def test_master_pause_success(monkeypatch, capsys):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(200, {
                   "status": "PAUSED",
                   "hbm_retained": True
               })):
        code = ctl.main(["pause", "--timeout=10"])
        assert code == 0
        captured = capsys.readouterr()
        assert "[CTL-Master] Success: Engine paused." in captured.out


def test_master_pause_error(monkeypatch, capsys):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(504, {
                   "detail": "Drain timed out"
               })):
        code = ctl.main(["pause", "--timeout=10"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error pausing engine (HTTP 504)" in captured.out


def test_master_resume_success(monkeypatch, capsys):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(200, {
                   "status": "SERVING"
               })):
        code = ctl.main(["resume"])
        assert code == 0
        captured = capsys.readouterr()
        assert "[CTL-Master] Success: Serving resumed with 0ms warmup." in captured.out


def test_probe_readiness_serving(monkeypatch):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(200, {
                   "status": "ready"
               })):
        assert ctl.main(["probe", "--type=readiness"]) == 0


def test_probe_readiness_paused(monkeypatch):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(503, {
                   "detail": "Paused"
               })):
        assert ctl.main(["probe", "--type=readiness"]) == 1


def test_probe_status_paused(monkeypatch):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(200, {
                   "status": "PAUSED"
               })):
        assert ctl.main(["probe", "--type=paused"]) == 0
        assert ctl.main(["probe", "--type=running"]) == 1


def test_probe_status_running(monkeypatch):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    with patch("tpu_inference.tools.ctl.call_api",
               return_value=(200, {
                   "status": "SERVING"
               })):
        assert ctl.main(["probe", "--type=running"]) == 0
        assert ctl.main(["probe", "--type=paused"]) == 1
