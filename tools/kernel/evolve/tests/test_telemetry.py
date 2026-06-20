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
"""Tests for telemetry writer + analyzer."""

import time

from tools.kernel.evolve.telemetry.writer import (TelemetryEvent,
                                                  TelemetryWriter, load_events,
                                                  summarize)


def test_writer_appends_to_disk(tmp_path):
    p = tmp_path / "tel.jsonl"
    with TelemetryWriter(p) as w:
        w.emit(
            TelemetryEvent(timestamp=time.time(),
                           run_id="r",
                           kernel="k",
                           shape_key="s",
                           genome_id="g1",
                           parent_ids=[],
                           generation=1,
                           island_id=0,
                           diff_summary="",
                           status="VERIFIED",
                           fitness_ns=100.0,
                           p50_ns=90,
                           p95_ns=120,
                           cosine=1.0,
                           max_abs_diff=0.001,
                           wall_time_s=5.0))
    assert p.exists()
    events = load_events(p)
    assert len(events) == 1
    assert events[0]["fitness_ns"] == 100.0


def test_writer_survives_crash_safely(tmp_path):
    p = tmp_path / "tel.jsonl"
    w = TelemetryWriter(p)
    w.emit(
        TelemetryEvent(timestamp=time.time(),
                       run_id="r",
                       kernel="k",
                       shape_key="s",
                       genome_id="g1",
                       parent_ids=[],
                       generation=0,
                       island_id=0,
                       diff_summary="",
                       status="VERIFIED",
                       fitness_ns=100.0,
                       p50_ns=None,
                       p95_ns=None,
                       cosine=None,
                       max_abs_diff=None,
                       wall_time_s=1.0))
    # Don't close (simulate crash).
    events = load_events(p)
    assert len(events) >= 1


def test_summarize_groups_by_kernel_and_shape(tmp_path):
    p = tmp_path / "tel.jsonl"
    with TelemetryWriter(p) as w:
        for fitness in [100.0, 200.0, 50.0]:
            w.emit(
                TelemetryEvent(timestamp=time.time(),
                               run_id="r",
                               kernel="rpa",
                               shape_key="shape_A",
                               genome_id="g",
                               parent_ids=[],
                               generation=1,
                               island_id=0,
                               diff_summary="",
                               status="VERIFIED",
                               fitness_ns=fitness,
                               p50_ns=None,
                               p95_ns=None,
                               cosine=None,
                               max_abs_diff=None,
                               wall_time_s=1.0))
        w.emit(
            TelemetryEvent(timestamp=time.time(),
                           run_id="r",
                           kernel="rpa",
                           shape_key="shape_B",
                           genome_id="g",
                           parent_ids=[],
                           generation=1,
                           island_id=0,
                           diff_summary="",
                           status="FAILED_NUMERICS",
                           fitness_ns=None,
                           p50_ns=None,
                           p95_ns=None,
                           cosine=None,
                           max_abs_diff=None,
                           wall_time_s=1.0))
    s = summarize(load_events(p))
    assert s["kernels"]["rpa"]["shape_A"]["verified"] == 3
    assert s["kernels"]["rpa"]["shape_A"]["best_fitness_ns"] == 50.0
    assert s["kernels"]["rpa"]["shape_B"]["verified"] == 0
