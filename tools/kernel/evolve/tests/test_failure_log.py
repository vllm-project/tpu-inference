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
"""Tests for the failure-learning log."""

from tools.kernel.evolve.mutator.failure_log import FailureLog


def test_verified_rate():
    log = FailureLog()
    for _ in range(7):
        log.record(rule_name="r", status="VERIFIED", fitness_ns=100.0)
    for _ in range(3):
        log.record(rule_name="r", status="FAILED_NUMERICS")
    assert log.verified_rate("r") == 0.7


def test_dead_rule_detection():
    log = FailureLog()
    # Below min_observations: not dead even if all fail.
    for _ in range(5):
        log.record(rule_name="few", status="FAILED_DIFF")
    assert not log.is_dead_rule("few")
    # Enough observations and zero verified: definitely dead.
    for _ in range(12):
        log.record(rule_name="bad", status="FAILED_NUMERICS")
    assert log.is_dead_rule("bad")
    # Healthy rule: not dead.
    for _ in range(10):
        log.record(rule_name="good", status="VERIFIED", fitness_ns=1.0)
    assert not log.is_dead_rule("good")


def test_anti_patterns_summary():
    log = FailureLog()
    for _ in range(2):
        log.record(rule_name="bad_block", status="FAILED_NUMERICS")
        log.record(rule_name="bad_block", status="FAILED_COMPILE")
    for _ in range(8):
        log.record(rule_name="good_block", status="VERIFIED", fitness_ns=1.0)
    patterns = log.anti_patterns()
    assert any("bad_block" in p for p in patterns)
    assert not any("good_block" in p for p in patterns)


def test_jsonl_persistence(tmp_path):
    p = tmp_path / "fl.json"
    log = FailureLog(persist_path=p)
    log.record(rule_name="r1", status="VERIFIED", fitness_ns=500.0)
    log.record(rule_name="r1", status="FAILED_RUN")
    log.record(rule_name="r2", status="VERIFIED", fitness_ns=300.0)
    # Reload.
    log2 = FailureLog(persist_path=p)
    assert log2.verified_rate("r1") == 0.5
    assert log2.verified_rate("r2") == 1.0


def test_best_fitness_tracked():
    log = FailureLog()
    log.record(rule_name="r", status="VERIFIED", fitness_ns=500.0)
    log.record(rule_name="r", status="VERIFIED", fitness_ns=300.0)
    log.record(rule_name="r", status="VERIFIED", fitness_ns=700.0)
    stats = log.all_stats()
    assert stats[0].best_fitness_ns == 300.0
