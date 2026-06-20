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
"""Unit tests for the benchmark harness."""

import time

import numpy as np
import pytest

from tools.kernel.tuner.v1.bench.harness import BenchResult, block_all, measure


def test_measure_reports_p50_p95():
    delays = iter([0.001] * 30)  # 1 ms per call

    def fn():
        next(delays)
        time.sleep(0.001)
        return np.zeros((4, ), dtype=np.float32)

    res = measure(fn, warmup=2, iters=10)
    assert isinstance(res, BenchResult)
    assert res.iters == 9  # cold-start excluded
    assert res.p50_ns > 0
    assert res.p95_ns >= res.p50_ns
    assert res.mean_ns > 0


def test_measure_excludes_cold_start():
    rng = np.random.default_rng(0)
    timings: list[float] = []

    def fn():
        timings.append(rng.random())
        return np.zeros((1, ))

    res = measure(fn, warmup=1, iters=4, exclude_cold_start=True)
    # warmup=1 + iters=4 + 1 cold-start excluded = 4 calls total in `timings`
    # but the harness drops the first timed iter from the summary.
    assert res.iters == 3
    assert res.cold_start_excluded


def test_measure_with_iters_one_no_cold_start_exclusion():

    def fn():
        return np.zeros((1, ))

    res = measure(fn, warmup=0, iters=1, exclude_cold_start=True)
    # Can't drop cold-start with only one timed iter; keep it.
    assert res.iters == 1
    assert not res.cold_start_excluded


def test_measure_iters_must_be_positive():

    def fn():
        return np.zeros((1, ))

    with pytest.raises(ValueError):
        measure(fn, iters=0)


def test_block_all_recurses_into_tuple_and_dict():
    a = np.zeros((4, ))
    block_all((a, [a, a], {"x": a}))


def test_measure_returns_last_output():
    counter = {"i": 0}

    def fn():
        counter["i"] += 1
        return np.full((1, ), float(counter["i"]), dtype=np.float32)

    res = measure(fn, warmup=2, iters=5)
    # 2 warmup + 5 iters → counter == 7
    assert counter["i"] == 7
    assert float(res.output[0]) == 7.0
