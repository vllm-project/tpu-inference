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
"""Tests for the e2e benchmark harness — CPU-only paths only."""

import pytest

from tools.kernel.evolve import e2e_benchmark
from tools.kernel.evolve.e2e_benchmark import (_parse_throughput, _restore,
                                               run_e2e_benchmark)

_TINY_DIFF = ("--- a/x/y.py\n"
              "+++ b/x/y.py\n"
              "@@ -1,3 +1,3 @@\n"
              " def f():\n"
              "-    return 1\n"
              "+    return 2\n")


def test_parse_throughput_extracts_output_tps():
    s = ("Throughput: 1.23 requests/s, 456.78 total tokens/s, "
         "987.65 output tokens/s")
    assert _parse_throughput(s) == pytest.approx(987.65)


def test_parse_throughput_falls_back_to_alt_form():
    s = "Achieved 543.21 tokens/s on this trial"
    assert _parse_throughput(s) == pytest.approx(543.21)


def test_parse_throughput_returns_none_on_no_match():
    assert _parse_throughput("nothing here") is None


def test_restore_raises_on_sha_mismatch(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n")
    import hashlib
    orig_sha = hashlib.sha256(b"original\n").hexdigest()
    # Tamper between snapshot and restore
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        _restore(f, b"WRONG\n", orig_sha)


def test_run_e2e_restores_kernel_on_subprocess_failure(tmp_path, monkeypatch):
    kernel = tmp_path / "k.py"
    original = "def f():\n    return 1\n"
    kernel.write_text(original)
    diff = tmp_path / "d.diff"
    diff.write_text(_TINY_DIFF)

    def _boom(**kw):
        raise RuntimeError("vllm exploded")

    monkeypatch.setattr(e2e_benchmark, "_bench_n_trials", _boom)
    with pytest.raises(RuntimeError, match="exploded"):
        run_e2e_benchmark(model="fake",
                          kernel_path=kernel,
                          diff_path=diff,
                          tensor_parallel=1,
                          max_model_len=128,
                          max_tokens=8,
                          num_prompts=1,
                          n_trials=1)
    assert kernel.read_text() == original


def test_run_e2e_happy_path_returns_comparison(tmp_path, monkeypatch):
    kernel = tmp_path / "k.py"
    kernel.write_text("def f():\n    return 1\n")
    diff = tmp_path / "d.diff"
    diff.write_text(_TINY_DIFF)
    state = {"calls": 0}

    def _stub(**kw):
        state["calls"] += 1
        # First batch = baseline; second = patched (higher throughput)
        return ([100.0, 102.0, 99.0]
                if state["calls"] == 1 else [115.0, 117.0, 114.0])

    monkeypatch.setattr(e2e_benchmark, "_bench_n_trials", _stub)
    cmp = run_e2e_benchmark(model="fake",
                            kernel_path=kernel,
                            diff_path=diff,
                            tensor_parallel=1,
                            max_model_len=128,
                            max_tokens=8,
                            num_prompts=1,
                            n_trials=3)
    assert cmp.baseline.mean_tok_per_s == pytest.approx(100.333, rel=1e-3)
    assert cmp.patched.mean_tok_per_s == pytest.approx(115.333, rel=1e-3)
    assert cmp.speedup_mean > 1.10  # ~+15%
    # kernel restored
    assert kernel.read_text() == "def f():\n    return 1\n"
