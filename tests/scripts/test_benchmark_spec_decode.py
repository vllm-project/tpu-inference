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

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).parents[2] / "scripts" / "benchmarking" /
           "spec_decode" / "benchmark_spec_decode.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_spec_decode",
                                               _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
# dataclasses resolve string annotations through sys.modules.
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def _counter(name, value):
    return types.SimpleNamespace(name=name, value=value)


def _vector(name, values):
    return types.SimpleNamespace(name=name, values=values)


def _parse(*argv):
    return bench.create_parser().parse_args(list(argv))


# --------------------------------------------------------------------------
# Metric aggregation
# --------------------------------------------------------------------------


def test_collect_sums_counters_across_engines():
    # Two DP engines report separate series; per-position vectors differ in
    # length when one engine never reached the last draft position.
    metrics = [
        _counter(bench.NUM_DRAFTS, 10),
        _counter(bench.NUM_DRAFTS, 5),
        _counter(bench.NUM_DRAFT_TOKENS, 30),
        _counter(bench.NUM_DRAFT_TOKENS, 15),
        _counter(bench.NUM_ACCEPTED_TOKENS, 20),
        _counter(bench.NUM_ACCEPTED_TOKENS, 6),
        _vector(bench.ACCEPTED_PER_POS, [9, 7, 4]),
        _vector(bench.ACCEPTED_PER_POS, [4, 2]),
        _counter("vllm:generation_tokens", 999),
    ]

    counters = bench.collect_spec_decode_counters(metrics)

    assert counters == bench.SpecDecodeCounters(num_drafts=15,
                                                num_draft_tokens=45,
                                                num_accepted_tokens=26,
                                                accepted_per_pos=[13, 9, 4])


def test_collect_without_spec_metrics_is_empty():
    counters = bench.collect_spec_decode_counters(
        [_counter("vllm:generation_tokens", 100)])

    assert counters == bench.SpecDecodeCounters()


def test_counter_difference_removes_warmup():
    after = bench.SpecDecodeCounters(12, 36, 25, [11, 9, 5])
    warmup = bench.SpecDecodeCounters(2, 6, 4, [2, 2])

    assert after - warmup == bench.SpecDecodeCounters(10, 30, 21, [9, 7, 5])


def test_summarize_reports_rate_length_and_positions():
    summary = bench.summarize_counters(
        bench.SpecDecodeCounters(num_drafts=10,
                                 num_draft_tokens=30,
                                 num_accepted_tokens=21,
                                 accepted_per_pos=[9, 7, 5]))

    assert summary.acceptance_rate == pytest.approx(0.7)
    assert summary.mean_acceptance_length == pytest.approx(3.1)
    assert summary.per_position_acceptance == pytest.approx([0.9, 0.7, 0.5])


def test_summarize_all_rejected_still_counts_target_token():
    summary = bench.summarize_counters(
        bench.SpecDecodeCounters(4, 12, 0, [0, 0, 0]))

    assert summary.acceptance_rate == 0.0
    assert summary.mean_acceptance_length == 1.0


def test_summarize_without_drafts_raises():
    with pytest.raises(ValueError, match="No speculative drafts"):
        bench.summarize_counters(bench.SpecDecodeCounters())


def test_compute_speedup():
    assert bench.compute_speedup(1000.0, 1700.0) == pytest.approx(1.7)
    with pytest.raises(ValueError, match="positive"):
        bench.compute_speedup(0.0, 1700.0)


# --------------------------------------------------------------------------
# Prompt loading
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record, expected",
    [
        ("Solve 2+2", [{
            "role": "user",
            "content": "Solve 2+2"
        }]),
        ({
            "question": "Solve 3+3",
            "answer": "6"
        }, [{
            "role": "user",
            "content": "Solve 3+3"
        }]),
        ({
            "messages": [{
                "role": "system",
                "content": "Be brief."
            }, {
                "role": "user",
                "content": "Hi",
                "name": "x"
            }]
        }, [{
            "role": "system",
            "content": "Be brief."
        }, {
            "role": "user",
            "content": "Hi"
        }]),
    ],
)
def test_to_conversation(record, expected):
    assert bench.to_conversation(record, "question") == expected


@pytest.mark.parametrize("record", [
    {
        "answer": "6"
    },
    {
        "messages": []
    },
    {
        "messages": [{
            "content": "no role"
        }]
    },
    "   ",
    42,
])
def test_to_conversation_rejects_malformed(record):
    with pytest.raises(ValueError):
        bench.to_conversation(record, "question")


def test_load_conversations_from_jsonl(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(
        json.dumps({"question": f"q{i}"}) for i in range(5)) + "\n\n")

    args = _parse("--prompts-jsonl", str(path), "--num-prompts", "3")
    conversations = bench.load_conversations(args)

    assert [c[0]["content"] for c in conversations] == ["q0", "q1", "q2"]


def test_load_conversations_too_few_prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"question": "only one"}) + "\n")

    args = _parse("--prompts-jsonl", str(path), "--num-prompts", "2")
    with pytest.raises(ValueError, match="only 1"):
        bench.load_conversations(args)


# --------------------------------------------------------------------------
# Engine configuration and argument validation
# --------------------------------------------------------------------------


def test_baseline_engine_kwargs_have_no_spec_config():
    kwargs = bench.build_engine_kwargs(_parse(),
                                       max_num_seqs=64,
                                       num_speculative_tokens=0)

    assert "speculative_config" not in kwargs
    assert kwargs["max_num_seqs"] == 64
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["disable_log_stats"] is False


def test_spec_engine_kwargs():
    args = _parse("--method", "eagle3", "--draft-model", "org/draft",
                  "--kv-cache-dtype", "fp8", "--extra-engine-args",
                  '{"max_model_len": 2048, "enforce_eager": true}')

    kwargs = bench.build_engine_kwargs(args,
                                       max_num_seqs=8,
                                       num_speculative_tokens=4)

    assert kwargs["speculative_config"] == {
        "method": "eagle3",
        "num_speculative_tokens": 4,
        "model": "org/draft",
    }
    assert kwargs["kv_cache_dtype"] == "fp8"
    assert kwargs["max_model_len"] == 2048  # Extra args override defaults.
    assert kwargs["enforce_eager"] is True


@pytest.mark.parametrize("argv", [
    ["--num-speculative-tokens", "0"],
    ["--max-num-seqs", "0"],
    ["--n", "0"],
    ["--num-prompts", "2", "--num-warmup-prompts", "3"],
    ["--extra-engine-args", "[1, 2]"],
    ["--extra-engine-args", "{not json"],
])
def test_validate_args_rejects(argv):
    with pytest.raises(ValueError):
        bench.validate_args(_parse(*argv))


# --------------------------------------------------------------------------
# Sweep orchestration (engine runs replaced by canned results)
# --------------------------------------------------------------------------


def _fake_result(tok_per_s, counters=None, prefix_caching=False):
    return {
        "num_completions": 16,
        "num_output_tokens": 1600,
        "mean_output_len": 100.0,
        "elapsed_s": 1600 / tok_per_s,
        "output_tok_per_s": tok_per_s,
        "prefix_caching": prefix_caching,
        "counters": counters or {
            "num_drafts": 0,
            "num_draft_tokens": 0,
            "num_accepted_tokens": 0,
            "accepted_per_pos": [],
        },
    }


def test_main_compares_each_spec_run_to_its_own_baseline(
        tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"question": "q"}) + "\n")
    out = tmp_path / "out.json"
    throughput = {(4, 0): 100.0, (4, 2): 180.0, (8, 0): 200.0, (8, 2): 250.0}
    spec_counters = {
        "num_drafts": 10,
        "num_draft_tokens": 20,
        "num_accepted_tokens": 15,
        "accepted_per_pos": [9, 6],
    }
    calls = []

    def fake_run_isolated(engine_kwargs, conversations, args):
        k = engine_kwargs.get("speculative_config",
                              {}).get("num_speculative_tokens", 0)
        calls.append((engine_kwargs["max_num_seqs"], k))
        return _fake_result(throughput[calls[-1]],
                            spec_counters if k else None)

    monkeypatch.setattr(bench, "run_isolated", fake_run_isolated)

    assert bench.main([
        "--prompts-jsonl",
        str(prompts), "--num-prompts", "1", "--num-warmup-prompts", "0",
        "--num-speculative-tokens", "2", "--max-num-seqs", "4", "8",
        "--output-json",
        str(out)
    ]) == 0

    assert calls == [(4, 0), (4, 2), (8, 0), (8, 2)]
    results = json.loads(out.read_text())["results"]
    spec_rows = [r for r in results if r["method"] == "mtp"]
    assert [r["speedup"] for r in spec_rows] == pytest.approx([1.8, 1.25])
    assert spec_rows[0]["spec"]["acceptance_rate"] == pytest.approx(0.75)
    assert spec_rows[0]["spec"]["mean_acceptance_length"] == pytest.approx(2.5)
    assert "speedup" not in results[0]
    assert "1.80x" in capsys.readouterr().out


def test_prefix_caching_mismatch_is_flagged(tmp_path, monkeypatch, capsys):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"question": "q"}) + "\n")
    spec_counters = {
        "num_drafts": 1,
        "num_draft_tokens": 1,
        "num_accepted_tokens": 1,
        "accepted_per_pos": [1],
    }

    def fake_run_isolated(engine_kwargs, conversations, args):
        is_spec = "speculative_config" in engine_kwargs
        # Mirrors the TPU platform turning prefix caching off for hybrid
        # models only when speculative decoding is on.
        return _fake_result(100.0,
                            spec_counters if is_spec else None,
                            prefix_caching=not is_spec)

    monkeypatch.setattr(bench, "run_isolated", fake_run_isolated)

    bench.main([
        "--prompts-jsonl",
        str(prompts), "--num-prompts", "1", "--num-warmup-prompts", "0",
        "--enable-prefix-caching"
    ])

    assert "prefix caching differs" in capsys.readouterr().out
