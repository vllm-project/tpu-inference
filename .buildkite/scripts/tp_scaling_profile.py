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
"""Trace a warm generate of test_tp_performance's workload at one TP size.

Usage: tp_scaling_profile.py TP TAG. One untimed generate runs first, so the
traced one holds only steady-state serving, as in the test's timed pass. The
summary splits TPU 0's time by HLO op category (collectives, matmuls, Pallas
kernels, the rest) and the host's by engine step, to see where TP=8 falls
short of 8x TP=1 and where kube differs from bare metal.
"""
import collections
import glob
import json
import os
import sys
import time

# Set outright, as the test does: run_in_docker.sh exports MODEL_IMPL_TYPE=auto,
# which on bare metal would select a different model implementation.
os.environ["MODEL_IMPL_TYPE"] = "vllm"
os.environ["SKIP_JAX_PRECOMPILE"] = "0"
os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "1"
# No Python tracer: its per-call overhead would inflate host time.
os.environ["PYTHON_TRACER_LEVEL"] = "0"
os.environ["PROFILE_SINGLE_DEVICE"] = "1"

sys.path.insert(0, "/workspace/tpu_inference/tests/e2e")
from test_tensor_parallel import generate_test_prompts  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

tp, tag = int(sys.argv[1]), sys.argv[2]
prof_dir = f"/tmp/prof/{tag}"

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          tensor_parallel_size=tp,
          max_model_len=2048,
          max_num_batched_tokens=2048,
          max_num_seqs=256,
          gpu_memory_utilization=0.80,
          kv_cache_dtype="auto",
          enable_prefix_caching=False,
          profiler_config={
              "profiler": "torch",
              "torch_profiler_dir": prof_dir
          })
prompts = generate_test_prompts(256, 18)
sampling = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)

t0 = time.time()
llm.generate(prompts, sampling)
warmup_s = time.time() - t0
llm.start_profile()
t0 = time.time()
llm.generate(prompts, sampling)
elapsed = time.time() - t0
llm.stop_profile()
llm.llm_engine.engine_core.shutdown()
print(f"[{tag}] TP={tp} warm-up generate {warmup_s:.2f}s, "
      f"traced generate {elapsed:.2f}s")


def profile_data_cls():
    import jax
    if hasattr(jax.profiler, "ProfileData"):
        return jax.profiler.ProfileData
    from jax._src.lib import _profile_data
    return _profile_data.ProfileData


def stats_of(event):
    try:
        return {k: v for k, v in event.stats}
    except Exception:
        return {}


def category(name, stats):
    """Coarse bucket for one HLO op, from xprof's category or the op name."""
    cat = str(stats.get("hlo_category", "")).lower()
    key = f"{cat} {name.lower()}"
    for coll in ("all-reduce", "all-gather", "reduce-scatter",
                 "collective-permute", "all-to-all"):
        if coll in key:
            return "collective:" + coll
    if "custom" in key or "pallas" in key or "mosaic" in key:
        return "kernel"
    if "convolution" in key or "dot" in key:
        return "matmul"
    return "other"


paths = sorted(glob.glob(f"{prof_dir}/**/*.xplane.pb", recursive=True),
               key=os.path.getmtime)
if not paths:
    sys.exit(f"[{tag}] no xplane.pb under {prof_dir}")
data = profile_data_cls().from_file(paths[-1])

summary = {
    "tag": tag,
    "tp": tp,
    "warmup_s": round(warmup_s, 3),
    "generate_s": round(elapsed, 3)
}
for plane in data.planes:
    if plane.name.startswith("/device:TPU:0"):
        summary["tpu_lines"] = [line.name for line in plane.lines]
        for line in plane.lines:
            ev = sorted(line.events, key=lambda e: e.start_ns)
            if not ev:
                continue
            if line.name == "XLA Modules":
                span = (ev[-1].start_ns + ev[-1].duration_ns -
                        ev[0].start_ns) / 1e9
                by = collections.defaultdict(float)
                for e in ev:
                    by[e.name.split("(")[0][:60]] += e.duration_ns / 1e6
                summary["modules"] = {
                    "n": len(ev),
                    "span_s": round(span, 3),
                    "busy_s": round(sum(e.duration_ns for e in ev) / 1e9, 3),
                    "top_ms": sorted(((k, round(v, 1)) for k, v in by.items()),
                                     key=lambda kv: -kv[1])[:8],
                }
            elif line.name == "XLA Ops":
                cats = collections.defaultdict(float)
                names = collections.defaultdict(lambda: [0, 0.0, ""])
                sample_stats = None
                for e in ev:
                    st = stats_of(e)
                    if sample_stats is None and st:
                        sample_stats = sorted(st)[:20]
                    ms = e.duration_ns / 1e6
                    c = category(e.name, st)
                    cats[c] += ms
                    n = names[e.name.split(".")[0][:50]]
                    n[0] += 1
                    n[1] += ms
                    n[2] = c
                summary["ops"] = {
                    "n": len(ev),
                    "total_ms": round(sum(cats.values()), 1),
                    "by_category_ms": {k: round(v, 1)
                                       for k, v in sorted(cats.items())},
                    "top": sorted(((k, c, round(t, 1), cnt)
                                   for k, (cnt, t, c) in names.items()),
                                  key=lambda x: -x[2])[:25],
                    "stat_keys": sample_stats,
                }
    if plane.name.startswith("/host:CPU"):
        steps = sorted((e for line in plane.lines for e in line.events
                        if e.name.startswith("execute_model:")),
                       key=lambda e: e.start_ns)
        if not steps:
            continue
        dur = [e.duration_ns / 1e6 for e in steps]
        gaps = [(b.start_ns - (a.start_ns + a.duration_ns)) / 1e6
                for a, b in zip(steps, steps[1:])]
        toks = []
        for e in steps:
            try:
                toks.append(int(e.name.split(",")[1].split()[0]))
            except (IndexError, ValueError):
                toks.append(-1)
        full = [t for t in toks if t >= 1024]
        summary["host"] = {
            "steps": len(steps),
            "steps_ge_1024_tokens": len(full),
            "tokens_total": sum(t for t in toks if t > 0),
            "execute_model_total_ms": round(sum(dur), 1),
            "execute_model_p50_ms": round(sorted(dur)[len(dur) // 2], 2),
            "gap_total_ms": round(sum(gaps), 1),
            "gap_p50_ms": round(sorted(gaps)[len(gaps) // 2], 2) if gaps else 0,
        }
print("SCALING_SUMMARY " + json.dumps(summary))
