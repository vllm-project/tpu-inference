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
"""Trace the TP=8 leg of test_tp_performance and summarize the trace.

Usage: tp8_profile.py TAG. Run it twice in one pod to compare the pod's first
TP=8 generate with its second: device time per XLA module on TPU 0 against
host time per engine step, to see which side the lost time is on.
"""
import collections
import glob
import json
import os
import statistics
import sys
import time

os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("SKIP_JAX_PRECOMPILE", "0")
os.environ.setdefault("VLLM_XLA_CHECK_RECOMPILATION", "1")
# A Python tracer adds per-call overhead to the host side being measured.
os.environ.setdefault("PYTHON_TRACER_LEVEL", "0")
os.environ.setdefault("PROFILE_SINGLE_DEVICE", "1")

sys.path.insert(0, "/workspace/tpu_inference/tests/e2e")
from test_tensor_parallel import generate_test_prompts  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

tag = sys.argv[1]
prof_dir = f"/tmp/prof/{tag}"

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=8,
          max_model_len=2048, max_num_batched_tokens=2048, max_num_seqs=256,
          gpu_memory_utilization=0.80, kv_cache_dtype="auto",
          enable_prefix_caching=False,
          profiler_config={"profiler": "torch", "torch_profiler_dir": prof_dir})
prompts = generate_test_prompts(256, 18)
sampling = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)

llm.start_profile()
t0 = time.time()
llm.generate(prompts, sampling)
elapsed = time.time() - t0
llm.stop_profile()
llm.llm_engine.engine_core.shutdown()
print(f"[{tag}] TP=8 generate {elapsed:.2f}s (traced)")


def profile_data_cls():
    import jax
    if hasattr(jax.profiler, "ProfileData"):
        return jax.profiler.ProfileData
    from jax._src.lib import _profile_data
    return _profile_data.ProfileData


paths = sorted(glob.glob(f"{prof_dir}/**/*.xplane.pb", recursive=True),
               key=os.path.getmtime)
if not paths:
    sys.exit(f"[{tag}] no xplane.pb under {prof_dir}")
data = profile_data_cls().from_file(paths[-1])


def pct(xs, p):
    s = sorted(xs)
    return s[int(p * (len(s) - 1))] if s else 0.0


summary = {"tag": tag, "generate_s": round(elapsed, 3)}
for plane in data.planes:
    if plane.name.startswith("/device:TPU:0"):
        for line in plane.lines:
            if line.name != "XLA Modules":
                continue
            ev = sorted(line.events, key=lambda e: e.start_ns)
            if not ev:
                continue
            span = (ev[-1].start_ns + ev[-1].duration_ns - ev[0].start_ns) / 1e9
            busy = sum(e.duration_ns for e in ev) / 1e9
            by = collections.defaultdict(list)
            for e in ev:
                by[e.name.split("(")[0][:60]].append(e.duration_ns / 1e6)
            summary["device"] = {
                "modules": len(ev), "span_s": round(span, 3),
                "busy_s": round(busy, 3),
                "first20_ms": [round(e.duration_ns / 1e6, 2) for e in ev[:20]],
                "top": sorted(({"name": k, "n": len(v),
                                "total_ms": round(sum(v), 1),
                                "p50_ms": round(pct(v, 0.5), 3),
                                "max_ms": round(max(v), 2)}
                               for k, v in by.items()),
                              key=lambda d: -d["total_ms"])[:12],
            }
    if plane.name.startswith("/host:CPU"):
        steps = []
        for line in plane.lines:
            steps += [e for e in line.events
                      if e.name.startswith("execute_model:")]
        steps.sort(key=lambda e: e.start_ns)
        if steps:
            dur = [e.duration_ns / 1e6 for e in steps]
            gaps = [(b.start_ns - (a.start_ns + a.duration_ns)) / 1e6
                    for a, b in zip(steps, steps[1:])]
            summary["host"] = {
                "steps": len(steps), "total_ms": round(sum(dur), 1),
                "p50_ms": round(pct(dur, 0.5), 2),
                "p90_ms": round(pct(dur, 0.9), 2),
                "max_ms": round(max(dur), 2),
                "gap_total_ms": round(sum(gaps), 1),
                "gap_p50_ms": round(pct(gaps, 0.5), 2),
                "first20": [(e.name[len("execute_model: "):],
                             round(e.duration_ns / 1e6, 2)) for e in steps[:20]],
                "mean_ms_by_decile": [
                    round(statistics.mean(dur[i * len(dur) // 10:
                                              (i + 1) * len(dur) // 10] or [0]), 2)
                    for i in range(10)],
            }
print("TRACE_SUMMARY " + json.dumps(summary))
