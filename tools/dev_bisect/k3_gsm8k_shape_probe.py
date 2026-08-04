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
"""gsm8k-shape consistency probe for the sliced K3 at pod topology.

Reproduces the 779 workload shape (chunked prefill of ~1k-token prompts at
512 budget, concurrency 8, long greedy decode) and measures per-prompt
consistency between:
  S  - the 8 prompts submitted ONE AT A TIME (no churn, no mixed batches)
  C  - the same prompts submitted concurrently (chunked prefill + decode mix)
  C2 - phase C again (determinism under identical submission)

A state-corruption mechanism (slot collision, cross-request bleed, chunked
fixup bug) shows up as LARGE C-vs-S divergence on multiple prompts and/or
C-vs-C2 nondeterminism; greedy near-ties on the degenerate slice produce
only occasional small tail drift.

Knobs (env): PROBE_BLOCKS (pin; empty = unpinned/compact), PROBE_TAG.
Output: JSON at /tmp/gsmprobe_<tag>.json
"""
import json
import os

import numpy as np  # noqa: E402

blocks = os.environ.get("PROBE_BLOCKS")
tag = os.environ.get("PROBE_TAG", "untagged")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.plugins import load_general_plugins  # noqa: E402

load_general_plugins()
from tpu_inference.models.vllm.experimental import \
    register_models  # noqa: E402

register_models()

kwargs = {}
if blocks:
    kwargs["num_gpu_blocks_override"] = int(blocks)

llm = LLM(
    model="gs://tpu-commons-ci/moonshootai/kimi/k3-sliced",
    trust_remote_code=True,
    load_format="runai_streamer",
    max_model_len=2048,
    max_num_batched_tokens=512,
    max_num_seqs=64,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.8,
    enable_prefix_caching=False,
    async_scheduling=False,
    additional_config={
        "sharding": {
            "sharding_strategy": {
                "tensor_parallelism": 2,
                "expert_parallelism": 4,
                "enable_dp_attention": False,
            }
        },
    },
    **kwargs,
)

rng = np.random.default_rng(42)
N, PLEN, OUT = 8, 1024, 64
# in-vocab, avoid special/reserved low ids and the eos region
prompts = [[int(t) for t in rng.integers(1000, 150000, size=PLEN)]
           for _ in range(N)]
sp = SamplingParams(temperature=0.0,
                    max_tokens=OUT,
                    ignore_eos=True,
                    logprobs=1)


def _extract(out):
    toks = [int(t) for t in out.outputs[0].token_ids]
    lps = []
    for tok, lpd in zip(toks, out.outputs[0].logprobs or []):
        ent = lpd.get(tok)
        lps.append(float(ent.logprob) if ent is not None else float("nan"))
    return toks, lps


def run_sequential():
    outs = []
    for p in prompts:
        o = llm.generate([TokensPrompt(prompt_token_ids=p)],
                         sp,
                         use_tqdm=False)
        outs.append(_extract(o[0]))
    return outs


def run_concurrent():
    o = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                     sp,
                     use_tqdm=False)
    return [_extract(x) for x in o]


S = run_sequential()
C = run_concurrent()
C2 = run_concurrent()


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


report = {"tag": tag, "blocks": blocks or "compact", "prompts": N}
for name, ref, cmp_ in [("C_vs_S", S, C), ("C2_vs_C", C, C2)]:
    divs = [first_div(a[0], b[0]) for a, b in zip(ref, cmp_)]
    # logprob delta of the CHOSEN token over the agreeing prefix: fp noise
    # for batch-composition near-ties, large/growing for state corruption
    lp_deltas = []
    for (ta, la), (tb, lb), d in zip(ref, cmp_, divs):
        end = d if d >= 0 else min(len(la), len(lb))
        deltas = [abs(x - y) for x, y in zip(la[:end], lb[:end])]
        lp_deltas.append(max(deltas) if deltas else 0.0)
    report[name] = {
        "first_divergence_per_prompt": divs,
        "identical_prompts": sum(int(d == -1) for d in divs),
        "max_lp_delta_common_prefix": lp_deltas,
    }
report["S"] = S
report["C"] = C
path = f"/tmp/gsmprobe_{tag}.json"
json.dump(report, open(path, "w"))
print("[gsmprobe]", tag, "blocks=", blocks or "compact")
print("[gsmprobe] C_vs_S ident:", report["C_vs_S"]["identical_prompts"],
      "divs:", report["C_vs_S"]["first_divergence_per_prompt"])
print("[gsmprobe] C_vs_S max|dlp| prefix:",
      ["%.2e" % x for x in report["C_vs_S"]["max_lp_delta_common_prefix"]])
print("[gsmprobe] C2_vs_C ident:", report["C2_vs_C"]["identical_prompts"])
print("[gsmprobe] saved", path)
