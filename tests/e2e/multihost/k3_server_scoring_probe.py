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
"""Teacher-forced scoring stability probe against a live server.

Repeats waves of the same 8 fixed token-id sequences through the
completions endpoint with echo+logprobs (no sampling: max_tokens=1, only
the echoed prompt logprobs are compared). Any wave-over-wave drift in a
sequence's summed prompt logprob is a state/e2e-numeric instability that
argmax-based evals cannot separate from near-ties. On a healthy server
this is bit-stable (verified on an 8-chip slice through 60 waves / ~500
churned requests); drift here with a simultaneously collapsed gsm8k
isolates the corruption to the decode path.

Env: SCORING_WAVES (default 12), SCORING_BASE (default
http://localhost:8000), SCORING_MODEL (default the K3 GCS path).
Prints per-wave max |delta| and exits 0 always (measurement, not a gate).
"""
import concurrent.futures
import json
import os
import urllib.request

import numpy as np

BASE = os.environ.get("SCORING_BASE", "http://localhost:8000")
MODEL = os.environ.get("SCORING_MODEL",
                       "gs://tpu-commons-ci/moonshootai/kimi/k3")
W = int(os.environ.get("SCORING_WAVES", "12"))

rng = np.random.default_rng(20260810)
PROMPTS = [[int(t) for t in rng.integers(1000, 150000, size=1024)]
           for _ in range(8)]


def score_one(prompt_ids):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    lps = out["choices"][0]["logprobs"]["token_logprobs"]
    # First entry is None (no logprob for the first token); the last entry
    # belongs to the generated token — sum the echoed prompt portion only.
    prompt_lps = [x for x in lps[:len(prompt_ids)] if x is not None]
    return float(sum(prompt_lps))


def wave():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(score_one, PROMPTS))


def main():
    print(f"[scoring-probe] {W} waves x 8 fixed 1024-token sequences")
    ref = None
    worst_overall = 0.0
    for w in range(W):
        try:
            sums = wave()
        except Exception as exc:  # noqa: BLE001
            print(f"[scoring-probe] wave={w} REQUEST FAILED: {exc}")
            print("[scoring-probe] RESULT: INCOMPLETE")
            return
        if ref is None:
            ref = sums
            print("[scoring-probe] wave=0 ref sums:",
                  ["%.3f" % s for s in sums])
            continue
        deltas = [abs(a - b) for a, b in zip(sums, ref)]
        mx = max(deltas)
        worst_overall = max(worst_overall, mx)
        flag = "  <-- DRIFT" if mx > 1e-3 else ""
        print(f"[scoring-probe] wave={w} max|dLP|={mx:.6f}{flag}")
    print(f"[scoring-probe] RESULT: worst={worst_overall:.6f} "
          f"({'BIT-STABLE' if worst_overall <= 1e-3 else 'DRIFTING'})")


if __name__ == "__main__":
    main()
