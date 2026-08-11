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

Mechanism: vLLM's `prompt_logprobs` completions extension (the engine
path this instrument was validated on) with echo+logprobs as a fallback.
HTTP errors print the full response body — a swallowed 500 here has
already cost one green-but-dead validation run.

Env: SCORING_WAVES (default 12), SCORING_BASE (default
http://localhost:8000), SCORING_MODEL (default the K3 GCS path).
Exit codes: 0 = measurement completed (BIT-STABLE or DRIFTING — drift is
a result, not a failure); 2 = INCOMPLETE (the instrument itself failed;
a green build must not hide a dead instrument).
"""
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

BASE = os.environ.get("SCORING_BASE", "http://localhost:8000")
MODEL = os.environ.get("SCORING_MODEL",
                       "gs://tpu-commons-ci/moonshootai/kimi/k3")
W = int(os.environ.get("SCORING_WAVES", "12"))

rng = np.random.default_rng(20260810)
PROMPTS = [[int(t) for t in rng.integers(1000, 150000, size=1024)]
           for _ in range(8)]

MECHANISM = os.environ.get("SCORING_MECHANISM", "prompt_logprobs")


def _post(payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode(errors="replace")[:2000]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"HTTP {e.code} from /v1/completions; response body: "
            f"{detail!r}") from e


def score_one(prompt_ids):
    global MECHANISM
    if MECHANISM == "prompt_logprobs":
        out = _post({
            "model": MODEL,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        })
        plps = out["choices"][0].get("prompt_logprobs")
        if plps is None:
            raise RuntimeError(
                "server returned no prompt_logprobs; response keys: "
                f"{sorted(out['choices'][0])}")
        total = 0.0
        for entry in plps:
            if entry:
                total += min(float(v["logprob"]) for v in entry.values())
        return total
    # Fallback: echo the prompt and read token_logprobs.
    out = _post({
        "model": MODEL,
        "prompt": prompt_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "echo": True,
        "logprobs": 1,
    })
    lps = out["choices"][0]["logprobs"]["token_logprobs"]
    prompt_lps = [x for x in lps[:len(prompt_ids)] if x is not None]
    return float(sum(prompt_lps))


def wave():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(score_one, PROMPTS))


def main():
    global MECHANISM
    print(f"[scoring-probe] {W} waves x 8 fixed 1024-token sequences "
          f"(mechanism={MECHANISM})")
    ref = None
    worst_overall = 0.0
    for w in range(W):
        try:
            sums = wave()
        except Exception as exc:  # noqa: BLE001
            if w == 0 and MECHANISM == "prompt_logprobs":
                print(f"[scoring-probe] prompt_logprobs failed ({exc}); "
                      "falling back to echo+logprobs")
                MECHANISM = "echo"
                try:
                    sums = wave()
                except Exception as exc2:  # noqa: BLE001
                    print(f"[scoring-probe] fallback also failed: {exc2}")
                    print("[scoring-probe] RESULT: INCOMPLETE")
                    sys.exit(2)
            else:
                print(f"[scoring-probe] wave={w} REQUEST FAILED: {exc}")
                print("[scoring-probe] RESULT: INCOMPLETE")
                sys.exit(2)
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
