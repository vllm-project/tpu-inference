#!/usr/bin/env python3
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
"""Measure the mismatch rate a disaggregated run cannot beat.

Comparing a disaggregated deployment's text against a single-node baseline
cannot on its own tell a corrupted KV handoff from greedy decoding flipping a
near-tied argmax. These two controls remove the KV transfer from the picture
entirely: every byte of KV is computed locally by the engine that uses it, so
any mismatch they report is attributable to the shape of the prefill alone.

  C1  chunking control. Two identical servers differing only in
      --max-num-batched-tokens (unchunked vs chunked). Same weights, same block
      size, no prefix caching, no disaggregation. A mismatch here means that
      splitting a prefill across forward passes is by itself enough to change
      the output.

  C2  prefix-cache control. One server with prefix caching on. The same prompt
      is sent twice: the first call prefills all N tokens, the second reuses the
      cached KV -- bit-identical by construction -- and recomputes only the
      tail. Identical bytes, different query-chunk shape.

Reading the result. A disaggregated run changes the execution path at least as
much as these controls do, so it cannot be expected to beat them. If the
controls sit near the disaggregated rate, the mismatches are argmax instability
and the transfer is not implicated. If the controls are near zero while the
disaggregated rate is not, the transferred bytes are suspect.

Prompt distribution dominates both. Prompts of random single letters leave the
top two logits routinely tied to within a bf16 ULP, so they flip under any
change of execution path; prose does not.
"""

import argparse
import json
import sys

from pd_probe import (build_prompt_ids, compare_logprobs, completion,
                      first_top_logprobs)
from transformers import AutoTokenizer


def run_pair(url_a,
             url_b,
             model,
             tok,
             lengths,
             styles,
             n,
             out_len,
             label,
             same_server_twice=False):
    """Compare two endpoints (or one endpoint called twice) over the sweep."""
    rows = []
    for style in styles:
        for n_tok in lengths:
            mism = tok1_mm = 0
            deltas = []
            for i in range(n):
                # Distinct seed space per control, so C2's "fresh" call really is
                # a cache miss rather than a prompt the sweep already warmed.
                seed = hash((label, style, n_tok, i)) & 0xFFFFFFFF
                ids = build_prompt_ids(tok, n_tok, style, seed)
                try:
                    a = completion(url_a, model, ids, out_len, None)
                    b = completion(url_b, model, ids, out_len, None)
                    if a["choices"][0]["text"] != b["choices"][0]["text"]:
                        mism += 1
                    la = completion(url_a, model, ids, 1, 5)
                    lb = completion(url_b, model, ids, 1, 5)
                    ta, topa = first_top_logprobs(la)
                    tb, topb = first_top_logprobs(lb)
                    if ta != tb:
                        tok1_mm += 1
                    deltas.append(compare_logprobs(topa, topb)[1])
                except Exception as e:  # noqa: BLE001
                    print(f"  [{label} {style} {n_tok} #{i}] error: {e!r}")
            med = sorted(deltas)[len(deltas) // 2] if deltas else float("nan")
            rows.append((label, style, n_tok, mism, tok1_mm, n, med))
            print(f"{label:<14} {style:<10} {n_tok:>5}  text_mm={mism}/{n}  "
                  f"tok1_mm={tok1_mm}/{n}  med_lp_delta={med:.2e}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unchunked-url", required=True, help="C1 server A")
    ap.add_argument("--chunked-url", required=True, help="C1 server B")
    ap.add_argument("--prefixcache-url", required=True, help="C2 server")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lengths", default="128,256,384,512")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--output-length", type=int, default=32)
    ap.add_argument("--style",
                    choices=["gibberish", "text", "both"],
                    default="both")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    styles = ["gibberish", "text"] if args.style == "both" else [args.style]
    tok = AutoTokenizer.from_pretrained(args.model)

    print("=== C1: unchunked vs chunked prefill (no disagg, no transfer) ===")
    r1 = run_pair(args.unchunked_url, args.chunked_url, args.model, tok,
                  lengths, styles, args.num_prompts, args.output_length,
                  "C1-chunking")

    print("\n=== C2: prefix-cache miss vs hit on one server "
          "(bit-identical KV, different query shape) ===")
    r2 = run_pair(args.prefixcache_url,
                  args.prefixcache_url,
                  args.model,
                  tok,
                  lengths,
                  styles,
                  args.num_prompts,
                  args.output_length,
                  "C2-prefixcache",
                  same_server_twice=True)

    def rate(rows):
        m = sum(r[3] for r in rows)
        t = sum(r[5] for r in rows)
        return m / t if t else float("nan"), m, t

    c1, m1, t1 = rate(r1)
    c2, m2, t2 = rate(r2)
    print("\n=== Control summary ===")
    print(f"C1 chunking      mismatch rate: {c1:.3f}  ({m1}/{t1})")
    print(f"C2 prefix-cache  mismatch rate: {c2:.3f}  ({m2}/{t2})")
    print(
        "\nCompare these against the disaggregated rate measured on the same "
        "model and prompts.\nA comparable rate means argmax instability, not a "
        "transfer fault; a near-zero rate\nhere alongside a much higher "
        "disaggregated rate means the transferred bytes are\nworth "
        "investigating.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"c1": r1, "c2": r2}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
