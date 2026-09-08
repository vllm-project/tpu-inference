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
"""Localise a disaggregated-vs-baseline output difference.

When a disaggregated deployment's text differs from a single-node baseline, the
cause is either the transferred KV or greedy decoding flipping a near-tied
argmax. Text equality alone cannot tell them apart. This probe posts the prompt
as an explicit list of token ids -- which the OpenAI completions API accepts --
so the token count is exact rather than whatever a word-count happened to
tokenize to, and runs two measurements over it.

Measurement A -- block-alignment sweep.
  The producer drops the trailing partial block (tpu_connector.request_finished:
  `computed_block_ids = block_ids if all_full else block_ids[:-1]`) and the
  consumer independently derives the same boundary via
  round_down(len(prompt), block_size). At a prompt length that is an exact
  multiple of block_size the decoder therefore has *zero* tail to recompute; at
  any other length it recomputes the remainder. Mismatches confined to
  non-multiples point at the recomputed tail; mismatches at exact multiples too
  point at the transferred bytes.

Measurement B -- first-token logprobs.
  Sub-ULP drift shows up as near-tied top-1/top-2 logits with the two endpoints
  agreeing to ~1e-3. Wrong KV shows up as a large delta whatever the tie
  structure. One call per endpoint settles which of the two is happening,
  without any KV-level tooling.

Prompt distribution is a variable, not a constant. --style gibberish emits
random single letters, whose top two logits are routinely tied to within a bf16
ULP; --style text uses prose. Running both quantifies how much of a rate is an
artefact of the prompts.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict

import requests
from transformers import AutoTokenizer

# Real prose, so that --style text exercises a normal (non-near-tied) logit
# distribution. Content is irrelevant; it only has to tokenize to enough tokens.
_PROSE = """
The history of computing hardware spans the development of machines able to
automate calculation. Early aids such as the abacus gave way to mechanical
calculators, and then to programmable machines whose behaviour was determined by
stored instructions rather than by physical rearrangement. The shift from vacuum
tubes to transistors, and then to integrated circuits, reduced cost and power draw
by orders of magnitude while raising reliability. Later, the separation of memory
from arithmetic became the dominant constraint on performance, and much of the
subsequent design effort went into hiding the latency of that separation through
caches, pipelining, and speculative execution. Accelerators returned to a different
tradeoff, spending area on arithmetic throughput and on very wide memory interfaces,
and pushing the burden of scheduling back onto the compiler and the programmer.
"""


def build_prompt_ids(tok, num_tokens: int, style: str, seed: int) -> list[int]:
    """Return exactly `num_tokens` token ids."""
    rng = random.Random(seed)
    if style == "gibberish":
        # Random single lowercase letters joined by spaces: the near-tie-rich
        # distribution. Over-generate, then truncate to an exact token count.
        words = [
            rng.choice("abcdefghijklmnopqrstuvwxyz")
            for _ in range(num_tokens * 2 + 32)
        ]
        text = " ".join(words)
    else:
        # Rotate the starting point so prompts differ between samples.
        body = " ".join(_PROSE.split())
        reps = math.ceil((num_tokens * 8) / max(len(body), 1)) + 1
        text = " ".join([body] * reps)
        off = rng.randrange(0, max(len(body), 1))
        text = text[off:]

    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < num_tokens:
        raise RuntimeError(
            f"only produced {len(ids)} tokens, needed {num_tokens}")
    return ids[:num_tokens]


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def completion(url: str, model: str, ids: list[int], max_tokens: int,
               logprobs: int | None) -> dict:
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    return post(url, payload)


def first_top_logprobs(resp: dict) -> tuple[str, dict[str, float]]:
    """(sampled token, {token: logprob}) for the first generated position."""
    lp = resp["choices"][0].get("logprobs")
    if not lp:
        raise RuntimeError(
            "endpoint returned no logprobs; is `logprobs` supported?")
    top = lp.get("top_logprobs") or []
    toks = lp.get("tokens") or []
    if not top or not toks:
        raise RuntimeError(f"malformed logprobs payload: {lp}")
    return toks[0], dict(top[0])


def compare_logprobs(base: dict[str, float],
                     disagg: dict[str, float]) -> tuple[float, float]:
    """(baseline top1-top2 gap, max |delta| over tokens common to both)."""
    ordered = sorted(base.values(), reverse=True)
    gap = (ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")
    shared = set(base) & set(disagg)
    delta = max((abs(base[t] - disagg[t]) for t in shared),
                default=float("nan"))
    return gap, delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-url", required=True)
    ap.add_argument("--disagg-url", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lengths",
                    type=str,
                    default="64,127,128,129,192,200,255,256,257,384,512",
                    help="exact prompt token counts to sweep")
    ap.add_argument("--num-prompts",
                    type=int,
                    default=10,
                    help="prompts per length")
    ap.add_argument("--output-length", type=int, default=32)
    ap.add_argument("--style",
                    choices=["gibberish", "text", "both"],
                    default="both")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    styles = ["gibberish", "text"] if args.style == "both" else [args.style]

    print(f"model={args.model} block_size={args.block_size}")
    print(f"baseline={args.baseline_url}")
    print(f"disagg  ={args.disagg_url}")
    tok = AutoTokenizer.from_pretrained(args.model)

    records = []
    # (style, length) -> counters
    agg: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "n": 0,
        "text_mismatch": 0,
        "tok1_mismatch": 0,
        "gaps": [],
        "deltas": []
    })

    for style in styles:
        for n_tok in lengths:
            for i in range(args.num_prompts):
                seed = hash((style, n_tok, i)) & 0xFFFFFFFF
                ids = build_prompt_ids(tok, n_tok, style, seed)
                rec = {
                    "style": style,
                    "num_tokens": n_tok,
                    "sample": i,
                    "aligned": n_tok % args.block_size == 0
                }

                # --- Measurement A: greedy text equality -------------------
                try:
                    b = completion(args.baseline_url, args.model, ids,
                                   args.output_length, None)
                    d = completion(args.disagg_url, args.model, ids,
                                   args.output_length, None)
                    bt = b["choices"][0]["text"]
                    dt = d["choices"][0]["text"]
                    rec["text_match"] = (bt == dt)
                    if bt != dt:
                        common = 0
                        for cb, cd in zip(bt, dt):
                            if cb != cd:
                                break
                            common += 1
                        rec["common_prefix_chars"] = common
                        rec["baseline_text"] = bt
                        rec["disagg_text"] = dt
                except Exception as e:  # noqa: BLE001
                    rec["text_error"] = repr(e)

                # --- Measurement B: first-token logprobs -------------------
                try:
                    bl = completion(args.baseline_url, args.model, ids, 1, 5)
                    dl = completion(args.disagg_url, args.model, ids, 1, 5)
                    b_tok, b_top = first_top_logprobs(bl)
                    d_tok, d_top = first_top_logprobs(dl)
                    gap, delta = compare_logprobs(b_top, d_top)
                    rec.update({
                        "tok1_match": b_tok == d_tok,
                        "baseline_top2_gap": gap,
                        "logprob_max_delta": delta,
                        "baseline_tok1": b_tok,
                        "disagg_tok1": d_tok
                    })
                except Exception as e:  # noqa: BLE001
                    rec["logprob_error"] = repr(e)

                records.append(rec)
                a = agg[(style, n_tok)]
                a["n"] += 1
                if rec.get("text_match") is False:
                    a["text_mismatch"] += 1
                if rec.get("tok1_match") is False:
                    a["tok1_mismatch"] += 1
                if isinstance(rec.get("baseline_top2_gap"), float):
                    a["gaps"].append(rec["baseline_top2_gap"])
                if isinstance(rec.get("logprob_max_delta"), float):
                    a["deltas"].append(rec["logprob_max_delta"])

    # --- Report ------------------------------------------------------------
    def med(xs):
        return float("nan") if not xs else sorted(xs)[len(xs) // 2]

    print(
        "\n=== Measurement A/B by prompt length "
        f"(block_size={args.block_size}; 'aligned' = whole prompt transferred, "
        "decoder recomputes nothing) ===")
    hdr = (f"{'style':<10} {'tokens':>7} {'aligned':>8} {'text_mm':>9} "
           f"{'tok1_mm':>9} {'med top2 gap':>13} {'med lp delta':>13}")
    print(hdr)
    print("-" * len(hdr))
    for style in styles:
        for n_tok in lengths:
            a = agg[(style, n_tok)]
            if not a["n"]:
                continue
            print(f"{style:<10} {n_tok:>7} "
                  f"{str(n_tok % args.block_size == 0):>8} "
                  f"{a['text_mismatch']:>4}/{a['n']:<4} "
                  f"{a['tok1_mismatch']:>4}/{a['n']:<4} "
                  f"{med(a['gaps']):>13.5f} {med(a['deltas']):>13.2e}")

    # --- Verdict -----------------------------------------------------------
    aligned = [r for r in records if r["aligned"]]
    unaligned = [r for r in records if not r["aligned"]]

    def mm_rate(rs):
        seen = [r for r in rs if "text_match" in r]
        return (sum(1 for r in seen if not r["text_match"]) /
                len(seen)) if seen else float("nan")

    all_deltas = [
        r["logprob_max_delta"] for r in records
        if isinstance(r.get("logprob_max_delta"), float)
        and not math.isnan(r["logprob_max_delta"])
    ]
    mismatch_gaps = [
        r["baseline_top2_gap"] for r in records if r.get("tok1_match") is False
        and isinstance(r.get("baseline_top2_gap"), float)
    ]

    print("\n=== Summary ===")
    print(f"mismatch rate, block-aligned prompts   : {mm_rate(aligned):.3f} "
          f"(n={len(aligned)})")
    print(f"mismatch rate, unaligned prompts       : {mm_rate(unaligned):.3f} "
          f"(n={len(unaligned)})")
    print(f"max first-token logprob delta (all)    : "
          f"{max(all_deltas) if all_deltas else float('nan'):.3e}")
    print(f"median top1-top2 gap where tok1 flipped: {med(mismatch_gaps):.5f} "
          f"(n={len(mismatch_gaps)})")
    print(
        "\nRead: a near-zero aligned rate, a tiny logprob delta and a near-zero "
        "top2 gap at the\n      flips mean the differences are argmax "
        "instability under a changed prefill shape.\n      A non-zero aligned "
        "rate, or a large logprob delta, means the transferred KV is\n      "
        "worth investigating.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(records, f, indent=1)
        print(f"\nper-prompt records -> {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
