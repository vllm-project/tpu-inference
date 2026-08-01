# Copyright 2025 Google LLC
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
"""Correctness probes for a model served across several hosts.

Runs against an already-serving OpenAI-compatible endpoint (the completions
API only) and checks invariants that hold regardless of the host count, so a
multi-host serve can be compared against a single-host one:

  health        the endpoint reports ready and names exactly one model
  determinism   the same greedy request twice returns the same text
  isolation     N concurrent requests each answer their OWN prompt; no
                response may contain another request's unique marker,
                run twice: equal-length prompts, then a wide spread of
                prompt lengths in one batch
  long          a long greedy generation terminates and does not collapse
                into a single repeated token
  needle        a fact planted at several depths of a long prompt is
                retrieved verbatim (skipped unless --needle-tokens is given)

Greedy decoding makes every check above exact, EXCEPT that token-level
equality across *different batch shapes* is not asserted anywhere: padding
buckets change XLA reduction order, so argmax can flip on near-ties. That is
a property of batched serving, not of the model, so the isolation probe keys
on unique markers rather than on token equality.

``--allow-incoherent`` drops only the checks that need the served model to be
a working language model (repeating its own marker, producing long
non-degenerate text, retrieving the needle) and keeps every structural one, so
a checkpoint that is deliberately not a language model -- a depth-sliced
snapshot cut down to fit one host -- can still be probed for state leaks
across concurrent requests.

Exits non-zero on the first failed probe, and prints every observed value it
checked so a failure can be diagnosed from the log alone.
"""

import argparse
import json
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

# Prompts that a coherent instruct model continues recognizably. They are only
# used for the determinism probe, which compares the server against itself.
_PROMPTS = [
    "Question: What is 17 multiplied by 24?\nAnswer:",
    "The capital city of Japan is",
    "In Python, the shortest way to reverse a string s is",
]


class ProbeFailure(AssertionError):
    """A probe's invariant did not hold. The message names the invariant."""


def _complete(base_url: str,
              model: str,
              prompt: str,
              max_tokens: int,
              timeout: float = 1800.0) -> dict:
    """One greedy completion. Returns the first choice."""
    resp = requests.post(
        f"{base_url}/v1/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
        }),
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise ProbeFailure(
            f"[probe] completions returned HTTP {resp.status_code} for a "
            f"{len(prompt)}-char prompt: {resp.text[:500]}")
    return resp.json()["choices"][0]


def probe_health(base_url: str) -> str:
    """Endpoint is ready and serves exactly one model. Returns its id."""
    resp = requests.get(f"{base_url}/health", timeout=60)
    if resp.status_code != 200:
        raise ProbeFailure(
            f"[probe:health] /health returned HTTP {resp.status_code}")
    models = requests.get(f"{base_url}/v1/models", timeout=60).json()["data"]
    ids = [m["id"] for m in models]
    if len(ids) != 1:
        raise ProbeFailure(
            f"[probe:health] expected exactly one served model, got {ids}")
    print(f"[probe:health] OK, serving {ids[0]!r}")
    return ids[0]


def probe_determinism(base_url: str, model: str, max_tokens: int) -> None:
    """Greedy decoding is reproducible when the request is repeated."""
    for prompt in _PROMPTS:
        first = _complete(base_url, model, prompt, max_tokens)
        second = _complete(base_url, model, prompt, max_tokens)
        print(f"[probe:determinism] prompt={prompt!r}\n"
              f"  run1 ({first['finish_reason']}): {first['text']!r}\n"
              f"  run2 ({second['finish_reason']}): {second['text']!r}")
        if not first["text"].strip():
            raise ProbeFailure(
                f"[probe:determinism] empty completion for {prompt!r} "
                f"(finish_reason={first['finish_reason']})")
        if first["text"] != second["text"]:
            raise ProbeFailure(
                "[probe:determinism] the same greedy request returned two "
                f"different completions for {prompt!r}:\n"
                f"  {first['text']!r}\n  {second['text']!r}")
    print(f"[probe:determinism] OK, {len(_PROMPTS)} prompts reproducible")


def _markers(count: int, seed: int = 0) -> list:
    """`count` distinct pronounceable markers, e.g. 'zubmiq'."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        marker = "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
        if marker not in out:
            out.append(marker)
    return out


def probe_isolation(base_url: str,
                    model: str,
                    concurrency: int,
                    ragged: bool = False,
                    require_coherence: bool = True) -> None:
    """Concurrent requests do not leak each other's state.

    Each request carries a unique marker and is asked to repeat it. A response
    that contains another request's marker means state crossed request slots —
    the failure mode that a sharded state pool, or slot indices that are
    global where they should be rank-local, would produce.

    With `ragged`, the requests are padded to widely different prompt lengths.
    Equal-length prompts exercise none of the ragged bookkeeping: it is unequal
    sequence lengths in one batch that make per-request offsets and per-rank
    packing disagree, so this variant is where a wrong slice shows up.

    `require_coherence=False` keeps the leak check — the invariant this probe
    exists for — but drops the "the response repeats its own marker" check,
    which needs a model that follows the instruction. Depth-sliced or otherwise
    truncated checkpoints emit meaningless text by construction, and for those
    the leak check is still meaningful while the echo check is not.
    """
    tag = "ragged" if ragged else "isolation"
    markers = _markers(concurrency, seed=1 if ragged else 0)
    filler = "The shelves hold old paper records nobody has opened in years. "
    prompts = []
    for i, m in enumerate(markers):
        # 0, 10, 20, ... filler sentences (~13 tokens each), so the longest
        # request in the batch is ~1000 tokens and fits a 2048 context.
        pad = filler * (i * 10)
        prompts.append(
            f"{pad}Repeat the word {m} five times, separated by spaces.\n{m}"
            if ragged else
            f"Repeat the word {m} five times, separated by spaces.\n{m}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        choices = list(
            pool.map(lambda p: _complete(base_url, model, p, 32), prompts))

    for i, (marker, choice) in enumerate(zip(markers, choices)):
        text = choice["text"]
        print(f"[probe:{tag}] req {i} marker={marker} "
              f"prompt_chars={len(prompts[i])} -> {text!r}")
        foreign = [m for m in markers if m != marker and m in text]
        if foreign:
            raise ProbeFailure(
                f"[probe:{tag}] request {i} (marker {marker}) emitted "
                f"another request's marker(s) {foreign} — state crossed "
                f"request slots. Text: {text!r}")
        if marker not in text:
            if require_coherence:
                raise ProbeFailure(
                    f"[probe:{tag}] request {i} never repeated its own "
                    f"marker {marker}; got {text!r}")
            print(f"[probe:{tag}] req {i} did not repeat its own marker "
                  "(not asserted: --allow-incoherent)")
    print(f"[probe:{tag}] OK, {concurrency} concurrent requests, "
          "no marker crossed requests")


def probe_long_generation(base_url: str,
                          model: str,
                          max_tokens: int,
                          require_coherence: bool = True) -> None:
    """A long generation terminates and stays non-degenerate.

    `require_coherence=False` keeps the request-completes check and still
    prints every observed value, but does not assert the output is long or
    non-degenerate: a truncated model legitimately produces degenerate text.
    """
    prompt = ("Write a detailed explanation of how a hash table works, "
              "including collision handling.\n")
    choice = _complete(base_url, model, prompt, max_tokens)
    text = choice["text"]
    words = text.split()
    print(f"[probe:long] finish_reason={choice['finish_reason']} "
          f"chars={len(text)} words={len(words)}")
    print(f"[probe:long] text: {text!r}")
    if not words:
        raise ProbeFailure(f"[probe:long] empty completion "
                           f"(finish_reason={choice['finish_reason']})")
    top_share = max(words.count(w) for w in set(words)) / len(words)
    if not require_coherence:
        print(f"[probe:long] OK (coherence not asserted: --allow-incoherent), "
              f"{len(words)} words, most frequent is {top_share:.0%}")
        return
    if len(words) < max_tokens // 8:
        raise ProbeFailure(
            f"[probe:long] asked for {max_tokens} tokens but got only "
            f"{len(words)} words (finish_reason={choice['finish_reason']})")
    if top_share > 0.5:
        raise ProbeFailure(
            f"[probe:long] degenerate output: one word is {top_share:.0%} of "
            f"the {len(words)} generated words")
    print(f"[probe:long] OK, most frequent word is {top_share:.0%} of output")


def probe_needle(base_url: str, model: str, needle_tokens: int) -> None:
    """A fact planted mid-context is retrieved after a long prefill.

    Approximates token count with a filler sentence of known length; the exact
    prompt length is printed so the log records what was actually run.
    """
    filler = ("The archive room is quiet and the shelves are full of old "
              "paper records that nobody has opened in years. ")
    # ~24 tokens per filler sentence; deliberately conservative so the prompt
    # lands under, not over, the requested budget.
    repeats = max(1, needle_tokens // 24)
    secret = "84713"
    for depth_pct in (10, 50, 90):
        cut = repeats * depth_pct // 100
        haystack = (filler * cut + f"The vault access code is {secret}. " +
                    filler * (repeats - cut))
        prompt = (f"{haystack}\n"
                  "Question: What is the vault access code? "
                  "Answer with the number only.\nAnswer:")
        choice = _complete(base_url, model, prompt, 16)
        text = choice["text"]
        print(f"[probe:needle] depth={depth_pct}% prompt_chars={len(prompt)} "
              f"-> {text!r}")
        if secret not in text:
            raise ProbeFailure(
                f"[probe:needle] code {secret} planted at {depth_pct}% depth "
                f"was not retrieved; got {text!r}")
    print("[probe:needle] OK, retrieved at 10%/50%/90% depth")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model",
                        default=None,
                        help="Served model id; discovered from /v1/models "
                        "when omitted.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--long-tokens", type=int, default=512)
    parser.add_argument("--needle-tokens",
                        type=int,
                        default=0,
                        help="Prompt budget for the needle probe. 0 skips it.")
    parser.add_argument(
        "--allow-incoherent",
        action="store_true",
        help="Do not assert that the served model produces meaningful text. "
        "Every structural invariant still runs and still fails the probe: "
        "the endpoint answers, greedy decoding is reproducible, concurrent "
        "requests do not leak each other's markers, a long request comes "
        "back non-empty. Use it for a checkpoint that is known not to be a "
        "working language model -- a depth-sliced snapshot cut down so a "
        "single host can serve it -- where the load and serve paths are "
        "under test and the text is expected to be meaningless.")
    args = parser.parse_args()

    served = probe_health(args.base_url)
    model = args.model or served
    coherent = not args.allow_incoherent
    if not coherent:
        print("[probe] --allow-incoherent: text-quality assertions are off; "
              "structural invariants still enforced")

    try:
        probe_determinism(args.base_url, model, max_tokens=32)
        probe_isolation(args.base_url,
                        model,
                        args.concurrency,
                        require_coherence=coherent)
        probe_isolation(args.base_url,
                        model,
                        args.concurrency,
                        ragged=True,
                        require_coherence=coherent)
        probe_long_generation(args.base_url,
                              model,
                              args.long_tokens,
                              require_coherence=coherent)
        if args.needle_tokens and coherent:
            probe_needle(args.base_url, model, args.needle_tokens)
        elif args.needle_tokens:
            print("[probe:needle] SKIPPED (--allow-incoherent: retrieval is "
                  "not defined for a model that is not a language model)")
        else:
            print("[probe:needle] SKIPPED (--needle-tokens not set)")
    except ProbeFailure as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    print("All probes passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
