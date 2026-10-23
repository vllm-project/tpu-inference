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
"""Greedy-output correctness check: disaggregated P/D against a single-node baseline.

Prompts are built as an explicit list of token ids, which the completions API
accepts, so the token count of a prompt is exact. That matters because every
boundary in the connector is expressed in tokens and blocks: `request_finished()`
drops the trailing partial block and `get_num_new_matched_tokens()` mirrors it with
`round_down(len(prompt), block_size)`. A prompt specified in words leaves the token
count unknown and variable.

Prompts default to real prose. Random-letter prompts are off-manifold, and the
model's top-1 and top-2 logits are then frequently tied to within a bfloat16 ULP, so
greedy argmax is settled by float noise. Controls that contain no KV transfer at all
-- unchunked versus chunked prefill on two identical servers, and prefix-cache miss
versus hit on a single server, where the reused KV is bit-identical by construction
-- mismatch at a comparable rate on that distribution. Greedy text equality over
random letters therefore measures argmax stability under any change of execution
path, not KV correctness. `--prompt_style gibberish` is kept for that stability
probe; pair it with a non-zero `--max_mismatch_rate`.

For the same reason a mismatch rate of exactly zero is not achievable in general, so
`--max_mismatch_rate` (default 0.0) is a floor you can raise. `--control_url` is the
better option: it measures this run's own instability floor, on the same model,
prompts and hardware, and gates on `disagg <= floor`.

The shape of a failure says more than its rate. A transfer fault diverges at
character 0 and at a rate near 1.0; argmax instability diverges deep into the
continuation at a few percent. Each mismatch is reported with the number of leading
characters the two outputs share, for that reason.

To diagnose a failure rather than merely detect one, `pd_probe.py` gives per-length
mismatch rates and first-token logprob deltas across the block boundary, and
`numerics_control.py` runs the same comparison with the KV transfer removed.
"""

import argparse
import json
import logging
import math
import random

import requests

# Real prose, so the default probe distribution is on-manifold and the model is
# confident. The content is irrelevant; it only has to tokenize to enough tokens.
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


def build_prompt_ids(tokenizer, num_tokens, style, seed):
    """Return exactly `num_tokens` token ids, so block boundaries are addressable."""
    rng = random.Random(seed)
    if style == "gibberish":
        words = [
            rng.choice("abcdefghijklmnopqrstuvwxyz")
            for _ in range(num_tokens * 2 + 32)
        ]
        text = " ".join(words)
    else:
        body = " ".join(_PROSE.split())
        reps = math.ceil((num_tokens * 8) / max(len(body), 1)) + 1
        # Rotate the starting point so that successive prompts differ.
        text = " ".join([body] * reps)[rng.randrange(0, max(len(body), 1)):]

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < num_tokens:
        raise RuntimeError(f"produced {len(ids)} tokens, needed {num_tokens}")
    return ids[:num_tokens]


def send_request(url, prompt_ids, model_name, output_length):
    """Sends a request to the LLM and returns the response."""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt_ids,
        "max_tokens": output_length,
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        # Assuming the response format is similar to OpenAI's
        return response.json()["choices"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Unexpected response format - {e}"


def main(args):
    """Main function to run the correctness test."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mismatches = 0
    control_mismatches = 0
    for i in range(args.num_requests):
        logging.info(f"Sending request {i+1}/{args.num_requests}...")
        prompt_ids = build_prompt_ids(tokenizer, args.input_length,
                                      args.prompt_style, i)

        baseline_response = send_request(args.baseline_url, prompt_ids,
                                         args.model, args.output_length)
        disagg_response = send_request(args.disagg_url, prompt_ids, args.model,
                                       args.output_length)

        if args.control_url:
            # The floor for this run. Send the same prompt to the control server
            # twice: the first call is a cache miss and prefills all N tokens, the
            # second is a hit and recomputes only the tail from the cached -- and
            # therefore bit-identical -- KV. Comparing the two isolates the change
            # of execution path with the byte question settled a priori, which is
            # the closest transfer-free analogue of what the disagg decoder does.
            # (Comparing the control against the baseline instead would just be two
            # independent cold prefills, which agree, and would measure nothing.)
            control_miss = send_request(args.control_url, prompt_ids,
                                        args.model, args.output_length)
            control_hit = send_request(args.control_url, prompt_ids,
                                       args.model, args.output_length)
            if control_miss != control_hit:
                control_mismatches += 1

        if baseline_response != disagg_response:
            mismatches += 1
            common = 0
            for cb, cd in zip(baseline_response, disagg_response):
                if cb != cd:
                    break
                common += 1
            logging.info(f"  MISMATCH FOUND for prompt {i+1}:")
            # Where they diverge is the useful part: divergence at character 0
            # points at the handoff, divergence deep into the continuation points
            # at drift accumulated during decode.
            logging.info(f"    Diverged after {common} identical characters")
            logging.info(f"    Baseline: {baseline_response}")
            logging.info(f"    Disagg:   {disagg_response}")
        else:
            logging.info(f"  Responses match for prompt {i+1}.")

    rate = mismatches / args.num_requests if args.num_requests else 0.0
    logging.info("\n--- Test Summary ---")
    logging.info(f"style={args.prompt_style} input_tokens={args.input_length} "
                 f"output_tokens={args.output_length}")
    if mismatches == 0:
        logging.info("All responses matched! The services are consistent.")
    else:
        logging.info(
            f"{mismatches}/{args.num_requests} requests had mismatched "
            f"responses (rate {rate:.3f}).")
    threshold = args.max_mismatch_rate
    if args.control_url:
        control_rate = (control_mismatches /
                        args.num_requests if args.num_requests else 0.0)
        logging.info(
            f"control (cache miss vs hit) mismatch rate, this run's floor: "
            f"{control_mismatches}/{args.num_requests} "
            f"({control_rate:.3f})")
        # Gate relative to the floor actually measured here, not to an absolute
        # constant: greedy argmax is not stable to execution-path changes even
        # when the KV is bit-identical, and how unstable depends on the model,
        # the prompt distribution and the hardware.
        threshold = max(threshold, control_rate)
    logging.info("--------------------")

    if rate > threshold:
        raise Exception(
            f"Mismatch rate {rate:.3f} exceeds threshold {threshold:.3f}. Use "
            f"pd_probe.py to localise it across the block boundary, and "
            f"numerics_control.py to establish this run's noise floor, before "
            f"concluding the KV transfer is at fault. A real transfer bug "
            f"diverges at character 0 and at a rate near 1.0; divergence deep "
            f"into the continuation at a few percent is argmax instability.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a correctness test between two LLM services.")
    parser.add_argument("--num_requests",
                        type=int,
                        default=20,
                        help="Number of requests to send.")
    parser.add_argument("--input_length",
                        type=int,
                        default=256,
                        help="Exact length of each input prompt, in tokens.")
    parser.add_argument("--output_length",
                        type=int,
                        default=32,
                        help="Number of tokens to generate per response.")
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="text",
        choices=["text", "gibberish"],
        help="'text' is real prose and is the correctness gate. "
        "'gibberish' is random letters: an argmax-stability probe "
        "that even bit-identical-KV controls fail, so pair it "
        "with a non-zero --max_mismatch_rate.")
    parser.add_argument("--max_mismatch_rate",
                        type=float,
                        default=0.0,
                        help="Raise if the mismatch rate exceeds this. Note "
                        "that a rate of exactly 0 is not always achievable "
                        "even with bit-identical KV; prefer --control_url.")
    parser.add_argument("--control_url",
                        type=str,
                        default=None,
                        help="Optional endpoint used to measure this run's "
                        "instability floor. It must be a server started with "
                        "--enable-prefix-caching: each prompt is sent to it "
                        "twice, so the second call recomputes only the tail "
                        "from bit-identical cached KV, and any disagreement "
                        "between the two is argmax instability with the KV "
                        "held constant. The gate becomes disagg <= floor.")
    parser.add_argument("--baseline_url",
                        type=str,
                        default="http://localhost:9400/v1/completions",
                        help="URL of the baseline LLM service.")
    parser.add_argument("--disagg_url",
                        type=str,
                        default="http://localhost:8000/v1/completions",
                        help="URL of the disaggregated LLM service.")
    parser.add_argument("--model",
                        type=str,
                        default="Qwen/Qwen3-0.6B",
                        help="Name of the model to use for the requests.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)
