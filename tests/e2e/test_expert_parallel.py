# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest


# Worker function to run inference (executed when script is run as main)
def run_inference_worker(parallel_size: int, enable_ep: bool):
    try:
        from vllm import LLM, EngineArgs, SamplingParams
    except ImportError:
        sys.exit(1)

    # Model Qwen1.5-MoE-A2.7B
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team?",
        "In Greek mythology, who is the god of the sea?",
        "What is the capital of Australia?",
        "What is the largest planet in our solar system?",
        "Who developed the theory of general relativity?",
    ]

    # Suppress JAX compilation logs
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    # In vLLM on TPU, 'tensor_parallel_size' defines the distributed mesh size.
    # When enable_expert_parallel is True, experts are sharded across this mesh.
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=128,
        tensor_parallel_size=parallel_size,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        enable_prefix_caching=False,
        additional_config={},
        kv_cache_dtype="auto",
        enable_expert_parallel=enable_ep,
    )

    llm = LLM(**asdict(engine_args))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=True,
        logprobs=1,
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        logprobs = output.outputs[0].logprobs
        top_logprobs = []
        if logprobs:
            for token_logprob_dict in logprobs:
                for token, lp in token_logprob_dict.items():
                    top_logprobs.append((token, lp.logprob))
                    break

        results.append({"text": text, "logprobs": top_logprobs})

    print(json.dumps(results))


# Helper to invoke this script in a subprocess
def run_inference_subprocess(parallel_size: int, enable_ep: bool):
    env = os.environ.copy()

    # Run self as a script
    # Arguments: parallel_size, enable_ep (as '1' or '0')
    cmd = [
        sys.executable, __file__, "--worker",
        str(parallel_size), "1" if enable_ep else "0"
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(
            f"Worker script failed with return code {result.returncode}")

    # Parse the last line as JSON
    lines = result.stdout.strip().split('\n')
    json_line = lines[-1]
    try:
        return json.loads(json_line)
    except json.JSONDecodeError:
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        print("STDOUT:", result.stdout)
        raise RuntimeError("Could not find valid JSON in output")


def test_expert_parallelism_correctness():
    # Baseline: TP=1 (Single Chip, EP Disabled)
    # This acts as the Ground Truth for model correctness.
    print("Running Baseline (TP=1, EP=False)...")
    baseline_results = run_inference_subprocess(parallel_size=1,
                                                enable_ep=False)

    # EP Run: TP=4 (4 Chips, EP Enabled)
    # We use 4 chips so that the 60 experts can be evenly divided (60/4 = 15).
    # Since we only enable expert parallel, this is effectively testing EP.
    print("Running Expert Parallel (TP=4, EP=True)...")
    ep_results = run_inference_subprocess(parallel_size=4, enable_ep=True)

    assert len(baseline_results) == len(ep_results)
    num_prompts = len(baseline_results)

    text_matches = 0
    text_mismatches = 0
    max_logprob_diff = 0.0
    logprob_mismatches = 0

    print(f"Comparing {num_prompts} prompts...")
    for i, (base, ep) in enumerate(zip(baseline_results, ep_results)):
        if base['text'] == ep['text']:
            text_matches += 1
        else:
            text_mismatches += 1
            print(f"Mismatch in prompt {i}:")
            print(f"  Baseline: {base['text']}")
            print(f"  EP:       {ep['text']}")

        if base['logprobs'] and ep['logprobs']:
            for (t1, lp1), (t2, lp2) in zip(base['logprobs'], ep['logprobs']):
                diff = abs(lp1 - lp2)
                max_logprob_diff = max(max_logprob_diff, diff)
                if diff > 1e-3:
                    logprob_mismatches += 1
                    # Only print significant differences to avoid spam
                    if diff > 0.1:
                        print(
                            f"  Significant logprob diff in prompt {i} for token '{t1}' vs '{t2}': {lp1:.4f} vs {lp2:.4f} (diff: {diff:.4f})"
                        )

    print("✓ Correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")
    print(f"  Significant logprob mismatches (>1e-3): {logprob_mismatches}")

    # Allow for some variance due to potential numerical differences
    # but most outputs should match with greedy sampling
    text_match_rate = text_matches / len(baseline_results)
    assert text_match_rate >= 0.9, f"Text match rate {text_match_rate:.2%} is too low"

    # Log probabilities should be very close (allow small numerical errors)
    assert max_logprob_diff < 1.0, f"Max logprob difference {max_logprob_diff} is too large"


if __name__ == '__main__':
    # If run with --worker, execute inference logic logic
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        p_size = int(sys.argv[2])
        is_ep = bool(int(sys.argv[3]))
        run_inference_worker(p_size, is_ep)
    else:
        # Otherwise, run pytest
        # Use -s to show output
        sys.exit(pytest.main(["-v", "-s", __file__]))
