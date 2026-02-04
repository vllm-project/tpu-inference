# -*- coding: utf-8 -*-
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
"""
Pytest comparison test: baseline vs expert-parallel (EP) lm_eval results.

This test runs lm_eval with two configurations:
- Baseline: TP=1, enable_expert_parallel=0
- EP: TP=4, enable_expert_parallel=1

Then compares the flex_score and strict_score metrics.
The EP version should not drop below 90% of the baseline scores.

"""

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoreMetrics:
    """Container for flex_score and strict_score with their stderr."""
    flex_score: float
    flex_stderr: float
    strict_score: float
    strict_stderr: float


def cleanup_tpu_resources():
    """Clean up TPU lockfile and kill lingering processes."""
    # Remove TPU lockfile
    subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"],
                   capture_output=True,
                   check=False)

    # Kill any lingering Python processes
    subprocess.run(["sudo", "pkill", "-9", "python3"],
                   capture_output=True,
                   check=False)

    # Wait a bit for resources to free up
    time.sleep(3)


def run_lm_eval_via_script(model_name: str,
                           tensor_parallel_size: int,
                           max_model_len: int,
                           max_num_batched_tokens: int,
                           max_gen_toks: int,
                           enable_expert_parallel: int,
                           use_moe_ep_kernel: int = 0) -> ScoreMetrics:
    """
    Run lm_eval via check_lm_eval.sh script and extract scores.
    
    Uses the existing shell script to ensure proper environment setup.
    """
    script = Path(__file__).parent / "check_lm_eval.sh"

    # Build arguments for the shell script
    cmd = [
        "bash",
        str(script),
        "--model_name",
        model_name,
        "--use_moe_ep_kernel",
        str(use_moe_ep_kernel),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--max_model_len",
        str(max_model_len),
        "--max_num_batched_tokens",
        str(max_num_batched_tokens),
        "--max_gen_toks",
        str(max_gen_toks),
        "--enable_expert_parallel",
        str(enable_expert_parallel),
        "--flex_threshold",
        "0.0",  # Set to 0 so script doesn't fail
        "--strict_threshold",
        "0.0",
    ]

    # Run the script
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr

    # Parse output using same regex pattern as check_lm_eval.sh
    # Look for lines like: "Extracted flexible-extract score: 0.4481"
    # And the table output: |gsm8k_cot |flexible-extract|0.4481|±  |0.0137|
    flex_match = re.search(
        r"flexible-extract.*?\|\s*([0-9.]+)\s*\|\s*±\s*\|\s*([0-9.]+)", output)
    strict_match = re.search(
        r"strict-match.*?\|\s*([0-9.]+)\s*\|\s*±\s*\|\s*([0-9.]+)", output)

    if not flex_match or not strict_match:
        # Try alternate parsing from "Extracted ..." lines
        flex_alt = re.search(r"Extracted flexible-extract score:\s*([0-9.]+)",
                             output)
        strict_alt = re.search(r"Extracted strict-match score:\s*([0-9.]+)",
                               output)
        if flex_alt and strict_alt:
            # Approximate stderr as 0.01 (we don't extract it from alternate format)
            return ScoreMetrics(
                flex_score=float(flex_alt.group(1)),
                flex_stderr=0.01,
                strict_score=float(strict_alt.group(1)),
                strict_stderr=0.01,
            )
        raise RuntimeError(
            f"Failed to extract scores from lm_eval output:\n{output}")

    return ScoreMetrics(
        flex_score=float(flex_match.group(1)),
        flex_stderr=float(flex_match.group(2)),
        strict_score=float(strict_match.group(1)),
        strict_stderr=float(strict_match.group(2)),
    )


# @pytest.mark.ep_comparison
def test_ep_comparison():
    """
    Compare baseline vs EP lm_eval results.
    
    Baseline: TP=1, enable_expert_parallel=0
    EP: TP=4, enable_expert_parallel=1
    
    Assert that EP scores are at least 90% of baseline.
    """
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"

    # Clean up TPU resources before starting
    # cleanup_tpu_resources()

    # Baseline configuration: TP=1, no expert parallel
    print("\n=== Running BASELINE evaluation (TP=1, EP=0) ===")
    baseline_scores = run_lm_eval_via_script(
        model_name=model_name,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_gen_toks=128,
        enable_expert_parallel=0,
        use_moe_ep_kernel=0,
    )
    print(
        f"Baseline scores: flex={baseline_scores.flex_score:.4f}±{baseline_scores.flex_stderr:.4f}, "
        f"strict={baseline_scores.strict_score:.4f}±{baseline_scores.strict_stderr:.4f}"
    )

    # Clean up between runs
    cleanup_tpu_resources()

    # EP configuration: TP=4, enable expert parallel
    print("\n=== Running EP evaluation (TP=4, EP=1) ===")
    ep_scores = run_lm_eval_via_script(
        model_name=model_name,
        tensor_parallel_size=4,
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_gen_toks=128,
        enable_expert_parallel=1,
        use_moe_ep_kernel=1,
    )
    print(
        f"EP scores: flex={ep_scores.flex_score:.4f}±{ep_scores.flex_stderr:.4f}, "
        f"strict={ep_scores.strict_score:.4f}±{ep_scores.strict_stderr:.4f}")

    # Compare scores by similarity to baseline (100% = identical, cannot exceed)
    # Similarity = 1 - |ratio - 1|. Require similarity >= 90%.
    threshold = 0.90

    flex_ratio = ep_scores.flex_score / baseline_scores.flex_score if baseline_scores.flex_score > 0 else 0.0
    strict_ratio = ep_scores.strict_score / baseline_scores.strict_score if baseline_scores.strict_score > 0 else 0.0

    flex_similarity = 1.0 - abs(flex_ratio - 1.0)
    strict_similarity = 1.0 - abs(strict_ratio - 1.0)

    print("\n=== Comparison Results ===")
    print(
        f"Flex ratio: {flex_ratio*100:.1f}% | similarity: {flex_similarity*100:.1f}% "
        f"(threshold: {threshold*100:.0f}%)")
    print(
        f"Strict ratio: {strict_ratio*100:.1f}% | similarity: {strict_similarity*100:.1f}% "
        f"(threshold: {threshold*100:.0f}%)")

    assert flex_similarity >= threshold, (
        f"EP flexible-extract too far from baseline: {ep_scores.flex_score:.4f} vs baseline {baseline_scores.flex_score:.4f} "
        f"(similarity {flex_similarity*100:.1f}% < {threshold*100:.0f}%)")

    assert strict_similarity >= threshold, (
        f"EP strict-match too far from baseline: {ep_scores.strict_score:.4f} vs baseline {baseline_scores.strict_score:.4f} "
        f"(similarity {strict_similarity*100:.1f}% < {threshold*100:.0f}%)")
