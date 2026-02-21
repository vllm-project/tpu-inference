# Update Sequence Parallelism (SP) Support Status

## Description
This PR updates the support status for Sequence Parallelism (SP) in `parallelism_support_matrix.csv` from unverified/untested to `🧪 Experimental`.

## Context & Motivation
Currently, SP passes correctness tests (as seen in PR #520). However, recent internal benchmarks on `v7x8` using `Qwen/Qwen2.5-32B` have shown a performance regression compared to the baseline (10.04 req/s with SP vs. 12.65 req/s without). This contradicts earlier reports of a 6-12% performance improvement.

Given this discrepancy and the current lack of active community demand for SP, we are pausing deep performance debugging. 

This matrix update reflects the accurate state of the feature: it is functionally correct but unoptimized. 

## Community Call to Action
We have opened a tracking issue [#1749](https://github.com/vllm-project/tpu-inference/issues/1749) to gather community feedback. We are asking users to:
1. **Upvote** the issue if SP is a blocker for their use case.
2. **Share Benchmark Data** if they are using SP on different models or hardware, helping us understand if the performance regressions are isolated or systemic.

## Changes Made
- Changed the CorrectnessTest and PerformanceTest status for the `SP` entry in `support_matrices/parallelism_support_matrix.csv` to `🧪 Experimental`.
- We will be adding a legend definition to our documentation reflecting that `🧪` means "Experimental (Unoptimized) - Vote to prioritize".
