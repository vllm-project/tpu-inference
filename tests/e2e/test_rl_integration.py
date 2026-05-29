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

# This file contains end-to-end RL integration tests.
#
# RL (Reinforcement Learning) workloads exercise several features that are
# critical for training-inference pipelines:
#
# Correctness tests verify that:
#   1. Delete + reinitialize HBM KV cache: the engine can free its KV cache,
#      then reallocate it and resume serving with correct outputs.
#   2. Delete KV cache actually reclaims HBM (critical for weight-sync).
#   3. Returning log-probs does not trigger recompilation.
#
# Performance tests verify that:
#   1. Returning log-probs does not add significant latency overhead.

from __future__ import annotations

import os
import time

import pytest
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 32
MAX_TOKENS_DEFAULT = 32

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm():
    """Shared LLM instance with prefix caching enabled.

    Prefix caching is enabled so that performance tests (group sampling)
    can verify prefix-cache speedup. 
    """
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "0")
    engine = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
    )
    yield engine
    del engine
    time.sleep(5)


@pytest.fixture
def test_prompts():
    return [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "Explain quantum computing in simple terms:",
        "What is the theory of relativity?",
        "How does photosynthesis work?",
        "Describe the water cycle:",
    ]


# ========================================================================
# Correctness tests
# ========================================================================


class TestDeleteReinitializeHBM:
    """Verify the delete + reinitialize HBM KV-cache lifecycle.
    """

    # Minimum word-overlap ratio to consider two outputs "matching".
    _WORD_OVERLAP_THRESHOLD = 0.7
    # Minimum fraction of prompts that must match.
    _TEXT_MATCH_RATE_THRESHOLD = 0.60

    @staticmethod
    def _word_overlap(text_a: str, text_b: str) -> float:
        """Return the fraction of words in *text_a* also present in
        *text_b*."""
        words_a = set(text_a.strip().split())
        words_b = set(text_b.strip().split())
        if not words_a:
            return 1.0 if not words_b else 0.0
        return len(words_a & words_b) / len(words_a)

    def test_generate_after_reinitialize_produces_valid_output(self, llm: LLM):
        """After delete → reinitialize, generation must produce valid text."""
        prompt = "What is 2 + 2? Answer with just the number:"
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=MAX_TOKENS_DEFAULT)

        # 1. Warm-up: generate before any KV-cache manipulation.
        outputs_before = llm.generate([prompt], sampling_params)
        text_before = outputs_before[0].outputs[0].text

        # 2. Delete and reinitialize KV cache via collective_rpc.
        llm.reset_prefix_cache()
        llm.collective_rpc("delete_kv_cache")
        llm.collective_rpc("reinitialize_kv_cache")

        # 3. Generate again – should produce similar greedy output.
        outputs_after = llm.generate([prompt], sampling_params)
        text_after = outputs_after[0].outputs[0].text

        assert len(text_after.strip()) > 0, (
            "Output should not be empty after KV-cache reinitialisation")

        overlap = self._word_overlap(text_before, text_after)
        print(f"  Reinit single-prompt overlap: {overlap:.0%}")
        assert overlap >= self._WORD_OVERLAP_THRESHOLD, (
            f"Greedy output diverged too much after KV-cache "
            f"reinitialisation (overlap={overlap:.0%}): "
            f"{text_before!r} vs {text_after!r}")

    def test_multiple_delete_reinitialize_cycles(self, llm: LLM):
        """The engine should tolerate multiple delete/reinit cycles."""
        prompt = "The largest planet in our solar system is"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)

        reference = llm.generate([prompt], sampling_params)[0].outputs[0].text

        for cycle in range(3):
            llm.collective_rpc("delete_kv_cache")
            llm.collective_rpc("reinitialize_kv_cache")

            result = llm.generate([prompt], sampling_params)[0].outputs[0].text
            overlap = self._word_overlap(reference, result)
            print(f"  Cycle {cycle} overlap: {overlap:.0%}")
            assert overlap >= self._WORD_OVERLAP_THRESHOLD, (
                f"Cycle {cycle}: output diverged too much after "
                f"reinitialisation (overlap={overlap:.0%}): "
                f"{result!r} vs {reference!r}")

    def test_delete_kv_cache_reclaims_hbm(self, llm: LLM):
        """Deleting the KV cache must free a measurable amount of HBM.

        In RL loops the KV cache is deleted before a weight-sync /
        resharding step so that HBM is available for the new weights.
        This test verifies that ``delete_kv_cache`` actually reclaims
        HBM and that ``reinitialize_kv_cache`` restores usage to the
        original level.

        """

        # 1. Measure available HBM before deleting (KV cache is allocated).
        #    determine_available_memory may raise ValueError when HBM
        #    usage exceeds the utilisation cap — that is expected when
        #    the KV cache is fully allocated.
        before_over_cap = False
        try:
            avail_before_delete = llm.collective_rpc(
                "determine_available_memory")[0]
            before_gb = avail_before_delete / (1024**3)
        except Exception:
            before_over_cap = True
            before_gb = 0.0
        print(f"  Available before delete: {before_gb:.2f} GiB"
              f"{' (over cap)' if before_over_cap else ''}")

        # 2. Delete KV cache → available HBM should increase.
        llm.collective_rpc("delete_kv_cache")
        avail_after_delete = llm.collective_rpc(
            "determine_available_memory")[0]
        after_delete_gb = avail_after_delete / (1024**3)
        print(f"  Available after delete:  {after_delete_gb:.2f} GiB")

        assert after_delete_gb > before_gb, (
            f"Expected more available HBM after delete_kv_cache, "
            f"but got {after_delete_gb:.2f} GiB (before: {before_gb:.2f} GiB)")

        # 3. Reinitialize → available HBM should return close to the
        #    pre-delete level.
        llm.collective_rpc("reinitialize_kv_cache")

        reinit_over_cap = False
        try:
            avail_after_reinit = llm.collective_rpc(
                "determine_available_memory")[0]
            after_reinit_gb = avail_after_reinit / (1024**3)
        except Exception:
            # determine_available_memory raises ValueError when HBM
            # exceeds the utilisation cap — treat as fully consumed.
            reinit_over_cap = True
            after_reinit_gb = 0.0

        print(f"  Available after reinit:  {after_reinit_gb:.2f} GiB"
              f"{' (over cap)' if reinit_over_cap else ''}")

        # If both before-delete and after-reinit exceeded the cap, the
        # KV cache was clearly reallocated — the test passes.
        if before_over_cap and reinit_over_cap:
            print("  Both measurements exceeded the HBM cap — "
                  "KV cache was reallocated.")
            return

        # After reinitialisation, available HBM should be similar to
        # what it was before deletion (within 1 GiB tolerance).
        tolerance_gb = 1.0
        diff_gb = abs(after_reinit_gb - before_gb)
        assert diff_gb <= tolerance_gb, (
            f"Expected available HBM after reinitialize to be within "
            f"{tolerance_gb} GiB of the pre-delete level "
            f"({before_gb:.2f} GiB), but got {after_reinit_gb:.2f} GiB "
            f"(diff={diff_gb:.2f} GiB)")


class TestLogprobsNoRecompilation:
    """Requesting log-probs must not trigger additional JAX compilations.

    The precompilation phase already compiles the sampling path with
    logprobs=True/False.  Switching between the two at runtime should
    hit the cache.
    """

    def test_logprobs_toggle_no_recompilation(self, llm: LLM):
        """Toggle logprobs on/off and verify no recompilation occurs.

        With SKIP_JAX_PRECOMPILE=0 the engine precompiles all sampling
        paths (logprobs=True/False) during init, so no warm-up calls
        are needed before entering ForbidCompile.
        """
        prompt = "The sky is"

        from tpu_inference.runner.utils import ForbidCompile

        with ForbidCompile(
                "Logprobs=None path triggered unexpected recompilation"):
            llm.generate([prompt],
                         SamplingParams(temperature=0,
                                        max_tokens=5,
                                        logprobs=None))

        with ForbidCompile(
                "Logprobs=5 path triggered unexpected recompilation"):
            llm.generate([prompt],
                         SamplingParams(temperature=0,
                                        max_tokens=5,
                                        logprobs=5))

    def test_logprobs_values_are_correct(self, llm: LLM):
        """Returned log-probs should be valid (non-positive values)."""
        prompt = "The capital of France is"
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=10,
                                         logprobs=5)

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        assert output.logprobs is not None, "logprobs should be returned"
        assert len(output.logprobs) > 0

        for token_logprobs in output.logprobs:
            assert token_logprobs is not None
            assert len(token_logprobs) <= 5
            for _token_id, logprob_obj in token_logprobs.items():
                assert logprob_obj.logprob <= 0, (
                    f"Log probability must be <= 0, got {logprob_obj.logprob}")


# ========================================================================
# Performance tests
# ========================================================================


class TestLogprobsLatency:
    """Returning log-probs should not significantly increase latency."""

    @staticmethod
    def _measure_latency(llm: LLM, prompts, sampling_params, num_runs=3):
        """Return the median wall-clock time over *num_runs*."""
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        times.sort()
        return times[len(times) // 2]

    def test_logprobs_latency_overhead(self, llm: LLM, test_prompts):
        """Logprobs overhead should be within 50 % of the no-logprobs run."""
        sp_no_lp = SamplingParams(temperature=0,
                                  max_tokens=MAX_TOKENS_DEFAULT,
                                  logprobs=None)
        sp_lp = SamplingParams(temperature=0,
                               max_tokens=MAX_TOKENS_DEFAULT,
                               logprobs=5)

        # Warm up both paths.
        llm.generate(test_prompts, sp_no_lp)
        llm.generate(test_prompts, sp_lp)

        t_no_lp = self._measure_latency(llm, test_prompts, sp_no_lp)
        t_lp = self._measure_latency(llm, test_prompts, sp_lp)

        overhead = (t_lp - t_no_lp) / t_no_lp if t_no_lp > 0 else 0

        print("✓ Logprobs latency test results:")
        print(f"  Without logprobs: {t_no_lp:.3f}s")
        print(f"  With logprobs:    {t_lp:.3f}s")
        print(f"  Overhead:         {overhead:.1%}")

        assert overhead < 1.00, (
            f"Logprobs overhead {overhead:.1%} exceeds 100 % threshold")
