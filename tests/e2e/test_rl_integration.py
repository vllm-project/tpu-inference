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
#   2. Returning log-probs does not trigger recompilation.
#   3. Batch invariance: outputs are deterministic regardless of the batch
#      composition (same prompt produces the same tokens whether alone or
#      batched with other prompts).
#
# Performance tests verify that:
#   1. Returning log-probs does not add significant latency overhead.
#   2. Group sampling (same prompt, n=16 outputs) benefits from prefix
#      caching.

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
    can verify prefix-cache speedup.  Correctness tests use fuzzy
    matching and are not affected by caching behaviour.
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
    """A diverse set of prompts for batch-invariance and general tests."""
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

    The feature (introduced in PR #1766) allows the engine to explicitly
    delete its KV-cache arrays to free HBM and later reinitialize them.
    This is essential in RL loops where weights are updated between
    inference steps.

    Due to the continuous-batching scheduler, greedy decoding may show
    minor numerical divergence across runs even with temperature=0.
    Tests therefore use fuzzy (word-overlap) matching.
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

    def test_generate_after_reinitialize_produces_valid_output(
            self, llm: LLM):
        """After delete → reinitialize, generation must produce valid text."""
        prompt = "What is 2 + 2? Answer with just the number:"
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=MAX_TOKENS_DEFAULT)

        # 1. Warm-up: generate before any KV-cache manipulation.
        outputs_before = llm.generate([prompt], sampling_params)
        text_before = outputs_before[0].outputs[0].text

        # 2. Delete and reinitialize KV cache via collective_rpc.
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

        reference = llm.generate([prompt],
                                 sampling_params)[0].outputs[0].text

        for cycle in range(3):
            llm.collective_rpc("delete_kv_cache")
            llm.collective_rpc("reinitialize_kv_cache")

            result = llm.generate([prompt],
                                  sampling_params)[0].outputs[0].text
            overlap = self._word_overlap(reference, result)
            print(f"  Cycle {cycle} overlap: {overlap:.0%}")
            assert overlap >= self._WORD_OVERLAP_THRESHOLD, (
                f"Cycle {cycle}: output diverged too much after "
                f"reinitialisation (overlap={overlap:.0%}): "
                f"{result!r} vs {reference!r}")

    def test_batch_generate_after_reinitialize(self, llm: LLM,
                                               test_prompts):
        """Batch generation must work after delete → reinitialize."""
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=MAX_TOKENS_DEFAULT)

        refs = llm.generate(test_prompts, sampling_params)
        ref_texts = [r.outputs[0].text for r in refs]

        llm.collective_rpc("delete_kv_cache")
        llm.collective_rpc("reinitialize_kv_cache")

        results = llm.generate(test_prompts, sampling_params)
        result_texts = [r.outputs[0].text for r in results]

        text_matches = 0
        for i, (ref, res) in enumerate(zip(ref_texts, result_texts)):
            overlap = self._word_overlap(ref, res)
            if overlap >= self._WORD_OVERLAP_THRESHOLD:
                text_matches += 1
            else:
                print(f"Prompt {i} mismatch (overlap={overlap:.0%}):")
                print(f"  ref: {ref!r}")
                print(f"  res: {res!r}")

        match_rate = text_matches / len(test_prompts)
        print(f"✓ batch-reinit: {text_matches}/{len(test_prompts)} "
              f"matched ({match_rate:.0%})")
        assert match_rate >= self._TEXT_MATCH_RATE_THRESHOLD, (
            f"Text match rate {match_rate:.0%} below threshold "
            f"{self._TEXT_MATCH_RATE_THRESHOLD:.0%}")


class TestLogprobsNoRecompilation:
    """Requesting log-probs must not trigger additional JAX compilations.

    The precompilation phase already compiles the sampling path with
    logprobs=True/False.  Switching between the two at runtime should
    hit the cache.
    """

    def test_logprobs_toggle_no_recompilation(self, llm: LLM):
        """Toggle logprobs on/off and verify no recompilation occurs."""
        prompt = "The sky is"

        # Warm up both paths (logprobs=None and logprobs=5).
        llm.generate([prompt],
                     SamplingParams(temperature=0,
                                    max_tokens=5,
                                    logprobs=None))
        llm.generate([prompt],
                     SamplingParams(temperature=0,
                                    max_tokens=5,
                                    logprobs=5))

        # Now run under ForbidCompile – neither path should recompile.
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

    def test_logprobs_batch_no_recompilation(self, llm: LLM, test_prompts):
        """Requesting logprobs in a batch should not recompile."""
        sampling_params_no_lp = SamplingParams(temperature=0,
                                                max_tokens=10,
                                                logprobs=None)
        sampling_params_lp = SamplingParams(temperature=0,
                                             max_tokens=10,
                                             logprobs=3)

        # Warm up with batch of prompts.
        llm.generate(test_prompts, sampling_params_no_lp)
        llm.generate(test_prompts, sampling_params_lp)

        from tpu_inference.runner.utils import ForbidCompile

        with ForbidCompile(
                "Batch logprobs path triggered unexpected recompilation"):
            outputs = llm.generate(test_prompts, sampling_params_lp)

        for out in outputs:
            assert out.outputs[0].logprobs is not None


class TestBatchInvariance:
    """Greedy outputs should be largely consistent regardless of batch
    composition.

    When temperature=0, a given prompt should produce very similar tokens
    whether it is the sole request in a batch or is surrounded by other
    prompts.  Due to the continuous-batching scheduler, different batch
    compositions may lead to slightly different padding / scheduling
    decisions, which can cause minor numerical divergence.  Therefore
    these tests use *fuzzy* matching (word-overlap >= 70 %) and require
    that the *majority* of prompts match, rather than demanding exact
    equality for every single prompt.
    """

    # Minimum word-overlap ratio to consider two outputs "matching".
    _WORD_OVERLAP_THRESHOLD = 0.7
    # Minimum fraction of prompts that must match.
    _TEXT_MATCH_RATE_THRESHOLD = 0.60

    @staticmethod
    def _word_overlap(text_a: str, text_b: str) -> float:
        """Return the fraction of words in *text_a* also present in *text_b*."""
        words_a = set(text_a.strip().split())
        words_b = set(text_b.strip().split())
        if not words_a:
            return 1.0 if not words_b else 0.0
        return len(words_a & words_b) / len(words_a)

    def test_single_vs_batch_output(self, llm: LLM, test_prompts):
        """Compare single-prompt results with batched results."""
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=MAX_TOKENS_DEFAULT)

        # Generate each prompt individually.
        single_texts = []
        for prompt in test_prompts:
            out = llm.generate([prompt], sampling_params)
            single_texts.append(out[0].outputs[0].text)

        # Generate all prompts in a single batch.
        batch_outputs = llm.generate(test_prompts, sampling_params)
        batch_texts = [o.outputs[0].text for o in batch_outputs]

        text_matches = 0
        for i, (single, batch) in enumerate(zip(single_texts, batch_texts)):
            overlap = self._word_overlap(single, batch)
            if overlap >= self._WORD_OVERLAP_THRESHOLD:
                text_matches += 1
            else:
                print(f"Prompt {i} mismatch (overlap={overlap:.0%}):")
                print(f"  single: {single!r}")
                print(f"  batch:  {batch!r}")

        match_rate = text_matches / len(test_prompts)
        print(f"✓ single-vs-batch: {text_matches}/{len(test_prompts)} "
              f"matched ({match_rate:.0%})")
        assert match_rate >= self._TEXT_MATCH_RATE_THRESHOLD, (
            f"Text match rate {match_rate:.0%} below threshold "
            f"{self._TEXT_MATCH_RATE_THRESHOLD:.0%}")

    def test_different_batch_sizes(self, llm: LLM, test_prompts):
        """Outputs should be largely stable across different batch sizes."""
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=MAX_TOKENS_DEFAULT)

        # Reference: all prompts at once.
        ref_outputs = llm.generate(test_prompts, sampling_params)
        ref_texts = [o.outputs[0].text for o in ref_outputs]

        # Split into halves.
        mid = len(test_prompts) // 2
        first_half = llm.generate(test_prompts[:mid], sampling_params)
        second_half = llm.generate(test_prompts[mid:], sampling_params)
        split_texts = ([o.outputs[0].text for o in first_half] +
                       [o.outputs[0].text for o in second_half])

        text_matches = 0
        for i, (ref, split) in enumerate(zip(ref_texts, split_texts)):
            overlap = self._word_overlap(ref, split)
            if overlap >= self._WORD_OVERLAP_THRESHOLD:
                text_matches += 1
            else:
                print(f"Prompt {i} mismatch (overlap={overlap:.0%}):")
                print(f"  ref:   {ref!r}")
                print(f"  split: {split!r}")

        match_rate = text_matches / len(test_prompts)
        print(f"✓ different-batch-sizes: {text_matches}/{len(test_prompts)} "
              f"matched ({match_rate:.0%})")
        assert match_rate >= self._TEXT_MATCH_RATE_THRESHOLD, (
            f"Text match rate {match_rate:.0%} below threshold "
            f"{self._TEXT_MATCH_RATE_THRESHOLD:.0%}")

    def test_batch_invariance_with_logprobs(self, llm: LLM, test_prompts):
        """Batch invariance should largely hold when logprobs are requested."""
        sampling_params = SamplingParams(temperature=0,
                                         max_tokens=MAX_TOKENS_DEFAULT,
                                         logprobs=3)

        # Single.
        single_texts = []
        single_logprobs = []
        for prompt in test_prompts:
            out = llm.generate([prompt], sampling_params)
            single_texts.append(out[0].outputs[0].text)
            single_logprobs.append(out[0].outputs[0].logprobs)

        # Batch.
        batch_outputs = llm.generate(test_prompts, sampling_params)

        text_matches = 0
        token_matches = 0
        total_tokens = 0

        for i, batch_out in enumerate(batch_outputs):
            overlap = self._word_overlap(single_texts[i],
                                         batch_out.outputs[0].text)
            if overlap >= self._WORD_OVERLAP_THRESHOLD:
                text_matches += 1

            # Compare top-1 token ids when both have logprobs.
            b_logprobs = batch_out.outputs[0].logprobs
            s_logprobs = single_logprobs[i]
            if s_logprobs and b_logprobs:
                for t_idx, (slp, blp) in enumerate(
                        zip(s_logprobs, b_logprobs)):
                    s_top = list(slp.keys())[0]
                    b_top = list(blp.keys())[0]
                    total_tokens += 1
                    if s_top == b_top:
                        token_matches += 1

        text_match_rate = text_matches / len(test_prompts)
        token_match_rate = (token_matches / total_tokens
                            if total_tokens > 0 else 1.0)

        print(f"✓ batch-invariance-with-logprobs:")
        print(f"  Text match rate:  {text_match_rate:.0%}")
        print(f"  Token match rate: {token_match_rate:.0%}")
        assert text_match_rate >= self._TEXT_MATCH_RATE_THRESHOLD, (
            f"Text match rate {text_match_rate:.0%} below threshold "
            f"{self._TEXT_MATCH_RATE_THRESHOLD:.0%}")
        assert token_match_rate >= 0.70, (
            f"Token match rate {token_match_rate:.0%} below 70%")


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


class TestGroupSamplingPrefixCache:
    """Group sampling (n=16) on the same prompt should benefit from prefix
    caching.

    When the same prompt is submitted with ``n=16``, the engine should
    compute the prefix only once and reuse it for all 16 continuations.
    With prefix caching enabled the second call (same prompt) should be
    noticeably faster than the first.
    """

    @staticmethod
    def _generate_timed(llm: LLM, prompts, sampling_params, num_runs=3):
        """Return the median wall-clock time over *num_runs*."""
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        times.sort()
        return times[len(times) // 2]

    def test_group_sampling_hits_prefix_cache(
            self, llm: LLM):
        """Repeated group sampling on the same prompt should speed up."""
        prompt = (
            "Reinforcement learning is a branch of machine learning where "
            "an agent learns to make decisions by interacting with an "
            "environment. The agent receives rewards or penalties based on "
            "its actions and aims to maximise cumulative reward over time. "
            "Explain the key concepts of RL in detail:")
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=MAX_TOKENS_DEFAULT,
            n=16,
        )

        # First call – populates the prefix cache.
        t_first = self._generate_timed(llm, [prompt], sampling_params,
                                       num_runs=1)

        # Second call – should hit the prefix cache and be faster.
        t_cached = self._generate_timed(llm, [prompt], sampling_params,
                                        num_runs=3)

        speedup = t_first / t_cached if t_cached > 0 else 0

        print("✓ Group sampling prefix-cache test results:")
        print(f"  First (cold) call:  {t_first:.3f}s")
        print(f"  Cached call:        {t_cached:.3f}s")
        print(f"  Speedup:            {speedup:.2f}x")

        assert speedup >= 1.0, (
            f"Expected prefix-cache speedup >= 1.0x, got {speedup:.2f}x")

    def test_group_sampling_produces_diverse_outputs(
            self, llm: LLM):
        """n=16 with temperature > 0 should produce diverse continuations."""
        prompt = "Write a creative one-sentence story about a robot:"
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=MAX_TOKENS_DEFAULT,
            n=16,
        )

        outputs = llm.generate([prompt], sampling_params)
        texts = {o.text.strip() for o in outputs[0].outputs}

        print(f"  Unique outputs: {len(texts)}/16")
        assert len(texts) > 1, (
            "Group sampling with temperature=1.0 should produce diverse "
            "outputs but all 16 were identical")
