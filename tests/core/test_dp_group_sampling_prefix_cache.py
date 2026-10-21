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

# Tests for group sampling with prefix caching.
#
# Correctness tests verify that:
#   1. Group sampling (n=16) with temperature > 0 produces diverse outputs.
#
# Performance tests verify that:
#   1. Group sampling (n=16) on the same prompt benefits from prefix caching.

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
    """Shared LLM instance with prefix caching enabled."""
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "0")
    engine = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=True,
        data_parallel_size=2,
    )
    yield engine
    engine.llm_engine.engine_core.shutdown()


class TestGroupSamplingPrefixCache:
    """Group sampling (n=16) on the same prompt should benefit from prefix
    caching and produce diverse outputs when temperature > 0.
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

    def test_group_sampling_produces_diverse_outputs(self, llm: LLM):
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
        assert len(texts) > 10, (
            "Group sampling with temperature=1.0 should produce diverse "
            "outputs but some were identical")

    def test_group_sampling_hits_prefix_cache(self, llm: LLM):
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

        # First call – populates the prefix cache for *prompt*.
        t_first = self._generate_timed(llm, [prompt],
                                       sampling_params,
                                       num_runs=1)

        # Second call (same prompt) – should hit the prefix cache.
        t_cached = self._generate_timed(llm, [prompt],
                                        sampling_params,
                                        num_runs=3)

        speedup = t_first / t_cached if t_cached > 0 else 0

        print("✓ Group sampling prefix-cache test results:")
        print(f"  First (cold) call:     {t_first:.3f}s")
        print(f"  Cached (same prompt):  {t_cached:.3f}s")
        print(f"  Speedup:               {speedup:.2f}x")

        assert speedup >= 1.0, (
            f"Expected cached call to be at least as fast as the first "
            f"call, got {speedup:.2f}x")
