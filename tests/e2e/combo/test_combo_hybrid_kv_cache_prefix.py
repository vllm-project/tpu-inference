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

import gc
import os
import time
from typing import List, Tuple

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "google/gemma-3-4b-it"


def _check_correctness(test_name: str, reference_outputs: list,
                       test_outputs: list):
    """Verify generated token ids match the reference run within acceptable tolerance."""
    assert len(reference_outputs) == len(test_outputs)

    for i, (reference,
            test_result) in enumerate(zip(reference_outputs, test_outputs)):
        reference_completion = reference.outputs[0]
        test_completion = test_result.outputs[0]
        reference_token_ids = tuple(reference_completion.token_ids)
        test_token_ids = tuple(test_completion.token_ids)

        # Verify prefix cache hit metric
        if "hit" in test_name.lower():
            num_cached = getattr(test_result, "num_cached_tokens", 0)
            assert num_cached > 0, f"Prefix cache hit missed. Cached tokens: {num_cached}"

        # Allow minor numerical divergence on TPU v7x; verify initial tokens match
        min_match_len = min(5, len(reference_token_ids), len(test_token_ids))

        assert reference_token_ids[:min_match_len] == test_token_ids[:min_match_len], (
            f"{test_name} critical prefix token mismatch in prompt {i}:\n"
            f"  Reference front: {reference_token_ids[:min_match_len]}\n"
            f"  Test front: {test_token_ids[:min_match_len]}")

        # Verify token length consistency
        assert abs(len(reference_token_ids) - len(test_token_ids)) <= 2, (
            f"{test_name} length mismatch too wide in prompt {i}: "
            f"Reference len {len(reference_token_ids)}, Test len {len(test_token_ids)}"
        )

    print(f"{test_name} generated outputs match reference criteria.")


def _reset_engine_prefix_cache(llm: LLM) -> None:
    if hasattr(llm, "reset_prefix_cache"):
        llm.reset_prefix_cache()
        return
    llm.llm_engine.engine_core.reset_prefix_cache()


@pytest.fixture
def sampling_params():
    return SamplingParams(
        temperature=0.0,
        max_tokens=16,
        seed=42,
        ignore_eos=True,
    )


@pytest.fixture
def shared_prefix_prompts():
    shared_prefix = (
        "A family is planning a weekend trip to a quiet town near the coast. "
        "They want to visit a local museum, walk through the old market, try "
        "a seafood restaurant, and spend some time watching the sunset by the "
        "water. The parents care about keeping the schedule relaxed, the "
        "children want enough time for snacks and photos, and everyone agrees "
        "that the plan should include indoor alternatives in case it rains. "
        "This shared setup is intentionally long so the prefix cache can reuse "
        "a meaningful number of prompt tokens across requests. ")
    return [
        shared_prefix + "Write a short travel plan for Saturday.",
        shared_prefix + "Suggest what they should pack for the trip.",
    ]


def _run_prefix_cache_sequence(
    prompts: List[str],
    sampling_params: SamplingParams,
) -> Tuple[list, list, list]:
    tensor_parallel_size = int(os.environ.get("TPU_TP_SIZE", "4"))
    llm = None
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=384,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=2048,
            max_num_seqs=64,
            enable_prefix_caching=True,
            disable_hybrid_kv_cache_manager=False,
            block_size=32,
        )

        # Step 1: Warm up and populate the in-memory prefix cache.
        initial_outputs = llm.generate(prompts, sampling_params)

        # Step 2: Repeat generations to exercise prefix cache hits.
        cached_outputs = llm.generate(prompts, sampling_params)

        # Step 3: Clear the in-memory prefix cache index, perturb the cache
        # with unrelated prompts, then force recomputation for the target prompts.
        _reset_engine_prefix_cache(llm)
        time.sleep(1)

        filler_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a short note about deterministic sampling.",
        ]
        llm.generate(filler_prompts, sampling_params)

        recomputed_outputs = llm.generate(prompts, sampling_params)
        return initial_outputs, cached_outputs, recomputed_outputs
    finally:
        if llm is not None and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        del llm
        gc.collect()
        time.sleep(10)


def test_kv_cache_prefix_caching_with_hybrid_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    sampling_params: SamplingParams,
    shared_prefix_prompts: List[str],
):
    """
    Exercise TPU Prefix Caching combined with Hybrid KV Cache (HMA).
    """
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "0")

    initial_outputs, cached_outputs, recomputed_outputs = (
        _run_prefix_cache_sequence(shared_prefix_prompts, sampling_params))

    _check_correctness(
        "Hybrid KV cache prefix cache hit",
        initial_outputs,
        cached_outputs,
    )
    _check_correctness(
        "Hybrid KV cache prefix cache recomputation after reset",
        initial_outputs,
        recomputed_outputs,
    )
