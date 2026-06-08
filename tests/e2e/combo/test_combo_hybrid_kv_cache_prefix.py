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

import time
from typing import List, Tuple

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "google/gemma-3-4b-it"


def _parse_outputs(outputs) -> Tuple[List[str], List[Tuple[int, ...]]]:
    texts = []
    token_ids = []
    for output in outputs:
        completion = output.outputs[0]
        texts.append(completion.text)
        token_ids.append(tuple(completion.token_ids))
    return texts, token_ids


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
        "water. ")
    return [
        shared_prefix + "Write a short travel plan for Saturday.",
        shared_prefix + "Suggest what they should pack for the trip.",
    ]


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

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=192,
        tensor_parallel_size=8,
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        enable_prefix_caching=True,
        disable_hybrid_kv_cache_manager=False,  # Enable Hybrid KV Cache Manager
    )

    try:
        # Step 1: Warm up and populate the in-memory prefix cache
        outputs1 = llm.generate(shared_prefix_prompts, sampling_params)
        texts1, tokens1 = _parse_outputs(outputs1)

        # Step 2: Repeat generations to verify prefix cache hits
        outputs2 = llm.generate(shared_prefix_prompts, sampling_params)
        texts2, tokens2 = _parse_outputs(outputs2)

        assert texts1 == texts2
        assert tokens1 == tokens2

        # Step 3: Clear the in-memory prefix cache index
        _reset_engine_prefix_cache(llm)
        time.sleep(1)

        filler_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a short note about deterministic sampling.",
        ]
        llm.generate(filler_prompts, sampling_params)

        # Step 4: Verify outputs match identically after recomputation
        outputs3 = llm.generate(shared_prefix_prompts, sampling_params)
        texts3, tokens3 = _parse_outputs(outputs3)

        assert texts1 == texts3
        assert tokens1 == tokens3
    finally:
        if 'llm' in locals() and hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        time.sleep(5)