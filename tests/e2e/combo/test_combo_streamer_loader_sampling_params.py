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
"""End-to-end integration tests to verify parity between runai_streamer loader
and standard HF loader on TPUs across different sampling configurations.
"""

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams

GCS_MODEL_NAME = "gs://vertex-model-garden-public-us/llama3/llama3-8b-hf"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]


def _sampling_cases() -> list[tuple[str, SamplingParams]]:
    # All sampling cases enforce temperature=0 (greedy decoding) because
    # non-deterministic sampling (temperature > 0) makes exact output token
    # verification infeasible between different loader runs.
    return [
        ("greedy", SamplingParams(temperature=0,
                                  max_tokens=10,
                                  ignore_eos=True)),
        ("top_p_greedy",
         SamplingParams(temperature=0,
                        top_p=0.1,
                        max_tokens=10,
                        ignore_eos=True)),
        ("top_k_greedy",
         SamplingParams(temperature=0, top_k=5, max_tokens=10,
                        ignore_eos=True)),
        ("logprobs",
         SamplingParams(temperature=0,
                        max_tokens=5,
                        logprobs=3,
                        ignore_eos=True)),
        ("prompt_logprobs",
         SamplingParams(temperature=0,
                        max_tokens=5,
                        prompt_logprobs=3,
                        ignore_eos=True)),
        ("combined_filters_with_logprobs",
         SamplingParams(temperature=0,
                        top_p=0.9,
                        top_k=50,
                        max_tokens=5,
                        logprobs=3,
                        ignore_eos=True)),
    ]


def _generate_outputs_by_case(model: str,
                              load_format: str | None = None
                              ) -> dict[str, list]:
    llm = None
    try:
        llm_kwargs = {
            "model": model,
            "max_model_len": 128,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 256,
        }
        if load_format is not None:
            llm_kwargs["load_format"] = load_format

        llm = LLM(**llm_kwargs)
        return {
            case_name: llm.generate(PROMPTS, sampling_params)
            for case_name, sampling_params in _sampling_cases()
        }
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def streamer_and_hf_outputs() -> dict[str, dict[str, list]]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Set fake credentials and emulator endpoints to prevent runai_streamer
        # from attempting to load real GCP credentials during unit tests.
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
        monkeypatch.setenv("RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS",
                           "true")
        monkeypatch.setenv("CLOUD_STORAGE_EMULATOR_ENDPOINT",
                           "https://storage.googleapis.com")
        monkeypatch.setenv("MODEL_IMPL_TYPE", "flax_nnx")

        yield {
            "gcs":
            _generate_outputs_by_case(
                GCS_MODEL_NAME,
                load_format="runai_streamer",
            ),
            "hf":
            _generate_outputs_by_case(HF_MODEL_NAME),
        }


def _assert_case_outputs_match(outputs_by_loader: dict[str, dict[str, list]],
                               case_name: str) -> tuple[list, list]:
    gcs_outputs = outputs_by_loader["gcs"][case_name]
    hf_outputs = outputs_by_loader["hf"][case_name]

    _assert_selected_tokens_match(case_name, gcs_outputs, hf_outputs)
    return gcs_outputs, hf_outputs


def _assert_selected_tokens_match(case_name: str, gcs_outputs: list,
                                  hf_outputs: list) -> None:
    assert len(gcs_outputs) == len(hf_outputs)

    for prompt_idx, (gcs_output,
                     hf_output) in enumerate(zip(gcs_outputs, hf_outputs)):
        gcs_completion = gcs_output.outputs[0]
        hf_completion = hf_output.outputs[0]

        assert tuple(gcs_completion.token_ids) == tuple(
            hf_completion.token_ids), (
                f"{case_name} token mismatch for prompt {prompt_idx}:\n"
                f"  GCS text: {gcs_completion.text!r}\n"
                f"  HF text: {hf_completion.text!r}\n"
                f"  GCS token ids: {gcs_completion.token_ids}\n"
                f"  HF token ids: {hf_completion.token_ids}")


def _assert_logprobs_are_valid(case_name: str, outputs: list) -> None:
    for prompt_idx, output in enumerate(outputs):
        completion = output.outputs[0]
        assert completion.logprobs is not None, (
            f"{case_name} missing logprobs for prompt {prompt_idx}")
        assert len(completion.logprobs) == len(completion.token_ids)
        for token_id, token_logprobs in zip(completion.token_ids,
                                            completion.logprobs):
            assert token_logprobs is not None
            assert token_id in token_logprobs
            for _token_id, logprob_obj in token_logprobs.items():
                assert logprob_obj.logprob <= 0


def _assert_prompt_logprobs_are_valid(case_name: str, outputs: list) -> None:
    for prompt_idx, output in enumerate(outputs):
        assert output.prompt_logprobs is not None, (
            f"{case_name} missing prompt_logprobs for prompt {prompt_idx}")
        # The first prompt token does not have prompt_logprobs because it has
        # no preceding context in the prompt, so vLLM returns None for it.
        assert output.prompt_logprobs[0] is None
        for token_logprobs in output.prompt_logprobs[1:]:
            assert token_logprobs is not None
            # With prompt_logprobs=3, we check <= 4 because vLLM returns up to 3 logprobs
            # from the top-3 token list, plus the logprob of the actual prompt token
            # that was processed, which might be outside the top-3 list (totaling 4).
            assert len(token_logprobs) <= 4
            for _token_id, logprob_obj in token_logprobs.items():
                assert logprob_obj.logprob <= 0


def test_runai_streamer_loader_matches_hf_with_greedy_sampling(
    streamer_and_hf_outputs: dict[str, dict[str, list]], ):
    """RunAI streamer loading must preserve deterministic sampling behavior."""
    _assert_case_outputs_match(streamer_and_hf_outputs, "greedy")


def test_runai_streamer_loader_matches_hf_with_greedy_sampling_filters(
    streamer_and_hf_outputs: dict[str, dict[str, list]], ):
    """Greedy top-p and top-k paths must match between loader implementations."""
    for case_name in ("top_p_greedy", "top_k_greedy"):
        _assert_case_outputs_match(streamer_and_hf_outputs, case_name)


def test_runai_streamer_loader_matches_hf_with_logprobs(
    streamer_and_hf_outputs: dict[str, dict[str, list]], ):
    """Generated-token logprobs must stay valid with streamed weights."""
    gcs_outputs, hf_outputs = _assert_case_outputs_match(
        streamer_and_hf_outputs, "logprobs")

    _assert_logprobs_are_valid("logprobs", gcs_outputs)
    _assert_logprobs_are_valid("logprobs", hf_outputs)


def test_runai_streamer_loader_matches_hf_with_prompt_logprobs(
    streamer_and_hf_outputs: dict[str, dict[str, list]], ):
    """Prompt-token logprobs must stay valid with streamed weights."""
    gcs_outputs, hf_outputs = _assert_case_outputs_match(
        streamer_and_hf_outputs, "prompt_logprobs")

    _assert_prompt_logprobs_are_valid("prompt_logprobs", gcs_outputs)
    _assert_prompt_logprobs_are_valid("prompt_logprobs", hf_outputs)


def test_runai_streamer_loader_matches_hf_with_combined_filters_and_logprobs(
    streamer_and_hf_outputs: dict[str, dict[str, list]], ):
    """Combined top-p/top-k filtering with logprobs must match HF loading."""
    case_name = "combined_filters_with_logprobs"
    gcs_outputs, hf_outputs = _assert_case_outputs_match(
        streamer_and_hf_outputs, case_name)

    _assert_logprobs_are_valid(case_name, gcs_outputs)
    _assert_logprobs_are_valid(case_name, hf_outputs)
