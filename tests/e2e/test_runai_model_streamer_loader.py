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

# This file contains end-to-end tests for the RunAI Model Streamer loader.
#
# The RunAI Model Streamer is a high-performance model loader that serves as an
# alternative to the default Hugging Face loader. Instead of downloading a model
# to local disk, it streams the weights from object storage (like GCS) into
# GPU memory. This streaming process is significantly faster than the
# traditional disk-based loading method.

# The tests in this file verify that loading model weights using the
# streamer produces the same results as loading the same model using the
# standard Hugging Face loader. This ensures the correctness of the streamer
# integration.

# The tests are performed by:
# 1. Loading a model from Google Cloud Storage using the `runai_streamer` format.
# 2. Generating output with this model.
# 3. Loading the same model from Hugging Face using the default loader.
# 4. Generating output with this second model.
# 5. Asserting that the outputs from both models are identical.

from __future__ import annotations

import time

import pytest
from vllm import LLM, SamplingParams


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)


# TODO(amacaskill): Once we have GKE owned GCS buckets, add a test for VLLM_XLA_CACHE_PATH set to GCS URI.


def test_correctness_jax_uni_proc_executor(
    sampling_config: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Compare the outputs of a model loaded from GCS via runai_model_streamer
    and a model loaded from Hugging Face. The outputs should be the same.
    These tests attempt to use tensor_parallel_size=1. The model is 16GB,
    # and v6e has 32GB of HBM, so it will fit.
    '''
    # TODO(amacaskill): Replace with GKE owned GCS bucket.
    gcs_model_name = "gs://vertex-model-garden-public-us/llama3/llama3-8b-hf"
    hf_model_name = "meta-llama/Meta-Llama-3-8B"
    prompt = "Hello, my name is"

    # Set ENV variables so that runai_model_streamer uses anonymous GCS access.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    monkeypatch.setenv("RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS", "true")
    monkeypatch.setenv("CLOUD_STORAGE_EMULATOR_ENDPOINT",
                       "https://storage.googleapis.com")
    gcs_llm = LLM(model=gcs_model_name,
                  load_format="runai_streamer",
                  max_model_len=128,
                  max_num_seqs=16,
                  max_num_batched_tokens=256)
    gcs_outputs = gcs_llm.generate([prompt], sampling_config)
    gcs_output_text = gcs_outputs[0].outputs[0].text
    del gcs_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=hf_model_name,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10)  # Wait for TPUs to be released

    assert gcs_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"GCS output: {gcs_output_text}, HF output: {hf_output_text}")


def test_correctness_torchax_uni_proc_executor(
    sampling_config: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Compare the outputs of a codegemma model loaded from GCS via runai_model_streamer
    and a model loaded from Hugging Face. The outputs should be the same.
    """
    # TODO(amacaskill): Replace with GKE owned GCS bucket.
    gcs_model_name = "gs://vertex-model-garden-public-us/codegemma/codegemma-2b"
    hf_model_name = "google/codegemma-2b"
    prompt = "def fibonacci("

    # Set ENV variables so that runai_model_streamer uses anonymous GCS access.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    monkeypatch.setenv("RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS", "true")
    monkeypatch.setenv("CLOUD_STORAGE_EMULATOR_ENDPOINT",
                       "https://storage.googleapis.com")

    gcs_llm = LLM(model=gcs_model_name,
                  load_format="runai_streamer",
                  max_model_len=128,
                  max_num_seqs=16,
                  max_num_batched_tokens=256)
    gcs_outputs = gcs_llm.generate([prompt], sampling_config)
    gcs_output_text = gcs_outputs[0].outputs[0].text
    del gcs_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=hf_model_name,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10)  # Wait for TPUs to be released

    assert gcs_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"GCS output: {gcs_output_text}, HF output: {hf_output_text}")


def test_correctness_torchax_ray_distributed_executor(
    sampling_config: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Compare the outputs of a codegemma model loaded from GCS via
    runai_model_streamer, and a model loaded from
    Hugging Face, both using RayDistributedExecutor. The outputs should be the same.
    Note that these are run on the same node since a multi-node TPU setup would
    require additional changes to the CI pipeline.
    """
    # TODO(amacaskill): Replace with GKE owned GCS bucket.
    gcs_model_name = "gs://vertex-model-garden-public-us/llama3/llama3-8b-hf"
    hf_model_name = "meta-llama/Meta-Llama-3-8B"
    prompt = "def fibonacci("

    # Set ENV variables so that runai_model_streamer uses anonymous GCS access.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fake-project")
    monkeypatch.setenv("RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS", "true")
    monkeypatch.setenv("CLOUD_STORAGE_EMULATOR_ENDPOINT",
                       "https://storage.googleapis.com")

    gcs_llm = LLM(model=gcs_model_name,
                  load_format="runai_streamer",
                  tensor_parallel_size=2,
                  max_model_len=128,
                  max_num_seqs=16,
                  max_num_batched_tokens=256)
    gcs_outputs = gcs_llm.generate([prompt], sampling_config)
    gcs_output_text = gcs_outputs[0].outputs[0].text
    del gcs_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=hf_model_name,
                 tensor_parallel_size=2,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10)  # Wait for TPUs to be released

    assert gcs_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"GCS output: {gcs_output_text}, HF output: {hf_output_text}")
