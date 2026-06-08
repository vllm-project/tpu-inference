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
from typing import Optional, Union

import pytest
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def parse_outputs(outputs):
    output_token_ids = []
    generated_texts = []
    for output in outputs:
        completion = output.outputs[0]
        generated_text = completion.text
        token_ids = completion.token_ids
        generated_texts.append(generated_text)
        output_token_ids.append(token_ids)
    return generated_texts, output_token_ids


def get_sampling_config():
    """deterministic sampling config"""
    return SamplingParams(temperature=0.0,
                          max_tokens=20,
                          seed=42,
                          ignore_eos=True)


def get_kv_transfer_config():
    """use TPUOffloadConnector"""
    return KVTransferConfig(
        kv_connector="TPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_module_path="tpu_inference.offload.tpu_offload_connector",
    )


def _test_combo_offload_prefix_spec_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    speculative_config: Optional[Union[dict, str]],
    cpu_chunks: str = "8",
    max_output_len: Optional[int] = None,
):
    sampling_config = get_sampling_config()
    if max_output_len is not None:
        sampling_config.max_tokens = max_output_len
    kv_transfer_config = get_kv_transfer_config()

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transformational for scientific research and computing.",
    ]
    num_requests = len(prompts)

    with monkeypatch.context():
        # Standard environmental configuration for JAX-TPU KV Offloading
        monkeypatch.setenv('SKIP_JAX_PRECOMPILE', '0')
        monkeypatch.setenv('TPU_OFFLOAD_SKIP_JAX_PRECOMPILE', '0')
        monkeypatch.setenv('TPU_OFFLOAD_DECODE_SAVE', '1')
        monkeypatch.setenv('TPU_OFFLOAD_BATCHED_SAVE', '0')
        monkeypatch.setenv('TPU_OFFLOAD_NUM_CPU_CHUNKS', cpu_chunks)

        # Ensure Hugging Face respects offline cache if specified
        monkeypatch.setenv("HF_HOME",
                           os.environ.get("HF_HOME", "~/.cache/huggingface"))
        monkeypatch.setenv(
            "HF_HUB_CACHE",
            os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))

        tensor_parallel_size = int(os.environ.get("TPU_TP_SIZE", "8"))

        llm = None
        try:
            llm = LLM(
                model=model_name,
                max_model_len=512,
                max_num_seqs=num_requests,
                async_scheduling=not speculative_config,
                tensor_parallel_size=tensor_parallel_size,
                enable_prefix_caching=True,
                kv_transfer_config=kv_transfer_config,
                speculative_config=speculative_config,
            )

            # --- Pass 1: Cold Generation (Calculates and offloads KV Cache) ---
            print(f"\n--- Pass 1: Generating for {num_requests} requests ---")
            t0 = time.time()
            outputs1 = llm.generate(prompts, sampling_config)
            pass1_time = time.time() - t0
            print(f"Pass 1 generation completed in {pass1_time:.4f} seconds")
            out_texts1, out_tokens1 = parse_outputs(outputs1)
            del outputs1
            time.sleep(5)

            # --- Resetting prefix cache in TPU HBM ---
            # Forces next pass to load prefix KV cache from Host CPU DRAM instead of recalculating
            print("\n--- Resetting prefix cache (evicting from TPU HBM) ---")
            llm.llm_engine.engine_core.reset_prefix_cache()
            time.sleep(2)

            # --- Pass 2: Warm Generation (Loads KV Cache from CPU DRAM) ---
            print(
                "\n--- Pass 2: Generating again (should load from CPU DRAM) ---"
            )
            t0 = time.time()
            outputs2 = llm.generate(prompts, sampling_config)
            pass2_time = time.time() - t0
            print(f"Pass 2 generation completed in {pass2_time:.4f} seconds")
            out_texts2, out_tokens2 = parse_outputs(outputs2)
            del outputs2
            time.sleep(1)

            print("\n" + "=" * 80)
            print("Accuracy Comparison Results")
            print(f"Pass 1 generate time: {pass1_time * 1000:.2f} ms")
            print(f"Pass 2 generate time: {pass2_time * 1000:.2f} ms")
            print("=" * 80)
            for i in range(len(out_texts1)):
                print(f"\nRequest {i}:")
                print(f"  Pass 1 Output Text:     {out_texts1[i]!r}")
                print(f"  Pass 2 Output Text:     {out_texts2[i]!r}")
                print(f"  Pass 1 Output Tokens:   {out_tokens1[i]}")
                print(f"  Pass 2 Output Tokens:   {out_tokens2[i]}")
            print("\n" + "=" * 80)

            # Output 1 and Output 2 must be bit-for-bit identical
            assert len(out_texts1) == len(out_texts2)
            assert len(out_tokens1) == len(out_tokens2)
            for i in range(len(out_texts1)):
                assert out_texts1[i] == out_texts2[
                    i], f"Text mismatch in request {i}"
                assert out_tokens1[i] == out_tokens2[
                    i], f"Token mismatch in request {i}"
        finally:
            if llm is not None and hasattr(llm.llm_engine, "shutdown"):
                llm.llm_engine.shutdown()
            del llm
            gc.collect()
            # Waiting for TPUs to be released.
            time.sleep(10)


def test_combo_ngram_llama_3b(monkeypatch: pytest.MonkeyPatch):
    """Tests Llama-3.2-3B-Instruct using Ngram Speculative Decoding, Prefix Caching, and KV Offloading"""
    model_name = os.environ.get("MODEL_NAME",
                                "meta-llama/Llama-3.2-3B-Instruct")
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 2,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": 4,
    }
    _test_combo_offload_prefix_spec_accuracy(
        monkeypatch=monkeypatch,
        model_name=model_name,
        speculative_config=speculative_config,
        cpu_chunks="8",
    )


def test_combo_eagle3_llama_8b(monkeypatch: pytest.MonkeyPatch):
    """Tests Llama-3 8B using Eagle3 Speculative Decoding, Prefix Caching, and KV Offloading"""
    model_name = os.environ.get("MODEL_NAME",
                                "meta-llama/Meta-Llama-3.1-8B-Instruct")
    draft_model = os.environ.get("DRAFT_MODEL_NAME",
                                 "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B")

    model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
    monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    speculative_config = {
        "method": "eagle3",
        "model": draft_model,
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    }
    _test_combo_offload_prefix_spec_accuracy(
        monkeypatch=monkeypatch,
        model_name=model_name,
        speculative_config=speculative_config,
        cpu_chunks="8",
    )
