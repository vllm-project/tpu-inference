# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import time

import pytest
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def parse_outputs(outputs):
    output_token_ids = []
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        completion = output.outputs[0]
        generated_text = completion.text
        token_ids = completion.token_ids
        print(
            f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\nToken IDs: {token_ids!r}"
        )
        generated_texts.append(generated_text)
        output_token_ids.append(token_ids)
    return generated_texts, output_token_ids


@pytest.fixture
def sampling_config():
    """deterministic sampling config"""
    return SamplingParams(temperature=0,
                          max_tokens=20,
                          seed=42,
                          ignore_eos=True)


@pytest.fixture
def kv_transfer_config():
    """use TPUOffloadConnector"""
    return KVTransferConfig(
        kv_connector="TPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_module_path="tpu_inference.offload.tpu_offload_connector",
    )


def _test_kv_cache_cpu_offloading_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
    swap_op_type: str,
    skip_precompile: str,
    decode_save: str,
    cpu_chunks: str,
    prompt_file: str,
):
    with monkeypatch.context():
        os.environ['SKIP_JAX_PRECOMPILE'] = '1'
        os.environ['TPU_OFFLOAD_SWAP_OP_TYPE'] = swap_op_type
        os.environ['TPU_OFFLOAD_SKIP_JAX_PRECOMPILE'] = skip_precompile
        os.environ['TPU_OFFLOAD_DECODE_SAVE'] = decode_save
        os.environ['TPU_OFFLOAD_NUM_CPU_CHUNKS'] = cpu_chunks
        llm = LLM(model="meta-llama/Llama-3.2-3B",
                  max_model_len=3072,
                  task="generate",
                  kv_transfer_config=kv_transfer_config)

        prompt = read_prompt_from_file(prompt_file)
        # 1st generate
        outputs = llm.generate([prompt], sampling_config)
        out_texts1, out_tokens1 = parse_outputs(outputs)
        time.sleep(1)

        # manually let llm scheduler's kv_cache_manager forget all prefixes' hash
        llm.llm_engine.engine_core.reset_prefix_cache()
        time.sleep(1)

        # 2nd generate
        outputs = llm.generate([prompt], sampling_config)
        out_texts2, out_tokens2 = parse_outputs(outputs)
        time.sleep(1)

        # TODO(jcgu): check some internal states to verify save and load operations.
        # output1 and output2 should be identical
        assert len(out_texts1) == len(out_texts2)
        assert len(out_tokens1) == len(out_tokens2)
        for text1, text2 in zip(out_texts1, out_texts2):
            assert text1 == text2
        for tokens1, tokens2 in zip(out_tokens1, out_tokens2):
            assert tokens1 == tokens2

        del llm
        # Waiting for TPUs to be released.
        time.sleep(20)


# This tests scenario where the KV cache size is smaller than the CPU RAM. To ensure a gap the CPU RAM has been set on the higher side while using a smaller prompt.
# The test does the following
#   1. generates tokens for the input prompt
#   2. clears HBM which forces clean up of KV cache from HBM
#   3. re-calculates tokens for input prompt
#   4. verifies tokens generated for 1. and 3. are identical when KV cache<CPU RAM
# def test_kv_cache_cpu_offloading_accuracy_smaller_then_cpu_ram(
#     monkeypatch: pytest.MonkeyPatch,
#     sampling_config: SamplingParams,
#     kv_transfer_config: KVTransferConfig,
# ):
#     swap_op_types = ["jax"]
#     decode_saves = ["0"]
#     skip_precompile = ["1"]
#     for swap_op_type, decode_save, _skip_precompile in itertools.product(
#             swap_op_types, decode_saves, skip_precompile):
#         _test_kv_cache_cpu_offloading_accuracy(
#             monkeypatch,
#             sampling_config,
#             kv_transfer_config,
#             swap_op_type,
#             _skip_precompile,
#             decode_save,
#             # The total CPU RAM size = # cpu chunks * cpu_chunk_size. cpu_chunk_size represent the number of tokens can fit into a single CPU RAM chunk.
#             # cpu_chunk_size for llama-3.2-3B(used above in test)= 256
#             # CPU RAM size = 4*256=1024 tokens
#             "4",  # TPU_OFFLOAD_NUM_CPU_CHUNKS
#             # Prompt length/#tokens: 246 tokens
#             "small_prompt.txt",
#         )


# This tests scenario where the KV cache size is larger than the CPU RAM. To ensure this the CPU RAM has been set on the lower side while using a larger prompt.
# The test does the following
#   1. generates tokens for the input prompt
#   2. clears HBM which forces clean up of KV cache from HBM, since KV cache size > CPU RAM the tokens spills over CPU RAM
#   3. re-calculates tokens for input prompt, since teh KV cache size > CPU RAM, it loads any available tokens from CPU RAM and re-calculates remaining tokens lost due to spillover
#   4. verifies tokens generated for 1. and 3. are identical when KV cache>CPU RAM
def test_kv_cache_cpu_offloading_accuracy_larger_than_cpu_ram(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
):
    swap_op_types = ["jax"]
    decode_saves = ["0"]
    skip_precompile = ["1"]
    for swap_op_type, decode_save, _skip_precompile in itertools.product(
            swap_op_types, decode_saves, skip_precompile):
        _test_kv_cache_cpu_offloading_accuracy(
            monkeypatch,
            sampling_config,
            kv_transfer_config,
            swap_op_type,
            _skip_precompile,
            decode_save,
            # The total CPU RAM size = # cpu chunks * cpu_chunk_size. cpu_chunk_size represent the number of tokens can fit into a single CPU RAM chunk.
            # cpu_chunk_size for llama-3.2-3B(used above in test)= 256
            # CPU RAM size = 4*256=1024 tokens
            "10",  # TPU_OFFLOAD_NUM_CPU_CHUNKS
            # Large prompt details: 2042 tokens
            "large_prompt.txt",
        )


def read_prompt_from_file(file_name):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "prompt_files", file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
