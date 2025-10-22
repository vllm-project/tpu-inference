# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
                          max_tokens=10,
                          seed=42,
                          ignore_eos=True)


@pytest.fixture
def kv_transfer_config():
    """use TPUConnector from tpu_connector_local"""
    return KVTransferConfig(
        kv_connector="TPUConnector",
        kv_role="kv_both",
        kv_connector_module_path=
        "tpu_inference.distributed.tpu_connector_local",
    )


def test_kv_cache_cpu_offloading_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    kv_transfer_config: KVTransferConfig,
):
    with monkeypatch.context():
        os.environ['SKIP_JAX_PRECOMPILE'] = '1'
        llm = LLM(model="meta-llama/Llama-3.2-3B",
                  max_model_len=1024,
                  tensor_parallel_size=1,
                  task="generate",
                  kv_transfer_config=kv_transfer_config)

        # 1st generate
        prompt = "Every Bill which shall have passed the House of Representatives and the Senate, shall, before it become a Law, be presented to the President of the United States; If he approve he shall sign it, but if not he shall return it, with his Objections to that House in which it shall have originated, who shall enter the Objections at large on their Journal, and proceed to reconsider it. If after such Reconsideration two thirds of that House shall agree to pass the Bill, it shall be sent, together with the Objections, to the other House, by which it shall likewise be reconsidered, and if approved by two thirds of that House, it shall become a Law. But in all such Cases the Votes of both Houses shall be determined by yeas and Nays, and the Names of the Persons voting for and against the Bill shall be entered on the Journal of each House respectively. If any Bill shall not be returned by the President within ten Days (Sundays excepted) after it shall have been presented to him, the Same shall be a Law, in like Manner as if he had signed it, unless the Congress by their Adjournment prevent its Return, in which Case"
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

        # output1 and output2 should be idential
        assert len(out_texts1) == len(out_texts2)
        assert len(out_tokens1) == len(out_tokens2)
        for text1, text2 in zip(out_texts1, out_texts2):
            assert text1 == text2
        for tokens1, tokens2 in zip(out_tokens1, out_tokens2):
            assert tokens1 == tokens2
