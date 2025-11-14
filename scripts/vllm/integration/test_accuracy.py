# Copied from vLLM: https://github.com/vllm-project/vllm/blob/839ab00/tests/entrypoints/llm/test_accuracy.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file test accuracy of the vLLM server via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

import threading

import lm_eval
import pytest
from vllm.platforms import current_platform

MODEL_NAMES = []
FP8_KV_MODEL_NAMES = []
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
_JSON_WRITE_LOCK = threading.Lock()


def run_test(model_name, expected_value, more_args=None):
    """Run the end to end accuracy test."""
    print(f"Running test for model: {model_name}")

    if model_name in ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen2.5-VL-7B-Instruct"]:
        model_args = f"pretrained={model_name},max_model_len=4096,max_num_batched_tokens=16384"
    else:
        model_args = f"pretrained={model_name},max_model_len=4096"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
    )

    measured_value = results["results"][TASK][FILTER]
    print(f"measured accuracy: {measured_value}")
    assert measured_value >= expected_value - RTOL, f"Expected: {expected_value} |  Measured: {measured_value}"


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 is currently only supported on CUDA and TPU")
def test_lm_eval_accuracy_v1_engine(monkeypatch: pytest.MonkeyPatch,
                                    request: pytest.FixtureRequest):
    """Run with the V1 Engine."""
    model = request.config.getoption("--model-name")
    print(f"Testing model: {model}...")

    tp_size = request.config.getoption("--tensor-parallel-size")
    expected_value = request.config.getoption("--expected-value")

    if expected_value is None:
        raise ValueError

    if tp_size is None:
        tp_size = 1
    elif tp_size < 1 or tp_size > 8:
        raise ValueError

    with monkeypatch.context() as _:
        more_args = None
        if current_platform.is_tpu():
            more_args = "max_model_len=2048,max_num_seqs=64"
            tp_size_str = f"tensor_parallel_size={tp_size}"
            more_args += ",{}".format(tp_size_str)

        print(f"common args: {more_args}")

        run_test(model, expected_value, more_args)


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 is currently only supported on CUDA and TPU")
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(
        monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Run with the V1 Engine."""
    fp8_kv_model = request.config.getoption("--fp8-kv-model-name")
    print(f"Testing fp8_kv_model: {fp8_kv_model}...")

    tp_size = request.config.getoption("--tensor-parallel-size")
    expected_value = request.config.getoption("--expected-value")

    if expected_value is None:
        raise ValueError

    if tp_size is None:
        tp_size = 1
    elif tp_size < 1 or tp_size > 8:
        raise ValueError

    with monkeypatch.context() as _:
        more_args = None
        if current_platform.is_tpu():
            more_args = "max_model_len=2048,max_num_seqs=128,kv_cache_dtype=fp8"
            tp_size_str = f"tensor_parallel_size={tp_size}"
            more_args += ",{}".format(tp_size_str)

        print(f"common args: {more_args}")

        run_test(fp8_kv_model, expected_value, more_args)
