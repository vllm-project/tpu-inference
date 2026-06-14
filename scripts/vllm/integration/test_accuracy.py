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

import os
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
    elif model_name in [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct"
    ]:
        model_args = f"pretrained={model_name},max_model_len=4096,add_bos_token=False"
    else:
        model_args = f"pretrained={model_name},max_model_len=4096"

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    apply_chat_template = os.environ.get("USE_CHAT_TEMPLATE", "0") == "1"
    if apply_chat_template:
        print("USE_CHAT_TEMPLATE=1: enabling apply_chat_template + "
              "fewshot_as_multiturn for lm_eval. Required for instruction-"
              "tuned BOS-sensitive models like gemma-4-it.")

    # Eval-geometry overrides. Reasoning models (Qwen3.x hybrid-thinking)
    # emit long chains of thought before the final answer; lm_eval's
    # default max_gen_toks (256) truncates mid-thought and both gsm8k
    # filters score ~0. EVAL_GEN_KWARGS lets CI raise the budget, e.g.
    # EVAL_GEN_KWARGS="max_gen_toks=4096". EVAL_LIMIT caps the number of
    # eval examples for faster diagnostic runs.
    gen_kwargs = os.environ.get("EVAL_GEN_KWARGS") or None
    if gen_kwargs:
        print(f"EVAL_GEN_KWARGS={gen_kwargs}")
    limit_env = os.environ.get("EVAL_LIMIT")
    limit = int(limit_env) if limit_env else None
    if limit:
        print(f"EVAL_LIMIT={limit}: evaluating a subset, accuracy is an estimate")
    # EVAL_LOG_SAMPLES=1: dump the first few prompt/generation pairs to the
    # CI log so eval-config failures (template issues, truncation, format
    # drift) are diagnosable without a local repro.
    log_samples = os.environ.get("EVAL_LOG_SAMPLES", "0") == "1"

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=apply_chat_template,
        gen_kwargs=gen_kwargs,
        limit=limit,
        log_samples=log_samples,
    )

    if log_samples:
        import json as _json
        samples = (results.get("samples") or {}).get(TASK, [])
        for i, s in enumerate(samples[:5]):
            args = s.get("arguments") or []
            ctx = ""
            gen_args = {}
            if args and isinstance(args[0], (list, tuple)):
                ctx = str(args[0][0])
                if len(args[0]) > 1:
                    gen_args = args[0][1]
            print(f"========== EVAL SAMPLE {i} ==========")
            print("PROMPT_TAIL:", repr(ctx[-300:]))
            print("GEN_ARGS:", _json.dumps(gen_args, default=str)[:600])
            print("RESPS:", _json.dumps(s.get("resps"), default=str)[:2500])
            print("FILTERED:", _json.dumps(s.get("filtered_resps"), default=str)[:300])
            print("TARGET_TAIL:", repr(str(s.get("target"))[-80:]))
            print(f"========== END SAMPLE {i} ==========")

    # gsm8k emits two filters: strict-match (default gate) and flexible-extract.
    # Print both so CI logs let reviewers compare across runs/kernels.
    task_results = results["results"][TASK]
    measured_value = task_results[FILTER]
    flex_value = task_results.get("exact_match,flexible-extract")
    print(f"measured accuracy: {measured_value}")
    print(f"measured accuracy (strict-match): {measured_value}")
    if flex_value is not None:
        print(f"measured accuracy (flexible-extract): {flex_value}")
    assert measured_value >= expected_value - RTOL, f"Expected: {expected_value} |  Measured: {measured_value}"


@pytest.mark.skip_global_cleanup
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
            # EVAL_MAX_MODEL_LEN: chat-template multiturn fewshot prompts +
            # reasoning-model thinking tokens don't fit the 2048 default.
            max_model_len = os.environ.get("EVAL_MAX_MODEL_LEN", "2048")
            more_args = f"max_model_len={max_model_len},max_num_seqs=64"
            tp_size_str = f"tensor_parallel_size={tp_size}"
            more_args += ",{}".format(tp_size_str)
            # EVAL_GPU_MEMORY_UTILIZATION: opt-in HBM headroom knob. Large MoE
            # models (e.g. Qwen3.6-35B-A3B) auto-size the KV cache to fill the
            # default budget, leaving no room for the compiled step program
            # (jit_step_fun_impl) and OOMing during warmup. Lowering the
            # utilization shrinks the (heavily over-provisioned) KV cache to
            # leave that headroom. Only applied when explicitly set so other
            # models keep the vLLM default.
            gpu_mem_util = os.environ.get("EVAL_GPU_MEMORY_UTILIZATION")
            if gpu_mem_util:
                more_args += f",gpu_memory_utilization={gpu_mem_util}"

        print(f"common args: {more_args}")

        run_test(model, expected_value, more_args)


@pytest.mark.skip_global_cleanup
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
