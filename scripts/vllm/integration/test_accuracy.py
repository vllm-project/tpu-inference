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

import lm_eval
import pytest
import json
import threading
import os

from pathlib import Path
from vllm.platforms import current_platform

MODEL_NAMES = [
    "Qwen/Qwen3-1.7B",
    "google/gemma-3-1b-it",
    # "meta-llama/Llama-3.1-8B-Instruct",
]
FP8_KV_MODEL_NAMES = [
    "Qwen/Qwen3-1.7B",
]
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
_JSON_WRITE_LOCK = threading.Lock()

EXPECTED_VALUES = {
    "Qwen/Qwen3-1.7B": 0.68,
    "google/gemma-3-1b-it": 0.25,
    "meta-llama/Llama-3.1-8B-Instruct": 0.76,
    "meta-llama/Llama-3.1-70B-Instruct": 0.876,
}

# Parametrize test cases based on CLI arguments or default values
def parametrize_by_cli_or_default(metafunc, fixture_name, cli_parameter, default_list):
    if fixture_name in metafunc.fixturenames:
        print(f"Checking CLI parameter '{cli_parameter}' for '{fixture_name}'")
        names_str = metafunc.config.getoption(cli_parameter)
        if names_str:
            print(f"Using '{cli_parameter}' parameter for '{fixture_name}'")
            param_list = [name.strip() for name in names_str.split(',') if name.strip()]
            metafunc.parametrize(fixture_name, param_list)
        else:
            print(f"Using default list for '{fixture_name}'")
            metafunc.parametrize(fixture_name, default_list)

def pytest_generate_tests(metafunc):
    parametrize_by_cli_or_default(metafunc, fixture_name="model", cli_parameter="--model-names", default_list=MODEL_NAMES)
    parametrize_by_cli_or_default(metafunc, fixture_name="fp8_kv_model", cli_parameter="--fp8-kv-model-names", default_list=FP8_KV_MODEL_NAMES)

# Write expected values to json file
# TBD: To support the functionality of connecting GPU and TPU expected values in the future
def write_expected_value_to_json(model_name, measured_value, json_filepath):
    with _JSON_WRITE_LOCK:
        data = {}
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"'{json_filepath}' not found or is empty/invalid. A new one will be created.")
            data = {}
        
        data[model_name] = measured_value
        
        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"Successfully updated '{json_filepath}' with the result for {model_name}.")
        except IOError as e:
            print(f"Error: Failed to write to file '{json_filepath}'. Reason: {e}")

# Read expected values from json file if exist
# TBD: To support the functionality of connecting GPU and TPU expected values in the future
def read_expected_value(expected_json_filepath=None):
    expected_values_data = {}
    if expected_json_filepath is None:
        expected_values_data = EXPECTED_VALUES
    else:
        path_obj = Path(expected_json_filepath)
        # Read expected values from json file if exist
        if path_obj.is_file() and os.path.getsize(expected_json_filepath) > 0:
            print(f"\n[Fixture] Loading from: {expected_json_filepath}")
            with open(expected_json_filepath, 'r', encoding='utf-8') as f:
                expected_values_data = json.load(f)
        else:
            raise FileNotFoundError(f"Expected values file not found: {expected_json_filepath}")
    return expected_values_data


def run_test(model_name, expected_values_data, expected_json_filepath, more_args=None):
    """Run the end to end accuracy test."""
    print(f"Running test for model: {model_name}")

    model_args = f"pretrained={model_name},max_model_len=4096"
    
    download_path = "/mnt/disks/persist"
    # download_path = "/tmp/hf_model"
    if os.path.isdir(download_path) and os.access(download_path, os.R_OK) and os.access(download_path, os.W_OK):
        model_args = f"{model_args},download_dir={download_path}"
    
    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
    )

    # Execute default behavior when `expected_json_filepath` is not set.
    if expected_json_filepath is None:
        print(f"Execute default behavior")
        measured_value = results["results"][TASK][FILTER]
        assert model_name in EXPECTED_VALUES, (
            f"Cannot find the expected value for the model {model_name=}")
        expected_value = EXPECTED_VALUES[model_name]
        assert (measured_value - RTOL < expected_value
                and measured_value + RTOL > expected_value
                ), f"Expected: {expected_value} |  Measured: {measured_value}"
    else:
        print(f"Execute specific models behavior")
        measured_value = results["results"][TASK][FILTER]
        expected_value = expected_values_data.get(model_name)

        # Model expected value not exist, write in file
        if model_name not in expected_values_data:
            print(f"Warning: No expected value found for {model_name}. "
                "Skipping accuracy check.")
            print(f"Measured value: {measured_value}")
            write_expected_value_to_json(model_name, measured_value, expected_json_filepath)

        else:
            print(f"Found expected value! {model_name=}, {measured_value=}, {expected_value=}")
            assert (measured_value - RTOL < expected_value
                and measured_value + RTOL > expected_value
                ), f"Expected: {expected_value} |  Measured: {measured_value}"

@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 is currently only supported on CUDA and TPU")
def test_lm_eval_accuracy_v1_engine(model, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Run with the V1 Engine."""
    print(f"Testing model: {model}...")

    tp_size = request.config.getoption("--tensor-parallel-size")
    expected_json_filepath = request.config.getoption("--expected-values-file")
            
    expected_values_data = read_expected_value(expected_json_filepath)

    if tp_size is None:
        tp_size = 1
    elif tp_size < 1 or tp_size > 8:
        raise ValueError

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        
        more_args = None
        if current_platform.is_tpu():
            more_args = "max_model_len=2048,max_num_seqs=64"
            tp_size_str = f"tensor_parallel_size={tp_size}"
            more_args += ",{}".format(tp_size_str)
        
        print(f"common args: {more_args}")

        run_test(model, expected_values_data, expected_json_filepath, more_args)


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 is currently only supported on CUDA and TPU")
def test_lm_eval_accuracy_v1_engine_fp8_kv_cache(
        fp8_kv_model, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Run with the V1 Engine."""
    print(f"Testing fp8_kv_model: {fp8_kv_model}...")

    tp_size = request.config.getoption("--tensor-parallel-size")
    expected_json_filepath = request.config.getoption("--expected-values-file")

    expected_values_data = read_expected_value(expected_json_filepath)

    if tp_size is None:
        tp_size = 1
    elif tp_size < 1 or tp_size > 8:
        raise ValueError

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        more_args = None
        if current_platform.is_tpu():
            more_args = "max_model_len=2048,max_num_seqs=128,kv_cache_dtype=fp8"
            tp_size_str = f"tensor_parallel_size={tp_size}"
            more_args += ",{}".format(tp_size_str)
            
        print(f"common args: {more_args}")

        run_test(fp8_kv_model, expected_values_data, expected_json_filepath, more_args)