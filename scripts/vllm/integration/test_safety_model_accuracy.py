# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test validates the classification accuracy of any safety model configured
in MODEL_CONFIG_MAP. It dynamically loads the full AILuminate dataset,
applies the model's specific chat template, and asserts the final classification
accuracy against the CI's required minimum threshold.
"""
from unittest.mock import patch

import pandas as pd
import pytest
from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

from tpu_inference.core import disagg_utils

# --- HAZARD MAPPING ---
# This mapping is based on the MLCommons taxonomy and is needed for the data loader.
# The ailuminate test set is a standard dataset for safety models
HAZARD_MAPPING = {
    "vcr": "S1",
    "src": "S3",
    "cse": "S4",
    "ssh": "S11",
    "iwp": "S9",
    "ipv": "S8",
    "dfm": "S5",
    "ncr": "S2",
    "hte": "S10",
    "prv": "S7",
    "spc_ele": "S13",
    "spc_fin": "S6",
    "spc_hlt": "S6",
    "spc_lgl": "S6",
    "sxc_prn": "S12",
}

# --- MODEL CONFIGURATION SUITES (Generic Setup) ---


def get_llama_guard_4_config():
    """Configuration specific to the Llama Guard 4 model."""
    return {
        "MODEL_NAME_TAG": "Llama-Guard-4-12B",
        # Arguments needed for tokenizer.apply_chat_template
        "TEMPLATE_ARGS": {
            "categories": {
                "S1": "Violent Crimes.",
                "S2": "Non-Violent Crimes.",
                "S3": "Sex Crimes.",
                "S4": "Child Exploitation.",
                "S5": "Defamation.",
                "S6": "Specialized Advice.",
                "S7": "Privacy.",
                "S8": "Intellectual Property.",
                "S9": "Indiscriminate Weapons.",
                "S10": "Hate.",
                "S11": "Self-Harm.",
                "S12": "Sexual Content.",
                "S13": "Elections.",
                "S14": "Code Interpreter Abuse."
            }
        },
        "ARCHITECTURE_OVERRIDES": {
            "architectures": ["Llama4ForConditionalGeneration"]
        },
    }


# Dictionary to map the --model argument to the configuration
MODEL_CONFIG_MAP = {
    "meta-llama/Llama-Guard-4-12B": get_llama_guard_4_config(),
    # Add other safety models here when implementing them:
    # "another/Safety-Model-1B": get_another_safety_model_config(),
}

# --- DATA LOADING FUNCTION ---


def load_data_from_csv(file_path: str) -> list[tuple[str, str]]:
    """Loads AILuminate CSV from GCS or local path and formats into (raw_prompt, expected_output) tuples."""

    print(f"Loading full dataset from: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(
            f"Error reading dataset from {file_path}. Ensure file exists and gcsfs is installed: {e}"
        )

    test_cases = []
    for _, row in df.iterrows():
        prompt_text = row["prompt_text"]
        hazard_abbrev = row["hazard"]

        # Determine expected output format (safe or unsafe\nS#)
        if hazard_abbrev == "no-hazard":
            expected_output = "safe"
        elif hazard_abbrev in HAZARD_MAPPING:
            s_code = HAZARD_MAPPING[hazard_abbrev]
            expected_output = f"unsafe\n{s_code}"
        else:
            continue  # Skip unknown hazards

        # Store as (raw_prompt, expected_output)
        test_cases.append((prompt_text, expected_output))

    print(f"Successfully loaded {len(test_cases)} valid test cases.")
    return test_cases


# --- FIXTURES ---


@pytest.fixture
def model_config_fixture(request: pytest.FixtureRequest):
    """Retrieves the configuration based on the model name passed via CLI."""
    model_name = request.config.getoption("--model-name")

    if model_name not in MODEL_CONFIG_MAP:
        pytest.fail(
            f"Configuration not found for model: {model_name}. Please update MODEL_CONFIG_MAP."
        )

    return MODEL_CONFIG_MAP[model_name]


@pytest.fixture
def safety_test_data_loader(request):
    """
    Dynamically loads the full AILuminate dataset from the CLI-specified path (GCS or local)
    and provides structured prompts and expected outputs.
    """
    # Get the dynamic dataset path passed via the shell script CLI
    dataset_path = request.config.getoption("--dataset-path")

    if not dataset_path:
        pytest.skip(
            "The --dataset-path argument is required for data loading.")

    # Load the full dataset (Pandas handles GCS URI)
    full_test_cases = load_data_from_csv(dataset_path)

    raw_prompts = [case[0] for case in full_test_cases]
    expected_outputs = [case[1] for case in full_test_cases]

    # Transform raw text into the vLLM conversation structure
    conversations = [[{
        "role": "user",
        "content": [{
            "type": "text",
            "text": raw_text
        }]
    }] for raw_text in raw_prompts]

    return conversations, expected_outputs, len(full_test_cases)


# --- TEST FUNCTION ---


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test is designed for TPU environment")
@pytest.mark.parametrize("disagg_enabled",
                         [True])  # Ensure we test the disaggregated path
def test_safety_model_accuracy_check(
        monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
        safety_test_data_loader, model_config_fixture,
        disagg_enabled):  # Inject the configuration here
    """
    Runs offline inference on the specified safety model and asserts the accuracy
    against the minimum expected threshold.
    """

    # 1. Read parameters from request.config (passed via CLI)
    model_name = request.config.getoption("--model-name")
    tp_size = request.config.getoption("--tensor-parallel-size")
    expected_threshold = request.config.getoption("--expected-value")

    if expected_threshold is None:
        pytest.fail(
            "The --expected-value (MINIMUM_ACCURACY_THRESHOLD) must be set.")

    # Standard parameters (fixed for this classification model type)
    max_tokens = 128
    temp = 0.0

    # Data unpacked
    conversations, expected_outputs, total_tests = safety_test_data_loader

    # 2. Setup LLM Args using dynamic config
    CONFIG = model_config_fixture
    llm_args = {
        "model": model_name,
        "max_model_len": 2048,
        "tensor_parallel_size": tp_size,
        "hf_overrides":
        CONFIG["ARCHITECTURE_OVERRIDES"],  # Use dynamic override
        "max_num_batched_tokens": 4096,
        "dtype": "bfloat16"
    }

    # 3. Initialize LLM (Mocking the disagg path if necessary)
    if disagg_utils.is_disagg_enabled():
        # NOTE: This assumes the required Disagg classes are accessible/mocked by the runner
        from tpu_inference.core.core_tpu import (DisaggEngineCore,
                                                 DisaggEngineCoreProc)
        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), patch(
                "vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
            llm = LLM(**llm_args)
    else:
        llm = LLM(**llm_args)

    # 4. Prepare Prompts (Tokenization)
    tokenizer = llm.llm_engine.tokenizer
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = temp
    sampling_params.max_tokens = max_tokens

    prompts = []
    for conv in conversations:
        # Use dynamic template args from the loaded config
        prompt_str = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            **CONFIG["TEMPLATE_ARGS"]  # Use dynamically loaded categories
        )
        tokenized_prompt = tokenizer.encode(prompt_str,
                                            add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=tokenized_prompt))

    # 5. Run Inference
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # 6. Calculate Accuracy
    passed_tests = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        expected_text = expected_outputs[i]

        if generated_text == expected_text:
            passed_tests += 1

    actual_accuracy = passed_tests / total_tests

    print(f"\n--- ACCURACY DIAGNOSTICS ---"
          f"\nTotal Test Cases: {total_tests}"
          f"\nPassed Cases: {passed_tests}"
          f"\nACTUAL_ACCURACY: {actual_accuracy:.4f}"
          f"\nMIN_THRESHOLD: {float(expected_threshold):.4f}"
          f"\n----------------------------")

    # 7. Assert against threshold (using Pytest standard assertion)
    assert actual_accuracy >= float(expected_threshold), (
        f"Accuracy check failed. Actual: {actual_accuracy:.4f} "
        f"is below expected minimum: {float(expected_threshold):.4f}")
