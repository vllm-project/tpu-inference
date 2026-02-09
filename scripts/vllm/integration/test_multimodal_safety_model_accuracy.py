# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test validates the classification accuracy of multimodal safety models
configured in MODEL_CONFIG_MAP. It dynamically loads the MM-SafetyBench dataset,
applies the model's specific chat template, and asserts the final classification
accuracy against the CI's required minimum threshold.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import pytest
from PIL import Image
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
        "MODEL_NAME_TAG": "meta-llama/Llama-Guard-4-12B",
        # Arguments needed for tokenizer.apply_chat_template
        "TEMPLATE_ARGS": {
            "categories": {
                "S1": "Violent Crimes.",
                "S2":
                "Non-Violent Crimes.",  # MLCommons hazards taxonomy: https://the-ai-alliance.github.io/trust-safety-user-guide/exploring/mlcommons-ailuminate/
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
}

# --- DATA LOADING FUNCTION ---


def load_mm_safety_bench(
        image_dir: Path) -> List[Tuple[List[Dict[str, Any]], str]]:
    """
    Loads MM-SafetyBench.
    Robustly searches for 'processed_questions' and handles nested unzip structures.
    """
    # Potential locations for processed_questions
    # Location 1: /tmp/MM-SafetyBench/data/processed_questions
    path_attempt_1 = image_dir.parent / "processed_questions"
    # Location 2: /tmp/MM-SafetyBench/processed_questions (Root of repo)
    path_attempt_2 = image_dir.parent.parent / "processed_questions"

    if path_attempt_1.exists():
        questions_dir = path_attempt_1
    elif path_attempt_2.exists():
        questions_dir = path_attempt_2
    else:
        # Failure: List directories to help debugging in CI logs
        debug_path = image_dir.parent.parent
        print(f"DEBUG: Contents of {debug_path}:")
        if debug_path.exists():
            for item in debug_path.iterdir():
                print(f" - {item}")

        pytest.fail(f"Could not find 'processed_questions' dir. \n"
                    f"Checked:\n 1. {path_attempt_1}\n 2. {path_attempt_2}")

    print(f"Found processed_questions at: {questions_dir}")

    test_cases = []

    # Iterate over every scenario JSON file (e.g., 01-Illegal_Activity.json)
    json_files = sorted(questions_dir.glob("*.json"))

    if not json_files:
        pytest.fail(
            f"Found 'processed_questions' directory at {questions_dir}, but it contains no .json files."
        )

    for json_file in json_files:
        scenario_name = json_file.stem  # e.g., "01-Illegal_Activity"

        with open(json_file, "r") as f:
            data = json.load(f)

        # MM-SafetyBench JSONs are dictionaries keyed by ID strings ("0", "1", "2"...)
        for question_id, entry in data.items():
            # 1. Get the prompt
            prompt_text = entry.get("Rephrased Question(SD)") or entry.get(
                "Rephrased Question")

            # 2. Find the image
            img_filename = f"{question_id}.jpg"

            # Check logical paths (Standard, Flattened, Nested Zip variants)
            possible_paths = [
                image_dir / scenario_name / "SD" / img_filename,
                image_dir / scenario_name / img_filename,
                image_dir / scenario_name / "SD_TYPO" / img_filename,
                # Fix for the nested folder found in debug: MM-SafetyBench(imgs)
                image_dir / "MM-SafetyBench(imgs)" / scenario_name / "SD" /
                img_filename,
                image_dir / "MM-SafetyBench(imgs)" / scenario_name /
                img_filename,
            ]

            final_image_path = None
            for p in possible_paths:
                if p.exists():
                    final_image_path = p
                    break

            # If image missing, skip (prevents partial test cases)
            if final_image_path is None:
                continue

            # 3. Construct the conversation
            conversation = [{
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": prompt_text
                }, {
                    "type": "image",
                    "image_path": final_image_path
                }]
            }]

            # 4. Set Expected Label
            # A safety model (Llama Guard) *should* classify these attacks as "unsafe".
            test_cases.append((conversation, "unsafe"))

    if len(test_cases) == 0:
        # Print debug info before failing
        print(f"DEBUG: Listing contents of image_dir ({image_dir}):")
        if image_dir.exists():
            for child in image_dir.iterdir():
                print(
                    f"  - {child.name} ({'dir' if child.is_dir() else 'file'})"
                )
        pytest.fail(
            f"Loaded 0 test cases. Images were not found in {image_dir} matching JSON entries."
        )

    print(f"Loaded {len(test_cases)} test cases from MM-SafetyBench.")
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
def multimodal_safety_test_data_loader(request):
    """
    Dynamically loads the MM-SafetyBench dataset from the CLI-specified path (local)
    and provides structured multimodal prompts and expected outputs.
    """
    # Get the dynamic dataset path passed via the shell script CLI
    image_dir_path = request.config.getoption("--image-dir")
    num_test_cases = request.config.getoption("--num-test-cases")

    if not image_dir_path:
        pytest.skip(
            "The --image-dir argument is required for multimodal data loading."
        )
    image_dir = Path(image_dir_path)

    # Load the full dataset
    full_test_cases = load_mm_safety_bench(image_dir)

    # Truncate the dataset if --num-test-cases is provided
    if num_test_cases is not None:
        full_test_cases = full_test_cases[:num_test_cases]

    conversations = [case[0] for case in full_test_cases]
    expected_outputs = [case[1] for case in full_test_cases]
    total_tests = len(full_test_cases)

    return conversations, expected_outputs, total_tests


# --- TEST FUNCTION ---


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test is designed for TPU environment")
@pytest.mark.parametrize("disagg_enabled",
                         [True])  # Ensure we test the disaggregated path
def test_multimodal_safety_model_accuracy_check(
        monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
        multimodal_safety_test_data_loader, model_config_fixture,
        disagg_enabled):  # Inject the configuration here
    """
    Runs offline inference on the specified multimodal safety model and asserts the accuracy
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
    conversations, expected_outputs, total_tests = multimodal_safety_test_data_loader

    # 2. Setup LLM Args using dynamic config
    CONFIG = model_config_fixture
    llm_args = {
        "model": model_name,
        "max_model_len": 4096,
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
        # Extract image objects and prepare for TokensPrompt
        image_objects = []
        for content_block in conv[0]["content"]:
            if content_block.get("type") == "image":
                image_objects.append(Image.open(content_block["image_path"]))

        prompt_str = tokenizer.apply_chat_template(conv,
                                                   tokenize=False,
                                                   add_generation_prompt=True,
                                                   **CONFIG["TEMPLATE_ARGS"])
        tokenized_prompt = tokenizer.encode(prompt_str,
                                            add_special_tokens=False)

        multi_modal_data = {
            "image":
            image_objects[0] if len(image_objects) == 1 else image_objects
        } if image_objects else None

        prompts.append(
            TokensPrompt(prompt_token_ids=tokenized_prompt,
                         multi_modal_data=multi_modal_data))

    # 5. Run Inference
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # 6. Calculate Accuracy
    passed_tests = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        expected_text = expected_outputs[i]

        test_passed = False
        if generated_text == expected_text:
            test_passed = True
        elif expected_text.startswith("unsafe") and generated_text.startswith(
                "unsafe"):
            test_passed = True

        if test_passed:
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