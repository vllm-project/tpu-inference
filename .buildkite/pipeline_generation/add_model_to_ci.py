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

import argparse
import sys
from enum import Enum
from pathlib import Path

# ANSI colors for better terminal readability
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
DIM = "\033[2m"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR_REL = Path(".buildkite/models")
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR_REL


class ModelType(str, Enum):
    TPU_OPTIMIZED = "tpu-optimized"
    VLLM_NATIVE = "vllm-native"


class ModelCategory(str, Enum):
    TEXT_ONLY = "text-only"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    DIFFUSION = "diffusion"


class HostScale(str, Enum):
    SMALL = "small"
    LARGE = "large"


MODEL_TYPE_TO_TEMPLATE = {
    ModelType.TPU_OPTIMIZED.value: "tpu_optimized_model_template.yml",
    ModelType.VLLM_NATIVE.value: "vllm_native_model_template.yml",
}

# User Correction: Use double curly braces for shell variables to avoid Python format conflicts
HOST_SCALE_TO_SETTINGS = {
    HostScale.SMALL.value: {
        "queue": "${TPU_QUEUE_SINGLE:-tpu_v6e_queue}",
        "tp_size": "${TENSOR_PARALLEL_SIZE_SINGLE:-1}",
    },
    HostScale.LARGE.value: {
        "queue": "${TPU_QUEUE_MULTI:-tpu_v6e_8_queue}",
        "tp_size": "${TENSOR_PARALLEL_SIZE_MULTI:-8}",
    },
}


def get_interactive_input():
    """
    Provides a guided, step-by-step configuration flow in the terminal.
    Validates user input and provides immediate feedback on the consequences of choices.
    """
    header_width = 60
    title = "Model CI Configuration Wizard"
    print(f"\n{BOLD}{'=' * header_width}{RESET}")
    print(f"{BOLD}{title.center(header_width)}{RESET}")
    print(f"{BOLD}{'=' * header_width}{RESET}")
    print(f"Target Directory: {YELLOW}{OUTPUT_DIR_REL}{RESET}")

    # --- STEP 1: HuggingFace Model ID ---
    print(
        f"\n{BOLD}[Step 1/4]{RESET} {CYAN}What is the full model name on HuggingFace?{RESET}"
    )
    print(
        f"   {YELLOW}Hint: Please ensure to use the full name (e.g., meta-llama/Llama-3.1-8B)\n"
    )
    while True:
        name = input(f"{BOLD}>> {RESET}").strip()
        if name:
            print(f"{GREEN}✓ Model ID recorded.{RESET}")
            break
        print(f"{RED}❌ Error: Model name cannot be empty.{RESET}\n")

    # --- STEP 2: Model Type ---
    print(f"\n{BOLD}[Step 2/4]{RESET} {CYAN}What is the model type?{RESET}\n")
    print(
        f"   [1] {BOLD}tpu-optimized{RESET} : TPU-specific optimizations (Optimizations for TPU. Includes unit, accuracy, and perf tests.)"
    )
    print(
        f"   [2] {BOLD}vllm-native{RESET}   : Upstream vLLM definition (Upstream vLLM definition. Includes unit and accuracy tests.)\n"
    )
    while True:
        choice = input(f"{BOLD}Select (1-2): {RESET}").strip()
        if choice == '1':
            m_type = ModelType.TPU_OPTIMIZED.value
            print(f"{GREEN}✓ Mode: {BOLD}tpu-optimized{RESET}{RESET}")
            break
        elif choice == '2':
            m_type = ModelType.VLLM_NATIVE.value
            print(f"{GREEN}✓ Mode: {BOLD}vllm-native{RESET}{RESET}")
            break
        print(f"{RED}❌ Invalid entry. Please enter 1 or 2.{RESET}\n")

    # --- STEP 3: Model Category ---
    print(
        f"\n{BOLD}[Step 3/4]{RESET} {CYAN}What category is this model?{RESET}")
    print(
        f"   {YELLOW}Note: This sets the 'Type' column in the support matrix.{RESET}\n"
    )
    print(f"   [1] {BOLD}text-only{RESET}")
    print(f"   [2] {BOLD}multimodal{RESET}")
    print(f"   [3] {BOLD}embedding{RESET}")
    print(f"   [4] {BOLD}diffusion{RESET}\n")
    while True:
        choice = input(f"{BOLD}Select (1-2): {RESET}").strip()
        if choice == '1':
            m_cat = ModelCategory.TEXT_ONLY.value
            print(f"{GREEN}✓ Category: {BOLD}text-only{RESET}")
            break
        elif choice == '2':
            m_cat = ModelCategory.MULTIMODAL.value
            print(f"{GREEN}✓ Category: {BOLD}multimodal{RESET}")
            break
        elif choice == '3':
            m_cat = ModelCategory.EMBEDDING.value
            print(f"{GREEN}✓ Category: {BOLD}embedding{RESET}")
            break
        elif choice == '4':
            m_cat = ModelCategory.DIFFUSION.value
            print(f"{GREEN}✓ Category: {BOLD}diffusion{RESET}")
            break
        print(f"{RED}❌ Invalid entry. Please enter 1 or 2.{RESET}\n")

    # --- STEP 4: Host Scale ---
    print(
        f"\n{BOLD}[Step 4/4]{RESET} {CYAN}Specify the host scale for running tests:{RESET}"
    )
    print(
        f"   {YELLOW}Hint: Choose the hardware scale based on your model's requirements{RESET}\n"
    )
    print(
        f"   [1] {BOLD}Small scale{RESET} : Runs tests on small scale queue. (v6e: tpu_v6e_queue, v7x: tpu_v7x_2_queue)"
    )
    print(
        f"   [2] {BOLD}Large scale{RESET}  : Runs tests on large scale queue. (v6e: tpu_v6e_8_queue, v7x: tpu_v7x_8_queue)\n"
    )
    while True:
        choice = input(f"{BOLD}Select (1-2): {RESET}").strip()
        if choice == '1':
            m_scale = HostScale.SMALL.value
            print(f"{GREEN}✓ Scale: {BOLD}small scale{RESET}")
            break
        elif choice == '2':
            m_scale = HostScale.LARGE.value
            print(f"{GREEN}✓ Scale: {BOLD}large scale{RESET}")
            break
        print(f"{RED}❌ Invalid entry. Please enter 1 or 2.{RESET}\n")

    return name, m_type, m_cat, m_scale


def generate_from_template(model_name: str, model_type: str,
                           model_category: str, host_scale: str) -> None:
    template_path = SCRIPT_DIR / MODEL_TYPE_TO_TEMPLATE[model_type]
    if not template_path.is_file():
        print(f"{RED}Error: Template path '{template_path}' invalid.{RESET}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    sanitized_model_name = model_name.replace("/", "_").replace(".", "_")
    host_scale_settings = HOST_SCALE_TO_SETTINGS[host_scale]

    try:
        generated_content = template_content.format(
            MODEL_NAME=model_name,
            CATEGORY=model_category,
            SANITIZED_MODEL_NAME=sanitized_model_name,
            QUEUE=host_scale_settings["queue"],
            TP_SIZE=host_scale_settings["tp_size"],
        )
    except KeyError as e:
        print(f"{RED}Error: Missing placeholder {e} in template.{RESET}")
        sys.exit(1)

    generated_filepath = OUTPUT_DIR / f"{sanitized_model_name}.yml"
    with open(generated_filepath, 'w', encoding='utf-8') as f:
        f.write(generated_content)

    # Final success and instruction block
    print(
        f"\n{GREEN}✅ Success!{RESET} Config file generated at: {YELLOW}{OUTPUT_DIR_REL}/{sanitized_model_name}.yml{RESET}"
    )

    print(f"\n{BOLD}📋 FINAL STEPS:{RESET}")
    print(" Please open the generated file and complete these TODOs:")
    print("  1. Set the unit test command for your model.")
    print(
        f"  2. Define the accuracy target ({CYAN}MINIMUM_ACCURACY_THRESHOLD{RESET})."
    )

    # Conditional step: Performance is only for tpu-optimized models
    if model_type == ModelType.TPU_OPTIMIZED.value:
        print(
            f"  3. Define the performance target ({CYAN}MINIMUM_THROUGHPUT_THRESHOLD{RESET})."
        )

    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Add Buildkite yml config file.")
    parser.add_argument("--model-name", type=str)
    parser.add_argument('--type', choices=[t.value for t in ModelType])
    parser.add_argument('--category', choices=[c.value for c in ModelCategory])
    parser.add_argument('--host-scale', choices=[s.value for s in HostScale])

    args = parser.parse_args()

    if not args.model_name:
        model_name, model_type, model_category, host_scale = get_interactive_input(
        )
    else:
        # Fallback to defaults if partial CLI args provided
        model_name = args.model_name
        model_type = args.type or ModelType.TPU_OPTIMIZED.value
        model_category = args.category or ModelCategory.TEXT_ONLY.value
        host_scale = args.host_scale or HostScale.SMALL.value

    generate_from_template(model_name, model_type, model_category, host_scale)


if __name__ == "__main__":
    main()
