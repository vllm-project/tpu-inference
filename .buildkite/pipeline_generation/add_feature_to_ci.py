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

import argparse
import sys
from enum import Enum
from pathlib import Path

# ANSI Color and Style Definitions
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Script and directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR_BASE = SCRIPT_DIR.parent


class FeatureCategory(str, Enum):
    FEATURE = "feature support matrix"
    KERNEL = "kernel support matrix"
    PARALLELISM = "parallelism support matrix"
    QUANTIZATION = "quantization support matrix"
    KERNEL_MICROBENCHMARKS = "kernel support matrix microbenchmarks"
    RL = "rl support matrix"


class HostScale(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


# Maps host scale to Buildkite settings with shell defaults
HOST_SCALE_TO_SETTINGS = {
    HostScale.SINGLE.value: {
        "queue": "${{TPU_QUEUE_SINGLE:-tpu_v6e_queue}}",
        "tp_size": "${{TENSOR_PARALLEL_SIZE_SINGLE:-1}}",
    },
    HostScale.MULTI.value: {
        "queue": "${{TPU_QUEUE_MULTI:-tpu_v6e_8_queue}}",
        "tp_size": "${{TENSOR_PARALLEL_SIZE_MULTI:-8}}",
    },
}

# Map categories to templates
CATEGORY_TO_TEMPLATE = {
    FeatureCategory.FEATURE.value: "feature_template.yml",
    FeatureCategory.KERNEL.value: "feature_template.yml",
    FeatureCategory.QUANTIZATION.value: "feature_template.yml",
    FeatureCategory.KERNEL_MICROBENCHMARKS.value: "feature_template.yml",
    FeatureCategory.PARALLELISM.value: "parallelism_template.yml",
    FeatureCategory.RL.value: "feature_template.yml",
}

# Map feature categories to their respective output directories
CATEGORY_TO_DIR = {
    FeatureCategory.FEATURE.value: "features",
    FeatureCategory.KERNEL.value: "features",
    FeatureCategory.PARALLELISM.value: "parallelism",
    FeatureCategory.QUANTIZATION.value: "quantization",
    FeatureCategory.KERNEL_MICROBENCHMARKS.value: "kernel_microbenchmarks",
    FeatureCategory.RL.value: "rl",
}


def get_interactive_input():
    """
    Guides the user through a series of prompts to configure a new feature.
    Provides immediate feedback and explains directory mappings.
    """
    header_width = 60
    title = "Feature CI Configuration Wizard"

    print(f"\n{BOLD}{'=' * header_width}{RESET}")
    print(f"{BOLD}{title.center(header_width)}{RESET}")
    print(f"{BOLD}{'=' * header_width}{RESET}")

    # --- STEP 1: Feature Name ---
    print(
        f"\n{BOLD}[Step 1/4]{RESET} {CYAN}What is the name of the feature?{RESET}"
    )
    print(
        f"   {YELLOW}Note: Special characters and spaces will be sanitized to underscores.{RESET}\n"
    )
    while True:
        name = input(f"{BOLD}>> {RESET}").strip()
        if name:
            print(f"{GREEN}✓ Feature name recorded.{RESET}")
            break
        print(f"{RED}❌ Error: Feature name is required.{RESET}")

    # --- STEP 2: Feature Category ---
    print(f"\n{BOLD}[Step 2/4]{RESET} {CYAN}Select Feature Category{RESET}")
    print(
        f"  {YELLOW}Determines where the YAML is placed (e.g. features/, parallelism/, etc.){RESET}\n"
    )

    categories = list(FeatureCategory)
    for i, cat in enumerate(categories, 1):
        print(f"  [{i}] {cat.value}")

    while True:
        choice = input(
            f"\n{BOLD}Select (1-{len(categories)}): {RESET}").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(categories):
            category = categories[int(choice) - 1].value
            print(f"{GREEN}✓ Category: {BOLD}{category}{RESET}")
            break
        print(f"{RED}❌ Invalid selection.{RESET}")

    # --- STEP 3: Group Name (Conditional) ---
    group = None
    if category == FeatureCategory.KERNEL_MICROBENCHMARKS.value:
        print(
            f"\n{BOLD}[Step 3/4]{RESET} {CYAN}Select or Create Kernel Group{RESET}"
        )
        print(
            f"  {YELLOW}Organizes microbenchmarks for a specific kernel into a directory.{RESET}\n"
        )

        # Scan for existing folders in .buildkite/kernel_microbenchmarks/
        microbench_base = SCRIPT_DIR.parent / "kernel_microbenchmarks"
        existing_groups = []
        if microbench_base.exists():
            existing_groups = sorted(
                [d.name for d in microbench_base.iterdir() if d.is_dir()])

        # List existing folders as options
        for i, folder in enumerate(existing_groups, 1):
            print(f"  [{i}] {folder}")

        # Add the option to create a new folder
        new_option_idx = len(existing_groups) + 1
        print(
            f"  [{new_option_idx}] {BOLD}Other: Define a new group name{RESET}"
        )

        while True:
            choice = input(
                f"\n{BOLD}Select (1-{new_option_idx}): {RESET}").strip()
            if choice.isdigit() and 1 <= int(choice) <= new_option_idx:
                idx = int(choice)
                if idx == new_option_idx:
                    # User wants to input a NEW folder name
                    print(
                        f"\n{YELLOW}Enter the new kernel group name (e.g., all_gather_matmul):{RESET}\n"
                    )
                    while True:
                        group = input(f"{BOLD}>> {RESET}").strip()
                        if group:
                            print(
                                f"{GREEN}✓ New group '{group}' created.{RESET}"
                            )
                            break
                        print(f"{RED}❌ Error: Group name is required.{RESET}")
                else:
                    # User selected an existing folder
                    group = existing_groups[idx - 1]
                    print(
                        f"{GREEN}✓ Using existing group: {BOLD}{group}{RESET}")
                break
            print(f"{RED}❌ Invalid selection.{RESET}")
    else:
        print(
            f"\n{BOLD}[Step 3/4]{RESET} {DIM}Group Name: (Not required for this category){RESET}"
        )

    # --- STEP 4: Host Scale (Now matches model script style) ---
    print(
        f"\n{BOLD}[Step 4/4]{RESET} {CYAN}Specify the host scale for running tests:{RESET}"
    )
    print(
        f"   {YELLOW}Hint: Choose the hardware scale based on your feature's requirements{RESET}\n"
    )
    print(
        f"   [1] {BOLD}single{RESET} : Runs tests on a single TPU host. (v6e: tpu_v6e_queue, v7x: tpu_v7x_2_queue)"
    )
    print(
        f"   [2] {BOLD}multi{RESET}  : Runs tests on multiple TPU hosts. (v6e: tpu_v6e_8_queue, v7x: tpu_v7x_8_queue)\n"
    )
    while True:
        choice = input(f"{BOLD}Select (1-2): {RESET}").strip()
        if choice == '1':
            m_scale = HostScale.SINGLE.value
            print(f"{GREEN}✓ Scale: {BOLD}single host{RESET}")
            break
        elif choice == '2':
            m_scale = HostScale.MULTI.value
            print(f"{GREEN}✓ Scale: {BOLD}multi host{RESET}")
            break
        print(f"{RED}❌ Invalid entry. Please enter 1 or 2.{RESET}\n")

    return name, category, group, m_scale


def generate_from_template(feature_name: str,
                           feature_category: str,
                           host_scale: str,
                           group: str | None = None) -> None:
    """
    Substitutes template placeholders and writes the YAML file to the correct path.
    """
    template_filename = CATEGORY_TO_TEMPLATE.get(feature_category)
    template_path = SCRIPT_DIR / template_filename

    if not template_path.is_file():
        print(f"{RED}Error: Template file '{template_path}' missing.{RESET}")
        sys.exit(1)

    # Determine relative and absolute paths for file output
    base_dir_name = CATEGORY_TO_DIR.get(feature_category)
    rel_output_path = Path(".buildkite") / base_dir_name
    output_dir = SCRIPT_DIR.parent / base_dir_name

    if group:
        output_dir = output_dir / group
        rel_output_path = rel_output_path / group

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        print(f"{RED}Error reading template: {e}{RESET}")
        sys.exit(1)

    # Sanitize name for filename and Buildkite step key restrictions
    sanitized_name = feature_name.replace("/", "_").replace(".", "_").replace(
        " ", "_").replace(":", "-")

    # Get settings based on host scale
    settings = HOST_SCALE_TO_SETTINGS[host_scale]

    try:
        # Format the content using template variables
        generated_content = template_content.format(
            FEATURE_NAME=feature_name,
            CATEGORY=feature_category,
            SANITIZED_FEATURE_NAME=sanitized_name,
            QUEUE=settings["queue"],
            TP_SIZE=settings["tp_size"],
        )
    except KeyError as e:
        print(
            f"{RED}Error: Missing placeholder {e} in the template file.{RESET}"
        )
        sys.exit(1)

    generated_filepath = output_dir / f"{sanitized_name}.yml"

    try:
        with open(generated_filepath, 'w', encoding='utf-8') as f:
            f.write(generated_content)

        # Success summary
        print(
            f"\n{GREEN}✅ Success!{RESET} Config file generated at: {YELLOW}{rel_output_path}/{sanitized_name}.yml{RESET}"
        )

        print(f"\n{BOLD}📋 FINAL STEPS:{RESET}")
        print("  Please open the generated file and complete these TODOs:")
        print("  1. Set the correctness test command for your feature.")
        print("  2. Set the performance test command for your feature.")
        print("")
    except Exception as e:
        print(f"{RED}Error writing output file: {e}{RESET}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add Buildkite yml config file.")
    parser.add_argument("--feature-name", type=str, help="Feature name")
    parser.add_argument('--category',
                        choices=[f.value for f in FeatureCategory],
                        help="Feature category")
    parser.add_argument('--group', type=str, help="Kernel group name")
    parser.add_argument('--host-scale',
                        choices=[s.value for s in HostScale],
                        help="Target host scale")

    args = parser.parse_args()

    # Launch interactive mode if required arguments are missing
    if not args.feature_name:
        feature_name, category, group, host_scale = get_interactive_input()
    else:
        # Fallback for automated usage
        feature_name = args.feature_name
        category = args.category or FeatureCategory.FEATURE.value
        group = args.group
        host_scale = args.host_scale or HostScale.SINGLE.value

    # Validation: Groups are mandatory for microbenchmarks
    if category == FeatureCategory.KERNEL_MICROBENCHMARKS.value and not group:
        print(f"{RED}❌ Error: --group is required for microbenchmarks.{RESET}")
        sys.exit(1)

    generate_from_template(
        feature_name=feature_name,
        feature_category=category,
        host_scale=host_scale,
        group=group,
    )


if __name__ == "__main__":
    main()
