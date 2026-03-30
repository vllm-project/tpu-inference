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

SCRIPT_DIR = Path(__file__).resolve().parent


class FeatureCategory(str, Enum):
    FEATURE = "feature support matrix"
    KERNEL = "kernel support matrix"
    PARALLELISM = "parallelism support matrix"
    QUANTIZATION = "quantization support matrix"
    KERNEL_MICROBENCHMARKS = "kernel support matrix microbenchmarks"
    RL = "rl support matrix"


# Map categories to templates
CATEGORY_TO_TEMPLATE = {
    FeatureCategory.FEATURE.value: "feature_template.yml",
    FeatureCategory.KERNEL.value: "feature_template.yml",
    FeatureCategory.QUANTIZATION.value: "feature_template.yml",
    FeatureCategory.KERNEL_MICROBENCHMARKS.value: "feature_template.yml",
    FeatureCategory.PARALLELISM.value: "parallelism_template.yml",
    FeatureCategory.RL.value: "feature_template.yml",
}

# Map feature categories to their respective output directories.
CATEGORY_TO_DIR = {
    FeatureCategory.FEATURE.value: "features",
    FeatureCategory.KERNEL.value: "features",
    FeatureCategory.PARALLELISM.value: "parallelism",
    FeatureCategory.QUANTIZATION.value: "quantization",
    FeatureCategory.KERNEL_MICROBENCHMARKS.value: "kernel_microbenchmarks",
    FeatureCategory.RL.value: "rl",
}


def generate_from_template(feature_name: str,
                           feature_category: str,
                           group: str | None = None) -> None:
    """
    Generates a buildkite yml file from feature template.
    Args:
        feature_name (str): The name of the feature.
        feature_category (str): The category of the feature.
        group (str, optional): The group for kernel microbenchmarks. Defaults to None.
    """
    print(f"--- Starting to generate for Feature '{feature_name}' ---")

    # Determine template path based on category
    template_filename = CATEGORY_TO_TEMPLATE.get(feature_category)
    if not template_filename:
        print(f"Error: No template found for category '{feature_category}'.")
        sys.exit(1)
    template_path = SCRIPT_DIR / template_filename

    # Check if the template file exists.
    if not template_path.is_file():
        print(
            f"Error: Template file '{template_path}' invalid. Did you remove it by accident?"
        )
        sys.exit(1)

    # Determine output directory based on category
    base_output_dir_name = CATEGORY_TO_DIR.get(feature_category)
    if not base_output_dir_name:
        print(f"Error: Invalid feature category '{feature_category}'.")
        sys.exit(1)

    output_dir = SCRIPT_DIR.parent / base_output_dir_name
    if feature_category == FeatureCategory.KERNEL_MICROBENCHMARKS.value:
        if not group:
            print(
                "Error: --group must be specified for 'kernel support matrix microbenchmarks' category."
            )
            sys.exit(1)
        output_dir = output_dir / group

    # Ensure the output directory exists. If not, create it.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the content of the template file.
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print("Read template file successfully.")
    except Exception as e:
        print(f"Error reading template file: {e}")
        sys.exit(1)

    # replace characters to satisfy filename and buildkite step key naming restrictions
    sanitized_feature_name = feature_name.replace("/", "_").replace(
        ".", "_").replace(" ", "_").replace(":", "-")

    # Substitute the placeholders with the provided arguments.
    format_args = {
        "FEATURE_NAME": feature_name,
        "CATEGORY": feature_category,
        "SANITIZED_FEATURE_NAME": sanitized_feature_name,
    }
    try:
        generated_content = template_content.format(**format_args)
        print("File content generated.")
    except KeyError as e:
        print(
            f"Error: A placeholder key {e} was not found in the provided arguments."
        )
        print(
            "Please check for mismatches between your template file and script."
        )
        sys.exit(1)

    generated_filepath = output_dir / f"{sanitized_feature_name}.yml"

    # Write the generated content to the file.
    try:
        with open(generated_filepath, 'w', encoding='utf-8') as f:
            f.write(generated_content)
        print(f"✅ Success! Config file generated at: '{generated_filepath}'")
    except Exception as e:
        print(f"Error writing output file to {generated_filepath}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add Buildkite yml config file for new feature.")

    # Add the command-line arguments. Both are required.
    parser.add_argument("--feature-name",
                        type=str,
                        required=True,
                        help="The name of the feature.")
    parser.add_argument(
        '--category',
        choices=[
            FeatureCategory.FEATURE.value,
            FeatureCategory.KERNEL.value,
            FeatureCategory.PARALLELISM.value,
            FeatureCategory.QUANTIZATION.value,
            FeatureCategory.KERNEL_MICROBENCHMARKS.value,
            FeatureCategory.RL.value,
        ],
        default=FeatureCategory.FEATURE.value,
        help=
        f'[OPTIONAL] Category of feature. (Default: {FeatureCategory.FEATURE.value})'
    )
    parser.add_argument(
        '--group',
        type=str,
        help=
        "[OPTIONAL] For 'kernel support matrix microbenchmarks' category, specify the group name (subdirectory name)."
    )
    args = parser.parse_args()

    if args.category == FeatureCategory.KERNEL_MICROBENCHMARKS.value and not args.group:
        parser.error(
            "--group is required when category is 'kernel support matrix microbenchmarks'"
        )

    generate_from_template(
        feature_name=args.feature_name,
        feature_category=args.category,
        group=args.group,
    )


if __name__ == "__main__":
    main()
