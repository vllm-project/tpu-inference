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

from constant import QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "models"


class ModelType(str, Enum):
    TPU_OPTIMIZED = "tpu-optimized"
    VLLM_NATIVE = "vllm-native"


class ModelCategory(str, Enum):
    TEXT_ONLY = "text-only"
    MULTIMODAL = "multimodal"


MODEL_TYPE_TO_TEMPLATE = {
    ModelType.TPU_OPTIMIZED.value: "tpu_optimized_model_template.yml",
    ModelType.VLLM_NATIVE.value: "vllm_native_model_template.yml",
}


def generate_from_template(model_name: str, queue: str, model_type: str,
                           model_category: str) -> None:
    """
    Generates a buildkite yml file from model template.
    Args:
        model_name (str): The full name of the model on Hugging Face.
        queue (str): The buildkite queue to run tests for this model on.
        model_type (str): The type of model (tpu-optimized or vllm-native).
    """
    if queue not in QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP:
        print(
            f"Queue {queue} not previously registered on Buildkite. If you added a queue, please add it to QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP"
        )
        sys.exit(1)

    print(f"Starting to generate for model '{model_name}'")

    # Check if the template file exists.
    template_path = SCRIPT_DIR / MODEL_TYPE_TO_TEMPLATE[model_type]
    if not template_path.is_file():
        print(
            f"Error: Template path '{template_path}' invalid. Did you remove it by accident?"
        )
        sys.exit(1)

    # Ensure the output directory exists. If not, create it.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read the content of the template file.
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print("Read template file successfully.")
    except Exception as e:
        print(f"Error reading template file: {e}")
        sys.exit(1)

    # replace characters to satisfy filename and buildkite step key naming restrictions
    sanitized_model_name = model_name.replace("/", "_").replace(".", "_")

    # Substitute the placeholders with the provided arguments.
    try:
        generated_content = template_content.format(
            MODEL_NAME=model_name,
            CATEGORY=model_category,
            SANITIZED_MODEL_NAME=sanitized_model_name,
            QUEUE=queue,
            TENSOR_PARALLEL_SIZE=QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP[queue],
        )
        print("File content generated.")
    except KeyError as e:
        print(
            f"Error: A placeholder key {e} was not found in the provided arguments."
        )
        print(
            "Please check for mismatches between your template file and script."
        )
        sys.exit(1)

    generated_filepath = OUTPUT_DIR / f"{sanitized_model_name}.yml"

    print("Writing output file")
    try:
        with open(generated_filepath, 'w', encoding='utf-8') as f:
            f.write(generated_content)
        print(f"✅ Success! Config file generated at: '{generated_filepath}'")
    except Exception as e:
        print(f"Error writing output file to {generated_filepath}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add Buildkite yml config file for new model.")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=
        "The full name of the model on Hugging Face (ex: 'meta-llama/Llama-3.1-8B')."
    )
    parser.add_argument(
        "--queue",
        type=str,
        required=True,
        help="The name of the agent queue to use (ex: 'tpu_v6e_queue')")
    parser.add_argument(
        '--type',
        choices=[ModelType.TPU_OPTIMIZED.value, ModelType.VLLM_NATIVE.value],
        default='tpu-optimized',
        help=
        '[OPTIONAL] Type of model. Must be tpu-optimized or vllm-native. (Default: tpu-optimized)'
    )
    parser.add_argument(
        '--category',
        choices=[
            ModelCategory.TEXT_ONLY.value, ModelCategory.MULTIMODAL.value
        ],
        default='text-only',
        help=
        '[OPTIONAL] Category of model. Must be "text-only" or "multimodal". (Default: text-only)'
    )

    args = parser.parse_args()
    generate_from_template(model_name=args.model_name,
                           queue=args.queue,
                           model_type=args.type,
                           model_category=args.category)


if __name__ == "__main__":
    main()
