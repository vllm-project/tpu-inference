import argparse
import sys
from pathlib import Path

from constant import QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "feature_template.yml"
OUTPUT_DIR = SCRIPT_DIR.parent / "features"


def generate_from_template(feature_name: str, queue: str) -> None:
    """
    Generates a buildkite yml file from feature template.
    Args:
        feature_name (str): The name of the feature.
        queue (str): The buildkite queue to run tests for this feature on.
    """
    if queue not in QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP:
        print(
            f"Queue {queue} not previously registered on Buildkite. If you added a queue, please add it to QUEUE_TO_TENSOR_PARALLEL_SIZE_MAP"
        )
        sys.exit(1)

    print(f"--- Starting to generate for Feature '{feature_name}' ---")

    # Check if the template file exists.
    if not TEMPLATE_PATH.is_file():
        print(
            f"Error: Template file '{TEMPLATE_PATH}' invalid. Did you remove it by accident?"
        )
        sys.exit(1)

    # Ensure the output directory exists. If not, create it.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read the content of the template file.
    try:
        with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print("Read template file successfully.")
    except Exception as e:
        print(f"Error reading template file: {e}")
        sys.exit(1)

    # replace characters to satisfy filename and buildkite step key naming restrictions
    sanitized_feature_name = feature_name.replace("/", "_").replace(
        ".", "_").replace(" ", "_").replace(":", "-")

    # Substitute the placeholders with the provided arguments.
    try:
        generated_content = template_content.format(
            FEATURE_NAME=feature_name,
            SANITIZED_FEATURE_NAME=sanitized_feature_name,
            QUEUE=queue,
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

    generated_filepath = OUTPUT_DIR / f"{sanitized_feature_name}.yml"

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
        "--queue",
        type=str,
        required=True,
        help="The name of the agent queue to use (ex: 'tpu_v6e_queue')")

    args = parser.parse_args()
    generate_from_template(feature_name=args.feature_name, queue=args.queue)


if __name__ == "__main__":
    main()
