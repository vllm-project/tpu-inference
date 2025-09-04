import argparse
from pathlib import Path

# Define the template filename and output directory as constants for easy modification.
TEMPLATE_FILENAME = "buildkite_ci_model_template.yml"
OUTPUT_DIR = Path("models")

def generate_from_template(model_name: str, queue: str) -> None:
    """
    Generates a buildkite yml file from a template.

    Args:
        model_name (str): The model_name parameter.
        queue (str): The Queue parameter.
    """
    print(f"--- Starting to generate for model '{model_name}' ---")

    # Check if the template file exists.
    template_path = Path(TEMPLATE_FILENAME)
    if not template_path.is_file():
        print(f"Error: Template file '{TEMPLATE_FILENAME}' not found!")
        return

    # Ensure the output directory exists. If not, create it.
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' is ready.")

    # Read the content of the template file.
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print("Template file read successfully.")
    except Exception as e:
        print(f"Error reading template file: {e}")
        return

    # Replace '/' and "." with an underscore for valid filenames and buildkite's key.
    safe_model_name = model_name.replace("/", "_").replace(".", "_")

    # Substitute the placeholders with the provided arguments.
    try:
        generated_content = template_content.format(
            MODEL_NAME=model_name,
            SAFE_MODEL_NAME=safe_model_name,
            QUEUE=queue,
        )
        print("Parameter substitution complete.")
    except KeyError as e:
        print(f"Error: A placeholder key {e} was not found in the provided arguments.")
        print("Please check for mismatches between your template file and script.")
        return

    # Define the output filename and path.
    # The filename is based on the model_name with a .yml extension.
    output_filename = f"{safe_model_name}.yml"
    output_path = OUTPUT_DIR / output_filename

    # Write the generated content to the file.
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(generated_content)
        print(f"✅ Success! Config file generated at: '{output_path}'")
    except Exception as e:
        print(f"Error writing output file: {e}")

    print("-" * 40 + "\n")

def main():
    """
    Main function to parse command-line arguments and run the generator.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Buildkite CI config file from a template."
    )

    # Add the command-line arguments. Both are required.
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="""
             The name of the model to use in the template (e.g., 'meta-llama/Llama-3.1-8B-Instruct'). 
             If have '/' or '.' in the model name, it will be replaced with '_' in the generated file name.
        """
    )
    parser.add_argument(
        "--queue",
        type=str,
        required=True,
        help="The name of the agent queue to use (e.g., 'tpu_v6e_queue' or 'tpu_v6e_8_queue')."
    )

    args = parser.parse_args()
    generate_from_template(model_name=args.model_name, queue=args.queue)

if __name__ == "__main__":
    main()
