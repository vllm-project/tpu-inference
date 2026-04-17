import argparse
import yaml
import os

def generate_cases(output_path):
    # 1. Logic to generate your test cases
    # This is where you would iterate through your tuning configurations
    test_cases = [
        {"name": "gmm_case_1", "params": "--size 1024 --dtype float32"},
        {"name": "gmm_case_2", "params": "--size 2048 --dtype bfloat16"},
    ]

    # 2. Logic to update your GCP Database
    print(f"Connecting to GCP Project to sync {len(test_cases)} cases...")
    # [Insert your DB insertion logic here]

    # 3. Create the Buildkite Pipeline Structure
    # This defines what the "next" steps in the pipeline will look like
    pipeline = {
        "steps": []
    }

    for case in test_cases:
        step = {
            "label": f":test_tube: Run {case['name']}",
            "agent": {"queue": "tpu_v7x_2_queue"}, # Adjust to your TPU queue
            "env": {
                "USE_PREBUILT_IMAGE": "1",
                "TPU_VERSION": "tpu7x"
            },
            "commands": [
                f".buildkite/scripts/run_in_docker.sh python3 tests/run_kernel_test.py {case['params']}"
            ]
        }
        pipeline["steps"].append(step)

    # 4. Write to the specified output path
    print(f"Writing generated pipeline to {output_path=}, {os.path.dirname(output_path)=}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
    
    print(f"Successfully generated dynamic pipeline at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GMM test cases and Buildkite YAML.")
    parser.add_argument(
        "--output-path", 
        type=str, 
        required=True, 
        help="Path where the generated .yml file should be saved"
    )
    args = parser.parse_args()
    generate_cases(args.output_path)