import argparse
import json
import os
import sys

def parse_mmlu_results(input_file):
    """
    Parses the raw results from an lm_eval MMLU run. It prints a
    machine-readable JSON object to stdout for automation and a human-readable
    summary to stderr. All metric keys in the JSON output are prefixed with
    'mmlu_' for consistency, and the main aggregate score is named 'mmlu_agg'.

    Args:
        input_file (str): Path to the input JSON file from lm_eval.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    results = data.get("results", {})
    groups = data.get("groups", {})
    
    summary = {}
    acc_key = "exact_match,strict_match"

    # Process all tasks and group aggregates from the 'results' dictionary.
    # The keys from lm-eval are already correctly prefixed (e.g., 'mmlu_humanities').
    for task, metrics in results.items():
        if acc_key in metrics:
            summary[task] = metrics[acc_key]

    # The overall aggregate is in the 'groups' dict under the key 'mmlu_llama'.
    # We rename it to 'mmlu_agg' for clarity in our database.
    if 'mmlu_llama' in groups and acc_key in groups['mmlu_llama']:
         summary['mmlu_agg'] = groups['mmlu_llama'][acc_key]
         # The 'mmlu' task is just the aggregate, so remove it to avoid duplication
         if 'mmlu_llama' in summary:
             del summary['mmlu_llama']

    # Print machine-readable JSON to stdout for the runner script
    print(json.dumps(summary))

    # Print human-readable summary to stderr for the user
    print("\n--- MMLU Results Summary ---", file=sys.stderr)
    print(f"File: {os.path.basename(input_file)}", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    
    if 'mmlu_agg' in summary:
        print(f"Overall MMLU Accuracy: {summary['mmlu_agg']:.4f}", file=sys.stderr)
        print("-" * 30, file=sys.stderr)

    print("Subtask Accuracies:", file=sys.stderr)
    # Sort the subtasks alphabetically for consistent and readable output
    for task, acc in sorted(summary.items()):
        if task != 'mmlu_agg':
            print(f"- {task}: {acc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse lm_eval MMLU results.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file from lm_eval.")
    args = parser.parse_args()

    parse_mmlu_results(args.input_file)