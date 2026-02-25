
import json
import argparse
import os
from datetime import datetime
import sys

def parse_results(file_path):
    """
    Parses a JSON results file from lm-evaluation-harness and extracts key information.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}", file=sys.stderr)
            sys.exit(1)

    # Extract the main task name (assuming one task per file)
    task_name = list(data['results'].keys())[0]
    results = data['results'][task_name]

    metrics_dict = {}
    metrics = [key for key in results.keys() if '_stderr' not in key and isinstance(results[key], (int, float))]
    for metric in metrics:
        metric_name = metric.replace(',none', '')
        metrics_dict[metric_name] = results[metric]
        stderr_key = metric.replace(',none', '_stderr,none')
        if stderr_key in results:
            metrics_dict[f"{metric_name}_stderr"] = results[stderr_key]

    # Print machine-readable JSON to stdout for the runner script
    print(json.dumps(metrics_dict))

    # Print human-readable summary to stderr for the user
    # Extract other key information
    model_name = data.get('model_name', 'N/A')
    num_fewshot = data.get('n-shot', {}).get(task_name, 'N/A')
    num_samples = data.get('n-samples', {}).get(task_name, {}).get('effective', 'N/A')
    git_hash = data.get('git_hash', 'N/A')
    total_time = data.get('total_evaluation_time_seconds', 'N/A')
    timestamp = data.get('date', 'N/A')
    
    if timestamp != 'N/A':
        try:
            # Attempt to convert from Unix timestamp
            timestamp = float(timestamp)
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            # Fallback for non-timestamp formats
            date_str = str(timestamp)
    else:
        date_str = 'N/A'


    print("--- LM Evaluation Harness Results Summary ---", file=sys.stderr)
    print(f"File:          {os.path.basename(file_path)}", file=sys.stderr)
    print(f"Model:         {model_name}", file=sys.stderr)
    print(f"Task:          {task_name}", file=sys.stderr)
    print(f"Date:          {date_str}", file=sys.stderr)
    print(f"Num Few-shot:  {num_fewshot}", file=sys.stderr)
    print(f"Num Samples:   {num_samples}", file=sys.stderr)

    if total_time != 'N/A':
        try:
            total_time = float(total_time)
            print(f"Total Time:    {total_time:.2f}s", file=sys.stderr)
        except (ValueError, TypeError):
            print(f"Total Time:    {total_time}", file=sys.stderr)
    
    print("-" * 43, file=sys.stderr)

    # Find all metrics that don't end with _stderr
    metrics = [key for key in results.keys() if '_stderr' not in key and isinstance(results[key], (int, float))]

    for metric in metrics:
        metric_name = metric.replace(',none', '')
        print(f"Metric:        {metric_name}", file=sys.stderr)
        print(f"Value:         {results[metric]:.4f}", file=sys.stderr)
        
        # Construct the corresponding stderr key
        stderr_key = metric.replace(',none', '_stderr,none')
        
        # Check if the stderr key exists in the results
        if stderr_key in results:
            print(f"Std. Err.:     {results[stderr_key]:.4f}", file=sys.stderr)
        
        print("-" * 43, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a JSON results file from lm-evaluation-harness."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the JSON results file.",
    )
    args = parser.parse_args()
    parse_results(args.file_path)
