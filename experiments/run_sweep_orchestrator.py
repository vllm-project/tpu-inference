"""
Orchestrates systematic parameter sweeps for TPU vLLM profiling

Reads a YAML configuration detailing grid search parameters (like batch sizes,
input/output lengths, and model kwargs) and iteratively launches the underlying
 script. It handles checkpointing/resuming from failed runs,
skips already-completed configurations by checking the resulting CSV, and manages
the trace configuration mappings.
"""

import yaml
import subprocess
import argparse
import os
import sys
import json
import itertools
import csv
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "tpu_profile_vllm.py")
DEFAULT_RESULT_DIR = os.path.join(BASE_DIR, "results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Base directory for results")
    parser.add_argument("--experiment-id", default=None, help="Pass to resume sweep from specific ID")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set tensor_parallel_size default to 4 to split model weights between 4 TPU devices. 
    tp_size = config.get("tensor_parallel_size", 4)
    dtype = config.get("dtype", "bfloat16")
    max_model_len = config.get("max_model_len", None)
    env_vars = config.get("env_vars", {})
    engine_args = config.get("engine_args", {})
    
    sweep = config.get("sweep_matrix", {})
    trace_configs = config.get("trace_configs", [])

    # Enforce all values as iterables natively 
    for k, v in sweep.items():
        if not isinstance(v, list):
            sweep[k] = [v]

    sweep_keys = list(sweep.keys())
    sweep_values = list(sweep.values())
    cross_product = list(itertools.product(*sweep_values))

    # Pre-Flight Validation for trace_configs
    for tc in trace_configs:
        # Normalize tc keys
        canonical_tc = {}
        for k, v in tc.items():
            if k == "jax_advanced_configuration": continue
            if k == "batch": k = "batch_size"
            if k in ["input", "inputs"]: k = "input_len"
            canonical_tc[k] = v
            
        for k, v in canonical_tc.items():
            if k not in sweep_keys or v not in sweep[k]:
                print(f"❌ ERROR: Trace config {tc} is invalid. {k}={v} is not in the generated sweep matrix boundaries.")
                return

    # Ensure that result directory exists for current experiments
    exp_id = args.experiment_id if args.experiment_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.result_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Ensure that csv file and trace directory exists to store experiment results
    csv_file = os.path.join(exp_dir, "results.csv")
    traces_dir = os.path.join(exp_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    
    # Check if sweep has been done before, if so which experiment runs were completed.
    completed_configs = []
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_configs.append(row)
    
    # Iterate through each experiment run in experiment sweep
    for combo in cross_product:
        current_config = dict(zip(sweep_keys, combo))

        # skip if experiment run has been done before
        skip = False
        for completed_row in completed_configs:
            if all(str(completed_row.get(k)) == str(v) for k, v in current_config.items()):
                skip = True
                break
                        
        if skip:
            print(f">>> [SKIPPED] Coordinate {current_config} already logged in {csv_file}.")
            continue

        # build script command
        print(f">>> Running iteration: {current_config}")
        cmd = [
            sys.executable, SCRIPT_PATH,
            "--csv-file", csv_file,
            "--sweep-metadata", json.dumps(current_config),
            "--tensor-parallel-size", str(tp_size),
            "--dtype", dtype,
            "--engine-args", json.dumps(engine_args)
        ]
        
        # Build CLI args generically
        for k, val in current_config.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(val)])
            
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
            
        # Target isolated JAX Traces
        for tc in trace_configs:
            # Normalize user trace matching to coordinate dictionary explicitly
            canonical_tc = {}
            for k, v in tc.items():
                if k == "jax_advanced_configuration": continue
                if k == "batch": k = "batch_size"
                if k in ["input", "inputs"]: k = "input_len"
                canonical_tc[k] = v
                
            # If everything in canonical_tc matches current_config natively
            if all(current_config.get(k) == v for k, v in canonical_tc.items()):
                trace_name = "_".join(f"{k}{v}".replace('/', '-') for k, v in canonical_tc.items())
                trace_dir = os.path.join(traces_dir, f"trace_{trace_name}")
                cmd.extend(["--trace", "--profile-result-dir", trace_dir])
                if "jax_advanced_configuration" in tc:
                    cmd.extend(["--jax-advanced-configuration", json.dumps(tc["jax_advanced_configuration"])])
                print(f"    >>> Profiling trace enabled! Dumps routing to {trace_dir}")
                break

        # Setup pure XLA orchestration environment
        env = os.environ.copy()
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        for k, v in env_vars.items():
            env[str(k)] = str(v)
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! CRASH DETECTED. This usually indicates a limits Wall.")
            print(f"!!! Cleaning TPU locks and safely moving to the next matrix dimension...")
            subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"])
                
    print(f"Sweep configuration {args.config} complete! Results written to {csv_file}")

if __name__ == "__main__":
    main()
