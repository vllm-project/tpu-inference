import yaml
import subprocess
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--result-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), help="Base directory for results")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = config.get("model", "google/gemma-4-12b-it")
    tp_size = config.get("tensor_parallel_size", 4)
    dtype = config.get("dtype", "bfloat16")
    max_model_len = config.get("max_model_len", None)
    env_vars = config.get("env_vars", {})
    engine_args = config.get("engine_args", {})
    
    sweep = config.get("sweep_matrix", {})
    batches = sweep.get("batches", config.get("batches", [1]))
    inputs = sweep.get("inputs", config.get("inputs", [128]))
    output_len = sweep.get("output_len", config.get("output_len", 64))
    
    # Clean model directory name and CSV file path
    model_dir_name = model.replace("/", "--")
    yaml_stem = os.path.splitext(os.path.basename(args.config))[0]
    model_result_dir = os.path.join(args.result_dir, model_dir_name)
    csv_file = os.path.join(model_result_dir, f"{yaml_stem}.csv")
    os.makedirs(model_result_dir, exist_ok=True)
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("Model,Batch_Size,Input_Len,Output_Len,Duration_s,Throughput_tok_s,Reproduction_Command\n")
    
    for b in batches:
        for i in inputs:
            skip = False
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    for line in f:
                        if line.startswith(f"{model},{b},{i},{output_len},"):
                            skip = True
                            break
                            
            if skip:
                print(f">>> [SKIPPED] Batch {b} | Input {i} already exists in {csv_file}.")
                continue
                
            print(f">>> Running Batch {b} | Input {i} | Output {output_len}")
            cmd = [
                "/mnt/pd/shen/vllm_env/bin/python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpu_profile_vllm.py"),
                "--model", model,
                "--input-len", str(i),
                "--output-len", str(output_len),
                "--batch-size", str(b),
                "--csv-file", csv_file,
                "--tensor-parallel-size", str(tp_size),
                "--dtype", dtype,
                "--engine-args", json.dumps(engine_args)
            ]
            if max_model_len:
                cmd.extend(["--max-model-len", str(max_model_len)])
            
            # Setup pure XLA orchestration environment
            env = os.environ.copy()
            env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            for k, v in env_vars.items():
                env[str(k)] = str(v)
            
            try:
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"!!! CRASH DETECTED at Batch {b} | Input {i}. This usually indicates a limits Wall.")
                print(f"!!! Cleaning TPU locks and safely moving to the next matrix dimension...")
                subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"])
                
    print(f"✅ Sweep configuration {args.config} complete! Results written to {csv_file}")

if __name__ == "__main__":
    main()
