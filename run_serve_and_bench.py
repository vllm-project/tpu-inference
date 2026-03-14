#
# python run_serve_and_bench.py --num_iterations=8 --random_range_ratio=0.8 --max_concurrency=512 --cleanup=True

import contextlib
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import requests
from absl import app
from absl import flags
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

_IN_OUT = flags.DEFINE_list("in_out", ["1024,8192", "8192,1024"], "Input output configs")
_NUM_ITERATIONS = flags.DEFINE_integer("num_iterations", 1, "Number of iterations to run for each configuration.")
_RANDOM_RANGE_RATIO = flags.DEFINE_float("random_range_ratio", 0.8, "Random range ratio for benchmark.")
_MAX_CONCURRENCY = flags.DEFINE_integer("max_concurrency", 64, "Maximum concurrency for the benchmark.")
_NUM_PROMPTS = flags.DEFINE_integer("num_prompts", 320, "Number of prompts to process.")

_ATTN_DP = flags.DEFINE_bool("attn_dp", False, "Enable attention data parallelism")
_ENABLE_EP = flags.DEFINE_bool("enable_ep", False, "Enable expert parallelism")
_ENABLE_XPROF = flags.DEFINE_bool("enable_xprof", False, "Enable xprof profiling")

_START_SERVER_ONCE = flags.DEFINE_bool("start_server_once", True, "Whether to start server once and run all benchmark, or start server and run one benchmark and then shutdown before start the server agian to run the next benchmark.")
_CLEANUP = flags.DEFINE_bool("cleanup", False, "Whether to clean up the result and profile directories before running.")

_BASE_DIR="/home/wyzhang_google_com/mnt/ullm/"
_EXP_BASE_DIR = flags.DEFINE_string("exp_base_dir", os.path.join(_BASE_DIR, "debug/variance-study-with-xprof"), "Base directory for experiments.")

def start_server(phased_profiling_dir, server_log_file):
    env = os.environ.copy()
    env.update({
        "USE_MOE_EP_KERNEL": "0",
        "MODEL_IMPL_TYPE": "vllm",
        "HF_HOME": "/home/wyzhang_google_com/ckpt/hf",
    })

    if _ENABLE_XPROF.value:
        env["PHASED_PROFILING_DIR"] = phased_profiling_dir

    if _ATTN_DP.value:
        env["NEW_MODEL_DESIGN"] = "True"
        
    server_cmd = [
        "vllm", "serve", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "--max-model-len=10240",
        "--max-num-seqs=512",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization=0.95",
        "--tensor-parallel-size=8",
        "--download-dir=/home/wyzhang_google_com/ckpt/hf",
        "--async-scheduling",
        "--port=8000",
        "--kv-cache-dtype=fp8",
        "--safetensors-load-strategy", "eager",
        "--load-format", "runai_streamer",
        "--model-loader-extra-config", '{"memory_limit":68719476736,"concurrency":4}',
    ]

    if _ENABLE_EP.value:
        server_cmd.append("--enable-expert-parallel")

    if _ATTN_DP.value:
        server_cmd.extend([
            "--max-num-batched-tokens=4096",
            "--additional-config", '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}',
        ])
    else:
        server_cmd.extend([
            "--max-num-batched-tokens=8192",
        ])

    print(f"[server] Starting server: {' '.join(server_cmd)}")
    with open(server_log_file, "w") as f:
        server_process = subprocess.Popen(server_cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return server_process, " ".join(server_cmd)


def wait_for_server(url, timeout=1800):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("[server] Server is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(60)
        print("[server] Waiting for server...")
    print("[server] Server failed to start.")
    return False


@contextlib.contextmanager
def server_manager(phased_profiling_dir, server_log_file):
    server_process, server_cmd = start_server(phased_profiling_dir, server_log_file)
    try:
        yield server_process, server_cmd
    finally:
        print("[server] Terminating server...")
        server_process.terminate()
        server_process.wait()
        time.sleep(15)


def get_latest_result_file(result_dir):
    list_of_files = glob.glob(os.path.join(result_dir, "*.json"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)


def parse_result(json_file, input_len, output_len):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Filter for scalar values to put in CSV
        row = {"input_len": input_len, "output_len": output_len}
        for k, v in data.items():
            if isinstance(v, (int, float, str)) and not isinstance(v, bool):
                row[k] = v
            
        # Sort keys to ensure consistent CSV order, keeping input/output len first
        keys = sorted([k for k in row.keys() if k not in ("input_len", "output_len")])
        keys = ["input_len", "output_len"] + keys
        
        header = ",".join(keys)
        values = ",".join(str(row[k]) for k in keys)
        return header, values
    except Exception as e:
        print(f"[bench] Error parsing result: {e}")
        return None, None


def run_benchmark(input_len, output_len, server_cmd, result_dir, bench_log_file):
    print(f"[bench] Running benchmark with input_len={input_len} and output_len={output_len}")
    subprocess.run(["date"], check=True)
    benchmark_cmd = [
        sys.executable, "../bench_serving/benchmark_serving.py",
        "--ignore-eos",
        "--model", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "--backend", "vllm",
        "--port", "8000",
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", str(_RANDOM_RANGE_RATIO.value),
        "--num-prompts", str(_NUM_PROMPTS.value),
        "--max-concurrency", str(_MAX_CONCURRENCY.value),
        "--save-result",
        "--result-dir", result_dir,
        "--temperature", "0.0",
    ]
    with open(bench_log_file, "w") as f:
        subprocess.run(benchmark_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    subprocess.run(["date"], check=True)
    
    latest_file = get_latest_result_file(result_dir)
    if latest_file:
        header, values = parse_result(latest_file, input_len, output_len)
        if header and values:
            print(f"[bench] Server command: {server_cmd}")
            print(f"[bench] Benchmark command: {' '.join(benchmark_cmd)}")
            print(f"[bench] Result for {input_len},{output_len}:")
            print(header)
            print(values)
            return header, values
    return None, None

def main(argv):
    del argv  # Unused.

    if _CLEANUP.value:
        if os.path.exists(_EXP_BASE_DIR.value):
            print(f"[cleanup] Cleaning up {_EXP_BASE_DIR.value}...")
            shutil.rmtree(_EXP_BASE_DIR.value)

    # Parse in_out flags
    flat_in_out = []
    for item in _IN_OUT.value:
        flat_in_out.extend(item.split(','))
    
    input_output_configs = []
    if len(flat_in_out) % 2 != 0:
        raise ValueError("in_out flag must contain pairs of input,output lengths.")
    
    for i in range(0, len(flat_in_out), 2):
        input_output_configs.append((int(flat_in_out[i]), int(flat_in_out[i+1])))

    benchmark_history = []
    csv_header = None

    def collect_result(h, v):
        nonlocal csv_header
        if h and v:
            if csv_header is None:
                csv_header = h
            benchmark_history.append(v)

    total_iterations = _NUM_ITERATIONS.value * len(input_output_configs)
    iteration_count = 0

    for i in range(_NUM_ITERATIONS.value):
        for input_len, output_len in input_output_configs:
            iteration_count += 1
            print(f"--- Running iteration {iteration_count}/{total_iterations} (config: {input_len},{output_len}, iter: {i+1}) ---")

            iter_dir = os.path.join(_EXP_BASE_DIR.value, f"exp-{iteration_count}")
            vllm_log_dir = os.path.join(iter_dir, "vllm")
            bench_log_dir = os.path.join(iter_dir, "bench_log")
            result_dir = os.path.join(iter_dir, "bench_results")
            phased_profiling_dir = os.path.join(iter_dir, "xprof")

            os.makedirs(vllm_log_dir, exist_ok=True)
            os.makedirs(bench_log_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            if _ENABLE_XPROF.value:
                os.makedirs(phased_profiling_dir, exist_ok=True)

            vllm_log_file = os.path.join(vllm_log_dir, "vllm.log")
            bench_log_file = os.path.join(bench_log_dir, f"bench_{iteration_count}.log")

            with server_manager(phased_profiling_dir, vllm_log_file) as (_, server_cmd):
                if not wait_for_server("http://localhost:8000/health"):
                    print(f"Server failed to start for iteration {iteration_count}. Skipping.")
                    continue

                try:
                    print(f"[bench] Server is ready. Starting benchmark for {input_len}, {output_len}...")
                    h, v = run_benchmark(input_len, output_len, server_cmd, result_dir, bench_log_file)
                    collect_result(h, v)
                except Exception as e:
                    print(f"[main] An error occurred during benchmark: {e}")

    if csv_header and benchmark_history:
        print("\n[bench] Benchmark History:")
        print(csv_header)
        for row in benchmark_history:
            print(row)

if __name__ == "__main__":
    app.run(main)