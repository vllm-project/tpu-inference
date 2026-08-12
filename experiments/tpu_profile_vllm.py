"""
Standalone execution and profiling script for vLLM on TPUs.

Initializes a HuggingFace model using vLLM, manages JAX profiler toggles, and
generates textual outputs. It measures raw tokens-per-second throughput and
optionally exports JAX XProf traces for analysis in TensorBoard.
Results are appended to a central CSV alongside the reproduction command.
"""
import argparse
import time
import os
import sys
import json
import csv
from vllm import LLM, SamplingParams
import jax
import numpy as np

def main(args):
    print(f"\n========== Benchmark Results ==========\nInitializing model: {args.model}")
    
    engine_kwargs = {"tensor_parallel_size": args.tensor_parallel_size, "dtype": args.dtype}
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len
        
    extra_args = json.loads(args.engine_args)
    engine_kwargs.update(extra_args)

    llm = LLM(model=args.model, **engine_kwargs)
    
    # Generate exact dummy token lengths according to input_len and batch_size 
    # Note: vocabulary size of Gemma 4 is 262144, far larger than our random range of 0 to 9999
    dummy_prompt_token_ids = np.random.randint(10000, size=(args.batch_size, args.input_len))
    prompts = [
        {"prompt_token_ids": [int(x) for x in batch]} 
        for batch in dummy_prompt_token_ids
    ]

    # Temperature controls shape of sampling probability distribution 
    #   higher temperature -> more deterministic sampling
    # Top p controls the size of the token candidate pool 
    #   0.95 keeps the top tokens whose probabilities add up to 95% of the total probability pool
    # ignore_eos=True ensures that model will keep generating until it hits the specified output length
    sampling_params = SamplingParams(temperature=0.8, 
                                     top_p=0.95, 
                                     max_tokens=args.output_len, 
                                     ignore_eos=True)

    # Warmup
    llm.generate(prompts=prompts, sampling_params=sampling_params)

    if args.trace:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.device_tracer_level = 1
        
        adv_config = json.loads(args.jax_advanced_configuration)
        if adv_config:
            options.advanced_configuration = adv_config
            
        base_dir = args.profile_result_dir
        os.makedirs(base_dir, exist_ok=True)
        jax.profiler.start_trace(base_dir, profiler_options=options)

    start_time = time.time()
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    end_time = time.time()

    if args.trace:
        jax.profiler.stop_trace()

    duration = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / duration

    print("\n========== Benchmark Results ==========")

    meta = {}
    if args.sweep_metadata:
        meta = json.loads(args.sweep_metadata)
    else:
        meta = {"model": args.model, "batch_size": args.batch_size, "input_len": args.input_len, "output_len": args.output_len}

    for k, v in meta.items():
        print(f"{k.capitalize():<25} {v}")

    print(f"{'Duration:':<25} {duration:.4f} s")
    print(f"{'Throughput:':<25} {throughput:.2f} tok/s")
    print("========================================\n")

    csv_file = args.csv_file if args.csv_file else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "summary_metrics.csv")
    csv_dir = os.path.dirname(csv_file)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    meta["Duration_s"] = f"{duration:.4f}"
    meta["Throughput_tok_s"] = f"{throughput:.2f}"
    
    # Extract only the file and the relevant args
    cmd_str = f"python3 {' '.join(sys.argv)}"
    meta["Reproduction_Command"] = cmd_str

    file_exists = os.path.exists(csv_file)
    
    if file_exists and os.path.getsize(csv_file) > 0:
        with open(csv_file, "r") as f_read:
            try:
                fieldnames = next(csv.reader(f_read))
            except StopIteration:
                fieldnames = list(meta.keys())
    else:
        # Rely on Python's guaranteed insertion order: 
        # sweep metadata is loaded first, metrics and command are appended last.
        fieldnames = list(meta.keys())

    dropped_keys = set(meta.keys()) - set(fieldnames)
    if dropped_keys:
        print(f"Warning: Dropping new keys not present in existing CSV: {dropped_keys}")

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', restval='')
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writeheader()
        writer.writerow(meta)

    print(f"Wrote metrics to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    # default tensor-parallel-size to 4 for 4 TPU devices
    parser.add_argument("--tensor-parallel-size", type=int, default=4) 
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--profile-result-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--csv-file", type=str, default=None, help="Explicit path to target CSV file")
    parser.add_argument("--engine-args", type=str, default="{}", help="JSON string of extra kwargs for LLM()")
    parser.add_argument("--jax-advanced-configuration", type=str, default="{}", help="Stringified JSON array to specify GFC counter options")
    parser.add_argument("--sweep-metadata", type=str, default="{}", help="JSON dict outlining precise configuration limits extracted dynamically for output dump")
    args = parser.parse_args()
    main(args)
