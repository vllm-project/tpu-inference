import argparse
import time
import os
import sys
import json
import csv
from vllm import LLM, SamplingParams
import jax

def main(args):
    print(f"\n========== Benchmark Results ==========\nInitializing model: {args.model}")
    
    engine_kwargs = {"tensor_parallel_size": args.tensor_parallel_size, "dtype": args.dtype}
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len
        
    extra_args = json.loads(args.engine_args)
    engine_kwargs.update(extra_args)

    llm = LLM(model=args.model, **engine_kwargs)
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ] * (args.batch_size // 2)
    
    if len(prompts) < args.batch_size:
        prompts.extend(["Fill"] * (args.batch_size - len(prompts)))

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.output_len)

    # Warmup
    llm.generate(prompts, sampling_params)

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
    outputs = llm.generate(prompts, sampling_params)
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
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(meta)

    print(f"Wrote metrics to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--profile-result-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--csv-file", type=str, default=None, help="Explicit path to target CSV file")
    parser.add_argument("--engine-args", type=str, default="{}", help="JSON string of extra kwargs for LLM()")
    parser.add_argument("--jax-advanced-configuration", type=str, default="{}", help="Stringified JSON array explicitly overriding profile configuration")
    parser.add_argument("--sweep-metadata", type=str, default="{}", help="JSON dict outlining precise configuration limits extracted dynamically for output dump")
    args = parser.parse_args()
    main(args)
