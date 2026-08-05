import argparse
import time
import os
import json
import numpy as np
import jax
from vllm import LLM, SamplingParams

def main(args):
    base_dir = args.profile_result_dir
    os.makedirs(base_dir, exist_ok=True)

    dummy_prompt_token_ids = np.random.randint(10000, size=(args.batch_size, args.input_len))
    dummy_prompts = [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]

    extra_engine_args = json.loads(args.engine_args) if args.engine_args else {}
    
    max_len = args.max_model_len if args.max_model_len else max(2048, args.input_len + args.output_len)

    print(f"Loading {args.model} on {args.tensor_parallel_size} TPUs (dtype={args.dtype}, max_model_len={max_len})...")
    if extra_engine_args:
        print(f"Extra engine args: {extra_engine_args}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_len,
        trust_remote_code=True,
        dtype=args.dtype,
        **extra_engine_args
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)

    print("Running warmup iteration...")
    _ = llm.generate(dummy_prompts, sampling_params)

    if args.trace:
        options = jax.profiler.ProfileOptions(
            host_tracer_level=2,
            device_tracer_level=1,
        )
        print(f"Starting JAX profiling trace to {base_dir}...")
        jax.profiler.start_trace(base_dir, options=options)

    start_time = time.perf_counter()
    outputs = llm.generate(dummy_prompts, sampling_params)
    duration = time.perf_counter() - start_time

    if args.trace:
        jax.profiler.stop_trace()

    total_tokens_generated = sum(len(out.outputs[0].token_ids) for out in outputs)
    throughput = total_tokens_generated / duration
    print("\n========== Benchmark Results ==========")
    print(f"Model:                     {args.model}")
    print(f"Batch Size:                {args.batch_size}")
    print(f"Input Length:              {args.input_len}")
    print(f"Output Length:             {args.output_len}")
    print(f"Duration:                  {duration:.4f} s")
    print(f"Throughput:                {throughput:.2f} tok/s")
    print("========================================\n")

    csv_file = args.csv_file if args.csv_file else os.path.join(base_dir, "summary_metrics.csv")
    csv_dir = os.path.dirname(csv_file)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a") as f:
        if not file_exists:
            f.write("Model,Batch_Size,Input_Len,Output_Len,Duration_s,Throughput_tok_s,Reproduction_Command\n")
        cmd_str = f"/mnt/pd/shen/baseline_v2/tpu_profile_vllm.py --model {args.model} --input-len {args.input_len} --output-len {args.output_len} --batch-size {args.batch_size} --csv-file {csv_file}"
        f.write(f"{args.model},{args.batch_size},{args.input_len},{args.output_len},{duration:.4f},{throughput:.2f},\"{cmd_str}\"\n")

    print(f"Wrote metrics to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--profile-result-dir", type=str, default="/mnt/pd/shen/baseline_v2/results")
    parser.add_argument("--csv-file", type=str, default=None, help="Explicit path to target CSV file")
    parser.add_argument("--engine-args", type=str, default="{}", help="JSON string of extra kwargs for LLM()")
    args = parser.parse_args()
    main(args)
