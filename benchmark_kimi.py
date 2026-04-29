#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Standalone benchmark for DeepSeek-R1 on TPU.
Measures the latency of loaded layers and the LAST output decode token.
Supports both JAX and vLLM Torch implementations.

Based on the user's provided configuration:
- Model: gs://tpu-commons-ci/deepseek/r1
- 1K input sequence length
- 8K output sequence length
- TP size 8
- max batched tokens 1024
- max seqs 64
- FP4 weight requantization (block size 512)
- FP8 KV cache
"""

import argparse
import datetime
import json
import os
import time


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek-R1 3 layers last token latency.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch"],
        default="torch",
        help=
        "Choose the backend: 'jax' for tpu-inference Flax implementation, 'torch' for vLLM Torchax implementation."
    )
    parser.add_argument("--model",
                        type=str,
                        default="moonshotai/Kimi-K2.6",
                        help="Path to the model checkpoint.")
    parser.add_argument("--tp-size",
                        type=int,
                        default=8,
                        help="Tensor parallel size.")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=128,
                        help="Max number of sequences.")
    parser.add_argument("--input-len",
                        type=int,
                        default=1,
                        help="Input prompt length.")
    parser.add_argument("--output-len",
                        type=int,
                        default=9 * 1024,
                        help="Output generation length.")
    parser.add_argument("--profiler-dir",
                        type=str,
                        default="jacobplatin/kimi-xprof",
                        help="Directory to save profiler traces.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1024,
                        help="Number of prompts to process.")
    parser.add_argument("--num-hidden-layers",
                        type=int,
                        default=5,
                        help="Number of hidden layers to load.")
    parser.add_argument(
        "--profiler-trigger-kv-len",
        type=int,
        default=9 * 1024 - 50,
        help=
        "Trigger profiling when batch size is max-num-seq * tp-size, all q len is 1, and all kv len >= this value."
    )

    # New arguments aligned with user command
    parser.add_argument("--moe-requantize-block-size",
                        type=str,
                        default="512",
                        help="MOE requantize block size.")
    parser.add_argument("--moe-requantize-weight-dtype",
                        type=str,
                        default="fp4",
                        help="MOE requantize weight dtype.")
    parser.add_argument("--new-model-design",
                        type=str,
                        default="1",
                        help="New model design flag.")
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=1024,
                        help="Max num batched tokens.")
    parser.add_argument("--kv-cache-dtype",
                        type=str,
                        default="fp8",
                        help="KV cache dtype.")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.95,
                        help="GPU memory utilization.")
    parser.add_argument("--enable-prefix-caching",
                        action="store_true",
                        default=False,
                        help="Enable prefix caching.")
    parser.add_argument("--no-enable-expert-parallel",
                        action="store_false",
                        dest="enable_expert_parallel",
                        default=True,
                        help="Disable expert parallel.")
    parser.add_argument(
        "--additional-config",
        type=str,
        default=
        '{"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}',
        help="Additional config as JSON string.")
    parser.add_argument("--limit-mm-per-prompt",
                        type=str,
                        default='{"image": 0, "video": 0, "vision_chunk": 0}',
                        help="Limit MM per prompt as JSON string.")
    parser.add_argument("--no-ignore-eos",
                        action="store_false",
                        dest="ignore_eos",
                        default=True,
                        help="Do not ignore EOS.")
    parser.add_argument("--force-moe-random-routing",
                        action="store_true",
                        default=True,
                        help="Force random routing in MoE.")
    parser.add_argument("--tag",
                        type=str,
                        default="",
                        help="Tag to append to the profiler timestamp directory.")

    args = parser.parse_args()

    # Set environment variables BEFORE importing vLLM or tpu_inference
    os.environ["MOE_REQUANTIZE_BLOCK_SIZE"] = args.moe_requantize_block_size
    os.environ[
        "MOE_REQUANTIZE_WEIGHT_DTYPE"] = args.moe_requantize_weight_dtype
    os.environ["NEW_MODEL_DESIGN"] = args.new_model_design
    os.environ[
        "FORCE_MOE_RANDOM_ROUTING"] = "1" if args.force_moe_random_routing else "0"
    os.environ["MLA_XPOSE_NTILE"] = "96"
    os.environ["KIMI_QUANTIZE_ATTN_TO_FP8"] = "1"

    # Set backend specific environment variable before importing vLLM
    if args.backend == "jax":
        os.environ["MODEL_IMPL_TYPE"] = "flax_nnx"
        print("Using JAX (Flax NNX) implementation from tpu-inference.")
    else:
        os.environ["MODEL_IMPL_TYPE"] = "vllm"
        print("Using vLLM Torch implementation.")

    from tpu_inference.platforms.tpu_platform import TpuPlatform
    import vllm.platforms
    tpu_platform_instance = TpuPlatform()
    vllm.platforms._current_platform = tpu_platform_instance
    vllm.platforms.current_platform = tpu_platform_instance

    # Delayed imports to ensure environment variables are respected by tpu_inference/vLLM
    from vllm import SamplingParams
    from vllm.config.profiler import ProfilerConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.tag:
        timestamp = f"{timestamp}_{args.tag}"

    # Handle GCS vs local path for timestamp directory
    if args.profiler_dir.startswith("gs://"):
        args.profiler_dir = f"{args.profiler_dir.rstrip('/')}/{timestamp}"
    else:
        args.profiler_dir = os.path.join(args.profiler_dir, timestamp)
        os.makedirs(args.profiler_dir, exist_ok=True)

    print(f"Profiler traces and logs will be saved to: {args.profiler_dir}")

    print(f"Initializing LLM Engine with {args.num_hidden_layers} LAYERS...")

    additional_config = json.loads(
        args.additional_config) if args.additional_config else {}

    limit_mm_per_prompt = json.loads(args.limit_mm_per_prompt) if args.limit_mm_per_prompt else {}

    engine_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.input_len + args.output_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # load_format=args.load_format,
        kv_cache_dtype=args.kv_cache_dtype,
        additional_config=additional_config,
        limit_mm_per_prompt=limit_mm_per_prompt,
        hf_overrides={"text_config": {"num_hidden_layers": args.num_hidden_layers}},
        enable_prefix_caching=args.enable_prefix_caching,
        enable_expert_parallel=args.enable_expert_parallel,
        seed=args.seed,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=args.profiler_dir,
        ),
    )

    engine_args = EngineArgs(**engine_kwargs)

    engine = LLMEngine.from_engine_args(engine_args)

    # Create dummy prompts of input_len
    prompt_token_ids = [1] * args.input_len
    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        ignore_eos=args.ignore_eos,
        temperature=0.0,
    )

    # Add requests to process
    for i in range(args.num_prompts):
        engine.add_request(request_id=f"request_{i}",
                           prompt=prompt_token_ids,
                           params=sampling_params)

    step_times = []
    print(f"Starting engine runs. Generating {args.output_len} tokens...")

    prev_output_lens = {}
    profile_started = False
    profile_triggered = False
    profile_steps = 0
    profiled_steps_logs = []

    while engine.has_unfinished_requests():
        start = time.time()

        request_outputs = engine.step()

        # Print sequence lengths (KV lengths and Q lengths) of each request in the batch
        batch_sz = len(request_outputs)
        all_pairs = []
        q_lens = []
        kv_lens = []
        for out in request_outputs:
            req_id = out.request_id
            prompt_len = len(
                out.prompt_token_ids) if out.prompt_token_ids else 0
            output_len = len(
                out.outputs[0].token_ids
            ) if out.outputs and out.outputs[0].token_ids else 0

            if req_id not in prev_output_lens:
                # First time we see this request, it just completed prefill
                q_len = prompt_len
            else:
                # Decode step
                q_len = output_len - prev_output_lens[req_id]

            prev_output_lens[req_id] = output_len
            kv_len = prompt_len + output_len

            all_pairs.append(f"[{kv_len}, {q_len}]")
            q_lens.append(q_len)
            kv_lens.append(kv_len)

        pairs_str = ", ".join(all_pairs)

        if batch_sz > 0:
            print(f"Step {len(step_times)}: b={batch_sz} {{ {pairs_str} }}")

        # Collect logs if profiling is active for this step
        if profile_started:
            log_line = f"Step {len(step_times)}: b={batch_sz} {{ {pairs_str} }}"
            print(f"[PROFILED] {log_line}")
            profiled_steps_logs.append(log_line)

        # Profiler trigger logic
        if args.profiler_trigger_kv_len is not None and not profile_triggered:
            cond1 = batch_sz == args.max_num_seqs * args.tp_size
            cond2 = all(q == 1 for q in q_lens) if q_lens else False
            cond3 = all(kv >= args.profiler_trigger_kv_len
                        for kv in kv_lens) if kv_lens else False

            if cond1 and cond2 and cond3:
                print(
                    f"Trigger condition met at step {len(step_times)}. Starting profile..."
                )
                engine.start_profile()
                profile_started = True
                profile_triggered = True
                profile_steps = 10  # Profile for 10 steps

        if profile_started:
            profile_steps -= 1
            if profile_steps == 0:
                print("Stopping profile...")
                engine.stop_profile()
                profile_started = False

                # Write logs to file
                # Write logs to file using Pandas to handle GCS automatically
                import pandas as pd
                df = pd.DataFrame({"logs": profiled_steps_logs})

                log_filename = "profiled_seq_lengths.log"

                if args.profiler_dir.startswith("gs://"):
                    gcs_dest = f"{args.profiler_dir}/{log_filename}"
                    print(f"Writing logs to {gcs_dest} using Pandas...")
                    try:
                        df.to_csv(gcs_dest, index=False, header=False)
                        print("Write complete.")
                    except Exception as e:
                        print(f"Error writing to GCS with Pandas: {e}")
                        # Fallback to local file
                        local_log_path = f"/tmp/{log_filename}"
                        df.to_csv(local_log_path, index=False, header=False)
                        print(f"Log file saved locally at: {local_log_path}")
                else:
                    # Local path
                    dest_path = os.path.join(args.profiler_dir, log_filename)
                    df.to_csv(dest_path, index=False, header=False)
                    print(f"Saved seq lengths log to {dest_path}")

        end = time.time()
        step_times.append(end - start)

        if len(step_times) % 100 == 0:
            print(
                f"Step {len(step_times)} completed. Current step latency: {(end - start)*1000:.2f} ms"
            )

    print("\n=== Benchmark Results ===")
    print(f"Total steps: {len(step_times)}")
    print(f"Prefill step latency: {step_times[0]*1000:.2f} ms")

    if len(step_times) > 1:
        print(
            f"Last decode step latency (Token {args.output_len}): {step_times[-1]*1000:.2f} ms"
        )

        decode_times = step_times[1:]
        avg_decode = sum(decode_times) / len(decode_times)
        print(f"Average decode step latency: {avg_decode*1000:.2f} ms")
    else:
        print("No decode steps were executed.")


if __name__ == "__main__":
    main()