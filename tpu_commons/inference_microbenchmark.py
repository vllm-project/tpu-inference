# SPDX-License-Identifier: Apache-2.0
"""
Runs a pure prefill and decode pass.

Exaple cmd: TPU_BACKEND_TYPE=jax python inference_microbenchmark.py

Options:
    --prompt: Input prompt (optional)
    --profile: Enable profiling (if enabled, must specify --profile-dir)
    --profile_dir: Directory to save profiling data
"""
import os
import time
import uuid

import jax
from transformers import AutoTokenizer
from utils_jax import (calculate_prefill_tflops_per_device,
                       get_kv_cache_size_bytes, get_model_size_bytes)
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request

# TODO: change back to  "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "I love to"
DEFAULT_BLOCK_SIZE = 64
WARMUP_ITERS = 2
BENCHMARK_ITERS = 10


def create_parser():
    """
    Create a parser for any CLI arguments to be passed to the EngineCore.
    """
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model=MODEL_NAME)
    parser.set_defaults(task="generate")
    parser.set_defaults(tensor_parallel_size=8)
    parser.set_defaults(max_num_seqs=1)
    parser.set_defaults(
        max_model_len=1024)  # TODO (jacobplatin): probably want to update this
    parser.set_defaults(max_num_batched_tokens=8192
                        )  # TODO (jacobplatin): probably want to update this

    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=str)

    return parser


def make_request(prompt_token_ids: list[int], sampling_params: SamplingParams,
                 tokenizer: AutoTokenizer) -> EngineCoreRequest:
    """
    Create an EngineCoreRequest for the given prompt and sampling parameters, which will
    be passed to the EngineCore.

    Args:
        prompt_token_ids: List of token IDs for the prompt.
        sampling_params: Sampling parameters for the request.

    Returns:
        EngineCoreRequest
    """
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=sampling_params,
        eos_token_id=tokenizer.eos_token_id,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def run_prefill(engine_core: EngineCore,
                tokenizer,
                prompt_ids: list[int],
                should_profile: bool,
                verbose: bool = True):
    """
    Run an isolated prefill pass.

    Args:
        engine_core: EngineCore instance.
        tokenizer: Tokenizer instance.
        prompt_ids: input prompt tokens.
        verbose: Whether to print verbose output.
    """
    if verbose:
        print("\n--- Running Prefill ---")

    request_id_client = str(uuid.uuid4())

    sampling_params_prefill = SamplingParams(max_tokens=len(prompt_ids) + 5,
                                             temperature=0.0,
                                             ignore_eos=True)

    engine_core_request = make_request(prompt_ids, sampling_params_prefill,
                                       tokenizer)
    engine_core_request.request_id = request_id_client
    engine_core_request.eos_token_id = tokenizer.eos_token_id

    # Convert from an EngineCore request to vllm.v1.request.Request (used in the v1 scheduler)
    # NOTE: can we just construct the Request directly?
    req = Request.from_engine_core_request(engine_core_request)

    engine_core.scheduler.add_request(req)

    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    num_model_params = engine_core.model_executor.driver_worker.worker.model_runner.total_model_params_num
    vllm_model_config = engine_core.vllm_config.model_config

    if should_profile:
        engine_core.profile(is_start=True)

    start_time = time.perf_counter()
    engine_core_output_step_0 = engine_core.step()
    end_time = time.perf_counter()

    if should_profile:
        engine_core.profile(is_start=False)

    prefill_average_ms = (end_time -
                          start_time) * 1000.0  # Calculate time in ms

    if len(engine_core_output_step_0[0][0].outputs) > 0:
        actual_processed_request_id = engine_core_output_step_0[0][0].outputs[
            0].request_id
    else:
        actual_processed_request_id = None  # No request processed, or problem

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    found_req_in_running = False
    for r in engine_core.scheduler.running:
        if r.request_id == actual_processed_request_id:
            found_req_in_running = True
            assert r.num_computed_tokens >= len(prompt_ids), \
                f"Prefill not fully computed. Computed: {r.num_computed_tokens}, Prompt length: {len(prompt_ids)}"
            break
    assert found_req_in_running, f"Request with ID {actual_processed_request_id} should be in scheduler.running after prefill step."

    # Should this be 128 because of prefill padding?
    total_prefill_tokens = engine_core.scheduler.running[0].num_computed_tokens

    if verbose:
        print("--- Prefill complete ---")

    total_tflops_per_device_value, _, _ = calculate_prefill_tflops_per_device(
        num_model_params, total_prefill_tokens, vllm_model_config, log=verbose)

    tflops_per_sec_per_device = total_tflops_per_device_value / (
        prefill_average_ms / 1000.0)

    if verbose:
        print(
            f"\nPrefill benchmark results for length {total_prefill_tokens}:\n"
            f"\tPrefill step average time: {prefill_average_ms:.3f} ms\n"
            f"\tPrefill total TFLOPs/device: {total_tflops_per_device_value:.3f}\n"
            f"\tPrefill TFLOPs/sec/device: {tflops_per_sec_per_device:.3f}\n\n"
        )


def run_decode(engine_core: EngineCore,
               tokenizer,
               prompt_ids: list[int],
               should_profile: bool,
               verbose: bool = True):
    """
    Run an isolated decode pass and measure its performance for a single execution.

    Args:
        engine_core: EngineCore instance.
        tokenizer: Tokenizer instance.
        prompt_ids: input prompt tokens.
        should_profile: Whether to enable profiling.
        verbose: Whether to print verbose output.
    """
    if verbose:
        print("\n--- Running Decode Benchmark (Single Execution) ---")

    assert len(engine_core.scheduler.waiting) == 0 and len(
        engine_core.scheduler.running) == 0

    request_id_decode_setup = str(uuid.uuid4())
    # Set max_tokens high enough for prompt + 1 decode token + buffer (since only 1 decode step)
    sampling_params_decode_setup = SamplingParams(max_tokens=len(prompt_ids) +
                                                  5,
                                                  temperature=0.0,
                                                  ignore_eos=True)

    engine_core_request_initial = make_request(prompt_ids,
                                               sampling_params_decode_setup,
                                               tokenizer)
    engine_core_request_initial.request_id = request_id_decode_setup
    engine_core_request_initial.eos_token_id = tokenizer.eos_token_id
    req_initial = Request.from_engine_core_request(engine_core_request_initial)

    engine_core.scheduler.add_request(req_initial)

    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    _ = engine_core.step()
    jax.block_until_ready(engine_core.scheduler.running)

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # Verify prefill is complete for this running request
    found_req_prefilled = False
    for r in engine_core.scheduler.running:
        if r.request_id == request_id_decode_setup:
            found_req_prefilled = True
            assert r.num_computed_tokens >= len(prompt_ids), \
                f"Prefill not fully computed for decode setup. Computed: {r.num_computed_tokens}, Prompt length: {len(prompt_ids)}"
            if verbose:
                print(
                    f"Request '{r.request_id}' prefill complete for decode benchmark setup."
                )
            break
    assert found_req_prefilled, f"Request with ID {request_id_decode_setup} should be in scheduler.running after prefill setup step."

    num_model_params = engine_core.model_executor.driver_worker.worker.model_runner.total_model_params_num
    vllm_model_config = engine_core.vllm_config.model_config

    model_size_bytes = get_model_size_bytes(num_model_params,
                                            vllm_model_config)
    kv_cache_size_bytes = get_kv_cache_size_bytes(
        engine_core.vllm_config.cache_config, vllm_model_config)

    if verbose:
        print("Executing single decode step...")

    if should_profile:
        engine_core.profile(is_start=True)

    start_time = time.perf_counter()
    engine_core_output_decode = engine_core.step()
    jax.block_until_ready(engine_core.scheduler.running)
    end_time = time.perf_counter()

    if should_profile:
        engine_core.profile(is_start=False)

    decode_average_ms = (end_time - start_time) * 1000.0

    tokens_this_step = 0
    if len(engine_core_output_decode[0][0].outputs) > 0:
        tokens_this_step = len(
            engine_core_output_decode[0][0].outputs[0].new_token_ids)
    total_tokens_decoded = tokens_this_step
    assert total_tokens_decoded == 1, "Should only have decoded 1 token per decode step"
    tokens_per_sec = total_tokens_decoded / (
        decode_average_ms / 1000.0) if decode_average_ms > 0 else 0

    # AR global batch size is the number of sequences concurrently decoding. For our test, it's 1.
    ar_global_batch_size = len(engine_core.scheduler.running) if len(
        engine_core.scheduler.running) > 0 else 1
    ar_average_ms_per_seq = decode_average_ms / ar_global_batch_size if ar_global_batch_size > 0 else 0

    # Calculate memory bandwidth
    seconds_per_step_actual = (decode_average_ms / 1000.0)
    GB_per_step_per_device = (model_size_bytes +
                              kv_cache_size_bytes) / 1e9 / jax.device_count()
    bw_per_device = GB_per_step_per_device / seconds_per_step_actual if seconds_per_step_actual > 0 else 0

    if verbose:
        print(
            f"\nDecode benchmark results:\n"
            f"\tAR step average time: {decode_average_ms:.3f} ms\n"
            f"\tAR step average time per seq: {ar_average_ms_per_seq:.3f} ms\n"
            f"\tAR global batch size: {ar_global_batch_size}\n"
            f"\tAR throughput: {tokens_per_sec:.3f} tokens/second\n"
            f"\tAR memory bandwidth per device: {bw_per_device:.3f} GB/s\n\n")


def clear_scheduler_state(engine_core: EngineCore):
    """
    Forcefully clears the active state of the scheduler for benchmarking purposes.

    Args:
        engine_core: EngineCore instance.
    """
    if len(engine_core.scheduler.waiting) > 0:
        engine_core.scheduler.waiting.clear()

    if len(engine_core.scheduler.running) > 0:
        engine_core.scheduler.running.clear()

    if hasattr(engine_core.scheduler, 'swapped') and len(
            engine_core.scheduler.swapped) > 0:
        engine_core.scheduler.swapped.clear()

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0


def main(args):
    """
    Main entry point for the script.

    Args:
        args: Dictionary of arguments.
    """
    should_profile = args.pop("profile", False)
    profile_dir = args.pop("profile_dir", None)
    prompt = args.pop("prompt", PROMPT)
    block_size = args.pop("block_size") or DEFAULT_BLOCK_SIZE

    if should_profile:
        assert profile_dir is not None, "Must specify profile_dir if profiling is enabled!"
        # NOTE: this must be set before the EngineCore is created
        # or else it won't be respected
        os.environ["VLLM_TORCH_PROFILER_DIR"] = profile_dir

    engine_args = EngineArgs(**args)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer(prompt).input_ids

    # TODO (jacobplatin): understand why this isn't being respected from the command line
    vllm_config.cache_config.block_size = block_size

    engine_core = EngineCore(vllm_config=vllm_config,
                             executor_class=executor_class,
                             log_stats=True)
    num_model_params = engine_core.model_executor.driver_worker.worker.model_runner.total_model_params_num
    print(f"Num model params: {num_model_params}")

    print("\n\n--- Starting Warmup ---")
    for _ in range(WARMUP_ITERS):
        # TODO: is this necessary?
        clear_scheduler_state(engine_core)
        run_prefill(engine_core, tokenizer, prompt_ids, False, verbose=False)
        clear_scheduler_state(engine_core)
        run_decode(engine_core, tokenizer, prompt_ids, False, verbose=False)

    print("--- Finished Warmup ---")

    clear_scheduler_state(engine_core)
    run_prefill(engine_core, tokenizer, prompt_ids, should_profile)

    clear_scheduler_state(engine_core)
    run_decode(engine_core, tokenizer, prompt_ids, should_profile)

    clear_scheduler_state(engine_core)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
