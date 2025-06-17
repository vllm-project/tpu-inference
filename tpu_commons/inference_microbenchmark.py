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
from utils_jax import calculate_prefill_tflops_per_device
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus

# TODO: change back to  "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "I love to"
DEFAULT_BLOCK_SIZE = 64


def create_parser():
    """
    Create a parser for any CLI arguments to be passed to the EngineCore.
    """
    parser = FlexibleArgumentParser()
    # Add engine args
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


def run_prefill(engine_core: EngineCore, tokenizer, prompt_ids: list[int],
                should_profile: bool):
    """
    Run an isolated prefill pass.

    Args:
        engine_core: EngineCore instance.
        tokenizer: Tokenizer instance.
        prompt_ids: input prompt tokens.
    """
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
    print(
        f"Added client request '{request_id_client}' to scheduler.waiting for prefill."
    )
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    # --- Retrieve necessary data for TFLOPs calculation directly inside the function ---
    # This maintains the function's original signature.
    num_model_params = engine_core.model_executor.driver_worker.worker.model_runner.total_model_params_num
    model_hf_config = engine_core.vllm_config.model_config.hf_config
    # ----------------------------------------------------------------------------------

    if should_profile:
        engine_core.profile(is_start=True)

    start_time = time.perf_counter()  # <--- Start timing
    engine_core_output_step_0 = engine_core.step()
    jax.block_until_ready(
        engine_core.scheduler.running)  # Block until JAX execution completes
    end_time = time.perf_counter()  # <--- Stop timing

    if should_profile:
        engine_core.profile(is_start=False)

    prefill_average_ms = (end_time -
                          start_time) * 1000.0  # Calculate time in ms

    if len(engine_core_output_step_0[0]
           [0].outputs) > 0:  # Accessing EngineCoreOutputs.outputs[0]
        # Access request_id from EngineCoreOutputs directly. This is the ID that was actually processed.
        actual_processed_request_id = engine_core_output_step_0[0][0].outputs[
            0].request_id
        print(
            f"Scheduler processed request with actual ID: {actual_processed_request_id}"
        )
    else:
        actual_processed_request_id = None  # No request processed, or problem

    print(
        f"Scheduler state after pure prefill step: waiting={len(engine_core.scheduler.waiting)}, running={len(engine_core.scheduler.running)}"
    )
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    found_req_in_running = False
    for r in engine_core.scheduler.running:
        if r.request_id == actual_processed_request_id:
            found_req_in_running = True
            assert r.num_computed_tokens >= len(prompt_ids), \
                f"Prefill not fully computed. Computed: {r.num_computed_tokens}, Prompt length: {len(prompt_ids)}"
            print(
                f"Request '{r.request_id}' prefill complete after first step: {r.num_computed_tokens}/{len(prompt_ids)} tokens computed."
            )
            break
    assert found_req_in_running, f"Request with ID {actual_processed_request_id} should be in scheduler.running after prefill step."

    # Should this be 128 because of prefill padding?
    total_prefill_tokens = engine_core.scheduler.running[0].num_computed_tokens

    print("Finished executing prefill.")

    # --- NEW: Calculate and Print TFLOPs metrics ---
    # Pass log=False for the main call to avoid duplicate breakdown print.
    total_tflops_per_device_value, _, _ = calculate_prefill_tflops_per_device(
        num_model_params, total_prefill_tokens, model_hf_config, log=True)

    # Calculate TFLOPs/sec/device
    tflops_per_sec_per_device = total_tflops_per_device_value / (
        prefill_average_ms / 1000.0)

    # --- Print Results in MaxText-like Format ---
    print(
        f"\nPrefill benchmark results for length {total_prefill_tokens}:\n"  # Use actual tokens computed for reporting
        f"\tPrefill step average time: {prefill_average_ms:.3f} ms\n"
        f"\tPrefill total TFLOPs/device: {total_tflops_per_device_value:.3f}\n"
        f"\tPrefill TFLOPs/sec/device: {tflops_per_sec_per_device:.3f}\n\n")


def run_decode(engine_core: EngineCore, tokenizer, prompt_ids: list[int],
               should_profile: bool):
    """
    Run an isolated decode pass.

    Args:
        engine_core: EngineCore instance.
        tokenizer: Tokenizer instance.
        prompt_ids: input prompt tokens.
    """
    print("\n--- Running Decode ---")
    # Generate the client-side request_id for the initial request
    # request_id_client_initial = str(uuid.uuid4())  # Keep this for reference

    # sampling_params_decode_setup = SamplingParams(max_tokens=len(prompt_ids) +
    #                                               5,
    #                                               temperature=0.0,
    #                                               ignore_eos=True)

    # engine_core_request_initial = make_request(prompt_ids,
    #                                            sampling_params_decode_setup, tokenizer)
    # engine_core_request_initial.request_id = request_id_client_initial  # Set our client ID
    # engine_core_request_initial.eos_token_id = tokenizer.eos_token_id
    # req_initial = Request.from_engine_core_request(engine_core_request_initial)
    # engine_core.scheduler.add_request(req_initial)
    # print(
    #     f"Added initial client request '{request_id_client_initial}' for prefill setup for decode."
    # )
    # TODO (jacobplatin): update this for future decoupled decodde/prefill support
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # engine_core_output_prefill_setup = engine_core.step()

    # # Extract the actual request ID processed in the prefill setup step
    # if len(engine_core_output_prefill_setup[0][0].outputs) > 0:
    #     actual_processed_request_id_prefill_setup = engine_core_output_prefill_setup[
    #         0][0].outputs[0].request_id
    #     print(
    #         f"Scheduler processed initial request with actual ID: {actual_processed_request_id_prefill_setup}"
    #     )
    # else:
    #     actual_processed_request_id_prefill_setup = None  # Problem

    # print(
    #     f"Scheduler state after prefill setup: waiting={len(engine_core.scheduler.waiting)}, running={len(engine_core.scheduler.running)}"
    # )
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # TODO (jacobplatin): this is the request that was processed in the prefill setup step
    # but ideally, we want to create a new request from scratch that solely does decode and
    # thus isn't dependent on the prefill setup
    actual_processed_request_id_prefill_setup = engine_core.scheduler.running[
        0].request_id

    found_req_in_running = False
    # Use the actual ID from the *output* for subsequent checks on scheduler.running
    for r in engine_core.scheduler.running:
        if r.request_id == actual_processed_request_id_prefill_setup:
            found_req_in_running = True
            assert r.num_computed_tokens >= len(prompt_ids), \
                f"Prefill not fully computed for decode setup. Computed: {r.num_computed_tokens}, Prompt length: {len(prompt_ids)}"
            print(
                f"Request '{r.request_id}' prefill complete for decode setup.")
            break
    assert found_req_in_running, f"Request with ID {actual_processed_request_id_prefill_setup} should be in scheduler.running after prefill setup step."  # <--- CHANGED: Updated error message

    if should_profile:
        engine_core.profile(is_start=True)
    engine_core_output_decode = engine_core.step()
    if should_profile:
        engine_core.profile(is_start=False)
    print(
        f"EngineCoreOutputs (Pure Decode Step):\n{engine_core_output_decode}")

    if actual_processed_request_id_prefill_setup:
        engine_core.scheduler.finish_requests(
            {actual_processed_request_id_prefill_setup},
            RequestStatus.FINISHED_LENGTH_CAPPED)
    print("Finished executing decode.")


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
    run_prefill(engine_core, tokenizer, prompt_ids, should_profile)
    run_decode(engine_core, tokenizer, prompt_ids, should_profile)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
