# Running cmd: TPU_BACKEND_TYPE=jax python inference_microbenchmark.py --profile
import os
import time
import uuid

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "Jalen Brunson is a good basketball player"


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model=MODEL_NAME)
    parser.set_defaults(task="generate")
    parser.set_defaults(tensor_parallel_size=8)
    parser.set_defaults(max_num_seqs=1)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--prompt", type=str, default=PROMPT)
    sampling_group.add_argument("--profile", action="store_true")
    sampling_group.add_argument("--profile-dir", type=str)

    return parser


def make_request(
    prompt_token_ids: list[int], sampling_params: SamplingParams
) -> EngineCoreRequest:  # <--- MODIFIED: Added sampling_params arg
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=
        sampling_params,  # <--- MODIFIED: Use passed sampling_params
        eos_token_id=
        None,  # Will be set to tokenizer.eos_token_id in scenario functions
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


# --- NEW HELPER FUNCTIONS FOR SCENARIOS ---


def run_pure_prefill_scenario(engine_core: EngineCore, tokenizer,
                              prompt_ids: list[int], initial_prompt: str):
    print("\n--- Running Pure Prefill Scenario ---")
    # Generate the client-side request_id, but be aware vLLM might internally use another
    request_id_client = str(uuid.uuid4())  # Keep this for reference if needed

    sampling_params_prefill = SamplingParams(max_tokens=len(prompt_ids) + 5,
                                             temperature=0.0)

    engine_core_request = make_request(prompt_ids, sampling_params_prefill)
    engine_core_request.request_id = request_id_client  # Ensure our client ID is set
    engine_core_request.eos_token_id = tokenizer.eos_token_id
    req = Request.from_engine_core_request(engine_core_request)

    engine_core.scheduler.add_request(req)
    print(
        f"Added client request '{request_id_client}' to scheduler.waiting for prefill."
    )
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    print("Executing first step (expected pure prefill)...")
    engine_core_output_step_0 = engine_core.step()

    # Extract the actual request ID used by the scheduler output
    # This ID comes from the NewRequestData that was just scheduled
    # Check if scheduled_new_reqs is not empty before accessing
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
        f"EngineCoreOutputs (Pure Prefill Step):\n{engine_core_output_step_0}")

    print(
        f"Scheduler state after pure prefill step: waiting={len(engine_core.scheduler.waiting)}, running={len(engine_core.scheduler.running)}"
    )
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    found_req_in_running = False
    # Use the actual ID from the *output* for subsequent checks on scheduler.running
    for r in engine_core.scheduler.running:
        if r.request_id == actual_processed_request_id:  # <--- CHANGED: Use actual_processed_request_id
            found_req_in_running = True
            assert r.num_computed_tokens >= len(prompt_ids), \
                f"Prefill not fully computed. Computed: {r.num_computed_tokens}, Prompt length: {len(prompt_ids)}"
            print(
                f"Request '{r.request_id}' prefill complete after first step: {r.num_computed_tokens}/{len(prompt_ids)} tokens computed."
            )
            break
    assert found_req_in_running, f"Request with ID {actual_processed_request_id} should be in scheduler.running after prefill step."  # <--- CHANGED: Updated error message
    print("Pure Prefill Scenario executed.")


def run_pure_decode_scenario(engine_core: EngineCore, tokenizer,
                             prompt_ids: list[int], initial_prompt: str):
    print("\n--- Running Pure Decode Scenario ---")
    # Generate the client-side request_id for the initial request
    request_id_client_initial = str(uuid.uuid4())  # Keep this for reference

    sampling_params_decode_setup = SamplingParams(max_tokens=len(prompt_ids) +
                                                  5,
                                                  temperature=0.0)

    engine_core_request_initial = make_request(prompt_ids,
                                               sampling_params_decode_setup)
    engine_core_request_initial.request_id = request_id_client_initial  # Set our client ID
    engine_core_request_initial.eos_token_id = tokenizer.eos_token_id
    req_initial = Request.from_engine_core_request(engine_core_request_initial)
    engine_core.scheduler.add_request(req_initial)
    print(
        f"Added initial client request '{request_id_client_initial}' for prefill setup for decode."
    )
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    print("Executing first step (completing prefill for decode setup)...")
    engine_core_output_prefill_setup = engine_core.step()

    # Extract the actual request ID processed in the prefill setup step
    if len(engine_core_output_prefill_setup[0][0].outputs) > 0:
        actual_processed_request_id_prefill_setup = engine_core_output_prefill_setup[
            0][0].outputs[0].request_id
        print(
            f"Scheduler processed initial request with actual ID: {actual_processed_request_id_prefill_setup}"
        )
    else:
        actual_processed_request_id_prefill_setup = None  # Problem

    print(
        f"EngineCoreOutputs (Prefill Setup Step):\n{engine_core_output_prefill_setup}"
    )

    print(
        f"Scheduler state after prefill setup: waiting={len(engine_core.scheduler.waiting)}, running={len(engine_core.scheduler.running)}"
    )
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    found_req_in_running = False
    # Use the actual ID from the *output* for subsequent checks on scheduler.running
    for r in engine_core.scheduler.running:
        if r.request_id == actual_processed_request_id_prefill_setup:  # <--- CHANGED: Use actual_processed_request_id_prefill_setup
            found_req_in_running = True
            assert r.num_computed_tokens >= len(prompt_ids), \
                f"Prefill not fully computed for decode setup. Computed: {r.num_computed_tokens}, Prompt length: {len(prompt_ids)}"
            print(
                f"Request '{r.request_id}' prefill complete for decode setup.")
            break
    assert found_req_in_running, f"Request with ID {actual_processed_request_id_prefill_setup} should be in scheduler.running after prefill setup step."  # <--- CHANGED: Updated error message

    print("Executing second step (expected pure decode)...")
    engine_core_output_decode = engine_core.step()
    print(
        f"EngineCoreOutputs (Pure Decode Step):\n{engine_core_output_decode}")

    # No direct asserts on scheduled_new_reqs/cached_reqs from EngineCoreOutputs.
    # We rely on the debug print from vllm/v1/engine/core.py and the state of scheduler.running

    print("Pure Decode Scenario executed.")

    # Clean up: manually mark the request as finished so the engine_core is clean for next runs/tests
    # The ID to mark finished is the actual ID processed by the scheduler
    if actual_processed_request_id_prefill_setup:
        engine_core.scheduler.finish_requests(
            {actual_processed_request_id_prefill_setup},
            RequestStatus.FINISHED_LENGTH_CAPPED)


def main(args):
    should_profile = args.pop("profile", False)
    profile_dir = args.pop("profile_dir", None)
    prompt = args.pop("prompt", PROMPT)
    engine_args = EngineArgs(**args)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompt_ids = tokenizer(prompt).input_ids

    if should_profile:
        assert profile_dir is not None, "Must specify profile_dir if profiling is enabled!"
        profile_dir = "test_profiles"
        os.environ["VLLM_TORCH_PROFILER_DIR"] = profile_dir

    # TODO (jacobplatin): understand why this isn't being respected
    vllm_config.cache_config.block_size = 64

    # --- Run Pure Prefill Scenario ---
    # Create a new EngineCore instance for the pure prefill test to ensure a clean state
    engine_core_prefill_test = EngineCore(vllm_config=vllm_config,
                                          executor_class=executor_class,
                                          log_stats=True)

    if should_profile:  # Apply profiling to this engine if enabled
        engine_core_prefill_test.profile(is_start=True)

    run_pure_prefill_scenario(engine_core_prefill_test, tokenizer, prompt_ids,
                              prompt)

    if should_profile:  # Stop profiling for this engine if enabled
        engine_core_prefill_test.profile(is_start=False)
        # Reset profile for the next engine if needed
        os.environ["VLLM_TORCH_PROFILER_DIR"] = ""  # Clear for next test

    # --- Run Pure Decode Scenario ---
    # Create another new EngineCore instance for the pure decode test
    engine_core_decode_test = EngineCore(vllm_config=vllm_config,
                                         executor_class=executor_class,
                                         log_stats=True)

    if should_profile:  # Apply profiling to this engine if enabled
        engine_core_decode_test.profile(is_start=True)

    run_pure_decode_scenario(engine_core_decode_test, tokenizer, prompt_ids,
                             prompt)

    if should_profile:  # Stop profiling for this engine if enabled
        engine_core_decode_test.profile(is_start=False)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
