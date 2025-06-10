# Running cmd: TPU_BACKEND_TYPE=jax  python inference_microbenchmark.py --profile
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
from vllm.v1.request import Request

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


def make_request(prompt_token_ids: list[int]) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=prompt_token_ids,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(max_tokens=1),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def main(args):
    # engine_args = EngineArgs(model=MODEL_NAME)
    # engine_args.from_cli_args(args)
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

    engine_core = EngineCore(vllm_config=vllm_config,
                             executor_class=executor_class,
                             log_stats=True)

    if should_profile:
        engine_core.profile(is_start=True)

    engine_core_request = make_request(prompt_ids)
    req = Request.from_engine_core_request(engine_core_request)

    # engine_core.add_request(engine_core_request)
    # Basically, all `add_request` does is create a SchedulerRequest and then add it to the scheduler,
    # so we'll do that here
    # This will give us more fine-tuned control over the scheduler, which controls prefill / decode
    engine_core.scheduler.add_request(req)
    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    engine_core_output_step_0 = engine_core.step()
    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 0

    print(engine_core_output_step_0)

    if should_profile:
        engine_core.profile(is_start=False)


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
