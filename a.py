# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from tpu_inference.core import disagg_utils
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.set_defaults(max_model_len=1024)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    # chat params
    chat_group = parser.add_argument_group("Chat parameters")
    chat_group.add_argument("--use-chat-template", action="store_true")
    # NOTE: a few models (like Qwen3.5) can use this to disable thinking,
    # e.g. --chat-template-kwargs='{"enable_thinking": false}'
    chat_group.add_argument('--chat-template-kwargs',
                            type=json.loads,
                            default={})

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens", 2048)
    temperature = args.pop("temperature", 0.0)
    top_p = args.pop("top_p", None)
    top_k = args.pop("top_k", None)
    use_chat_template = args.pop("use_chat_template", False)
    chat_template_kwargs = args.pop('chat_template_kwargs', {})

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object exactly matching the JSON
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = 2048  # From arg_1
    sampling_params.temperature = 0.0  # From arg_1
    sampling_params.stop = ["Question:"]  # From arg_1 'until'

    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # The exact pre-rendered prompt string extracted from "arg_0"
    # This includes the ChatML tags and the fatal <think> tags at the end.
    exact_lm_eval_prompt = """<|im_start|>system\nThe following are multiple choice questions (with answers) about psychology. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.<|im_end|>\n<|im_start|>user\nQuestion:\nPascale is interested in the processing strategies children use to learn new information. Pascale would best be classified as what type of psychologist?\nOptions:\nA. social\nB. school\nC. sociocultural\nD. forensic\nE. behaviorist\nF. health\nG. clinical\nH. cognitive\nI. psychoanalytic\nJ. developmental\nAnswer: Let's think step by step. We refer to Wikipedia articles on psychology for help. Sociocultural psychologist focuses on the effect of societal factors on people. Clinical psychologist focuses on people with mental issues. Cognitive psychologist focuses on how people think and learn, including the processing strategies. Behaviorist focuses more on the environment and experience effect on people. The answer is (H).\n\nQuestion:\nAccording to Caplan's model of consultee-centered case consultation, the consultant is primarily interested in\nOptions:\nA. identifying the causes and solutions of the client's presenting problems\nB. establishing a hierarchy of authority to enable effective decision making\nC. ensuring the consultee adheres strictly to a predetermined action plan\nD. proposing multiple alternative solutions for the consultee to choose from\nE. identifying the strengths and weaknesses of the consultee's current approach\nF. presenting a single, well-defined and unambiguous course of action for the consultant to overcome skills deficits\nG. developing a comprehensive treatment plan for the client\nH. identifying and eliminating the causes of the consultee's difficulties in handling a problem\nI. focusing on the consultant's personal growth and development\nJ. focusing on the relationship between the client and the consultee\nAnswer: Let's think step by step. We refer to Wikipedia articles on psychology for help. Caplan defines two type of consultation. Client-centered case consultation aims to handle client's problems, while consultee-centered case consultation aims to identify the reason of client's difficulty to solve problems. The answer is (H).\n\nQuestion:\nAccording to the Individuals with Disabilities Education Improvement Act, which of the following must an educational agency do before it changes the educational placement of a student with a disability?\nOptions:\nA. Notify the parents in writing\nB. Obtain the child's consent\nC. Obtain a court order\nD. Conduct a new evaluation of the child's disability\nE. Discuss with the child's psychologist\nF. Give the child a trial period in the new environment\nG. Obtain parental consent\nH. Notify the local education authority\nI. Arrange a meeting with all teachers and administrators\nJ. Obtain school board approval\nAnswer: Let's think step by step. We refer to Wikipedia articles on psychology for help. When the decision to change the educational placement of a student with a disability is made, the educational agency must notify the parents in writing on that date. The answer is (A).\n\nQuestion:\nAni believes that her attitudes and behavior play a central role in what happens to her. Such a belief is likely to be associated with\nOptions:\nA. low self-esteem.\nB. a strong id.\nC. a high level of anxiety.\nD. a strong superego.\nE. high levels of self-consciousness.\nF. an external locus of control.\nG. an inferiority complex.\nH. a low level of self-awareness.\nI. low self-efficacy.\nJ. an internal locus of control.\nAnswer: Let's think step by step. We refer to Wikipedia articles on psychology for help. People with an external locus of control believes fate and luck play an important role in their lives, while people with an internal locus of control believes they control their lives. The answer is (J).\n\nQuestion:\nIn terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:\nOptions:\nA. individualism and long-term orientation.\nB. individualism and power distance.\nC. uncertainty avoidance.\nD. long-term orientation.\nE. individualism.\nF. individualism and masculinity.\nG. long-term orientation and uncertainty avoidance.\nH. power distance.\nI. power distance and masculinity.\nJ. N/A\nAnswer: Let's think step by step. We refer to Wikipedia articles on psychology for help. US scores highest on individualism among the five cultural dimensions. The answer is (E).\n\nQuestion:\nA person who received a Level 4 rating on the Rancho Los Amigos Scale of Cognitive Functioning:\nOptions:\nA. is unresponsive to all stimuli but shows signs of basic physiological functioning such as breathing and heartbeat.\nB. is highly responsive and coherent, but cannot remember or integrate past and recent events.\nC. is functioning at an intellectual level that is average for his/her age, education, and demographic background.\nD. is responsive to stimuli but cannot remember or integrate past and recent events, and exhibits impaired judgment.\nE. is confused and incoherent, may exhibit bizarre behavior, and is unable to care for him/herself.\nF. is functioning at an intellectual level that is superior for his/her age, education, and demographic background.\nG. is nonresponsive to visual or auditory stimuli and seems to be in a state of deep sleep.\nH. is alert and oriented and can remember and integrate remote and recent events but may have some impairment in judgment, planning, and abstract reasoning.\nAnswer: Let's think step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"""

    prompts = [exact_lm_eval_prompt]

    profiler_config = llm.llm_engine.vllm_config.profiler_config
    if profiler_config.profiler == "torch":
        llm.start_profile()

    # We MUST use generate() here. If we use chat(), vLLM will wrap the text again.
    logger.info(
        "Using LLM generate API for inference to simulate lm_eval behavior")
    outputs = llm.generate(prompts, sampling_params)

    if profiler_config.profiler == "torch":
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text:\n{generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    # Skip long warmup for local simple test.
    os.environ.setdefault('SKIP_JAX_PRECOMPILE', '1')

    parser = create_parser()
    args: dict = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(args)
    else:
        from unittest.mock import patch

        from tpu_inference.core.core_tpu import (DisaggEngineCore,
                                                 DisaggEngineCoreProc)

        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), patch(
                "vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
            main(args)
