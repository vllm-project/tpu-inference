# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import vllm.envs as envs
from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

from tpu_inference.core import disagg_utils


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
    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        """<|begin_of_text|><|header_start|>system<|header_end|>You are an expert in abstract algebra who answers multiple-choice questions.<|eot|><|header_start|>user<|header_end|>The following are multiple choice questions (with answers) about abstract_algebra.

    Q: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
    A: 0
    B: 1
    C: 2
    D: 3
    Answer: B

    Q: "Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G."
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: B

    Q: Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: C

    Q: Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: A

    Q: Find the characteristic of the ring 2Z.
    A: 0
    B: 3
    C: 12
    D: 30
    Answer: A

    Q: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
    A: 0
    B: 4
    C: 2
    D: 6
    Answer:<|eot|><|header_start|>assistant<|header_end|>""",

        """<|begin_of_text|><|header_start|>system<|header_end|>You are an expert in abstract algebra who answers multiple-choice questions.<|eot|><|header_start|>user<|header_end|>The following are multiple choice questions (with answers) about abstract_algebra.

    Q: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
    A: 0
    B: 1
    C: 2
    D: 3
    Answer: B

    Q: "Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G."
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: B

    Q: Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: C

    Q: Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: A

    Q: Find the characteristic of the ring 2Z.
    A: 0
    B: 3
    C: 12
    D: 30
    Answer: A

    Q: Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.
    A: 8
    B: 2
    C: 24
    D: 120
    Answer:<|eot|><|header_start|>assistant<|header_end|>""",

        """<|begin_of_text|><|header_start|>system<|header_end|>You are an expert in abstract algebra who answers multiple-choice questions.<|eot|><|header_start|>user<|header_end|>The following are multiple choice questions (with answers) about abstract_algebra.

    Q: Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
    A: 0
    B: 1
    C: 2
    D: 3
    Answer: B

    Q: "Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G."
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: B

    Q: Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: C

    Q: Statement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.
    A: True, True
    B: False, False
    C: True, False
    D: False, True
    Answer: A

    Q: Find the characteristic of the ring 2Z.
    A: 0
    B: 3
    C: 12
    D: 30
    Answer: A

    Q: Find all zeros in the indicated finite field of the given polynomial with coefficients in that field. x^5 + 3x^3 + x^2 + 2x in Z_5
    A: 0
    B: 1
    C: 0,1
    D: 0,4
    Answer:<|eot|><|header_start|>assistant<|header_end|>"""
    ]

    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    if envs.VLLM_TORCH_PROFILER_DIR is not None:
        llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    # Skip long warmup for local simple test.
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'

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
