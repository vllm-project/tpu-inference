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

import argparse
import os
import sys

# Ensure we can import other scripts in this directory
sys.path.append(os.path.dirname(__file__))

from benchmark_core import RequestFuncOutput
from benchmark_dataset import MMMUProDataset
from benchmark_utils import eval_accuracy_mmmu_pro
from transformers import AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--mmmu-pro-subset",
                        type=str,
                        default="standard (10 options)")
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    # 1. Initialize the LLM engine
    # We pass the relevant parsed arguments to EngineArgs
    # We filter out the custom args we added above
    engine_args_dict = vars(args).copy()
    custom_args = [
        "num_prompts", "mmmu_pro_subset", "output_len", "temperature"
    ]
    for arg in custom_args:
        engine_args_dict.pop(arg, None)

    engine_args = EngineArgs(**engine_args_dict)
    print("Initializing LLM engine...")
    llm = LLM(**vars(engine_args))

    # 2. Load dataset
    print(f"Loading MMMU-Pro dataset subset: {args.mmmu_pro_subset}...")
    dataset = MMMUProDataset(subset=args.mmmu_pro_subset)

    try:
        # Pass the tokenizer from the LLM instance to satisfy the sample signature
        tokenizer = llm.get_tokenizer()
    except AttributeError:
        # Fallback if tokenizer isn't exposed directly
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        output_len=args.output_len,
    )

    # 3. Prepare inputs and perform generation
    messages = []
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.output_len,
    )

    for req in requests:
        # The mmmu dataset returns dicts in req.messages for chat requests:
        # [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]}]
        msg = req.messages
        messages.append(msg)

    print(f"Generating answers for {len(messages)} prompts...")

    # We use llm.chat since MMMU-Pro is a chat-based multimodal dataset
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)

    # 4. Evaluate the outputs
    request_outputs = []
    for request, output in zip(requests, outputs):
        generated_text = output.outputs[0].text

        # Package the result to match what eval_accuracy_mmmu_pro expects
        result = RequestFuncOutput(
            generated_text=generated_text,
            success=True,
            latency=0.0,  # Not measuring latency here
            ttft=0.0,
            output_tokens=len(output.outputs[0].token_ids),
            itl=[],
            tpot=0.0,
            error="",
            input_request=request)
        request_outputs.append(result)

    print("\nEvaluating Accuracy...")
    metrics = eval_accuracy_mmmu_pro(request_outputs)

    print("\n============ Offline Accuracy Benchmark Result ============")
    for key, value in metrics.items():
        print(f"{key:<40} {value}")


if __name__ == '__main__':
    main()
