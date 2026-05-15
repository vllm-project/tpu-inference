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
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate decode-only JSONL dataset for vLLM benchmark")
    parser.add_argument("--input-len",
                        required=True,
                        type=int,
                        help="Exact number of input tokens")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5000,
        help=
        "Total number of prompts to generate. Must be > benchmark requests to avoid vLLM oversampling.",
    )
    parser.add_argument(
        "--num-distinct",
        required=True,
        type=int,
        help=
        "Suggested to be equal to --max-num-seqs to maximize the randomness while ensuring all requests can be safely cached without eviction.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/tmp/decode_only_dataset.jsonl",
        help="Path to save the JSONL",
    )
    args = parser.parse_args()

    print(
        f"[Data Prep] Generating {args.num_distinct} distinct random prompts (Len: {args.input_len})..."
    )

    # Generate `num_distinct` completely different random token arrays to force realistic hardware utilization
    # We avoid 0-100 because they are often reserved for special tokens (BOS, EOS, PAD).
    random_matrix = np.random.randint(low=100,
                                      high=50000,
                                      size=(args.num_distinct, args.input_len))
    distinct_prompts = [row.tolist() for row in random_matrix]

    # vLLM 'custom' dataset expects JSONL format and natively supports "prompt_token_ids".
    # We cycle through the distinct prompts sequentially (e.g., A, B, C, D, A, B, C, D...).
    with open(args.output_file, "w") as f:
        for i in range(args.num_prompts):
            prompt_ids = distinct_prompts[i % args.num_distinct]
            json_record = {"prompt_token_ids": prompt_ids}
            f.write(json.dumps(json_record) + "\n")

    print(
        "[Data Prep] Successfully generated Custom JSONL dataset using raw Token IDs."
    )
    print(
        f"[Data Prep] Total Lines: {args.num_prompts} | Distinct Request Types: {args.num_distinct}"
    )
    print(f"[Data Prep] Saved to {args.output_file}")


if __name__ == "__main__":
    main()
