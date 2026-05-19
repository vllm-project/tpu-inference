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
from transformers import AutoTokenizer

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
        help="Suggested to be equal to --max-num-seqs to avoid eviction."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/tmp/decode_only_dataset.jsonl",
        help="Path to save the JSONL"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="HuggingFace model ID or path for Tokenizer"
    )
    parser.add_argument(
        "--shrink-ratio", 
        type=float, 
        default=0.95, 
        help="Ratio to shrink initial random array to prevent BPE fragmentation overflow"
    )
    args = parser.parse_args()

    print(f"[Data Prep] Loading Tokenizer for {args.model}...")
    # Load the tokenizer to perform the decoding
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Calculate the safe initial length to account for BPE expansion
    safe_initial_len = int(args.input_len * args.shrink_ratio)
    print(f"[Data Prep] Target Limit: {args.input_len} tokens. Generating arrays of {safe_initial_len} (Ratio: {args.shrink_ratio})...")

    # Generate `num_distinct` completely different random token arrays
    random_matrix = np.random.randint(low=100,
                                      high=50000,
                                      size=(args.num_distinct, safe_initial_len))
    
    distinct_prompts_text = []
    print("[Data Prep] Decoding random token arrays into strings...")
    for i, row in enumerate(random_matrix):
        prompt_ids = row.tolist()
        # Decode the random integers directly into a raw string
        decoded_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        distinct_prompts_text.append(decoded_text)
        # Log the actual re-tokenized length
        actual_len = len(tokenizer(decoded_text).input_ids)
        length_diff = args.input_len - actual_len
        print(f"  -> Prompt {i+1}: Original IDs={args.input_len} | Re-tokenized Length={actual_len} | Diff=-{length_diff} tokens")

    # We cycle through the distinct prompts sequentially.
    with open(args.output_file, "w") as f:
        for i in range(args.num_prompts):
            prompt_text = distinct_prompts_text[i % args.num_distinct]
            
            # vLLM's dataset parser natively reads standard strings from the 'prompt' key
            json_record = {
                "prompt": prompt_text
            }
            f.write(json.dumps(json_record) + "\n")

    print("[Data Prep] Successfully generated Custom JSONL dataset using decoded random text.")
    print(f"[Data Prep] Total Lines: {args.num_prompts} | Distinct Request Types: {args.num_distinct}")
    print(f"[Data Prep] Saved to {args.output_file}")


if __name__ == "__main__":
    main()