import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Generate decode-only JSONL dataset with raw Token IDs")
    parser.add_argument("--input-len",
                        required=True,
                        type=int,
                        help="Exact number of input tokens")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5000,
        help="Total number of prompts to generate."
    )
    parser.add_argument(
        "--num-distinct",
        required=True,
        type=int,
        help="Number of distinct prompts for caching."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/tmp/decode_only_dataset.jsonl",
        help="Path to save the JSONL"
    )
    args = parser.parse_args()

    print(f"[Data Prep] Generating {args.num_distinct} distinct random token arrays (Exact Len: {args.input_len})...")

    # Generate distinct random token arrays (avoiding 0-100 reserved special tokens)
    random_matrix = np.random.randint(low=100,
                                      high=50000,
                                      size=(args.num_distinct, args.input_len))
    
    distinct_prompts_ids = [row.tolist() for row in random_matrix]

    print(f"[Data Prep] Writing {args.num_prompts} records to JSONL...")
    with open(args.output_file, "w") as f:
        for i in range(args.num_prompts):
            # We assign the raw LIST OF INTEGERS directly to the "prompt" key!
            json_record = {
                "prompt": distinct_prompts_ids[i % args.num_distinct]
            }
            f.write(json.dumps(json_record) + "\n")

    print("[Data Prep] Successfully generated Custom JSONL dataset using raw Token IDs.")
    print(f"[Data Prep] Saved to {args.output_file}")

if __name__ == "__main__":
    main()
