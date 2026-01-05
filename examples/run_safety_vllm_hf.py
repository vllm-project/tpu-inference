# Copyright 2025 Google LLC
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

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image

# vLLM Imports
from vllm import LLM, SamplingParams

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-Guard-4-12B"
# CRITICAL: Use 4 chips for sharding (Fixes OOM on v6e-8)
TENSOR_PARALLEL_SIZE = 8 
MAX_MODEL_LEN = 8192      # We can now use full 8k context!

def load_mm_safety_bench(image_dir: Path) -> List[Tuple[List[Dict[str, Any]], str]]:
    """
    Loads MM-SafetyBench from the specific directory structure:
    root/
      ├── images/
      │     └── {scenario_name}/SD/{id}.jpg  (Standard Benchmark Structure)
      └── processed_questions/
            └── {scenario_name}.json
    """
    # The user passes 'images' dir, but we need the root to find 'processed_questions'
    dataset_root = image_dir.parent
    questions_dir = dataset_root / "processed_questions"
    
    if not questions_dir.exists():
        if (image_dir / "processed_questions").exists():
            dataset_root = image_dir
            questions_dir = image_dir / "processed_questions"
            image_dir = image_dir / "images"
        else:
            raise FileNotFoundError(f"Could not find 'processed_questions' dir relative to {image_dir}")

    test_cases = []
    
    # Iterate over every scenario JSON file
    for json_file in sorted(questions_dir.glob("*.json")):
        scenario_name = json_file.stem 
        
        with open(json_file, "r") as f:
            data = json.load(f)
            
        for question_id, entry in data.items():
            # 1. Get the prompt
            prompt_text = entry.get("Rephrased Question(SD)") or entry.get("Rephrased Question")
            
            # 2. Find the image
            img_filename = f"{question_id}.jpg"
            possible_paths = [
                image_dir / scenario_name / "SD" / img_filename,
                image_dir / scenario_name / img_filename,
                image_dir / scenario_name / "SD_TYPO" / img_filename,
            ]
            
            final_image_path = None
            for p in possible_paths:
                if p.exists():
                    final_image_path = p
                    break
            
            if final_image_path is None:
                continue

            # 3. Construct the conversation (Standard Format)
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image_path": str(final_image_path)} # Ensure path is string
                ]
            }]
            
            # 4. Set Expected Label
            test_cases.append((conversation, "unsafe"))

    print(f"Loaded {len(test_cases)} test cases from MM-SafetyBench.")
    return test_cases


def main(image_dir_path: str):
    print(f"Initializing vLLM with Model: {MODEL_ID}...")
    print(f"Sharding across {TENSOR_PARALLEL_SIZE} chips.")

    # 1. Initialize vLLM Engine
    # This automatically handles loading weights and sharding memory across all 4 chips.
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True # Often recommended for TPU stability
    )

    image_dir = Path(image_dir_path)
    test_cases = load_mm_safety_bench(image_dir)

    total_tests = len(test_cases)
    passed_tests = 0

    # Define Generation Parameters
    sampling_params = SamplingParams(
        temperature=0.0,       # Greedy decoding (deterministic)
        max_tokens=20,         # Short output for safety labels
        ignore_eos=False
    )

    print("-" * 80)
    
    for i, (conversation, expected_text) in enumerate(test_cases):
        print(f"Running Test Case {i+1}/{total_tests}...")
        
        user_content = conversation[0]["content"]
        
        # Extract Prompt and Image for vLLM format
        prompt_text = ""
        image_input = None
        
        for block in user_content:
            if block["type"] == "text":
                prompt_text = block["text"]
            elif block["type"] == "image":
                try:
                    image_input = Image.open(block["image_path"]).convert("RGB")
                except Exception as e:
                    print(f"  [Error] Could not load image: {e}")
                    image_input = Image.new('RGB', (224, 224), color='gray')

        # Construct vLLM Chat Input
        # We put the image first logic here implicitly via the multi_modal_data dict,
        # but the prompt template usually handles placement.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"}, # Placeholder for vLLM to insert image tokens
                ]
            }
        ]

        # Generate
        outputs = llm.chat(
            messages=messages,
            multi_modal_data={"image": image_input} if image_input else None,
            sampling_params=sampling_params
        )

        generated_text = outputs[0].outputs[0].text.strip()

        # --- VERIFICATION ---
        print(f"\nGenerated: {generated_text!r}")
        print(f"Expected:  {expected_text!r}")

        match = False
        if generated_text == expected_text:
            match = True
        elif expected_text.startswith("unsafe") and generated_text.startswith("unsafe"):
            match = True

        if match:
            print("Result: PASSED")
            passed_tests += 1
        else:
            print("Result: FAILED")

        print("-" * 80)

    accuracy = (passed_tests / total_tests) * 100
    print(f"Final Accuracy: {passed_tests}/{total_tests} = {accuracy:.2f}%")


if __name__ == "__main__":
    IMAGE_DIR = "/home/jiries_google_com/mm-safetybench/images"
    if os.path.isdir(IMAGE_DIR):
        main(IMAGE_DIR)
    else:
        print(
            f"Please update IMAGE_DIR in the script. Current path not found: {IMAGE_DIR}"
        )