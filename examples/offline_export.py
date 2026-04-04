#!/usr/bin/env python3
"""Hello world example for TPU Model Export on CPU."""

import os
from vllm import LLM
from tpu_inference.runner.compilation_manager import export_func_map
from tpu_inference.export import serving_model
import shutil


def main():
    # Use workspace folder for export to see results on host
    export_path = os.environ.get("TPU_INFERENCE_EXPORT_PATH") or "/workspace/tpu_inference/exported_model_dir"
    print(f"Starting TPU Model Export on CPU. Target path: {export_path}")
    
    # Set environment variables if not already set by run_export_docker.sh
    if not os.environ.get("TPU_INFERENCE_EXPORT_PATH"):
        os.environ["TPU_INFERENCE_EXPORT_PATH"] = export_path
    if not os.environ.get("TPU_INFERENCE_EXPORT_TOPOLOGY"):
        os.environ["TPU_INFERENCE_EXPORT_TOPOLOGY"] = "TPU7x:2x4"


    
    # Initialize LLM with dummy weights to bypass loading
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_format="dummy",
        max_model_len=1024,
    )

    if os.path.exists(export_path):
        print(f"Success! Export directory created at {export_path}")
        print("Contents:")
        for f in os.listdir(export_path):
             print(f"  - {f}")
    else:
        print(f"Error: Export directory {export_path} not found.")

if __name__ == "__main__":
    main()
