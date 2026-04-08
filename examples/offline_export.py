#!/usr/bin/env python3
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
"""Hello world example for TPU Model Export on CPU."""

import os

from vllm import LLM


def main():
    # Use workspace folder for export to see results on host
    export_path = os.environ.get(
        "TPU_INFERENCE_EXPORT_PATH"
    ) or "/workspace/tpu_inference/exported_model_dir"
    print(f"Starting TPU Model Export on CPU. Target path: {export_path}")

    # Set environment variables if not already set by run_export_docker.sh
    if not os.environ.get("TPU_INFERENCE_EXPORT_PATH"):
        os.environ["TPU_INFERENCE_EXPORT_PATH"] = export_path
    if not os.environ.get("TPU_INFERENCE_EXPORT_TOPOLOGY"):
        os.environ["TPU_INFERENCE_EXPORT_TOPOLOGY"] = "TPU7x:2x4"

    # Initialize LLM with dummy weights to bypass loading
    LLM(
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
