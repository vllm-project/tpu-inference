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

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def quantize_absmax(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to float8_e4m3fn with absmax scaling."""
    # e4m3 max representable value is 448.0
    amax = tensor.abs().max(dim=-1, keepdim=True).values
    scale = (amax / 448.0).clamp(min=1e-12)
    quantized = (tensor / scale).to(torch.float8_e4m3fn)
    # vLLM FP8 expects scale to be float32, and 1D for most loaders
    return quantized, scale.squeeze().to(torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        type=Path,
                        required=True,
                        help="Source model ID or local path")
    parser.add_argument("--output",
                        type=Path,
                        required=True,
                        help="Path to save the quantized model")
    args = parser.parse_args()

    if not args.source.exists():
        repo_id = str(args.source)
        LOGGER.info(f"Downloading {repo_id}")
        source_root = Path(snapshot_download(repo_id))
    else:
        source_root = args.source

    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(source_root, args.output)

    for st_path in sorted(args.output.glob("*.safetensors")):
        updates = {}
        with safe_open(st_path, framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)

                is_vision_component = "vision_tower" in key or "embed_vision" in key

                if not is_vision_component and (
                    ("proj" in key and "weight" in key) or "experts" in key):
                    LOGGER.info(f"Quantizing {key} of shape {tensor.shape}")
                    q_weight, scale = quantize_absmax(tensor)

                    # For standard vLLM FP8 format
                    if not key.endswith(".weight"):
                        weight_key = key + ".weight"
                        scale_key = key + ".weight_scale_inv"
                    else:
                        weight_key = key
                        scale_key = key.replace(".weight", ".weight_scale_inv")

                    updates[weight_key] = q_weight
                    updates[scale_key] = scale
                else:
                    updates[key] = tensor

        save_file(updates, st_path, metadata=metadata)
        LOGGER.info(f"Updated {st_path.name}")

    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "version": "1.0"
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    LOGGER.info(f"Quantized FP8 checkpoint written to {args.output}")


if __name__ == "__main__":
    main()
