# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools


def patch_vllm_input_processor_for_block_diffusion() -> None:
    """Reject resumable requests before they reach the diffusion runner."""
    from vllm.v1.engine.input_processor import InputProcessor

    if getattr(InputProcessor, "_block_diffusion_validation_patched", False):
        return

    original_process_inputs = InputProcessor.process_inputs

    @functools.wraps(original_process_inputs)
    def patched_process_inputs(processor_self, *args, **kwargs):
        if "resumable" in kwargs:
            resumable = bool(kwargs["resumable"])
        else:
            # `resumable` is the eleventh positional argument after `self`.
            resumable = len(args) > 10 and bool(args[10])
        additional_config = getattr(processor_self.vllm_config,
                                    "additional_config", {}) or {}
        raw_strategy = additional_config.get("generation_strategy")
        strategy = getattr(raw_strategy, "value", raw_strategy)
        if resumable and strategy == "block_diffusion":
            raise ValueError(
                "block_diffusion does not support resumable or streaming-input "
                "requests")
        return original_process_inputs(processor_self, *args, **kwargs)

    InputProcessor.process_inputs = patched_process_inputs
    InputProcessor._block_diffusion_validation_patched = True
