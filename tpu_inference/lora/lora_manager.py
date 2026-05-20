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

import re
from typing import List, Optional
import torchax
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from tpu_inference import envs

# MaxText/Pax style name to HuggingFace/vLLM canonical target module mapping.
MAXTEXT_TO_HF_LORA_MAPPING = {
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "out": "o_proj",
    "wi_0": "gate_proj",
    "wi_1": "up_proj",
    "wo": "down_proj",
    "embed": "embed_tokens",
    "lm_head": "lm_head",
}


class TPULRUCacheWorkerLoRAManager(LRUCacheWorkerLoRAManager):
    """
    TPU-specific wrapper to ensure dummy LoRA creation happens 
    within the torchax environment.
    """

    def add_dummy_lora(self, lora_request, rank: int) -> bool:
        with torchax.default_env():
            return super().add_dummy_lora(lora_request, rank)


def parse_lora_module_path_env() -> Optional[List[str]]:
    """Parses LORA_MODULE_PATH env var into vLLM canonical target_modules list."""
    env_val = envs.LORA_MODULE_PATH
    if not env_val:
        return None

    target_modules = set()

    # Extract regex groups like (query|key|value|out)
    if '(' in env_val or ')' in env_val:
        matches = re.findall(r'\(([^)]+)\)', env_val)
        for group in matches:
            for module in group.split('|'):
                target_modules.add(MAXTEXT_TO_HF_LORA_MAPPING.get(module, module))
    else:
        # No parentheses found, assume standard list
        # Split by comma or pipe
        parts = re.split(r'[,|]', env_val)
        for part in parts:
            part = part.strip()
            if part:
                target_modules.add(MAXTEXT_TO_HF_LORA_MAPPING.get(part, part))
    return list(target_modules) if target_modules else None
