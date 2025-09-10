from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from vllm.lora.request import LoRARequest

from tpu_commons.models.vllm.sharding import (
    extract_all_buffers, shard_lora_weights_and_move_to_tpu)

if TYPE_CHECKING:
    from tpu_commons.runner.tpu_jax_runner import TPUModelRunner


class LoraUtils:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

    def set_active_loras(self, num_scheduled_tokens_per_req,
                         total_num_scheduled_tokens,
                         padded_total_num_scheduled_tokens):
        # We need to respect padding when activating LoRA adapters
        padded_num_scheduled_tokens_per_req = np.copy(
            num_scheduled_tokens_per_req
        )  # Copying to avoid accidental state corruption bugs
        padded_num_scheduled_tokens_per_req[-1] += \
            padded_total_num_scheduled_tokens - total_num_scheduled_tokens

        prompt_lora_mapping: tuple[int, ...]  # of size input_batch.num_reqs
        token_lora_mapping: tuple[int,
                                  ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
        prompt_lora_mapping, token_lora_mapping, lora_requests = \
                            self.runner.input_batch.make_lora_inputs(padded_num_scheduled_tokens_per_req)
        # One should not put lora_manager.set_active_loras under torchax.default_env() because set_active_loras also load lora from disk and torchax currently does not support that.
        # Here we load the lora and set the lora weight to the linear layers.
        self.runner._set_active_loras(prompt_lora_mapping, token_lora_mapping,
                                      lora_requests)

        shard_lora_weights_and_move_to_tpu(self.runner.model.model,
                                           self.runner.mesh)

        params, buffers, _ = extract_all_buffers(self.runner.model.model)
        self.runner.state = {**params, **buffers}
