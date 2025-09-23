from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from torchax.interop import jax_view
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.request import LoRARequest

from tpu_commons.models.vllm.sharding import shard_model_to_tpu

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
        # One should not put lora_manager.set_active_loras under
        # torchax.default_env() because set_active_loras also load lora from
        # disk and torchax currently does not support that.  Here we load the
        # lora and set the lora weight to the linear layers.
        self.runner._set_active_loras(prompt_lora_mapping, token_lora_mapping,
                                      lora_requests)

        params_and_buffers = shard_model_to_tpu(self.runner.model.model,
                                                self.runner.mesh)
        self.runner.state = jax_view(params_and_buffers)

    def extract_lora_metadata(self):
        if self.runner.lora_config is None:
            return None

        metadata = {}
        punica_wrapper = None
        for _, m in self.runner.model.model.named_modules():
            if isinstance(m, BaseLinearLayerWithLoRA):
                assert getattr(
                    m, 'punica_wrapper', None
                ) is not None, 'A lora wrapper should have contained a punica_wrapper'
                punica_wrapper = m.punica_wrapper
                break
        assert punica_wrapper is not None, 'Should have been able to find a punica wrapper from the Lora wrapper.'

        # vars does not show inherited methods or class attributes but this is
        # fine because we only care about instance attributes.
        for k in vars(punica_wrapper):
            v = getattr(punica_wrapper, k, None)
            if k == 'device':  # Exclude string as it can't be traced by jax.jit
                continue
            metadata[k] = v
        return jax_view(metadata)


def replace_lora_metadata(model, metadata: dict, lora_config) -> dict:
    if lora_config is None:
        return {}

    original_metadata = {}
    punica_wrapper = None
    for _, m in model.named_modules():
        if isinstance(m, BaseLinearLayerWithLoRA):
            assert getattr(
                m, 'punica_wrapper', None
            ) is not None, 'A lora wrapper should have contained a punica_wrapper'
            punica_wrapper = m.punica_wrapper
            break
    assert punica_wrapper is not None, 'Should have been able to find a punica wrapper from the Lora wrapper.'

    for k in vars(punica_wrapper):
        if k in metadata:
            original_metadata[k] = getattr(punica_wrapper, k)
            setattr(punica_wrapper, k, metadata[k])
    return original_metadata
