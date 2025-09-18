from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec
from torch.utils import _pytree as pytree
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.lora.request import LoRARequest

from tpu_commons.models.vllm.sharding import (
    LORA_MODULE_TYPE_TO_WRAPPING_FUNC, extract_all_params_buffers, get_fqn,
    shard_lora_weights_and_move_to_tpu)

if TYPE_CHECKING:
    from tpu_commons.runner.tpu_jax_runner import TPUModelRunner

P = PartitionSpec


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

        params_and_buffers = self._process_params_and_buffers(
            self.runner.model.model, self.runner.mesh)
        self.runner.state = params_and_buffers

    def _process_params_and_buffers(self, model, mesh):

        def _is_unmoved_tensor(x):
            # tensors haven't been turned into torchax tensor are the ones not moved to TPU yet.
            return isinstance(
                x, torch.Tensor) and not isinstance(x, torchax.tensor.Tensor)

        def _move_to_tpu_replicated(x):
            # In certain cases, if t2j puts the tensor on CPU first and then device_put it to TPU, the tensor layout get messed up. To avoid that, we set jax default_device to TPU.
            with jax.default_device(jax.devices("tpu")[0]):
                x = t2j(x, use_dlpack=False)
            return torch_view(x).apply_jax_(jax.device_put,
                                            NamedSharding(mesh, P()))

        # Merely doing extract_all_buffers is not enough because we need to change the lora weights to torchax tensor.
        params, buffers = extract_all_params_buffers(model)
        params, buffers = pytree.tree_map_only(_is_unmoved_tensor,
                                               _move_to_tpu_replicated,
                                               (params, buffers))
        params_and_buffers = {**params, **buffers}
        return jax_view(params_and_buffers)

    def extract_lora_metadata(self):
        metadata = {}

        punica_wrapper = None
        for m_name, m in self.runner.model.model.named_modules():
            module_qualname = get_fqn(m)
            if module_qualname in LORA_MODULE_TYPE_TO_WRAPPING_FUNC:
                assert getattr(
                    m, 'punica_wrapper', None
                ) is not None, 'A lora wrapper should have contained a punica_wrapper'
                punica_wrapper = m.punica_wrapper
                break
        assert punica_wrapper is not None, "Should have been able to find a punica wrapper from the Lora wrapper."

        # vars does not show inherited methods or class attributes but this is fine because we only care about instance attributes.
        for k in vars(punica_wrapper):
            v = getattr(punica_wrapper, k, None)
            if k == "device":
                continue
            elif v is not None and isinstance(v, list):
                v = tuple(v)
            metadata[k] = v
        return metadata


# xw32: what's the type of the model here?
def replace_lora_metadata(model, metadata: dict) -> dict:
    original_metadata = {}

    punica_wrapper = None
    for m_name, m in model.named_modules():
        module_qualname = get_fqn(m)
        if module_qualname in LORA_MODULE_TYPE_TO_WRAPPING_FUNC:
            assert getattr(
                m, 'punica_wrapper', None
            ) is not None, 'A lora wrapper should have contained a punica_wrapper'
            punica_wrapper = m.punica_wrapper
            break
    assert punica_wrapper is not None, "Should have been able to find a punica wrapper from the Lora wrapper."

    for k in vars(punica_wrapper):
        if k in metadata:
            original_metadata[k] = getattr(punica_wrapper, k)
            setattr(punica_wrapper, k, metadata[k])
    return original_metadata
