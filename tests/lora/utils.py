# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights


# https://github.com/vllm-project/vllm/blob/279a5f31b3faa6f40759516efa5c742f637ab8b7/tests/lora/utils.py
class DummyLoRAManager:

    def __init__(self, device: torch.device = "cuda:0"):
        super().__init__()
        self._loras: dict[str, LoRALayerWeights] = {}
        self._device = device

    def set_module_lora(self, module_name: str, lora: LoRALayerWeights):
        self._loras[module_name] = lora

    def get_module_lora(self, module_name: str) -> LoRALayerWeights:
        return self._loras[module_name]

    def init_random_lora(
        self,
        module_name: str,
        weight: torch.Tensor,
        rank: int = 8,
    ):
        lora = LoRALayerWeights(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=torch.rand([rank, weight.shape[1]],
                              dtype=weight.dtype,
                              device=self._device),
            lora_b=torch.rand([weight.shape[0], rank],
                              dtype=weight.dtype,
                              device=self._device),
        )
        self.set_module_lora(module_name, lora)

        return lora

    def init_lora(
        self,
        module_name: str,
        input_dim: int,
        output_dim: int,
        rank=8,
        noop=False,
        embeddings_tensor=None,
    ):
        lora = LoRALayerWeights(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=torch.rand([rank, input_dim], device="cuda"),
            lora_b=torch.rand([output_dim, input_dim], device="cuda"),
            embeddings_tensor=embeddings_tensor,
        )
        self.set_module_lora(module_name, lora)
        return lora

    def reset_lora(self):
        self._loras = {}

    def init_packed_lora(
        self,
        module_name: str,
        input_dim: int,
        output_dims: list[int],
        noop_lora_index: list[int] | None = None,
        rank: int = 8,
    ):
        base_loras: list[LoRALayerWeights] = []
        noop_lora_index_set = set(noop_lora_index or [])

        for i, out_dim in enumerate(output_dims):
            base_lora = self.init_lora(
                module_name + "_000_" + str(i),
                input_dim,
                out_dim,
                rank=rank,
                noop=i in noop_lora_index_set,
            )
            base_loras.append(base_lora)
        packed_lora = PackedLoRALayerWeights.pack(base_loras)
        self.set_module_lora(module_name, packed_lora)
        return packed_lora
