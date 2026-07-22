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

import torch
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    if not hasattr(DeepseekV32IndexerCache, "kv_sharing_target_layer_name"):
        DeepseekV32IndexerCache.kv_sharing_target_layer_name = None
except Exception:
    pass

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


@SparseAttnIndexer.register_oot
class VllmSparseAttnIndexer(SparseAttnIndexer):

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_tpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.topk_indices_buffer
