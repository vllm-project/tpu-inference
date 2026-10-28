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
"""vLLM-registry compatibility shim for the JAX-native JinaBert model.

vLLM resolves architectures like "JinaBertForMaskedLM" through its own
ModelRegistry at config time (to determine, among other things, that this is
a pooling model). vLLM upstream has no JinaBert implementation, so we
register this shim. It is deliberately torch-only (no tpu_inference/JAX
imports): the registry inspects it — possibly in a subprocess — but never
instantiates or executes it. The real model is the JAX implementation in
`tpu_inference.models.jax.jina_bert`, dispatched via tpu-inference's own
registry.
"""

from typing import Any, Optional

import torch


class JinaBertForMaskedLM(torch.nn.Module):
    # Interface flags read by vLLM's registry inspection.
    is_pooling_model = True
    # Bidirectional encoder: vLLM then disables prefix caching and chunked
    # prefill (a cached prefix would reach the model as a partial prefill,
    # which mean pooling over bidirectional attention cannot serve).
    attn_type = "encoder_only"
    # Mean pooling. `default_seq_pooling_type` is the name read by current
    # vLLM; `default_pooling_type` is kept for older versions.
    default_seq_pooling_type = "MEAN"
    default_pooling_type = "MEAN"

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError(
            "JinaBert runs via the JAX backend; the PyTorch forward is a "
            "registry-compatibility stub.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "JinaBert runs via the JAX backend; embed_input_ids is a "
            "registry-compatibility stub.")

    def load_weights(self, *args, **kwargs):
        # Prevent vLLM from trying to load weights into this stub.
        return None


class JinaBertForSequenceClassification(JinaBertForMaskedLM):
    """Registry stub for JinaBert + mean pooling + linear head.

    The `...ForSequenceClassification` suffix makes vLLM default to
    `--runner pooling --convert classify`. The real model is
    `tpu_inference.models.jax.jina_bert.JinaBertForSequenceClassification`.
    """

