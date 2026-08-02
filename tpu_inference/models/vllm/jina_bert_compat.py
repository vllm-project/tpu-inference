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
"""vLLM-registry compatibility wrapper for the JAX-native JinaBert model.

vLLM resolves architectures like "JinaBertForMaskedLM" through its own
ModelRegistry at config time (to determine, among other things, that this is
a pooling model). vLLM upstream has no JinaBert implementation, so we
register this wrapper — it is never instantiated or executed by vLLM's
PyTorch backend; the real model is the JAX implementation in
`tpu_inference.models.jax.jina_bert`, dispatched via tpu-inference's own
registry. This mirrors what `model_loader.register_model` builds dynamically
for out-of-tree models.
"""

from typing import Any, Optional

import torch

from tpu_inference.models.jax.jina_bert import \
    JinaBertForMaskedLM as _JaxJinaBertForMaskedLM


class JinaBertForMaskedLM(_JaxJinaBertForMaskedLM, torch.nn.Module):
    is_pooling_model = True

    def __init__(self, *args, **kwargs):
        # Only torch.nn.Module init: this class exists purely to satisfy
        # vLLM's registry inspection and must not trigger JAX logic.
        torch.nn.Module.__init__(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        raise NotImplementedError(
            "This is a JAX model and does not implement the PyTorch forward.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "This is a JAX model and does not implement embed_input_ids.")

    def load_weights(self, *args, **kwargs):
        # Prevent vLLM from trying to load weights into this dummy class.
        return None
