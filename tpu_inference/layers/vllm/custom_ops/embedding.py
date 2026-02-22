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

from vllm.config.vllm import get_current_vllm_config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding)

from tpu_inference.layers.vllm.quantization.unquantized import \
    VllmUnquantizedEmbeddingMethod


@VocabParallelEmbedding.register_oot
class VllmVocabParallelEmbedding(VocabParallelEmbedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            # TODO(kyuyeunk): Cleanup process of passing mesh variable.
            vllm_config = get_current_vllm_config()
            mesh = vllm_config.quant_config.mesh
            self.quant_method = VllmUnquantizedEmbeddingMethod(mesh)

    def forward(self, input_):
        return super().forward(input_)


@ParallelLMHead.register_oot
class VllmParallelLMHead(ParallelLMHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            # TODO(kyuyeunk): Cleanup process of passing mesh variable.
            vllm_config = get_current_vllm_config()
            mesh = vllm_config.quant_config.mesh
            self.quant_method = VllmUnquantizedEmbeddingMethod(mesh)

    def forward(self, input_):
        return super().forward(input_)
