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
"""JinaBert v2 encoder megakernel (see README.md)."""

from tpu_inference.kernels.jina_bert_megakernel.kernel import (
    MAX_TOKENS, PRECISIONS, JinaBertPackedWeights, estimate_vmem_bytes,
    jina_bert_encoder_megakernel, mxu_dtype_for, pack_jina_bert_weights,
    unsupported_geometry_reason, unsupported_vmem_reason)

__all__ = [
    "MAX_TOKENS",
    "PRECISIONS",
    "JinaBertPackedWeights",
    "estimate_vmem_bytes",
    "jina_bert_encoder_megakernel",
    "mxu_dtype_for",
    "pack_jina_bert_weights",
    "unsupported_geometry_reason",
    "unsupported_vmem_reason",
]
