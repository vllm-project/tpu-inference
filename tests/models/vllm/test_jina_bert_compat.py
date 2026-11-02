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
"""vLLM registry entries for the JAX-native JinaBert architectures.

Runs anywhere (CPU is fine): only vLLM's ModelRegistry and the torch-only
registry stubs are involved.
"""

import pytest

import tpu_inference
from tpu_inference.models.vllm import jina_bert_compat
from tpu_inference.models.vllm.experimental import _TPU_VLLM_MODELS

# arch string -> registry stub class in jina_bert_compat.
JINA_ARCHS = {
    "JinaBertForMaskedLM": "JinaBertForMaskedLM",
    "JinaBertModel": "JinaBertForMaskedLM",
    "JinaBertForSequenceClassification": "JinaBertForSequenceClassification",
}


@pytest.mark.parametrize("arch,stub", JINA_ARCHS.items())
def test_archs_are_registered_with_vllm(arch, stub):
    from vllm import ModelRegistry

    # Done on `import tpu_inference` (every vLLM process, incl. the API
    # server) and by the vllm.general_plugins entry point.
    tpu_inference._register_vllm_compat_archs()
    assert arch in ModelRegistry.get_supported_archs()
    assert _TPU_VLLM_MODELS[arch] == (
        f"tpu_inference.models.vllm.jina_bert_compat:{stub}")


@pytest.mark.parametrize("stub", sorted(set(JINA_ARCHS.values())))
def test_stubs_declare_encoder_only_mean_pooling(stub):
    cls = getattr(jina_bert_compat, stub)
    # Encoder-only: vLLM disables prefix caching and chunked prefill, so
    # every prompt reaches the model (and mean pooling) in one piece.
    assert cls.is_pooling_model
    assert cls.attn_type == "encoder_only"
    assert cls.default_seq_pooling_type == "MEAN"
