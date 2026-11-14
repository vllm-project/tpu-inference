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
"""Tests for tpu_inference.models.vllm.transformers_backend.global_view."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

tf_base = pytest.importorskip("vllm.model_executor.models.transformers.base")
from vllm.config import ModelConfig  # noqa: E402
from vllm.model_executor.models.transformers.fusers import \
    base as fusers_base  # noqa: E402
from vllm.model_executor.models.transformers.fusers import \
    packed_qkv as fusers_packed_qkv  # noqa: E402

from tpu_inference.models.vllm import transformers_backend  # noqa: E402

# codegemma-2b: 8 attention heads, 1 KV head.
_MODEL_CONFIG = SimpleNamespace(use_mla=False,
                                model_arch_config=SimpleNamespace(
                                    total_num_attention_heads=8,
                                    total_num_kv_heads=1))
_TP2 = SimpleNamespace(tensor_parallel_size=2)


def _targets():
    return (tf_base.Base._create_attention_instances,
            fusers_base.local_output_sizes,
            fusers_packed_qkv.local_output_sizes)


def test_patches_apply_only_inside_the_context():
    before = _targets()
    with transformers_backend.global_view():
        inside = _targets()
        assert all(a is not b for a, b in zip(before, inside))
    assert _targets() == before


def test_split_prefers_tpu_split_sizes():
    with transformers_backend.global_view():
        src = fusers_base.local_output_sizes("qkv_proj")
        assert fusers_packed_qkv.local_output_sizes("qkv_proj") == src
    replicated_qkv = SimpleNamespace(tpu_split_sizes=[2048, 256, 256],
                                     output_sizes=[2048, 512, 512],
                                     tp_size=1)
    assert eval(src, {},
                {"self": SimpleNamespace(qkv_proj=replicated_qkv)}) == [
                    2048, 256, 256
                ]


def test_split_falls_back_to_vllm_formula():
    with transformers_backend.global_view():
        src = fusers_base.local_output_sizes("gate_up_proj")
    plain = SimpleNamespace(output_sizes=[1024, 1024], tp_size=2)
    assert eval(src, {},
                {"self": SimpleNamespace(gate_up_proj=plain)}) == [512, 512]


def test_attention_instances_see_global_head_counts():
    seen = {}

    def fake_create(self, prefix=""):
        seen["heads"] = ModelConfig.get_num_attention_heads(
            _MODEL_CONFIG, _TP2)
        seen["kv_heads"] = ModelConfig.get_num_kv_heads(_MODEL_CONFIG, _TP2)

    # global_view wraps whatever _create_attention_instances is when entered.
    with patch.object(tf_base.Base, "_create_attention_instances",
                      fake_create):
        with transformers_backend.global_view():
            tf_base.Base._create_attention_instances(object())
        # Outside global_view, head counts are per shard again.
        assert ModelConfig.get_num_attention_heads(_MODEL_CONFIG, _TP2) == 4
    assert seen == {"heads": 8, "kv_heads": 1}
