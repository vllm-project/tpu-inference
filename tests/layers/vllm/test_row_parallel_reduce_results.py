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
"""`VllmRowParallelLinear` keeps `defer_all_reduce` in sync with
`reduce_results`, including when it is changed after construction."""

from types import SimpleNamespace

import pytest
import torch

from tpu_inference.layers.vllm.custom_ops.linear import VllmRowParallelLinear


def _bare_layer(reduce_results: bool) -> VllmRowParallelLinear:
    # Skip vLLM's heavy __init__; only the attribute plumbing is under test.
    layer = VllmRowParallelLinear.__new__(VllmRowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.reduce_results = reduce_results
    layer.quant_method = SimpleNamespace(linear_config=SimpleNamespace(
        defer_all_reduce=None))
    layer._sync_defer_all_reduce()
    return layer


def test_defer_all_reduce_follows_reduce_results_at_init():
    assert _bare_layer(
        True).quant_method.linear_config.defer_all_reduce is False
    assert _bare_layer(
        False).quant_method.linear_config.defer_all_reduce is True


def test_defer_all_reduce_follows_reduce_results_set_after_init():
    # vLLM's Transformers modeling backend flips `down_proj.reduce_results`
    # to False on fused shared experts after the model is built.
    layer = _bare_layer(True)
    layer.reduce_results = False
    assert layer.reduce_results is False
    assert layer.quant_method.linear_config.defer_all_reduce is True
    layer.reduce_results = True
    assert layer.quant_method.linear_config.defer_all_reduce is False


def test_reduce_results_false_after_init_rejects_bias():
    layer = _bare_layer(True)
    layer.bias = torch.nn.Parameter(torch.zeros(4))
    layer.skip_bias_add = False
    with pytest.raises(ValueError):
        layer.reduce_results = False
    assert layer.quant_method.linear_config.defer_all_reduce is False
    # A skipped bias is returned to the caller, so deferring is fine.
    layer.skip_bias_add = True
    layer.reduce_results = False
    assert layer.quant_method.linear_config.defer_all_reduce is True
