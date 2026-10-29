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
"""Regression tests for b/579355339.

Upstream vLLM's ``RoutedExperts`` stores the hash-MoE routing table as a plain
attribute (``object.__setattr__(self, "hash_indices_table", ...)``) because the
gate owns the underlying ``nn.Parameter``. Assigning an ``nn.Parameter`` to that
name afterwards goes through ``nn.Module.register_parameter`` and raises
``KeyError: "attribute 'hash_indices_table' already exists"``. These tests pin
the sharding helper to the upstream convention.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe import RoutedExperts

from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias


def _make_routed_experts_like_upstream(hash_indices_table, bias):
    """Build a bare RoutedExperts the way upstream __init__ wires these two
    attributes, without needing a FusedMoEConfig / quant method."""
    module = RoutedExperts.__new__(RoutedExperts)
    nn.Module.__init__(module)
    # Upstream: gate owns the Parameter; RoutedExperts holds a plain reference.
    object.__setattr__(module, "hash_indices_table", hash_indices_table)
    # Upstream: plain assignment, i.e. registered as a parameter if Parameter.
    module.e_score_correction_bias = bias
    return module


def test_assigning_parameter_over_plain_hash_indices_table_raises():
    tid2eid = nn.Parameter(torch.randint(0, 8, (16, 2), dtype=torch.int32),
                           requires_grad=False)
    module = _make_routed_experts_like_upstream(tid2eid, None)
    # This is the exact failure mode from b/579355339.
    with pytest.raises(KeyError, match="hash_indices_table"):
        module.hash_indices_table = nn.Parameter(tid2eid.detach().clone(),
                                                 requires_grad=False)


def test_replicate_keeps_hash_table_unregistered_and_bias_registered():
    tid2eid = nn.Parameter(torch.randint(0, 8, (16, 2), dtype=torch.int32),
                           requires_grad=False)
    bias = nn.Parameter(torch.zeros(8, dtype=torch.float32),
                        requires_grad=False)
    module = _make_routed_experts_like_upstream(tid2eid, bias)

    def fake_replicate(t, mesh):
        return t.detach().clone()

    with patch(
            "tpu_inference.layers.vllm.process_weights.cleanup_sharding."
            "_shard_tensor_to_tpu_replicated",
            side_effect=fake_replicate):
        _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(
            module, MagicMock())

    # hash_indices_table was replaced (not the gate's object) and stays a
    # plain attribute, so the gate remains the sole owner of the Parameter.
    assert module.hash_indices_table is not tid2eid
    assert torch.equal(module.hash_indices_table, tid2eid)
    assert "hash_indices_table" not in dict(module.named_parameters())

    # e_score_correction_bias keeps upstream's registered-parameter semantics.
    assert isinstance(module.e_score_correction_bias, nn.Parameter)
    assert "e_score_correction_bias" in dict(module.named_parameters())


def test_replicate_is_noop_without_hash_table_or_bias():
    module = _make_routed_experts_like_upstream(None, None)
    with patch("tpu_inference.layers.vllm.process_weights.cleanup_sharding."
               "_shard_tensor_to_tpu_replicated") as replicate:
        _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(
            module, MagicMock())
    replicate.assert_not_called()
    assert module.hash_indices_table is None
