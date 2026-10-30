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
``KeyError: "attribute 'hash_indices_table' already exists"``. cleanup_sharding
therefore registers the TPU-replicated copy as the ``tpu_hash_indices_table``
buffer, which keeps it a ``functional_call`` input (not a jit constant).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe import RoutedExperts

from tpu_inference.layers.vllm.process_weights.cleanup_sharding import (
    _extract_all_params_buffers,
    _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias)

_REPLICATE = ("tpu_inference.layers.vllm.process_weights.cleanup_sharding."
              "_shard_tensor_to_tpu_replicated")


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


def _table():
    return nn.Parameter(torch.randint(0, 8, (16, 2), dtype=torch.int32),
                        requires_grad=False)


def test_assigning_parameter_over_plain_hash_indices_table_raises():
    module = _make_routed_experts_like_upstream(_table(), None)
    # This is the exact failure mode from b/579355339 and the reason the
    # replicated copy must live under a different name.
    with pytest.raises(KeyError, match="hash_indices_table"):
        module.hash_indices_table = _table()


def test_replicate_registers_tpu_buffer_and_keeps_bias_parameter():
    tid2eid = _table()
    bias = nn.Parameter(torch.zeros(8, dtype=torch.float32),
                        requires_grad=False)
    module = _make_routed_experts_like_upstream(tid2eid, bias)

    replicated = {}

    def fake_replicate(t, mesh):
        out = t.detach().clone()
        replicated[id(t)] = out
        return out

    with patch(_REPLICATE, side_effect=fake_replicate):
        _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(
            module, MagicMock())

    # The gate-owned reference is untouched and still not a parameter here.
    assert module.hash_indices_table is tid2eid
    params, buffers = _extract_all_params_buffers(module)
    assert "hash_indices_table" not in params

    # The replicated copy is a (non-persistent) buffer, so it is part of what
    # shard_model_to_tpu returns and functional_call feeds to the jitted step.
    assert module.tpu_hash_indices_table is replicated[id(tid2eid)]
    assert buffers["tpu_hash_indices_table"] is module.tpu_hash_indices_table
    assert "tpu_hash_indices_table" not in module.state_dict()
    # HASH_TABLE_ROW_MAJOR in vllm_model_wrapper keys on this suffix.
    assert "tpu_hash_indices_table".endswith("hash_indices_table")

    # e_score_correction_bias keeps upstream's registered-parameter semantics
    # (re-wrapped as a new nn.Parameter around the replicated tensor).
    assert isinstance(module.e_score_correction_bias, nn.Parameter)
    assert params["e_score_correction_bias"] is module.e_score_correction_bias
    assert module.e_score_correction_bias is not bias
    assert torch.equal(module.e_score_correction_bias, replicated[id(bias)])


def test_replicate_is_idempotent():
    module = _make_routed_experts_like_upstream(_table(), None)
    with patch(_REPLICATE, side_effect=lambda t, m: t.detach().clone()):
        _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(
            module, MagicMock())
        first = module.tpu_hash_indices_table
        _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(
            module, MagicMock())
    assert module.tpu_hash_indices_table is not first
    assert len(dict(module.named_buffers())) == 1


def test_replicate_is_noop_without_hash_table_or_bias():
    module = _make_routed_experts_like_upstream(None, None)
    with patch(_REPLICATE) as replicate:
        _replicate_fused_moe_hash_indices_tables_and_e_score_correction_bias(
            module, MagicMock())
    replicate.assert_not_called()
    assert module.hash_indices_table is None
    assert not hasattr(module, "tpu_hash_indices_table")
