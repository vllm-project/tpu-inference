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

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest
import torch

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.vllm.custom_ops import fused_moe as fm


def _make_runner(*,
                 shared_experts=None,
                 is_sequence_parallel=False,
                 moe_config=None) -> fm.VllmMoERunner:
    """A ``VllmMoERunner`` with only the attributes the tested methods touch.

    ``__init__`` pulls in a full vLLM MoE config, so the runner is built via
    ``__new__`` and the handful of attributes used by the reduction logic are
    set directly (mirroring ``test_gdn_attention_op.py``).
    """
    runner = fm.VllmMoERunner.__new__(fm.VllmMoERunner)
    runner._shared_experts = shared_experts
    runner.moe_config = moe_config or SimpleNamespace(
        is_sequence_parallel=is_sequence_parallel)
    return runner


class TestFusedOutputIsReduced:

    def test_true_when_no_shared_expert(self):
        runner = _make_runner(shared_experts=None)
        # Nothing to fuse with -> the kernel reduces its own output.
        with patch.object(fm, "_get_mesh", return_value=object()):
            assert runner._fused_output_is_reduced is True

    def test_true_under_attention_dp(self):
        runner = _make_runner(shared_experts=object())
        with patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "is_attn_dp", return_value=True):
            assert runner._fused_output_is_reduced is True

    @pytest.mark.parametrize("backend", [MoEBackend.GMM_EP, MoEBackend.GMM_TP])
    def test_attention_dp_overrides_gmm_defer(self, backend):
        # attention-DP takes precedence over the GMM defer: even in the one
        # scenario that would otherwise return False (a GMM backend whose
        # reduction axes match the shared expert's), DP resolves the reduction
        # separately so the kernel must still reduce its own output. Guards the
        # order of the checks -- the DP short-circuit must precede the GMM ones.
        runner = _make_runner(shared_experts=object())
        with patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "is_attn_dp", return_value=True), \
             patch.object(fm, "select_moe_backend_from_fused_moe_config",
                          return_value=backend), \
             patch.object(fm, "_gmm_and_shared_reduce_over_same_axes",
                          return_value=True):
            assert runner._fused_output_is_reduced is True

    def test_true_when_no_mesh(self):
        runner = _make_runner(shared_experts=object())
        with patch.object(fm, "_get_mesh", return_value=None):
            assert runner._fused_output_is_reduced is True

    @pytest.mark.parametrize("backend",
                             [MoEBackend.FUSED_MOE, MoEBackend.DENSE_MAT])
    def test_true_for_non_gmm_backends(self, backend):
        # Only GMM_EP / GMM_TP defer the all-reduce; others always reduce.
        runner = _make_runner(shared_experts=object())
        with patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "select_moe_backend_from_fused_moe_config",
                          return_value=backend):
            assert runner._fused_output_is_reduced is True

    @pytest.mark.parametrize("backend", [MoEBackend.GMM_EP, MoEBackend.GMM_TP])
    def test_true_when_gmm_reduces_different_axes(self, backend):
        runner = _make_runner(shared_experts=object())
        with patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "select_moe_backend_from_fused_moe_config",
                          return_value=backend), \
             patch.object(fm, "_gmm_and_shared_reduce_over_same_axes",
                          return_value=False):
            # The GMM's reduction can't be folded into the single MLP_TENSOR
            # all-reduce, so the kernel must reduce itself.
            assert runner._fused_output_is_reduced is True

    @pytest.mark.parametrize("backend", [MoEBackend.GMM_EP, MoEBackend.GMM_TP])
    def test_false_when_all_conditions_met(self, backend):
        runner = _make_runner(shared_experts=object())
        with patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "select_moe_backend_from_fused_moe_config",
                          return_value=backend), \
             patch.object(fm, "_gmm_and_shared_reduce_over_same_axes",
                          return_value=True):
            # Shared expert present, no DP, GMM backend, matching axes -> the
            # kernel skips its reduce and the late single collective handles it.
            assert runner._fused_output_is_reduced is False


# ---------------------------------------------------------------------------
# VllmMoERunner._maybe_reduce_shared_expert_output  (early path)
# ---------------------------------------------------------------------------
class TestMaybeReduceSharedExpertOutput:

    def test_passthrough_when_shared_output_none(self):
        runner = _make_runner()
        with patch.object(fm, "_all_reduce_over_tp") as reduce:
            assert runner._maybe_reduce_shared_expert_output(None) is None
        reduce.assert_not_called()

    def test_passthrough_under_sequence_parallel(self):
        # SP defers the reduction to a separate all-gather in the model.
        runner = _make_runner(is_sequence_parallel=True)
        shared = torch.ones(2, 3)
        with patch.object(fm.VllmMoERunner, "_fused_output_is_reduced",
                          new_callable=PropertyMock, return_value=True), \
             patch.object(fm, "_all_reduce_over_tp") as reduce:
            out = runner._maybe_reduce_shared_expert_output(shared)
        reduce.assert_not_called()
        assert out is shared

    def test_passthrough_when_fused_output_not_reduced(self):
        # The late path will reduce shared + fused together, so don't reduce
        # the shared output on its own here.
        runner = _make_runner()
        shared = torch.ones(2, 3)
        with patch.object(fm.VllmMoERunner, "_fused_output_is_reduced",
                          new_callable=PropertyMock, return_value=False), \
             patch.object(fm, "_all_reduce_over_tp") as reduce:
            out = runner._maybe_reduce_shared_expert_output(shared)
        reduce.assert_not_called()
        assert out is shared

    def test_reduces_shared_output_on_early_path(self):
        runner = _make_runner()
        shared = torch.ones(2, 3)
        reduced = torch.full((2, 3), 7.0)
        mesh = object()
        with patch.object(fm.VllmMoERunner, "_fused_output_is_reduced",
                          new_callable=PropertyMock, return_value=True), \
             patch.object(fm, "_get_mesh", return_value=mesh), \
             patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm, "_all_reduce_over_tp",
                          return_value=reduced) as reduce:
            out = runner._maybe_reduce_shared_expert_output(shared)
        reduce.assert_called_once_with(shared, mesh)
        assert out is reduced

    def test_reduces_shared_output_under_attention_dp(self):
        # Under attention DP the fused kernel reduces its own output, so the
        # shared-expert output is reduced separately on the early path. Unlike
        # the test above this drives the real ``_fused_output_is_reduced``
        # property (via ``is_attn_dp``) instead of mocking it, exercising the
        # attn-DP -> early-reduce routing end to end.
        runner = _make_runner(shared_experts=object())
        shared = torch.ones(2, 3)
        reduced = torch.full((2, 3), 7.0)
        mesh = object()
        with patch.object(fm, "_get_mesh", return_value=mesh), \
             patch.object(fm, "is_attn_dp", return_value=True), \
             patch.object(fm, "_all_reduce_over_tp",
                          return_value=reduced) as reduce:
            runner._maybe_reduce_shared_expert_output(shared)
        reduce.assert_not_called()


# ---------------------------------------------------------------------------
# VllmMoERunner._maybe_reduce_final_output  (late path)
# ---------------------------------------------------------------------------
class TestMaybeReduceFinalOutput:

    def test_attention_dp_only_truncates(self):
        # Under attention DP the reduction is a separate reduce-scatter in the
        # model; here only the padding is stripped.
        runner = _make_runner()
        states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        with patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "is_attn_dp", return_value=True), \
             patch.object(fm, "_all_reduce_over_tp") as reduce:
            out = runner._maybe_reduce_final_output(states, trunc_size=3)
        reduce.assert_not_called()
        torch.testing.assert_close(out, states[..., :3])

    def test_sequence_parallel_only_truncates(self):
        runner = _make_runner(is_sequence_parallel=True)
        states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        with patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm.VllmMoERunner, "_fused_output_is_reduced",
                          new_callable=PropertyMock, return_value=False), \
             patch.object(fm, "_all_reduce_over_tp") as reduce:
            out = runner._maybe_reduce_final_output(states, trunc_size=2)
        reduce.assert_not_called()
        torch.testing.assert_close(out, states[..., :2])

    def test_only_truncates_when_kernel_already_reduced(self):
        # The kernel reduced its output (early path handled the shared expert),
        # so no late collective is needed.
        runner = _make_runner()
        states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        with patch.object(fm, "_get_mesh", return_value=object()), \
             patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm.VllmMoERunner, "_fused_output_is_reduced",
                          new_callable=PropertyMock, return_value=True), \
             patch.object(fm, "_all_reduce_over_tp") as reduce:
            out = runner._maybe_reduce_final_output(states, trunc_size=2)
        reduce.assert_not_called()
        torch.testing.assert_close(out, states[..., :2])

    def test_reduces_then_truncates_on_late_path(self):
        runner = _make_runner()
        states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        reduced = torch.arange(100, 108, dtype=torch.float32).reshape(2, 4)
        mesh = object()
        with patch.object(fm, "_get_mesh", return_value=mesh), \
             patch.object(fm, "is_attn_dp", return_value=False), \
             patch.object(fm.VllmMoERunner, "_fused_output_is_reduced",
                          new_callable=PropertyMock, return_value=False), \
             patch.object(fm, "_all_reduce_over_tp",
                          return_value=reduced) as reduce:
            out = runner._maybe_reduce_final_output(states, trunc_size=3)
        reduce.assert_called_once_with(states, mesh)
        # Reduction happens first, then the padding is stripped.
        torch.testing.assert_close(out, reduced[..., :3])
