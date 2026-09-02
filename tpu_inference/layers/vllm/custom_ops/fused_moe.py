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

import torch
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from tpu_inference.layers.common.linear import sum_partials
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import ShardingAxisName, is_attn_dp
from tpu_inference.layers.vllm.interface.moe import \
    select_moe_backend_from_fused_moe_config
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)


def _get_mesh() -> Mesh | None:
    """The active JAX device mesh, or ``None`` outside a model context."""
    try:
        return get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return None


def _sum_partials(t: torch.Tensor) -> torch.Tensor:
    """Reduce the leading partial axis of a deferred (unreduced) output — a
    shared expert's down_proj and/or the fused experts — with one all-reduce
    (see linear.sum_partials)."""
    return torch_view(sum_partials(jax_view(t)))


def _gmm_and_shared_reduce_over_same_axes(mesh: Mesh,
                                          moe_backend: MoEBackend) -> bool:
    """Whether the GMM reduction collapses exactly the shared expert's TP axes
    (DENSE_TENSOR, the axis its linears shard over).

    Only axes that actually shard (size > 1) on ``mesh`` count, so axis-name
    tuples that differ only in size-1 axes still match.
    """
    gmm_axis = (ShardingAxisName.EXPERT if moe_backend == MoEBackend.GMM_EP
                else ShardingAxisName.MLP_TENSOR)

    def effective(axes):
        axes = (axes, ) if isinstance(axes, str) else axes
        return frozenset(a for a in axes
                         if get_mesh_shape_product(mesh, a) > 1)

    return effective(gmm_axis) == effective(ShardingAxisName.DENSE_TENSOR)


@MoERunner.register_oot
class VllmMoERunner(MoERunner):

    @property
    def _fused_output_is_reduced(self) -> bool:
        # Returns False -- i.e. the GMM kernel skips its own all-reduce and the
        # shared + fused outputs are reduced together by a single collective in
        # ``_maybe_reduce_final_output`` -- ONLY when every condition below
        # holds. Otherwise the fused output is reduced by the kernel itself.
        #
        #   1. a shared expert is present (else there is nothing to fuse with)
        #   2. attention-DP is disabled (DP resolves the reduction separately)
        #   3. the backend is GMM_EP / GMM_TP (only those honor defer_all_reduce;
        #      e.g. the FUSED_MOE kernel always reduces)
        #   4. the GMM reduction collapses the same mesh axes as the shared
        #      expert (MLP_TENSOR) -- else one all-reduce over MLP_TENSOR cannot
        #      stand in for the GMM's reduction
        mesh = _get_mesh()
        if mesh is None:
            return True

        if self._shared_experts is None:
            return True

        if is_attn_dp(mesh) or self.moe_config.is_sequence_parallel:
            return True

        moe_backend = select_moe_backend_from_fused_moe_config(self.moe_config)
        if moe_backend not in (MoEBackend.GMM_EP, MoEBackend.GMM_TP):
            return True

        # The late path folds the GMM's reduction into a single all-reduce over
        # MLP_TENSOR (the shared expert's TP axis). That is only valid when the
        # GMM effectively reduces the same axes; otherwise fall back to the
        # kernel reducing its own output. (GMM_TP always reduces over
        # MLP_TENSOR; GMM_EP reduces over EXPERT, which may differ.)
        if not _gmm_and_shared_reduce_over_same_axes(mesh, moe_backend):
            return True

        return False

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        """Early all-reduce path: reduce the shared-expert output on its own.

        When the fused kernel already reduced its output, the shared-expert
        output (a ``[n_shards, tokens, hidden]`` partial stack from its
        RowParallelLinear, whose all-reduce was skipped via
        ``reduce_results=False``) must be reduced separately so the two match
        before being summed downstream. Under sequence parallelism a
        separate all-gather step in the model handles this instead.
        """
        mesh = _get_mesh()

        if mesh is None:
            return shared_output

        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced

        # The partial stack is sharded over the shared expert's own TP axis
        # (DENSE_TENSOR; ATTN_HEAD under attention DP), so summing it is the
        # right reduction on every mesh — PCP is plain TP here.
        if (shared_output is not None and fused_output_is_reduced
                and not self.moe_config.is_sequence_parallel):
            shared_output = _sum_partials(shared_output)
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        """Late all-reduce path: reduce the combined (shared + fused) output.

        When the fused kernel did not reduce its output, the shared and fused
        partial stacks were summed while still unreduced and the combined
        result is all-reduced here in a single collective instead of two.
        Under attention-DP the reduction is handled by a separate
        reduce-scatter pass in the model, so only the padding is stripped here.
        """
        mesh = _get_mesh()
        if mesh is None:
            return states[..., :trunc_size]

        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced

        is_dp = is_attn_dp(mesh)
        is_sequence_parallel = self.moe_config.is_sequence_parallel

        if not is_dp and not is_sequence_parallel and not output_is_reduced:
            states = _sum_partials(states)
        return states[..., :trunc_size]
