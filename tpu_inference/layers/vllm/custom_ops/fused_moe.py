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

import jax
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, shard_map, torch_view
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import ShardingAxisName, is_attn_dp
from tpu_inference.layers.vllm.interface.moe import \
    select_moe_backend_from_fused_moe_config
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def _get_mesh() -> Mesh | None:
    """The active JAX device mesh, or ``None`` outside a model context."""
    try:
        return get_vllm_model_wrapper_context().mesh
    except AssertionError:
        return None


def _all_reduce_over_tp(t: torch.Tensor, mesh: Mesh) -> torch.Tensor:
    """All-reduce an unreduced local sum over the TP axis."""
    spec = P(ShardingAxisName.ATTN_DATA, None)

    @shard_map(mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)
    def _reduce(x):
        return jax.lax.psum(x, axis_name=ShardingAxisName.MLP_TENSOR)

    return torch_view(_reduce(jax_view(t)))


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
        mesh = _get_mesh()
        if mesh is None:
            return True

        if self._shared_experts is None or is_attn_dp(mesh):
            return True

        moe_backend = select_moe_backend_from_fused_moe_config(self.moe_config)
        if moe_backend not in (MoEBackend.GMM_EP, MoEBackend.GMM_TP):
            return True

        return False

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Early all-reduce path: reduce the shared-expert output on its own.

        When the fused kernel already reduced its output, the shared-expert
        output (produced unreduced by its RowParallelLinear, whose all-reduce
        was skipped via ``reduce_results=False``) must be reduced separately so the
        two match before being summed downstream. Under sequence parallelism a
        separate all-gather step in the model handles this instead.
        """
        if (shared_output is not None
                and not self.moe_config.is_sequence_parallel
                and self._fused_output_is_reduced):
            mesh = _get_mesh()
            if mesh is not None:
                shared_output = _all_reduce_over_tp(shared_output, mesh)
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int,
    ) -> torch.Tensor:
        """Late all-reduce path: reduce the combined (shared + fused) output.

        When the fused kernel did not reduce its output, the shared and fused
        outputs were summed while still unreduced and the combined result is
        all-reduced here in a single collective instead of two. Under attention-DP the
        reduction is handled by a separate reduce-scatter pass in the model, so
        only the padding is stripped here.
        """
        mesh = _get_mesh()
        if mesh is None:
            return states[..., :trunc_size]

        is_dp = is_attn_dp(mesh)
        is_sequence_parallel = self.moe_config.is_sequence_parallel
        is_fused_output_reduced = self._fused_output_is_reduced

        if not is_dp and not is_sequence_parallel and not is_fused_output_reduced:
            states = _all_reduce_over_tp(states, mesh)
        return states[..., :trunc_size]
