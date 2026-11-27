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
"""TPU custom op for GemmaRMSNorm using a one-pass Pallas kernel at prefill shapes.

SP-aware: caches vllm_config at __init__ time and resolves the active token
sharding spec (ATTN_DATA when enable_sp=False, MLP_TENSOR or EXPERT_DATA when
enable_sp=True) so shard_map introduces zero communication under both standard
TP and Sequence Parallelism.
"""

import jax
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from tpu_inference.kernels.layernorm.fused_gemma_rmsnorm import (
    DEFAULT_BLOCK_ROWS, fused_add_gemma_rms_norm, supports)
from tpu_inference.layers.common.sharding import (ShardingAxisName,
                                                  sp_token_axes)
from tpu_inference.layers.vllm.interface.moe import vllm_keeps_tokens_sharded
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from tpu_inference.utils import TPU_SECOND_LAST_MINOR, get_mesh_shape_product

MIN_LOCAL_ROWS = 512


@GemmaRMSNorm.register_oot
class TpuGemmaRMSNorm(GemmaRMSNorm):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps=eps)
        try:
            self._vllm_config = get_current_vllm_config()
        except Exception:
            self._vllm_config = None
        additional = (getattr(self._vllm_config, "additional_config", None)
                      if self._vllm_config is not None else None)
        self._enable_sp = (bool(additional.get("enable_sp", False))
                           if additional else False)

    def _resolve_norm_data_spec(
        self,
        x_rows: int,
        mesh: jax.sharding.Mesh,
    ) -> tuple[P, int]:
        if self._enable_sp and self._vllm_config is not None:
            if vllm_keeps_tokens_sharded(self._vllm_config, mesh):
                sp_axes = sp_token_axes(mesh, True, x_rows)
                if sp_axes is not None:
                    return P(sp_axes,
                             None), get_mesh_shape_product(mesh, sp_axes)
            else:
                tp_size = get_mesh_shape_product(mesh,
                                                 ShardingAxisName.MLP_TENSOR)
                if (tp_size > 1 and x_rows % tp_size == 0
                        and (x_rows // tp_size) >= TPU_SECOND_LAST_MINOR):
                    return P(ShardingAxisName.MLP_TENSOR, None), tp_size
        return P(ShardingAxisName.ATTN_DATA,
                 None), get_mesh_shape_product(mesh,
                                               ShardingAxisName.ATTN_DATA)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2 and (residual is None or residual.shape == x.shape):
            ctx = get_vllm_model_wrapper_context()
            mesh = ctx.mesh if ctx is not None else None
            if mesh is not None:
                data_spec, shard_div = self._resolve_norm_data_spec(
                    x.shape[0], mesh)
                local_rows = x.shape[0] // shard_div
            else:
                data_spec = None
                local_rows = x.shape[0]
            if supports(local_rows,
                        x.shape[1],
                        DEFAULT_BLOCK_ROWS,
                        min_rows=MIN_LOCAL_ROWS):
                jx = jax_view(x)
                jw = jax_view(self.weight)
                if mesh is not None:
                    w_spec = P(None)
                    if residual is not None:
                        jr = jax_view(residual)
                        jy, jro = jax.shard_map(
                            lambda a, b, w: fused_add_gemma_rms_norm(
                                a,
                                b,
                                w,
                                eps=self.variance_epsilon,
                                block_rows=DEFAULT_BLOCK_ROWS,
                            ),
                            mesh=mesh,
                            in_specs=(data_spec, data_spec, w_spec),
                            out_specs=(data_spec, data_spec),
                            check_vma=False,
                        )(jx, jr, jw)
                        return torch_view(jy), torch_view(jro)
                    jy = jax.shard_map(
                        lambda a, w: fused_add_gemma_rms_norm(
                            a,
                            None,
                            w,
                            eps=self.variance_epsilon,
                            block_rows=DEFAULT_BLOCK_ROWS,
                        ),
                        mesh=mesh,
                        in_specs=(data_spec, w_spec),
                        out_specs=data_spec,
                        check_vma=False,
                    )(jx, jw)
                    return torch_view(jy)
                if residual is not None:
                    jr = jax_view(residual)
                    jy, jro = fused_add_gemma_rms_norm(
                        jx,
                        jr,
                        jw,
                        eps=self.variance_epsilon,
                        block_rows=DEFAULT_BLOCK_ROWS,
                    )
                    return torch_view(jy), torch_view(jro)
                jy = fused_add_gemma_rms_norm(
                    jx,
                    None,
                    jw,
                    eps=self.variance_epsilon,
                    block_rows=DEFAULT_BLOCK_ROWS,
                )
                return torch_view(jy)
        return super().forward_native(x, residual)
