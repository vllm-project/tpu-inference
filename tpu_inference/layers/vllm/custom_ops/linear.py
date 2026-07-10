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
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, shard_map, torch_view
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import (
    reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)
from tpu_inference.utils import get_mesh_shape_product


@RowParallelLinear.register_oot
class VllmRowParallelLinear(RowParallelLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Only defer the all-reduce when the quant method's matmul actually
        # honors ``linear_config.defer_all_reduce``. Methods that don't (e.g.
        # the unquantized method, whose plain einsum is always all-reduced by
        # the GSPMD partitioner) would silently return an already-reduced
        # output, and any downstream "merged" reduce would double-count it.
        linear_config = getattr(self.quant_method, "linear_config", None)
        if linear_config is not None:
            supports_defer = getattr(self.quant_method,
                                     "supports_defer_all_reduce", False)
            linear_config.defer_all_reduce = (not self.reduce_results
                                              and supports_defer)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        return super().forward(input_)


@ColumnParallelLinear.register_oot
class VllmColumnParallelLinear(ColumnParallelLinear):

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        return super().forward(input_)


@ReplicatedLinear.register_oot
class VllmReplicatedLinear(ReplicatedLinear):

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        return super().forward(x)


@QKVParallelLinear.register_oot
class VllmQKVParallelLinear(QKVParallelLinear):
    """Pre-replicate KV heads when TP > total_num_kv_heads.

    With GQA/MQA and TP > total_num_kv_heads, `total_num_kv_heads` must be
    inflated to TP by replicating each KV head, otherwise a single KV head
    would end up split across multiple devices.

    vLLM's `QKVParallelLinear` already implements this replication, but it
    keys off `get_tensor_model_parallel_world_size()` — which returns 1 on
    TPU, because TPUWorker initializes torch.distributed with world_size=1.
    So we do the inflation here instead: pass `total_num_kv_heads * replicas` 
    to `super().__init__()`, and override `weight_loader[_v2]` to tile each 
    loaded K/V tensor by `replicas`.

    After super init, `self.total_num_kv_heads` is restored to the *real*
    value so vLLM's `_load_fused_module_from_checkpoint` (Phi-3-style fused
    QKV on disk) computes correct on-disk offsets. Other shape attributes
    (`num_kv_heads`, `output_sizes`, allocated buffers) stay inflated.
    """

    def __init__(self,
                 hidden_size: int,
                 head_size: int,
                 total_num_heads: int,
                 total_num_kv_heads: int | None = None,
                 *args,
                 **kwargs):
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        replicas = 1

        vllm_config = get_current_vllm_config()
        mesh = vllm_config.quant_config.mesh if vllm_config.quant_config else None
        if mesh is not None:
            tp = get_mesh_shape_product(vllm_config.quant_config.mesh,
                                        ShardingAxisName.ATTN_HEAD)
        else:
            tp = 1

        if tp > total_num_kv_heads:
            assert tp % total_num_kv_heads == 0, (
                f"tp_size ({tp}) must be divisible by total_num_kv_heads "
                f"({total_num_kv_heads}) for KV-head replication")
            replicas = tp // total_num_kv_heads

        super().__init__(hidden_size, head_size, total_num_heads,
                         total_num_kv_heads * replicas, *args, **kwargs)

        self.total_num_kv_heads = total_num_kv_heads
        self.num_kv_head_replicas = replicas
        self.tp_size = tp

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: str | None = None):
        if loaded_shard_id in ("k", "v") and self.num_kv_head_replicas > 1:
            loaded_weight = self._tile_kv(loaded_weight, param)
        return super().weight_loader(param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self,
                         param,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: str | None = None):
        if loaded_shard_id in ("k", "v") and self.num_kv_head_replicas > 1:
            loaded_weight = self._tile_kv(loaded_weight, param)
        return super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

    def _tile_kv(self, w: torch.Tensor, param) -> torch.Tensor:
        # Repeat each KV head along the param's output dim.
        # To replicate KV head when TP > total_num_kv_heads.
        dim = getattr(param, "output_dim", None)
        if dim is None or w.ndim <= dim:
            return w
        n = self.total_num_kv_heads
        output_size = w.shape[dim]
        if output_size < n or output_size % n != 0:
            return w
        r = self.num_kv_head_replicas
        rows_per_head = output_size // n
        front = tuple(w.shape[:dim])
        back = tuple(w.shape[dim + 1:])
        w = w.reshape(front + (n, rows_per_head) + back)
        w = w.repeat_interleave(r, dim=dim)
        return w.reshape(front + (n * r * rows_per_head, ) + back)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.num_kv_head_replicas == 1:
            return super().forward(x)

        out, bias = super().forward(x)
        out_jax = jax_view(out)

        out_jax = reorder_concatenated_tensor_for_sharding(
            out_jax,
            self.output_sizes,
            self.tp_size,
            dim=-1,
        )
        q_jax, k_jax, v_jax = slice_sharded_tensor_for_concatenation(
            out_jax,
            self.output_sizes,
            self.tp_size,
        )

        mesh = jax.sharding.get_abstract_mesh()
        # Split the `kv_head` mesh axis (size kv_size) into a pair
        # `(kv_head, replica)` of sizes `(kv_size // replicas, replicas)`.
        # The `replica` sub-axis is later replicated via `shard_map` with no
        # data movement, since `_tile_kv` already placed identical KV-head
        # copies on each replica-group of devices when TP > total_num_kv_heads.
        replicas = self.num_kv_head_replicas
        kv_head_axis = None
        for a in reversed(mesh.axis_names):
            if a in ShardingAxisName.ATTN_HEAD and get_mesh_shape_product(
                    mesh, a) >= replicas:
                kv_head_axis = a
                break

        if kv_head_axis is None:
            raise ValueError(
                f"Cannot find a mesh axis to split for KV-head replication: "
                f"no axis in {mesh.axis_names} contains "
                f"{ShardingAxisName.ATTN_HEAD} and has size >= {replicas}")

        replica_axis = 'replica'
        data_axis = ShardingAxisName.ATTN_DATA
        head_axis = ShardingAxisName.ATTN_HEAD

        i = mesh.axis_names.index(kv_head_axis)
        kv_size = mesh.axis_sizes[i]
        kv_type = mesh.axis_types[i]
        new_mesh = jax.sharding.AbstractMesh(
            mesh.axis_sizes[:i] + (kv_size // replicas, replicas) +
            mesh.axis_sizes[i + 1:],
            mesh.axis_names[:i] + (kv_head_axis, replica_axis) +
            mesh.axis_names[i + 1:],
            mesh.axis_types[:i] + (kv_type, kv_type) + mesh.axis_types[i + 1:],
        )
        if isinstance(head_axis, tuple):
            in_head_axis = list(head_axis)
            in_head_axis.insert(
                in_head_axis.index(kv_head_axis) + 1, replica_axis)
        else:
            in_head_axis = (head_axis, replica_axis)

        @shard_map(mesh=new_mesh,
                   in_specs=P(data_axis, in_head_axis),
                   out_specs=P(data_axis, head_axis),
                   check_vma=False)
        def _mark_kv_head_replicated(t):
            return t

        with jax.sharding.use_abstract_mesh(new_mesh):
            k_jax = _mark_kv_head_replicated(k_jax)
            v_jax = _mark_kv_head_replicated(v_jax)

        out_jax = jnp.concatenate([q_jax, k_jax, v_jax], axis=-1)
        return torch_view(out_jax), bias
