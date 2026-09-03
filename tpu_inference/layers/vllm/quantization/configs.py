# Copyright 2025 Google LLC
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

import torchax
from jax.sharding import Mesh, PartitionSpec
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig, RoutedExperts
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)

from tpu_inference.layers.common.process_weights.linear_weights import \
    get_model_matmul_fusion_assignment
from tpu_inference.layers.common.quantization.configs import QuantLinearConfig
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.utils import TPU_SECOND_LAST_MINOR, get_mesh_shape_product

# yapf: enable

P = PartitionSpec

logger = init_logger(__name__)


class VllmQuantLinearConfig(QuantLinearConfig):

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh, layer: LinearBase):
        assert isinstance(layer, LinearBase)

        # Softmax attention keeps its tokens split over pcp (ATTN_DATA) and
        # shards heads over ATTN_HEAD; every other linear — GDN projections,
        # dense MLPs, shared experts, routers — sees tokens replicated over
        # pcp (DENSE_DATA) and shards over DENSE_TENSOR, which also spans pcp.
        # A reduced row-parallel output is always DENSE_DATA (an attention
        # out-projection is gathered over pcp back onto the residual stream);
        # a deferred one is the partial stack [n_shards, tokens, out].
        if self._is_attention(layer):
            tensor_axis, tokens = (ShardingAxisName.ATTN_HEAD,
                                   ShardingAxisName.ATTN_DATA)
        else:
            tensor_axis, tokens = (ShardingAxisName.DENSE_TENSOR,
                                   ShardingAxisName.DENSE_DATA)

        deferred_output_sharding = None
        pin_input_replicated = False
        if isinstance(layer, RowParallelLinear):
            weight_sharding = P(tensor_axis, None)
            input_sharding = P(tokens, tensor_axis)
            output_sharding = P(ShardingAxisName.DENSE_DATA, None)
            deferred_output_sharding = P(tensor_axis, tokens, None)
        elif isinstance(layer, ColumnParallelLinear):
            weight_sharding = P(None, tensor_axis)
            input_sharding = P(tokens, None)
            output_sharding = P(tokens, tensor_axis)
            pin_input_replicated = self._is_attention(layer)
        elif isinstance(layer, ReplicatedLinear):
            weight_sharding = P(None, None)
            input_sharding = P(ShardingAxisName.DENSE_DATA, None)
            output_sharding = P(ShardingAxisName.DENSE_DATA, None)
        else:
            raise NotImplementedError(
                f"Unsupported linear layer type {type(layer)}")

        super().__init__(
            enable_sp=vllm_config.compilation_config.pass_config.enable_sp,
            output_sizes=[layer.output_size],
            weight_sharding=weight_sharding,
            input_sharding=input_sharding,
            output_sharding=output_sharding,
            deferred_output_sharding=deferred_output_sharding)
        self.pin_input_replicated = pin_input_replicated
        self.mesh = mesh
        self.tp_size = get_mesh_shape_product(self.mesh,
                                              ShardingAxisName.MLP_TENSOR)

        if isinstance(layer, RowParallelLinear):
            if self.enable_sp:
                self.sp_output_sharding = P(ShardingAxisName.MLP_TENSOR, None)
        elif isinstance(layer, ColumnParallelLinear):
            if self.enable_sp:
                self.sp_input_sharding = P(ShardingAxisName.MLP_TENSOR, None)

            if isinstance(layer, MergedColumnParallelLinear) or isinstance(
                    layer, QKVParallelLinear):
                self.output_sizes = layer.output_sizes

            self.fuse_matmuls = get_model_matmul_fusion_assignment(
                vllm_config.model_config.model,
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.parallel_config.tensor_parallel_size,
                layer._get_name())

        if isinstance(layer, QKVParallelLinear):
            self.num_proj = 3
        elif isinstance(layer, MergedColumnParallelLinear):
            self.num_proj = len(layer.output_sizes)
        else:
            self.num_proj = 1

        self.bias_sharding = P(self.weight_sharding[1])
        self.n_shards = get_mesh_shape_product(self.mesh,
                                               self.weight_sharding[1])

    @staticmethod
    def _is_attention(layer: LinearBase) -> bool:
        """Softmax-attention projection (q/k/v, o_proj)? o_proj and an MLP
        down_proj are both RowParallelLinear, so attention is told apart by
        the module path (self_attn.*, attn.*, attention.*); GDN's
        linear_attn.* / linear_attention.* is not attention in this sense."""
        if isinstance(layer, QKVParallelLinear):
            return True
        prefix = getattr(layer, "prefix", "")
        return (("attn" in prefix or "attention" in prefix)
                and "linear_attn" not in prefix
                and "linear_attention" not in prefix)

    def get_input_sharding(self, x: torchax.tensor.Tensor):
        if not self.enable_sp:
            return None
        token_num = x.shape[0]
        # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
        if token_num // self.tp_size < TPU_SECOND_LAST_MINOR:
            return None
        return self.sp_input_sharding

    def get_output_sharding(self, x: torchax.tensor.Tensor):
        if self.enable_sp:
            token_num = x.shape[0]
            # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
            if token_num // self.tp_size < TPU_SECOND_LAST_MINOR:
                return None
        return self.sp_output_sharding


class VllmQuantConfig:
    vllm_config: VllmConfig
    mesh: Mesh

    @classmethod
    def set_configs(cls, vllm_config: VllmConfig, mesh: Mesh):
        cls.vllm_config = vllm_config
        cls.mesh = mesh

    def get_linear_config(self, layer: LinearBase) -> VllmQuantLinearConfig:
        assert isinstance(layer, LinearBase)
        return VllmQuantLinearConfig(self.vllm_config, self.mesh, layer)

    def get_moe_config(self, layer: RoutedExperts) -> FusedMoEConfig:
        assert isinstance(layer, RoutedExperts)
        moe_config = layer.moe_config
        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        moe_config.moe_parallel_config.use_ep = use_ep
        return moe_config
