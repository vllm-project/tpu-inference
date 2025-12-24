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
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEConfig
# yapf: disable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.linear_common import \
    get_model_matmul_fusion_assignment
from tpu_inference.utils import TPU_SECOND_LAST_MINOR, get_mesh_shape_product

# yapf: enable

P = PartitionSpec

logger = init_logger(__name__)


class JaxCommonLinearConfig:

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh, layer: LinearBase):
        assert isinstance(layer, LinearBase)

        self.mesh = mesh
        self.output_sizes = [layer.output_size]
        self.weight_sharding = P(None, None)
        self.fuse_matmuls = True
        self.enable_sp = vllm_config.compilation_config.pass_config.enable_sp
        self.input_sharding = None
        self.output_sharding = None

        self.tp_size = get_mesh_shape_product(self.mesh,
                                              ShardingAxisName.MLP_TENSOR)

        if isinstance(layer, RowParallelLinear):
            self.weight_sharding = P(None, ShardingAxisName.ATTN_HEAD)
            if self.enable_sp:
                self.output_sharding = P(ShardingAxisName.MLP_TENSOR, None)
        elif isinstance(layer, ColumnParallelLinear):
            self.weight_sharding = P(ShardingAxisName.ATTN_HEAD, None)

            if self.enable_sp:
                self.input_sharding = P(ShardingAxisName.MLP_TENSOR, None)

            if isinstance(layer, MergedColumnParallelLinear) or isinstance(
                    layer, QKVParallelLinear):
                self.output_sizes = layer.output_sizes

            self.fuse_matmuls = get_model_matmul_fusion_assignment(
                vllm_config.model_config.model,
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.parallel_config.tensor_parallel_size,
                layer._get_name())
        elif isinstance(layer, ReplicatedLinear):
            self.weight_sharding = P(None, None)
        else:
            logger.warning(
                "Unsupported linear layer type of %s. Can potentially yield "
                " bad performance.", type(layer))

        self.bias_sharding = P(self.weight_sharding[0])
        self.n_shards = get_mesh_shape_product(self.mesh,
                                               self.weight_sharding[0])

    def get_input_sharding(self, x: torchax.tensor.Tensor):
        if self.enable_sp:
            token_num = x.shape[0]
            # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
            if token_num // self.tp_size >= TPU_SECOND_LAST_MINOR:
                return self.input_sharding
            else:
                return None
        return self.input_sharding

    def get_output_sharding(self, x: torchax.tensor.Tensor):
        if self.enable_sp:
            token_num = x.shape[0]
            # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
            if token_num // self.tp_size >= TPU_SECOND_LAST_MINOR:
                return self.output_sharding
            else:
                return None
        return self.output_sharding


class JaxCommonConfig:
    vllm_config: VllmConfig
    mesh: Mesh

    @classmethod
    def set_configs(cls, vllm_config: VllmConfig, mesh: Mesh):
        cls.vllm_config = vllm_config
        cls.mesh = mesh

    def get_linear_config(self, layer: LinearBase) -> JaxCommonLinearConfig:
        assert isinstance(layer, LinearBase)
        return JaxCommonLinearConfig(self.vllm_config, self.mesh, layer)

    def get_moe_config(self, layer: FusedMoE) -> FusedMoEConfig:
        assert isinstance(layer, FusedMoE)
        moe_config = layer.moe_config
        use_ep = self.vllm_config.parallel_config.enable_expert_parallel
        moe_config.moe_parallel_config.use_ep = use_ep
        return moe_config
