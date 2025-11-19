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

from tpu_inference.layers.vllm.linear_common import \
    get_model_matmul_fusion_assignment
from tpu_inference.utils import TPU_SECOND_LAST_MINOR

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
        self.enable_sequence_parallelism = vllm_config.compilation_config.pass_config.enable_sequence_parallelism
        self.input_sharding = None
        self.output_sharding = None

        if isinstance(layer, RowParallelLinear):
            self.weight_sharding = P(None, "model")
            if self.enable_sequence_parallelism:
                self.output_sharding = P("model", None)
        elif isinstance(layer, ColumnParallelLinear):
            self.weight_sharding = P("model", None)
            if self.enable_sequence_parallelism:
                self.input_sharding = P("model", None)

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
        if isinstance(self.weight_sharding[0], tuple):
            self.n_shards = 1
            for axis in self.weight_sharding[0]:
                self.n_shards *= self.mesh.shape.get(axis, 1)
        else:
            self.n_shards = self.mesh.shape.get(self.weight_sharding[0], 1)

    def get_input_sharding(self, x: torchax.tensor.Tensor):
        if self.enable_sequence_parallelism:
            token_num = x.shape[0]
            # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
            if token_num // self.mesh.shape["model"] >= TPU_SECOND_LAST_MINOR:
                return self.input_sharding
            else:
                return None
        return self.input_sharding

    def get_output_sharding(self, x: torchax.tensor.Tensor):
        if self.enable_sequence_parallelism:
            token_num = x.shape[0]
            # NOTE(chengjiyao): make sure the sharded token_num is larger than TPU_SECOND_LAST_MINOR
            if token_num // self.mesh.shape["model"] >= TPU_SECOND_LAST_MINOR:
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
