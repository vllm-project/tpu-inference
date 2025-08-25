import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.linear import QKVParallelLinear

from tpu_commons.models.vllm.jax_merged_column_parallel_linear_core import \
    JaxMergedColumnParallelLinearCore


class JaxQKVParallelLinear(JaxMergedColumnParallelLinearCore):

    def __init__(self, qkv_linear: torch.nn.Module, mesh: Mesh,
                 fuse_matmuls: bool, enable_sequence_parallelism: bool):
        assert isinstance(qkv_linear, QKVParallelLinear)
        super().__init__(
            qkv_linear,
            mesh,
            "JaxQKVParallelLinear",
            fuse_matmuls=fuse_matmuls,
            enable_sequence_parallelism=enable_sequence_parallelism)
