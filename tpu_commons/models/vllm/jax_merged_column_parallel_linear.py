import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

from tpu_commons.models.vllm.jax_merged_column_parallel_linear_core import \
    JaxMergedColumnParallelLinearCore


class JaxMergedColumnParallelLinear(JaxMergedColumnParallelLinearCore):

    def __init__(self, merged_col_parallel_linear: torch.nn.Module, mesh: Mesh,
                 fuse_matmuls: bool):
        assert isinstance(merged_col_parallel_linear,
                          MergedColumnParallelLinear)
        super().__init__(merged_col_parallel_linear,
                         mesh,
                         "JaxMergedColumnParallelLinear",
                         fuse_matmuls=fuse_matmuls)
