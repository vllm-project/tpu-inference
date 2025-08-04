import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.linear import QKVParallelLinear

from tpu_commons.models.vllm.jax_merged_column_parallel_linear_core import \
    JaxMergedColumnParallelLinearCore


class JaxQKVParallelLinear(JaxMergedColumnParallelLinearCore):

    def __init__(self, qkv_linear: torch.nn.Module, mesh: Mesh):
        assert isinstance(qkv_linear, QKVParallelLinear)
        super().__init__(qkv_linear, mesh, "JaxQKVParallelLinear")
        self.bias = qkv_linear.bias
        self.quant_method = qkv_linear.quant_method  # otherwise, it fails with an error https://paste.googleplex.com/5262933009104896
        self.weight = qkv_linear.weight
        self.gather_output = qkv_linear.gather_output
