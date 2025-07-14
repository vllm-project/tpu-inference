import tempfile

import jax
import pytest
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import QKVParallelLinear

from tpu_commons.models.vllm.jax_qkv_parallel_linear import \
    JaxQKVParallelLinear

P = PartitionSpec


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # QKVParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )

    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)


def _get_spmd_mesh():
    axis_names = ("data", "model")
    mesh_shape = (1, len(jax.devices()))
    return jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [_get_spmd_mesh()])
def test_jax_qkv_parallel_linear(bias, mesh):
    dtype = torch.bfloat16

    qkv_linear = QKVParallelLinear(
        hidden_size=4096,
        head_size=128,
        total_num_heads=32,
        total_num_kv_heads=8,
        bias=bias,
        params_dtype=dtype,
        return_bias=False,
    )
    qkv_linear.weight.data = torch.rand_like(qkv_linear.weight.data) / 10
    if bias:
        qkv_linear.bias.data = torch.rand_like(qkv_linear.bias.data)
    qkv_linear = qkv_linear.to('cpu')

    jax_qkv_linear = JaxQKVParallelLinear(qkv_linear, mesh=mesh)

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')
    output = qkv_linear(input_tensor)

    jax_input_tensor = torch_view(t2j(input_tensor))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    jax_output = jax_qkv_linear(jax_input_tensor)
    # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
    jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)
