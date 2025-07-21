import tempfile

import jax
import pytest
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

from tpu_commons.models.vllm.jax_merged_column_parallel_linear import \
    JaxMergedColumnParallelLinear

P = PartitionSpec


@pytest.fixture(scope="module", autouse=True)
def setup_torchax():
    """Enable torchax globally before all tests, disable after all tests."""
    torchax.enable_globally()
    yield
    torchax.disable_globally()


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
def test_jax_merged_column_parallel_linear(bias, mesh):
    dtype = torch.bfloat16

    merged_column_linear = MergedColumnParallelLinear(
        input_size=4096,
        output_sizes=[14336] * 2,
        bias=bias,
        params_dtype=dtype,
        return_bias=False,
    )
    merged_column_linear.weight.data = torch.rand_like(
        merged_column_linear.weight.data) / 10
    if bias:
        merged_column_linear.bias.data = torch.rand_like(
            merged_column_linear.bias.data)
    merged_column_linear = merged_column_linear.to('cpu')

    jax_merged_column_linear = JaxMergedColumnParallelLinear(
        merged_column_linear, mesh=mesh)

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')
    output = merged_column_linear(input_tensor)

    jax_input_tensor = torch_view(t2j(input_tensor))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    jax_output = jax_merged_column_linear(jax_input_tensor)
    # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
    jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)
