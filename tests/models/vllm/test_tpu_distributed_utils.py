# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from unittest.mock import patch

import jax
import pytest
import torch
import torchax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import QKVParallelLinear

from tpu_commons.distributed.tpu_distributed_utils import (
    XlaQKVParallelLinear, create_torchax_kv_cache,
    create_torchax_tensor_with_partition_spec)


@pytest.fixture(scope="module", autouse=True)
def setup_torchax():
    """Enable torchax globally before all tests, disable after all tests."""
    torchax.enable_globally()
    yield
    torchax.disable_globally()


@pytest.fixture(autouse=True)
def set_tpu_backend_env():
    """Set TPU_BACKEND_TYPE=torchax for all tests in this module."""
    with patch.dict(os.environ, {"TPU_BACKEND_TYPE": "torchax"}):
        yield


@pytest.mark.parametrize("mesh,partition_spec", [
    (None, None),
    (Mesh(jax.devices(), axis_names=('x', )), P(None, None, 'x', None)),
])
def test_create_torchax_kv_cache(mesh, partition_spec):
    kv_cache_shape = (1024, 16, 16, 128)
    dtype = torch.bfloat16
    tensor = create_torchax_kv_cache(kv_cache_shape, dtype, mesh,
                                     partition_spec)

    # Check the properties of the created tensor
    assert isinstance(tensor, torchax.tensor.Tensor)
    assert tensor.shape == kv_cache_shape
    assert tensor.dtype == dtype


@pytest.mark.parametrize("mesh,partition_spec", [
    (None, None),
    (Mesh(jax.devices(), axis_names=('x', )), P('x', )),
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_create_torchax_tensor_with_partition_spec(mesh, partition_spec,
                                                   dtype):
    shape = (1024, 1024)
    torch_t = torch.empty(shape, dtype=dtype)

    # Create a Torchax tensor with the specified partition spec
    tensor = create_torchax_tensor_with_partition_spec(torch_t, mesh,
                                                       partition_spec)

    # Check the properties of the created tensor
    assert isinstance(tensor, torchax.tensor.Tensor)
    assert tensor.shape == shape
    assert tensor.dtype == dtype


def test_create_torchax_tensor_with_partition_spec_value_error():
    """Test that ValueError is raised when mesh is None but partition_spec is not None/empty."""
    shape = (512, 256)
    dtype = torch.float32
    torch_t = torch.randn(shape, dtype=dtype)

    # Test with mesh=None and partition_spec not None
    with pytest.raises(
            ValueError,
            match="If mesh is None, partition_spec must also be None or empty"
    ):
        create_torchax_tensor_with_partition_spec(torch_t,
                                                  mesh=None,
                                                  partition_spec=('x', ))


# Test for XLA parallel layers


@pytest.fixture
def setup_environment():
    """Setup distributed environment for tests that need it."""
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
        yield


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("return_bias", [True, False])
@pytest.mark.parametrize("skip_bias_add", [True, False])
@pytest.mark.parametrize("use_mesh", [True, False])
@torch.no_grad()
def test_xla_qkv_linear(setup_environment, bias, return_bias, skip_bias_add,
                        use_mesh):
    torch.manual_seed(123)

    if use_mesh:
        mesh = Mesh(jax.devices(), axis_names=('x', ))
    else:
        mesh = None

    qkv_linear = QKVParallelLinear(
        hidden_size=4096,
        head_size=128,
        total_num_heads=32,
        total_num_kv_heads=8,
        bias=bias,
        params_dtype=torch.bfloat16,
        return_bias=return_bias,
        skip_bias_add=skip_bias_add,
    )

    qkv_linear.weight.data = torch.rand_like(qkv_linear.weight.data) / 10
    if bias:
        qkv_linear.bias.data = torch.rand_like(qkv_linear.bias.data)

    xla_qkv_linear = XlaQKVParallelLinear(qkv_linear, mesh=mesh)

    if mesh is None:
        xla_qkv_linear = xla_qkv_linear.to('jax')

    qkv_linear = qkv_linear.to('jax')
    input_tensor = torch.rand(10, 4096, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('jax')

    output = qkv_linear(input_tensor)
    if mesh is not None:
        input_tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
    xla_output = xla_qkv_linear(input_tensor)
    if return_bias:
        assert isinstance(xla_output, tuple) and isinstance(output, tuple)
        assert len(xla_output) == len(output) == 2
        assert torch.allclose(output[0].cpu(), xla_output[0].cpu())
        if (not skip_bias_add) or (not bias):
            assert output[1] is None and output[1] == xla_output[1]
        else:
            print("output[1]:", output[1])
            print("xla_output[1]:", xla_output[1])
            assert torch.allclose(output[1].cpu(), xla_output[1].cpu())
    else:
        assert torch.allclose(output.cpu(), xla_output.cpu())
