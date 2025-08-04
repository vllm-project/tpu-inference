import tempfile
from unittest.mock import patch

import jax
import pytest
import torch
import torchax
import utils as test_utils
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsW8A8Int8

from tpu_commons.models.vllm.jax_row_parallel_linear import \
    JaxRowParallelLinear

P = PartitionSpec


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # RowParallelLinear needs dist env to be initialized.
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


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh()])
def test_jax_row_parallel_linear(bias, mesh):
    dtype = torch.bfloat16

    row_linear = RowParallelLinear(
        input_size=4096,
        output_size=8192,
        bias=bias,
        params_dtype=dtype,
        return_bias=False,
    )

    row_linear.weight.data = torch.rand_like(row_linear.weight.data) / 10
    if bias:
        row_linear.bias.data = torch.rand_like(row_linear.bias.data)
    row_linear = row_linear.to('cpu')
    row_linear.quant_method.process_weights_after_loading(row_linear)

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')
    output = row_linear(input_tensor).to(dtype)

    # Set jax default device to workaround a layout bug in JAX 0.7.0 and earlier
    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        jax_row_linear = JaxRowParallelLinear(row_linear, mesh=mesh)
        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        jax_input_tensor.apply_jax_(jax.device_put,
                                    NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        jax_output = jax_row_linear(jax_input_tensor)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh()])
def test_jax_row_parallel_linear_w8a8_int8(bias, mesh):
    dtype = torch.bfloat16

    row_linear = RowParallelLinear(
        input_size=4096,
        output_size=8192,
        bias=bias,
        params_dtype=dtype,
        return_bias=False,
        quant_config=test_utils.gen_vllm_w8a8_int8_config(),
    )

    # Assert we're testing the right code path when quant config is set.
    assert isinstance(row_linear.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(row_linear.scheme, CompressedTensorsW8A8Int8)

    row_linear.weight.data = torch.randint_like(row_linear.weight.data,
                                                low=-128,
                                                high=128)
    row_linear.weight_scale.data = torch.rand_like(
        row_linear.weight_scale.data) / 10
    if bias:
        row_linear.bias.data = torch.rand_like(row_linear.bias.data)
    row_linear = row_linear.to('cpu')
    row_linear.quant_method.process_weights_after_loading(row_linear)

    input_tensor = torch.rand(16, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')
    # Overwrite the torch_xla kernel with a reference implementation, as it's difficult to call torch_xla in tpu_commons and we want to run the ref result on CPU.
    with patch(
            "vllm.model_executor.layers.quantization.kernels.scaled_mm.xla.XLAScaledMMLinearKernel.apply_weights",
            new=test_utils.quantized_matmul_ref):
        output = row_linear(input_tensor).to(dtype)

    # Set jax default device to workaround a layout bug in JAX 0.7.0 and earlier
    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        jax_row_linear = JaxRowParallelLinear(row_linear, mesh=mesh)
        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        jax_input_tensor.apply_jax_(jax.device_put,
                                    NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        jax_output = jax_row_linear(jax_input_tensor)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output, atol=5, rtol=0.1)
