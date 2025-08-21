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
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsW8A8Int8

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


@pytest.mark.skip(
    reason=
    "b/440248045. The failure is not caused by Rpav3. Will fix in another change."
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh()])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
def test_jax_qkv_parallel_linear(bias, mesh, fuse_matmuls):
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
    qkv_linear.quant_method.process_weights_after_loading(qkv_linear)

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')
    output = qkv_linear(input_tensor).to(dtype)

    # Set jax default device to workaround a layout bug in JAX 0.7.0 and earlier
    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        jax_qkv_linear = JaxQKVParallelLinear(qkv_linear, mesh, fuse_matmuls)
        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        jax_input_tensor.apply_jax_(jax.device_put,
                                    NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        jax_output = jax_qkv_linear(jax_input_tensor)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.skip(
    reason=
    "b/440248045. The failure is not caused by Rpav3. Will fix in another change."
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh()])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
def test_jax_qkv_parallel_linear_w8a8_int8(bias, mesh, fuse_matmuls):
    dtype = torch.bfloat16

    qkv_linear = QKVParallelLinear(
        hidden_size=4096,
        head_size=128,
        total_num_heads=32,
        total_num_kv_heads=8,
        bias=bias,
        params_dtype=dtype,
        return_bias=False,
        quant_config=test_utils.gen_vllm_w8a8_int8_config(),
    )

    # Assert we're testing the right code path when quant config is set.
    assert isinstance(qkv_linear.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(qkv_linear.scheme, CompressedTensorsW8A8Int8)

    qkv_linear.weight.data = torch.randint_like(qkv_linear.weight.data,
                                                low=-128,
                                                high=128)
    qkv_linear.weight_scale.data = torch.rand_like(
        qkv_linear.weight_scale.data) / 10
    if bias:
        qkv_linear.bias.data = torch.rand_like(qkv_linear.bias.data)
    qkv_linear = qkv_linear.to('cpu')
    qkv_linear.quant_method.process_weights_after_loading(qkv_linear)

    input_tensor = torch.rand(16, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')
    # Overwrite the torch_xla kernel with a reference implementation, as it's difficult to call torch_xla in tpu_commons and we want to run the ref result on CPU.
    with patch(
            "vllm.model_executor.layers.quantization.kernels.scaled_mm.xla.XLAScaledMMLinearKernel.apply_weights",
            new=test_utils.quantized_matmul_ref):
        output = qkv_linear(input_tensor).to(dtype)

    # Set jax default device to workaround a layout bug in JAX 0.7.0 and earlier
    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        jax_qkv_linear = JaxQKVParallelLinear(qkv_linear, mesh, fuse_matmuls)
        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        jax_input_tensor.apply_jax_(jax.device_put,
                                    NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        jax_output = jax_qkv_linear(jax_input_tensor)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output, atol=5, rtol=0.1)
