import tempfile

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
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.model_loader import get_model as vllm_get_model

from tpu_commons.models.vllm.quantization import get_tpu_quantization_config
from tpu_commons.models.vllm.quantization.unquantized import (
    JaxUnquantizedConfig, JaxUnquantizedLinearMethod)

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


@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
def test_quant_override(mesh):

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    assert isinstance(quant_config, JaxUnquantizedConfig)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


@pytest.mark.parametrize("model", ["Qwen/Qwen2-1.5B-Instruct"])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
def test_loading_model(model, mesh):
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config, mesh)
    vllm_config.device_config.device = "cpu"

    vllm_model = vllm_get_model(vllm_config=vllm_config)
    layers = test_utils.find_all_layer_type(vllm_model, LinearBase)
    for layer in layers:
        assert isinstance(layer.quant_config, JaxUnquantizedConfig)
        assert isinstance(layer.quant_method, JaxUnquantizedLinearMethod)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_jax_row_parallel_linear(bias, mesh, enable_sp):
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sequence_parallelism = enable_sp

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    with set_current_vllm_config(vllm_config):
        row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    weight_data = torch.rand_like(row_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(row_linear.bias.data)

    row_linear.weight.data = weight_data
    if bias:
        row_linear.bias.data = bias_data
    row_linear = row_linear.to('cpu')
    row_linear.quant_method.process_weights_after_loading(row_linear)
    output = row_linear(input_tensor).to(dtype)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_row_linear = RowParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    jax_row_linear.weight.data = weight_data
    if bias:
        jax_row_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_row_linear.quant_method,
                          JaxUnquantizedLinearMethod)
        jax_row_linear.quant_method.process_weights_after_loading(
            jax_row_linear)
        jax_output = jax_row_linear(jax_input_tensor)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_jax_column_parallel_linear(bias, mesh, enable_sp):
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sequence_parallelism = enable_sp

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    with set_current_vllm_config(vllm_config):
        column_linear = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    weight_data = torch.rand_like(column_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(column_linear.bias.data)

    column_linear.weight.data = weight_data
    if bias:
        column_linear.bias.data = bias_data
    column_linear = column_linear.to('cpu')
    column_linear.quant_method.process_weights_after_loading(column_linear)
    output = column_linear(input_tensor).to(dtype)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_column_linear = ColumnParallelLinear(
            input_size=4096,
            output_size=8192,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    jax_column_linear.weight.data = weight_data
    if bias:
        jax_column_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_column_linear.quant_method,
                          JaxUnquantizedLinearMethod)
        jax_column_linear.quant_method.process_weights_after_loading(
            jax_column_linear)
        jax_output = jax_column_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
def test_jax_qkv_parallel_linear(bias, mesh, enable_sp, fuse_matmuls):
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sequence_parallelism = enable_sp

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    with set_current_vllm_config(vllm_config):
        qkv_linear = QKVParallelLinear(
            hidden_size=4096,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    weight_data = torch.rand_like(qkv_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(qkv_linear.bias.data)

    qkv_linear.weight.data = weight_data
    if bias:
        qkv_linear.bias.data = bias_data
    qkv_linear = qkv_linear.to('cpu')
    qkv_linear.quant_method.process_weights_after_loading(qkv_linear)
    output = qkv_linear(input_tensor).to(dtype)

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_qkv_linear = QKVParallelLinear(
            hidden_size=4096,
            head_size=128,
            total_num_heads=32,
            total_num_kv_heads=8,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        jax_qkv_linear.quant_method.fuse_matmuls = fuse_matmuls

    jax_qkv_linear.weight.data = weight_data
    if bias:
        jax_qkv_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_qkv_linear.quant_method,
                          JaxUnquantizedLinearMethod)
        jax_qkv_linear.quant_method.process_weights_after_loading(
            jax_qkv_linear)
        jax_output = jax_qkv_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(jax.local_device_count())
])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_jax_merged_column_parallel_linear(bias, mesh, fuse_matmuls,
                                           enable_sp):
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sequence_parallelism = enable_sp

    input_tensor = torch.rand(10, 4096, dtype=dtype) / 10
    input_tensor = input_tensor.to('cpu')

    # Call vLLM code
    with set_current_vllm_config(vllm_config):
        merged_column_linear = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[14336] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
        )

    weight_data = torch.rand_like(merged_column_linear.weight.data) / 10
    if bias:
        bias_data = torch.rand_like(merged_column_linear.bias.data)

    merged_column_linear.weight.data = weight_data
    if bias:
        merged_column_linear.bias.data = bias_data
    merged_column_linear = merged_column_linear.to('cpu')
    merged_column_linear.quant_method.process_weights_after_loading(
        merged_column_linear)
    output = merged_column_linear(input_tensor).to(dtype)

    # Call tpu_commons code
    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        jax_merged_column_linear = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[14336] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        jax_merged_column_linear.quant_method.fuse_matmuls = fuse_matmuls

    jax_merged_column_linear.weight.data = weight_data
    if bias:
        jax_merged_column_linear.bias.data = bias_data

    jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
    jax_input_tensor.apply_jax_(jax.device_put,
                                NamedSharding(mesh, P(None, None)))
    with torchax.default_env():
        assert isinstance(jax_merged_column_linear.quant_method,
                          JaxUnquantizedLinearMethod)
        jax_merged_column_linear.quant_method.process_weights_after_loading(
            jax_merged_column_linear)
        jax_output = jax_merged_column_linear(jax_input_tensor)
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    torch.testing.assert_close(output, jax_output)
